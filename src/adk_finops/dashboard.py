# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.

"""Near-Live FinOps Web Dashboard for adk-finops.

Provides a zero-extra-dependency FastAPI + Uvicorn live dashboard that aggregates:
1. Live in-memory CostTracker sessions/turns (real-time mid-run visibility)
2. Local timestamped JSONL and CSV telemetry files (`logs/**/*.jsonl`, `logs/**/*.csv`)
3. Optional BigQuery table records (`ADK_FINOPS_BIGQUERY_TABLE`)

Can be launched either:
- Automatically in the background via `FinOpsCostPlugin(enable_dashboard=True, dashboard_port=8088)`
- Standalone via CLI: `adk-finops dashboard --log-dir logs --port 8088`
  or `python -m adk_finops.dashboard --log-dir logs --port 8088`
"""

from __future__ import annotations

import argparse
import csv
from datetime import datetime, timezone
import json
import logging
import os
from pathlib import Path
import socket
import threading
import time
from typing import Any

from .tracker import CostTracker

logger = logging.getLogger("adk_finops.dashboard")

_BQ_CACHE: dict[str, Any] = {"timestamp": 0.0, "rows": [], "error": None}
_BQ_CACHE_TTL_SEC = 15.0
_INGESTED_ROWS: dict[tuple[Any, ...], dict[str, Any]] = {}
_DASHBOARD_THREAD: threading.Thread | None = None
_DASHBOARD_URL: str | None = None


def _normalize_row(raw: dict[str, Any], source: str = "local") -> dict[str, Any]:
    """Normalizes a raw row from JSONL, CSV, in-memory CostTracker, or BigQuery."""
    tags_raw = raw.get("tags")
    tags_dict: dict[str, Any] = {}
    if isinstance(tags_raw, dict):
        tags_dict = tags_raw
    elif isinstance(tags_raw, str) and tags_raw.strip():
        try:
            tags_dict = json.loads(tags_raw)
        except Exception:
            tags_dict = {}

    status = raw.get("status") or tags_dict.get("status") or "success"
    status = str(status).strip().lower()

    is_failure_raw = raw.get("is_failure")
    if is_failure_raw is None:
        is_failure_raw = tags_dict.get("is_failure")
    if isinstance(is_failure_raw, str):
        is_failure = is_failure_raw.strip().lower() in ("true", "1", "yes")
    elif is_failure_raw is not None:
        is_failure = bool(is_failure_raw)
    else:
        is_failure = status in ("failed", "failure", "error", "aborted", "budget_exceeded", "timeout")

    error_msg = raw.get("error") or tags_dict.get("error") or ""

    def _to_int(val: Any) -> int:
        try:
            return int(float(val or 0))
        except Exception:
            return 0

    def _to_float(val: Any) -> float:
        try:
            return float(val or 0.0)
        except Exception:
            return 0.0

    agent_name = raw.get("agent_name")
    if isinstance(agent_name, str) and not agent_name.strip():
        agent_name = None

    model_name = raw.get("model_name")
    if isinstance(model_name, str) and not model_name.strip():
        model_name = None

    root_agent_name = raw.get("root_agent_name") or tags_dict.get("root_agent_name")
    if isinstance(root_agent_name, str) and not root_agent_name.strip():
        root_agent_name = None

    parent_agent_name = raw.get("parent_agent_name") or tags_dict.get("parent_agent_name")
    if isinstance(parent_agent_name, str) and not parent_agent_name.strip():
        parent_agent_name = None

    breakdown_by_agent_raw = raw.get("breakdown_by_agent")
    breakdown_by_agent: dict[str, Any] = {}
    if isinstance(breakdown_by_agent_raw, dict):
        breakdown_by_agent = breakdown_by_agent_raw
    elif isinstance(breakdown_by_agent_raw, str) and breakdown_by_agent_raw.strip():
        try:
            parsed_bba = json.loads(breakdown_by_agent_raw)
            if isinstance(parsed_bba, dict):
                breakdown_by_agent = parsed_bba
        except Exception:
            breakdown_by_agent = {}

    return {
        "timestamp": str(raw.get("timestamp") or datetime.now(timezone.utc).isoformat()),
        "session_id": str(raw.get("session_id") or "default_session"),
        "turn_id": str(raw.get("turn_id") or ""),
        "scope": str(raw.get("scope") or "session").lower(),
        "agent_name": agent_name,
        "root_agent_name": root_agent_name,
        "parent_agent_name": parent_agent_name,
        "agent_role": raw.get("agent_role"),
        "model_name": model_name,
        "status": status,
        "is_failure": is_failure,
        "error": str(error_msg) if error_msg else None,
        "prompt_tokens": _to_int(raw.get("prompt_tokens")),
        "completion_tokens": _to_int(raw.get("completion_tokens")),
        "thoughts_tokens": _to_int(raw.get("thoughts_tokens")),
        "cached_tokens": _to_int(raw.get("cached_tokens")),
        "total_tokens": _to_int(raw.get("total_tokens")),
        "llm_cost_usd": _to_float(raw.get("llm_cost_usd")),
        "tool_cost_usd": _to_float(raw.get("tool_cost_usd")),
        "total_cost_usd": _to_float(raw.get("total_cost_usd")),
        "gross_cost_usd": _to_float(raw.get("gross_cost_usd")),
        "savings_usd": _to_float(raw.get("savings_usd")),
        "savings_pct": _to_float(raw.get("savings_pct")),
        "tool_calls_count": _to_int(raw.get("tool_calls_count")),
        "budget_limit_usd": _to_float(raw.get("budget_limit_usd")) if raw.get("budget_limit_usd") not in (None, "") else None,
        "budget_utilization_pct": _to_float(raw.get("budget_utilization_pct")) if raw.get("budget_utilization_pct") not in (None, "") else None,
        "budget_exceeded": str(raw.get("budget_exceeded", "")).lower() in ("true", "1") if isinstance(raw.get("budget_exceeded"), str) else bool(raw.get("budget_exceeded", False)),
        "_breakdown_by_agent_keys": list(breakdown_by_agent.keys()) if breakdown_by_agent else [],
        "source": source,
    }


def _enrich_hierarchy(rows: list[dict[str, Any]]) -> list[dict[str, Any]]:
    """Infers and enriches Root (Parent) Agent -> Sub-Agent hierarchy across all session rows."""
    by_session_scope: dict[tuple[str, str], list[dict[str, Any]]] = {}
    for r in rows:
        key = (r.get("session_id") or "default_session", r.get("scope") or "session")
        by_session_scope.setdefault(key, []).append(r)

    root_keywords = ("coordinator", "root", "orchestrator", "supervisor", "router", "manager", "lead", "main", "planner")
    enriched: list[dict[str, Any]] = []

    for (sess_id, scope), group in by_session_scope.items():
        session_root: str | None = None
        # 1. Explicit root_agent_name from any row in the group
        for r in group:
            if r.get("root_agent_name"):
                session_root = r["root_agent_name"]
                break
        # 2. First key from breakdown_by_agent on the session aggregate row
        if not session_root:
            for r in group:
                bba_keys = r.get("_breakdown_by_agent_keys") or []
                if bba_keys:
                    for k in bba_keys:
                        if any(kw in k.lower() for kw in root_keywords):
                            session_root = k
                            break
                    if not session_root:
                        session_root = bba_keys[0]
                    break
        # 3. Heuristic match on agent names in the session
        agent_names_in_group = [r["agent_name"] for r in group if r.get("agent_name")]
        if not session_root and agent_names_in_group:
            for a_name in agent_names_in_group:
                if any(kw in a_name.lower() for kw in root_keywords):
                    session_root = a_name
                    break
            if not session_root:
                session_root = agent_names_in_group[0]
        if not session_root:
            session_root = "root_agent"

        has_rollup = any(r.get("agent_name") is None for r in group)
        for r in group:
            r["root_agent_name"] = r.get("root_agent_name") or session_root
            if r.get("agent_name") is None:
                r["agent_role"] = "root_rollup"
                r["parent_agent_name"] = None
            elif r["agent_name"] == r["root_agent_name"]:
                r["agent_role"] = "root_self"
                r["parent_agent_name"] = None
            else:
                r["agent_role"] = "sub_agent"
                r["parent_agent_name"] = r.get("parent_agent_name") or r["root_agent_name"]
            r.pop("_breakdown_by_agent_keys", None)
            enriched.append(r)

        # Synthesize overall parent rollup row if only per-agent rows were present
        if not has_rollup and group:
            first = group[0]
            synth = {
                "timestamp": max(r.get("timestamp", "") for r in group),
                "session_id": sess_id,
                "turn_id": first.get("turn_id", ""),
                "scope": scope,
                "agent_name": None,
                "root_agent_name": session_root,
                "parent_agent_name": None,
                "agent_role": "root_rollup",
                "model_name": None,
                "status": "failed" if any(r.get("is_failure") for r in group) else first.get("status", "success"),
                "is_failure": any(r.get("is_failure") for r in group),
                "error": next((r.get("error") for r in group if r.get("error")), None),
                "prompt_tokens": sum(r.get("prompt_tokens", 0) for r in group),
                "completion_tokens": sum(r.get("completion_tokens", 0) for r in group),
                "thoughts_tokens": sum(r.get("thoughts_tokens", 0) for r in group),
                "cached_tokens": sum(r.get("cached_tokens", 0) for r in group),
                "total_tokens": sum(r.get("total_tokens", 0) for r in group),
                "llm_cost_usd": round(sum(r.get("llm_cost_usd", 0.0) for r in group), 7),
                "tool_cost_usd": round(sum(r.get("tool_cost_usd", 0.0) for r in group), 7),
                "total_cost_usd": round(sum(r.get("total_cost_usd", 0.0) for r in group), 7),
                "gross_cost_usd": round(sum(r.get("gross_cost_usd", 0.0) for r in group), 7),
                "savings_usd": round(sum(r.get("savings_usd", 0.0) for r in group), 7),
                "savings_pct": 0.0,
                "tool_calls_count": sum(r.get("tool_calls_count", 0) for r in group),
                "budget_limit_usd": first.get("budget_limit_usd"),
                "budget_utilization_pct": first.get("budget_utilization_pct"),
                "budget_exceeded": any(r.get("budget_exceeded") for r in group),
                "source": first.get("source", "synthesized"),
            }
            enriched.append(synth)

    return sorted(enriched, key=lambda r: (r.get("timestamp", ""), 1 if r.get("agent_role") == "root_rollup" else 0), reverse=True)


def collect_telemetry_rows(
    log_dir: str | Path | None = "logs",
    include_bigquery: bool = True,
    bigquery_table: str | None = None,
) -> dict[str, Any]:
    """Collects and deduplicates telemetry rows from live memory, HTTP ingest, local files, and BigQuery."""
    deduped: dict[tuple[Any, ...], dict[str, Any]] = {}
    sources_found: list[str] = []

    # 0. Include any rows pushed via POST /api/ingest from remote org agents
    if _INGESTED_ROWS:
        for k, v in _INGESTED_ROWS.items():
            deduped[k] = dict(v)
        sources_found.append("http_ingest")

    # 1. Read local JSONL and CSV files across one or more comma-separated log_dir paths
    log_dirs = [p.strip() for p in str(log_dir).split(",") if p.strip()] if log_dir else []
    for d_str in log_dirs:
        base_path = Path(d_str)
        if base_path.exists():
            jsonl_files = sorted(base_path.rglob("*.jsonl"))
            csv_files = sorted(base_path.rglob("*.csv"))

            for jf in jsonl_files:
                try:
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = _normalize_row(json.loads(line), source=f"jsonl:{jf.parent.name}/{jf.name}")
                        key = (row["session_id"], row["scope"], row["agent_name"] or "__ALL__")
                        deduped[key] = row
                    if "local_jsonl" not in sources_found:
                        sources_found.append("local_jsonl")
                except Exception as e:
                    logger.debug(f"[FinOps Dashboard] Could not read {jf}: {e}")

            for cf in csv_files:
                try:
                    with cf.open("r", encoding="utf-8", newline="") as f:
                        for raw_row in csv.DictReader(f):
                            row = _normalize_row(raw_row, source=f"csv:{cf.parent.name}/{cf.name}")
                            key = (row["session_id"], row["scope"], row["agent_name"] or "__ALL__")
                            if key not in deduped:
                                deduped[key] = row
                    if "local_csv" not in sources_found:
                        sources_found.append("local_csv")
                except Exception as e:
                    logger.debug(f"[FinOps Dashboard] Could not read {cf}: {e}")

    # 2. Read live in-memory CostTracker sessions (real-time mid-run state)
    try:
        active_sessions = getattr(CostTracker, "_sessions", {})
        for sess_id in list(active_sessions.keys()):
            summary = CostTracker.get_summary(run_id=sess_id, session_id=sess_id, pop=False)
            if not summary:
                continue
            sess_info = summary.get("session", summary)
            budget_info = summary.get("budget", {})
            models_used = list(sess_info.get("breakdown_by_model", {}).keys())
            scope_model = models_used[0] if len(models_used) == 1 else None
            sess_root = sess_info.get("root_agent_name")
            agg_row = _normalize_row(
                {
                    "timestamp": datetime.now(timezone.utc).isoformat(),
                    "session_id": sess_id,
                    "scope": "session",
                    "agent_name": None,
                    "root_agent_name": sess_root,
                    "model_name": scope_model,
                    "status": sess_info.get("status", "pending"),
                    "is_failure": sess_info.get("is_failure", False),
                    "error": sess_info.get("error"),
                    **sess_info,
                    "budget_limit_usd": budget_info.get("budget_limit_usd"),
                    "budget_utilization_pct": budget_info.get("utilization_pct"),
                    "budget_exceeded": budget_info.get("exceeded", False),
                },
                source="live_memory",
            )
            deduped[(sess_id, "session", "__ALL__")] = agg_row

            for a_name, a_data in sess_info.get("breakdown_by_agent", {}).items():
                agent_row = _normalize_row(
                    {
                        "timestamp": datetime.now(timezone.utc).isoformat(),
                        "session_id": sess_id,
                        "scope": "session",
                        "agent_name": a_name,
                        "root_agent_name": a_data.get("root_agent_name") or sess_root,
                        "parent_agent_name": a_data.get("parent_agent_name"),
                        "model_name": a_data.get("model_name"),
                        "status": sess_info.get("status", "pending"),
                        "is_failure": sess_info.get("is_failure", False),
                        "error": sess_info.get("error"),
                        **a_data,
                    },
                    source="live_memory",
                )
                deduped[(sess_id, "session", a_name)] = agent_row
            if "live_memory" not in sources_found:
                sources_found.append("live_memory")
    except Exception as e:
        logger.debug(f"[FinOps Dashboard] In-memory inspection error: {e}")

    # 3. Optional BigQuery Live Query (cached for 15s)
    bq_error = None
    resolved_bq = bigquery_table or os.getenv("ADK_FINOPS_BIGQUERY_TABLE")
    if include_bigquery and resolved_bq:
        now = time.time()
        if now - _BQ_CACHE["timestamp"] > _BQ_CACHE_TTL_SEC:
            try:
                from google.cloud import bigquery

                parts = resolved_bq.split(".")
                client = bigquery.Client(project=parts[0] if len(parts) == 3 else None)
                query = f"""
                SELECT
                  CAST(timestamp AS STRING) AS timestamp,
                  session_id,
                  turn_id,
                  scope,
                  agent_name,
                  model_name,
                  COALESCE(JSON_VALUE(tags, '$.status'), 'success') AS status,
                  COALESCE(CAST(JSON_VALUE(tags, '$.is_failure') AS BOOL), FALSE) AS is_failure,
                  JSON_VALUE(tags, '$.error') AS error,
                  JSON_VALUE(tags, '$.root_agent_name') AS root_agent_name,
                  JSON_VALUE(tags, '$.parent_agent_name') AS parent_agent_name,
                  prompt_tokens,
                  completion_tokens,
                  thoughts_tokens,
                  cached_tokens,
                  total_tokens,
                  llm_cost_usd,
                  tool_cost_usd,
                  total_cost_usd,
                  gross_cost_usd,
                  savings_usd,
                  savings_pct,
                  tool_calls_count,
                  budget_limit_usd,
                  budget_utilization_pct,
                  budget_exceeded,
                  breakdown_by_agent,
                  tags
                FROM `{resolved_bq}`
                ORDER BY timestamp DESC
                LIMIT 300
                """
                bq_rows = [dict(r.items()) for r in client.query(query).result()]
                _BQ_CACHE["rows"] = bq_rows
                _BQ_CACHE["timestamp"] = now
                _BQ_CACHE["error"] = None
            except Exception as e:
                _BQ_CACHE["error"] = str(e)
        bq_error = _BQ_CACHE["error"]
        for raw_bq in _BQ_CACHE["rows"]:
            row = _normalize_row(raw_bq, source="bigquery")
            key = (row["session_id"], row["scope"], row["agent_name"] or "__ALL__")
            if key not in deduped:
                deduped[key] = row
        if _BQ_CACHE["rows"] and "bigquery" not in sources_found:
            sources_found.append("bigquery")

    rows = _enrich_hierarchy(list(deduped.values()))
    return {
        "rows": rows,
        "sources": sources_found,
        "bigquery_table": resolved_bq,
        "bigquery_error": bq_error,
        "updated_at": datetime.now().strftime("%H:%M:%S"),
    }


DASHBOARD_HTML = """<!DOCTYPE html>
<html lang="en" class="dark">
<head>
  <meta charset="UTF-8" />
  <meta name="viewport" content="width=device-width, initial-scale=1.0" />
  <title>ADK FinOps • Near-Live Cost & Token Tracker</title>
  <script src="https://cdn.tailwindcss.com"></script>
  <script src="https://cdn.jsdelivr.net/npm/chart.js@4.4.3/dist/chart.umd.min.js"></script>
  <style>
    body { background-color: #0b0f19; color: #e2e8f0; font-family: ui-sans-serif, system-ui, -apple-system, sans-serif; }
    .card { background: #111827; border: 1px solid #1f2937; border-radius: 0.75rem; }
    .pulse-dot { animation: pulse 2s infinite; }
    @keyframes pulse { 0%, 100% { opacity: 1; } 50% { opacity: 0.35; } }
  </style>
</head>
<body class="min-h-screen p-6">
  <!-- Header -->
  <div class="max-w-7xl mx-auto">
    <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-4 mb-6 pb-4 border-b border-gray-800">
      <div>
        <div class="flex items-center gap-3">
          <span class="text-2xl">💸</span>
          <h1 class="text-2xl font-bold tracking-tight text-white">ADK FinOps Near-Live Tracker</h1>
          <span class="inline-flex items-center gap-1.5 px-2.5 py-0.5 rounded-full text-xs font-medium bg-emerald-950 text-emerald-400 border border-emerald-800">
            <span id="liveDot" class="w-2 h-2 rounded-full bg-emerald-400 pulse-dot"></span>
            <span id="liveStatusText">LIVE (2s)</span>
          </span>
        </div>
        <p class="text-xs text-gray-400 mt-1">
          Sources: <span id="activeSources" class="text-indigo-400 font-mono">loading...</span>
          • Last Refreshed: <span id="lastUpdated" class="font-mono text-gray-300">--:--:--</span>
        </p>
      </div>

      <!-- Controls -->
      <div class="flex flex-wrap items-center gap-3">
        <label class="inline-flex items-center gap-2 text-xs bg-gray-900 border border-gray-700 px-3 py-1.5 rounded-lg cursor-pointer">
          <input type="checkbox" id="bqToggle" checked onchange="fetchMetrics()" class="rounded bg-gray-800 border-gray-600 text-indigo-500" />
          <span>Include BigQuery</span>
        </label>
        <button id="pauseBtn" onclick="togglePause()" class="text-xs bg-gray-800 hover:bg-gray-700 border border-gray-700 px-3 py-1.5 rounded-lg font-medium">
          ⏸ Pause Auto-Refresh
        </button>
        <button onclick="fetchMetrics()" class="text-xs bg-indigo-600 hover:bg-indigo-500 text-white px-3 py-1.5 rounded-lg font-medium">
          ↻ Refresh Now
        </button>
      </div>
    </div>

    <!-- Hierarchical Filter Bar -->
    <div class="card p-4 mb-6 grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-3">
      <div>
        <label class="block text-xs text-indigo-300 font-semibold mb-1">1. 👑 Root (Parent) Agent</label>
        <select id="rootAgentFilter" onchange="onRootAgentChange()" class="w-full bg-gray-900 border border-indigo-700/80 rounded-lg px-2.5 py-1.5 text-xs text-white font-medium">
          <option value="ALL">All Root Agents (Overall Rollup)</option>
        </select>
      </div>
      <div>
        <label class="block text-xs text-cyan-300 font-semibold mb-1">2. ↳ Sub-Agent Drilldown</label>
        <select id="subAgentFilter" onchange="onSubAgentChange()" class="w-full bg-gray-900 border border-cyan-800/70 rounded-lg px-2.5 py-1.5 text-xs text-gray-200">
          <option value="ALL">∑ Overall Parent Total (Root + Sub-Agents)</option>
        </select>
      </div>
      <div>
        <label class="block text-xs text-gray-300 font-medium mb-1">3. Session Filter (Scoped)</label>
        <select id="sessionFilter" onchange="renderDashboard()" class="w-full bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-200">
          <option value="ALL">All Sessions</option>
        </select>
      </div>
      <div>
        <label class="block text-xs text-gray-400 mb-1">4. Model Filter</label>
        <select id="modelFilter" onchange="renderDashboard()" class="w-full bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-200">
          <option value="ALL">All Models</option>
        </select>
      </div>
      <div>
        <label class="block text-xs text-gray-400 mb-1">5. Task Outcome</label>
        <select id="outcomeFilter" onchange="renderDashboard()" class="w-full bg-gray-900 border border-gray-700 rounded-lg px-2.5 py-1.5 text-xs text-gray-200">
          <option value="ALL">All Outcomes (Success + Wasted)</option>
          <option value="EFFECTIVE">✅ Effective Spend Only (Success)</option>
          <option value="WASTED">🔥 Wasted Spend Only (Failed / Error / Budget)</option>
        </select>
      </div>
    </div>

    <!-- KPI Cards -->
    <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-5 gap-4 mb-6">
      <div class="card p-4 border-indigo-900/60">
        <div class="flex items-center justify-between">
          <div class="text-xs text-gray-400 font-medium">Total Spend (USD)</div>
          <span id="kpiScopeBadge" class="text-[10px] px-1.5 py-0.5 rounded bg-indigo-950 text-indigo-300 border border-indigo-800 font-mono">∑ Parent Rollup</span>
        </div>
        <div id="kpiTotalCost" class="text-2xl font-bold text-white mt-1">$0.0000</div>
        <div id="kpiCostSub" class="text-xs text-gray-400 mt-1">LLM: $0.0000 • Tools: $0.0000</div>
      </div>
      <div class="card p-4 border-emerald-900/60">
        <div class="text-xs text-emerald-400 font-medium">✅ Effective Spend</div>
        <div id="kpiEffectiveCost" class="text-2xl font-bold text-emerald-400 mt-1">$0.0000</div>
        <div id="kpiEffectiveSub" class="text-xs text-gray-400 mt-1">0 successful runs</div>
      </div>
      <div class="card p-4 border-rose-900/60">
        <div class="text-xs text-rose-400 font-medium">🔥 Wasted Spend (Failed)</div>
        <div id="kpiWastedCost" class="text-2xl font-bold text-rose-400 mt-1">$0.0000</div>
        <div id="kpiWastedSub" class="text-xs text-rose-300/80 mt-1">0.0% of total spend</div>
      </div>
      <div class="card p-4 border-amber-900/60">
        <div class="text-xs text-amber-400 font-medium">💰 Context Cache Saved</div>
        <div id="kpiSavingsCost" class="text-2xl font-bold text-amber-400 mt-1">$0.0000</div>
        <div id="kpiSavingsSub" class="text-xs text-gray-400 mt-1">0 cached tokens</div>
      </div>
      <div class="card p-4">
        <div class="text-xs text-gray-400 font-medium">Total Tokens Processed</div>
        <div id="kpiTotalTokens" class="text-2xl font-bold text-indigo-400 mt-1">0</div>
        <div id="kpiTokenSub" class="text-xs text-gray-400 mt-1">In: 0 • Out: 0 • Think: 0</div>
      </div>
    </div>

    <!-- Root (Parent) Agent -> Sub-Agent Hierarchy Explorer -->
    <div class="card p-4 mb-6">
      <div class="flex flex-col sm:flex-row sm:items-center sm:justify-between gap-2 mb-3 pb-2 border-b border-gray-800">
        <div>
          <h2 class="text-sm font-semibold text-white flex items-center gap-2">
            <span>👑 Root (Parent) Agent ➔ Sub-Agents Hierarchy Rollup</span>
          </h2>
          <p class="text-xs text-gray-400">
            Shows overall parent-level spend & tokens (Root Orchestrator + all delegated Sub-Agents) and each child's share. Click any card to filter.
          </p>
        </div>
        <button onclick="resetHierarchyFilter()" class="text-xs bg-gray-800 hover:bg-gray-700 text-indigo-300 border border-gray-700 px-2.5 py-1 rounded-lg self-start sm:self-auto">
          Reset Agent Hierarchy Filter
        </button>
      </div>
      <div id="hierarchyContainer" class="space-y-4"></div>
    </div>

    <!-- Charts Row -->
    <div class="grid grid-cols-1 lg:grid-cols-3 gap-4 mb-6">
      <div class="card p-4">
        <h2 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">Spend Efficiency & ROI</h2>
        <div class="h-56 flex items-center justify-center">
          <canvas id="efficiencyChart"></canvas>
        </div>
      </div>
      <div class="card p-4">
        <h2 id="agentChartTitle" class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">Sub-Agent Cost Attribution ($)</h2>
        <div class="h-56">
          <canvas id="agentChart"></canvas>
        </div>
      </div>
      <div class="card p-4">
        <h2 class="text-xs font-semibold uppercase tracking-wider text-gray-400 mb-3">Token Composition by Model</h2>
        <div class="h-56">
          <canvas id="modelChart"></canvas>
        </div>
      </div>
    </div>

    <!-- Live Telemetry Table -->
    <div class="card overflow-hidden">
      <div class="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
        <h2 class="text-sm font-semibold text-white">Near-Live Session & Hierarchical Agent Telemetry</h2>
        <span id="rowCountBadge" class="text-xs text-gray-400 font-mono">0 records</span>
      </div>
      <div class="overflow-x-auto">
        <table class="w-full text-left border-collapse text-xs">
          <thead>
            <tr class="bg-gray-900/90 text-gray-400 border-b border-gray-800">
              <th class="py-2.5 px-3">Time (UTC)</th>
              <th class="py-2.5 px-3">Session ID</th>
              <th class="py-2.5 px-3">👑 Root (Parent) Agent</th>
              <th class="py-2.5 px-3">↳ Agent / Hierarchy Role</th>
              <th class="py-2.5 px-3">Model</th>
              <th class="py-2.5 px-3">Outcome</th>
              <th class="py-2.5 px-3 text-right">Tokens (In/Out/Think)</th>
              <th class="py-2.5 px-3 text-right">Cached</th>
              <th class="py-2.5 px-3 text-right">Total Cost</th>
              <th class="py-2.5 px-3 text-right">Saved</th>
              <th class="py-2.5 px-3">Source / Error</th>
            </tr>
          </thead>
          <tbody id="telemetryBody" class="divide-y divide-gray-800/70 font-mono"></tbody>
        </table>
      </div>
    </div>
  </div>

  <script>
    let rawRows = [];
    let isPaused = false;
    let effChart = null, agentChart = null, modelChart = null;

    function togglePause() {
      isPaused = !isPaused;
      document.getElementById('pauseBtn').innerText = isPaused ? '▶ Resume Auto-Refresh' : '⏸ Pause Auto-Refresh';
      document.getElementById('liveStatusText').innerText = isPaused ? 'PAUSED' : 'LIVE (2s)';
      document.getElementById('liveDot').className = isPaused
        ? 'w-2 h-2 rounded-full bg-amber-400'
        : 'w-2 h-2 rounded-full bg-emerald-400 pulse-dot';
    }

    async function fetchMetrics() {
      const includeBq = document.getElementById('bqToggle').checked;
      try {
        const res = await fetch(`/api/metrics?include_bigquery=${includeBq}`);
        const data = await res.json();
        rawRows = data.rows || [];
        document.getElementById('activeSources').innerText = (data.sources || []).join(', ') || 'waiting for runs...';
        document.getElementById('lastUpdated').innerText = data.updated_at || '--:--:--';
        updateFilterDropdowns();
        renderDashboard();
      } catch (e) {
        console.error('Dashboard fetch error:', e);
      }
    }

    function updateFilterDropdowns() {
      const rootAgents = new Set();
      rawRows.forEach(r => {
        if (r.root_agent_name) rootAgents.add(r.root_agent_name);
      });
      syncSelect('rootAgentFilter', 'All Root Agents (Overall Rollup)', [...rootAgents].map(ra => ({ value: ra, label: `👑 ${ra} (Parent Rollup)` })));
      populateDependentDropdowns();
    }

    function populateDependentDropdowns() {
      const rootVal = document.getElementById('rootAgentFilter').value;
      const subAgents = new Map();

      // 1. Populate Sub-Agents belonging to the selected Root Agent
      rawRows.forEach(r => {
        const rRoot = r.root_agent_name || 'root_agent';
        if (rootVal !== 'ALL' && rRoot !== rootVal) return;
        if (r.agent_name) {
          const isRootSelf = r.agent_name === rRoot;
          const label = isRootSelf
            ? `👑 ${r.agent_name} (Root Orchestrator Direct Only)`
            : (rootVal === 'ALL' ? `↳ 🤖 ${r.agent_name} [under ${rRoot}]` : `↳ 🤖 ${r.agent_name} (Sub-Agent)`);
          subAgents.set(r.agent_name, label);
        }
      });
      const subItems = [...subAgents.entries()].map(([val, label]) => ({ value: val, label }));
      const allSubLabel = rootVal === 'ALL'
        ? '∑ Overall Parent Total (All Root + Sub-Agents)'
        : `∑ ${rootVal} Overall Total (Parent + Sub-Agents)`;
      syncSelect('subAgentFilter', allSubLabel, subItems);

      // 2. Populate Sessions & Models scoped strictly to the selected Root Agent (and Sub-Agent if selected)
      const subVal = document.getElementById('subAgentFilter').value;
      const sessions = new Map();
      const models = new Set();

      rawRows.forEach(r => {
        const rRoot = r.root_agent_name || 'root_agent';
        if (rootVal !== 'ALL' && rRoot !== rootVal) return;
        if (subVal !== 'ALL' && r.agent_name !== subVal) return;
        if (r.session_id && !sessions.has(r.session_id)) {
          const sessLabel = rootVal === 'ALL'
            ? `${r.session_id} (${rRoot})`
            : r.session_id;
          sessions.set(r.session_id, sessLabel);
        }
        if (r.model_name) {
          models.add(r.model_name);
        }
      });

      const sessItems = [...sessions.entries()].map(([val, label]) => ({ value: val, label }));
      const allSessLabel = rootVal === 'ALL'
        ? `All Sessions (${sessItems.length})`
        : `All Sessions for ${rootVal} (${sessItems.length})`;
      syncSelect('sessionFilter', allSessLabel, sessItems);

      const modelItems = [...models].map(m => ({ value: m, label: m }));
      syncSelect('modelFilter', 'All Models', modelItems);
    }

    function onRootAgentChange() {
      document.getElementById('subAgentFilter').value = 'ALL';
      document.getElementById('sessionFilter').value = 'ALL';
      populateDependentDropdowns();
      renderDashboard();
    }

    function onSubAgentChange() {
      populateDependentDropdowns();
      renderDashboard();
    }

    function selectHierarchyTarget(rootName, subName) {
      document.getElementById('rootAgentFilter').value = rootName || 'ALL';
      populateDependentDropdowns();
      document.getElementById('subAgentFilter').value = subName || 'ALL';
      populateDependentDropdowns();
      renderDashboard();
    }

    function resetHierarchyFilter() {
      document.getElementById('rootAgentFilter').value = 'ALL';
      document.getElementById('subAgentFilter').value = 'ALL';
      document.getElementById('sessionFilter').value = 'ALL';
      populateDependentDropdowns();
      renderDashboard();
    }

    function syncSelect(id, allLabel, items) {
      const sel = document.getElementById(id);
      const cur = sel.value;
      sel.innerHTML = `<option value="ALL">${allLabel}</option>` +
        items.map(x => `<option value="${x.value}">${x.label}</option>`).join('');
      if (items.some(x => x.value === cur)) {
        sel.value = cur;
      } else {
        sel.value = 'ALL';
      }
    }

    function renderDashboard() {
      const sessVal = document.getElementById('sessionFilter').value;
      const rootVal = document.getElementById('rootAgentFilter').value;
      const subVal = document.getElementById('subAgentFilter').value;
      const modelVal = document.getElementById('modelFilter').value;
      const outcomeVal = document.getElementById('outcomeFilter').value;

      const sessionScopeRows = rawRows.filter(r => r.scope === 'session');
      const baseRows = sessionScopeRows.length > 0 ? sessionScopeRows : rawRows;

      // Filter rows by Session, Root Agent, and Outcome first (used for Hierarchy Tree & Charts)
      const parentMatchedRows = baseRows.filter(r => {
        if (sessVal !== 'ALL' && r.session_id !== sessVal) return false;
        if (rootVal !== 'ALL' && (r.root_agent_name || 'root_agent') !== rootVal) return false;
        if (outcomeVal === 'EFFECTIVE' && r.is_failure) return false;
        if (outcomeVal === 'WASTED' && !r.is_failure) return false;
        return true;
      });

      // Rows matching Sub-Agent and Model filters as well
      const filtered = parentMatchedRows.filter(r => {
        if (subVal !== 'ALL' && r.agent_name !== subVal) return false;
        if (modelVal !== 'ALL' && r.model_name !== modelVal) return false;
        return true;
      });

      // Determine KPI rows:
      // - When subVal === 'ALL' and modelVal === 'ALL', use root_rollup rows (agent_name === null) so KPIs show true Overall Parent-Level Spend!
      // - Otherwise sum the matching child/model rows.
      let kpiRows = [];
      if (subVal === 'ALL' && modelVal === 'ALL') {
        kpiRows = parentMatchedRows.filter(r => !r.agent_name);
        if (kpiRows.length === 0) kpiRows = parentMatchedRows.filter(r => r.agent_name);
        document.getElementById('kpiScopeBadge').innerText = rootVal === 'ALL'
          ? '∑ All Root Rollups'
          : `∑ ${rootVal} (Parent Total)`;
      } else {
        kpiRows = filtered.filter(r => r.agent_name);
        if (kpiRows.length === 0) kpiRows = filtered;
        document.getElementById('kpiScopeBadge').innerText = subVal !== 'ALL'
          ? `↳ ${subVal}`
          : `Model: ${modelVal}`;
      }

      let totalCost = 0, llmCost = 0, toolCost = 0, effCost = 0, wastedCost = 0, savedCost = 0;
      let totalTok = 0, inTok = 0, outTok = 0, thinkTok = 0, cachedTok = 0;
      let succCount = 0, failCount = 0;

      kpiRows.forEach(r => {
        totalCost += r.total_cost_usd || 0;
        llmCost += r.llm_cost_usd || 0;
        toolCost += r.tool_cost_usd || 0;
        savedCost += r.savings_usd || 0;
        totalTok += r.total_tokens || 0;
        inTok += r.prompt_tokens || 0;
        outTok += r.completion_tokens || 0;
        thinkTok += r.thoughts_tokens || 0;
        cachedTok += r.cached_tokens || 0;
        if (r.is_failure) {
          wastedCost += r.total_cost_usd || 0;
          failCount++;
        } else {
          effCost += r.total_cost_usd || 0;
          succCount++;
        }
      });

      const wastedPct = totalCost > 0 ? ((wastedCost / totalCost) * 100).toFixed(1) : '0.0';
      document.getElementById('kpiTotalCost').innerText = `$${totalCost.toFixed(4)}`;
      document.getElementById('kpiCostSub').innerText = `LLM: $${llmCost.toFixed(4)} • Grounding: $${toolCost.toFixed(4)}`;
      document.getElementById('kpiEffectiveCost').innerText = `$${effCost.toFixed(4)}`;
      document.getElementById('kpiEffectiveSub').innerText = `${succCount} successful record(s)`;
      document.getElementById('kpiWastedCost').innerText = `$${wastedCost.toFixed(4)}`;
      document.getElementById('kpiWastedSub').innerText = `${wastedPct}% of total spend (${failCount} failed)`;
      document.getElementById('kpiSavingsCost').innerText = `$${savedCost.toFixed(4)}`;
      document.getElementById('kpiSavingsSub').innerText = `${cachedTok.toLocaleString()} cached tokens`;
      document.getElementById('kpiTotalTokens').innerText = totalTok.toLocaleString();
      document.getElementById('kpiTokenSub').innerText = `In: ${inTok.toLocaleString()} • Out: ${outTok.toLocaleString()} • Think: ${thinkTok.toLocaleString()}`;

      renderHierarchyTree(parentMatchedRows, rootVal, subVal);

      // Sub-Agent Breakdown for Chart (when a Root Agent is selected, show its children even if subVal === 'ALL')
      const chartSourceRows = (subVal === 'ALL' ? parentMatchedRows : filtered).filter(r => r.agent_name && (modelVal === 'ALL' || r.model_name === modelVal));
      const agentMap = {};
      chartSourceRows.forEach(r => {
        const isRootSelf = r.agent_name === r.root_agent_name;
        const label = isRootSelf ? `👑 ${r.agent_name} (Direct)` : `↳ 🤖 ${r.agent_name}`;
        if (!agentMap[label]) agentMap[label] = { llm: 0, tool: 0, tokens: 0, model: r.model_name };
        agentMap[label].llm += r.llm_cost_usd || 0;
        agentMap[label].tool += r.tool_cost_usd || 0;
        agentMap[label].tokens += r.total_tokens || 0;
      });
      document.getElementById('agentChartTitle').innerText = rootVal === 'ALL'
        ? 'Root & Sub-Agent Cost Attribution ($)'
        : `Sub-Agents of 👑 ${rootVal} ($ Spend)`;

      // Model Breakdown
      const modelMap = {};
      (chartSourceRows.length ? chartSourceRows : kpiRows).forEach(r => {
        const m = r.model_name || 'gemini-2.5-flash';
        if (!modelMap[m]) modelMap[m] = { prompt: 0, completion: 0, thoughts: 0, cached: 0 };
        modelMap[m].prompt += r.prompt_tokens || 0;
        modelMap[m].completion += r.completion_tokens || 0;
        modelMap[m].thoughts += r.thoughts_tokens || 0;
        modelMap[m].cached += r.cached_tokens || 0;
      });

      updateCharts(effCost, wastedCost, savedCost, agentMap, modelMap);
      renderTable(subVal === 'ALL' ? parentMatchedRows.filter(r => modelVal === 'ALL' || !r.agent_name || r.model_name === modelVal) : filtered);
    }

    function renderHierarchyTree(rows, activeRoot, activeSub) {
      const container = document.getElementById('hierarchyContainer');
      const roots = {};

      rows.forEach(r => {
        const rootName = r.root_agent_name || 'root_agent';
        if (!roots[rootName]) {
          roots[rootName] = {
            rollupCost: 0,
            rollupTokens: 0,
            rollupSavings: 0,
            sessions: new Set(),
            children: {}
          };
        }
        roots[rootName].sessions.add(r.session_id);
        if (!r.agent_name) {
          roots[rootName].rollupCost += r.total_cost_usd || 0;
          roots[rootName].rollupTokens += r.total_tokens || 0;
          roots[rootName].rollupSavings += r.savings_usd || 0;
        } else {
          const cName = r.agent_name;
          if (!roots[rootName].children[cName]) {
            roots[rootName].children[cName] = {
              name: cName,
              isRootSelf: cName === rootName,
              cost: 0,
              llmCost: 0,
              toolCost: 0,
              tokens: 0,
              prompt: 0,
              completion: 0,
              thoughts: 0,
              models: new Set()
            };
          }
          const c = roots[rootName].children[cName];
          c.cost += r.total_cost_usd || 0;
          c.llmCost += r.llm_cost_usd || 0;
          c.toolCost += r.tool_cost_usd || 0;
          c.tokens += r.total_tokens || 0;
          c.prompt += r.prompt_tokens || 0;
          c.completion += r.completion_tokens || 0;
          c.thoughts += r.thoughts_tokens || 0;
          if (r.model_name) c.models.add(r.model_name);
        }
      });

      const rootNames = Object.keys(roots);
      if (rootNames.length === 0) {
        container.innerHTML = `<div class="text-xs text-gray-500 py-3">No hierarchical agent telemetry recorded yet.</div>`;
        return;
      }

      container.innerHTML = rootNames.map(rName => {
        const info = roots[rName];
        const childList = Object.values(info.children).sort((a, b) => (a.isRootSelf === b.isRootSelf) ? (b.cost - a.cost) : (a.isRootSelf ? -1 : 1));
        const sumChildrenCost = childList.reduce((acc, c) => acc + c.cost, 0);
        const parentTotalCost = info.rollupCost > 0 ? info.rollupCost : sumChildrenCost;
        const sumChildrenTokens = childList.reduce((acc, c) => acc + c.tokens, 0);
        const parentTotalTokens = info.rollupTokens > 0 ? info.rollupTokens : sumChildrenTokens;
        const subCount = childList.filter(c => !c.isRootSelf).length;
        const isSelectedRoot = activeRoot === rName && activeSub === 'ALL';

        const childrenHtml = childList.map(c => {
          const sharePct = parentTotalCost > 0 ? ((c.cost / parentTotalCost) * 100).toFixed(1) : '0.0';
          const isSelectedSub = activeSub === c.name;
          const roleBadge = c.isRootSelf
            ? `<span class="px-1.5 py-0.5 rounded text-[10px] bg-indigo-950 text-indigo-300 border border-indigo-800 font-semibold">👑 ROOT DIRECT</span>`
            : `<span class="px-1.5 py-0.5 rounded text-[10px] bg-cyan-950 text-cyan-300 border border-cyan-800 font-semibold">↳ 🤖 SUB-AGENT</span>`;
          const borderCls = isSelectedSub
            ? 'border-cyan-400 bg-cyan-950/30 ring-1 ring-cyan-400'
            : 'border-gray-800 bg-gray-900/70 hover:border-gray-600';
          const barColor = c.isRootSelf ? 'bg-indigo-500' : 'bg-cyan-500';
          return `
            <div onclick="selectHierarchyTarget('${rName}', '${c.name}')" class="cursor-pointer rounded-lg border p-3 transition ${borderCls}">
              <div class="flex items-center justify-between gap-2 mb-1">
                <div class="flex items-center gap-1.5 truncate">
                  ${roleBadge}
                  <span class="text-xs font-semibold text-white truncate">${c.name}</span>
                </div>
                <span class="text-xs font-bold text-white font-mono">$${c.cost.toFixed(5)}</span>
              </div>
              <div class="flex items-center justify-between text-[11px] text-gray-400 mb-1.5">
                <span class="truncate">Model: <span class="text-gray-300 font-mono">${[...c.models].join(', ') || 'default'}</span></span>
                <span class="text-cyan-300 font-mono">${sharePct}% of parent</span>
              </div>
              <div class="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden mb-1.5">
                <div class="${barColor} h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(2, parseFloat(sharePct)))}%"></div>
              </div>
              <div class="text-[10px] text-gray-400 font-mono flex justify-between">
                <span>Tokens: ${c.tokens.toLocaleString()}</span>
                <span>In:${c.prompt} Out:${c.completion} Think:${c.thoughts}</span>
              </div>
            </div>
          `;
        }).join('');

        const rootBorder = isSelectedRoot ? 'border-indigo-500 ring-1 ring-indigo-500/60' : 'border-gray-800';
        return `
          <div class="rounded-xl border ${rootBorder} bg-gray-950/60 p-4">
            <div class="flex flex-col md:flex-row md:items-center md:justify-between gap-3 pb-3 mb-3 border-b border-gray-800/80">
              <div class="flex flex-wrap items-center gap-2.5">
                <span class="px-2 py-0.5 rounded-md text-xs font-bold bg-amber-950 text-amber-300 border border-amber-700/80">
                  ∑ PARENT LEVEL ROLLUP
                </span>
                <span class="text-sm font-bold text-white font-mono">👑 ${rName}</span>
                <span class="text-xs text-gray-400">
                  (${subCount} Sub-Agent${subCount === 1 ? '' : 's'} • ${info.sessions.size} Session${info.sessions.size === 1 ? '' : 's'})
                </span>
              </div>
              <div class="flex flex-wrap items-center gap-4">
                <div class="text-right">
                  <div class="text-[10px] uppercase tracking-wider text-gray-400">Overall Parent Spend</div>
                  <div class="text-sm font-bold text-amber-300 font-mono">$${parentTotalCost.toFixed(5)}</div>
                </div>
                <div class="text-right">
                  <div class="text-[10px] uppercase tracking-wider text-gray-400">Overall Parent Tokens</div>
                  <div class="text-sm font-bold text-indigo-300 font-mono">${parentTotalTokens.toLocaleString()}</div>
                </div>
                <button onclick="selectHierarchyTarget('${rName}', 'ALL')" class="text-xs px-3 py-1.5 rounded-lg font-medium ${isSelectedRoot ? 'bg-indigo-600 text-white' : 'bg-gray-800 hover:bg-gray-700 text-indigo-300 border border-gray-700'}">
                  ${isSelectedRoot ? '✓ Viewing Parent Rollup' : 'Filter Root & Show Sub-Agents'}
                </button>
              </div>
            </div>
            <div class="grid grid-cols-1 sm:grid-cols-2 lg:grid-cols-4 gap-3">
              ${childrenHtml}
            </div>
          </div>
        `;
      }).join('');
    }

    function updateCharts(effCost, wastedCost, savedCost, agentMap, modelMap) {
      // 1. Efficiency Chart
      const effCtx = document.getElementById('efficiencyChart');
      if (effChart) effChart.destroy();
      effChart = new Chart(effCtx, {
        type: 'doughnut',
        data: {
          labels: ['Effective Spend ($)', 'Wasted Spend ($)', 'Cache Savings ($)'],
          datasets: [{
            data: [effCost, wastedCost, savedCost],
            backgroundColor: ['#10b981', '#f43f5e', '#f59e0b'],
            borderWidth: 0
          }]
        },
        options: { plugins: { legend: { position: 'bottom', labels: { color: '#9ca3af', font: { size: 11 } } } } }
      });

      // 2. Agent Chart
      const aNames = Object.keys(agentMap);
      const agCtx = document.getElementById('agentChart');
      if (agentChart) agentChart.destroy();
      agentChart = new Chart(agCtx, {
        type: 'bar',
        data: {
          labels: aNames,
          datasets: [
            { label: 'LLM Cost ($)', data: aNames.map(a => agentMap[a].llm), backgroundColor: '#6366f1' },
            { label: 'Tool/Grounding ($)', data: aNames.map(a => agentMap[a].tool), backgroundColor: '#ec4899' }
          ]
        },
        options: {
          indexAxis: 'y',
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { stacked: true, ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' } },
            y: { stacked: true, ticks: { color: '#e5e7eb' }, grid: { display: false } }
          },
          plugins: { legend: { labels: { color: '#9ca3af', font: { size: 11 } } } }
        }
      });

      // 3. Model Chart
      const mNames = Object.keys(modelMap);
      const mCtx = document.getElementById('modelChart');
      if (modelChart) modelChart.destroy();
      modelChart = new Chart(mCtx, {
        type: 'bar',
        data: {
          labels: mNames,
          datasets: [
            { label: 'Prompt', data: mNames.map(m => modelMap[m].prompt), backgroundColor: '#3b82f6' },
            { label: 'Completion', data: mNames.map(m => modelMap[m].completion), backgroundColor: '#10b981' },
            { label: 'Thinking', data: mNames.map(m => modelMap[m].thoughts), backgroundColor: '#a855f7' },
            { label: 'Cached', data: mNames.map(m => modelMap[m].cached), backgroundColor: '#f59e0b' }
          ]
        },
        options: {
          responsive: true,
          maintainAspectRatio: false,
          scales: {
            x: { stacked: true, ticks: { color: '#e5e7eb' } },
            y: { stacked: true, ticks: { color: '#9ca3af' }, grid: { color: '#1f2937' } }
          },
          plugins: { legend: { labels: { color: '#9ca3af', font: { size: 10 } } } }
        }
      });
    }

    function renderTable(rows) {
      document.getElementById('rowCountBadge').innerText = `${rows.length} records`;
      const tbody = document.getElementById('telemetryBody');
      tbody.innerHTML = rows.slice(0, 100).map(r => {
        const badgeClass = r.is_failure
          ? 'bg-rose-950 text-rose-400 border-rose-800'
          : 'bg-emerald-950 text-emerald-400 border-emerald-800';
        const outcomeLabel = (r.status || 'success').toUpperCase();
        const ts = (r.timestamp || '').replace('T', ' ').slice(0, 19);
        const rootLabel = `<span class="text-indigo-300 font-semibold">👑 ${r.root_agent_name || 'root_agent'}</span>`;
        let agentRoleCell = '';
        if (!r.agent_name) {
          agentRoleCell = `<span class="px-2 py-0.5 rounded bg-amber-950/90 text-amber-300 border border-amber-700/80 text-[11px] font-bold">∑ OVERALL PARENT TOTAL (${r.root_agent_name || 'root'})</span>`;
        } else if (r.agent_name === r.root_agent_name) {
          agentRoleCell = `<span class="text-indigo-200 font-semibold">👑 ${r.agent_name} <span class="text-[10px] text-indigo-400">[ROOT DIRECT]</span></span>`;
        } else {
          agentRoleCell = `<span class="text-cyan-300">↳ 🤖 ${r.agent_name} <span class="text-[10px] text-gray-400">[SUB-AGENT]</span></span>`;
        }
        return `<tr class="hover:bg-gray-900/50 ${!r.agent_name ? 'bg-amber-950/10' : ''}">
          <td class="py-2 px-3 text-gray-400">${ts}</td>
          <td class="py-2 px-3 text-indigo-300 font-medium">${r.session_id}</td>
          <td class="py-2 px-3">${rootLabel}</td>
          <td class="py-2 px-3">${agentRoleCell}</td>
          <td class="py-2 px-3 text-gray-300">${r.model_name || 'ALL'}</td>
          <td class="py-2 px-3"><span class="px-2 py-0.5 rounded border text-[10px] font-semibold ${badgeClass}">${outcomeLabel}</span></td>
          <td class="py-2 px-3 text-right text-gray-300">${(r.total_tokens||0).toLocaleString()} <span class="text-gray-500">(${r.prompt_tokens}/${r.completion_tokens}/${r.thoughts_tokens})</span></td>
          <td class="py-2 px-3 text-right text-amber-400">${(r.cached_tokens||0).toLocaleString()}</td>
          <td class="py-2 px-3 text-right font-semibold text-white">$${(r.total_cost_usd||0).toFixed(6)}</td>
          <td class="py-2 px-3 text-right text-emerald-400">${r.savings_usd > 0 ? '$' + r.savings_usd.toFixed(6) : '—'}</td>
          <td class="py-2 px-3 text-gray-400 truncate max-w-xs" title="${r.error || r.source}">${r.error ? '<span class="text-rose-400">⚠️ ' + r.error + '</span>' : r.source}</td>
        </tr>`;
      }).join('');
    }

    fetchMetrics();
    setInterval(() => { if (!isPaused) fetchMetrics(); }, 2000);
  </script>
</body>
</html>
"""


def create_dashboard_app(
    log_dir: str | Path = "logs",
    bigquery_table: str | None = None,
) -> Any:
    """Creates a FastAPI application serving the near-live FinOps dashboard."""
    from fastapi import FastAPI
    from fastapi.responses import HTMLResponse, JSONResponse

    app = FastAPI(title="ADK FinOps Live Tracker", version="0.4.0")

    @app.get("/", response_class=HTMLResponse)
    async def index() -> str:
        return DASHBOARD_HTML

    @app.get("/api/metrics", response_class=JSONResponse)
    async def api_metrics(include_bigquery: bool = True) -> dict[str, Any]:
        return collect_telemetry_rows(
            log_dir=log_dir,
            include_bigquery=include_bigquery,
            bigquery_table=bigquery_table,
        )

    @app.post("/api/ingest", response_class=JSONResponse)
    async def api_ingest(payload: dict[str, Any]) -> dict[str, Any]:
        """Allows remote agents across an organization to push telemetry rows directly to a central dashboard."""
        rows_in = payload.get("rows", [payload] if "session_id" in payload else [])
        count = 0
        for raw in rows_in:
            row = _normalize_row(raw, source="http_ingest")
            key = (row["session_id"], row["scope"], row["agent_name"] or "__ALL__")
            _INGESTED_ROWS[key] = row
            count += 1

        # Also persist ingested rows to local JSONL on the central dashboard server if log_dir is configured
        if count > 0 and log_dir:
            try:
                first_dir = str(log_dir).split(",")[0].strip()
                if first_dir:
                    persist_file = Path(first_dir) / "ingested_costs.jsonl"
                    persist_file.parent.mkdir(parents=True, exist_ok=True)
                    with persist_file.open("a", encoding="utf-8") as f:
                        for raw in rows_in:
                            f.write(json.dumps(raw, ensure_ascii=False) + "\n")
            except Exception as e:
                logger.debug(f"[FinOps Dashboard] Could not persist ingested rows: {e}")

        return {"status": "ok", "ingested": count}

    return app


def _find_free_port(preferred_port: int = 8088) -> int:
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        if s.connect_ex(("127.0.0.1", preferred_port)) != 0:
            return preferred_port
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.bind(("127.0.0.1", 0))
        return int(s.getsockname()[1])


def start_background_dashboard(
    port: int = 8088,
    host: str = "127.0.0.1",
    log_dir: str | Path = "logs",
    bigquery_table: str | None = None,
) -> str:
    """Starts the FastAPI live dashboard in a non-blocking daemon thread."""
    global _DASHBOARD_THREAD, _DASHBOARD_URL
    if _DASHBOARD_THREAD is not None and _DASHBOARD_THREAD.is_alive() and _DASHBOARD_URL:
        return _DASHBOARD_URL

    actual_port = _find_free_port(port)
    _DASHBOARD_URL = f"http://{host}:{actual_port}"

    def _run_server() -> None:
        try:
            import uvicorn

            app = create_dashboard_app(log_dir=log_dir, bigquery_table=bigquery_table)
            config = uvicorn.Config(
                app=app,
                host=host,
                port=actual_port,
                log_level="warning",
            )
            server = uvicorn.Server(config)
            server.run()
        except Exception as e:
            logger.warning(f"[FinOps Dashboard] Could not start background server: {e}")

    _DASHBOARD_THREAD = threading.Thread(target=_run_server, daemon=True, name="adk-finops-dashboard")
    _DASHBOARD_THREAD.start()

    # Wait up to 1.5s for uvicorn socket to bind
    for _ in range(15):
        with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
            if s.connect_ex((host, actual_port)) == 0:
                break
        time.sleep(0.1)

    msg = f"📊 [FinOps Live Dashboard] Near-live cost tracker running at {_DASHBOARD_URL}"
    print(msg, flush=True)
    logger.info(msg)
    return _DASHBOARD_URL


def cli_main() -> None:
    """CLI entrypoint for `adk-finops dashboard` or `python -m adk_finops.dashboard`."""
    parser = argparse.ArgumentParser(
        prog="adk-finops",
        description="ADK FinOps Near-Live Cost & Token Dashboard",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="dashboard",
        help="Command to execute (default: dashboard)",
    )
    parser.add_argument(
        "--port",
        type=int,
        default=int(os.getenv("ADK_FINOPS_DASHBOARD_PORT", "8088")),
        help="Port to run the dashboard on (default: 8088)",
    )
    parser.add_argument(
        "--host",
        type=str,
        default="127.0.0.1",
        help="Host interface to bind (default: 127.0.0.1)",
    )
    parser.add_argument(
        "--log-dir",
        type=str,
        default=os.getenv("ADK_FINOPS_LOG_DIR", "logs"),
        help="Directory containing local timestamped JSONL/CSV FinOps logs (default: logs)",
    )
    parser.add_argument(
        "--bigquery-table",
        type=str,
        default=os.getenv("ADK_FINOPS_BIGQUERY_TABLE"),
        help="Optional BigQuery table ID (project.dataset.table) for historical + cloud sync",
    )
    args = parser.parse_args()

    import uvicorn

    app = create_dashboard_app(log_dir=args.log_dir, bigquery_table=args.bigquery_table)
    print(f"\n📊 ADK FinOps Near-Live Dashboard running at: http://{args.host}:{args.port}")
    print(f"   • Watching local log directory: {Path(args.log_dir).resolve()}")
    if args.bigquery_table:
        print(f"   • BigQuery table configured: {args.bigquery_table}")
    print("   • Press Ctrl+C to stop.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    cli_main()
