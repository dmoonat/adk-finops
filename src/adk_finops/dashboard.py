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

from collections import OrderedDict
import hmac

from .exporters.bigquery import validate_bq_table_id
from .tracker import CostTracker

logger = logging.getLogger("adk_finops.dashboard")

_BQ_CACHE: dict[str, Any] = {"timestamp": 0.0, "rows": [], "error": None}
_BQ_CACHE_TTL_SEC = 15.0
MAX_INGESTED_ROWS = 5_000
MAX_INGEST_BATCH_ROWS = 250
_INGESTED_ROWS: OrderedDict[tuple[Any, ...], dict[str, Any]] = OrderedDict()
_INGESTED_ROWS_LOCK = threading.Lock()
_DASHBOARD_THREAD: threading.Thread | None = None
_DASHBOARD_URL: str | None = None


def ingest_row_in_memory(raw: dict[str, Any], source: str = "http_ingest") -> dict[str, Any]:
    """Thread-safely normalizes and inserts a row into _INGESTED_ROWS with LRU eviction."""
    row = _normalize_row(raw, source=source)
    key = (row["session_id"], row["scope"], row["agent_name"] or "__ALL__")
    with _INGESTED_ROWS_LOCK:
        if key in _INGESTED_ROWS:
            _INGESTED_ROWS.move_to_end(key)
        _INGESTED_ROWS[key] = row
        while len(_INGESTED_ROWS) > MAX_INGESTED_ROWS:
            _INGESTED_ROWS.popitem(last=False)
    return row


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

    breakdown_by_tool_raw = raw.get("breakdown_by_tool")
    breakdown_by_tool: dict[str, Any] = {}
    if isinstance(breakdown_by_tool_raw, dict):
        breakdown_by_tool = breakdown_by_tool_raw
    elif isinstance(breakdown_by_tool_raw, str) and breakdown_by_tool_raw.strip():
        try:
            parsed_bbt = json.loads(breakdown_by_tool_raw)
            if isinstance(parsed_bbt, dict):
                breakdown_by_tool = parsed_bbt
        except Exception:
            breakdown_by_tool = {}

    if not breakdown_by_tool and breakdown_by_agent:
        for a_key, a_val in breakdown_by_agent.items():
            if isinstance(a_val, dict) and isinstance(a_val.get("tools"), dict) and a_val["tools"]:
                breakdown_by_tool[a_key] = a_val["tools"]

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
        "tool_calls_count": _to_int(raw.get("tool_calls_count") or raw.get("total_tool_calls") or raw.get("tool_calls")),
        "budget_limit_usd": _to_float(raw.get("budget_limit_usd")) if raw.get("budget_limit_usd") not in (None, "") else None,
        "budget_utilization_pct": _to_float(raw.get("budget_utilization_pct")) if raw.get("budget_utilization_pct") not in (None, "") else None,
        "budget_exceeded": str(raw.get("budget_exceeded", "")).lower() in ("true", "1") if isinstance(raw.get("budget_exceeded"), str) else bool(raw.get("budget_exceeded", False)),
        "breakdown_by_tool": breakdown_by_tool,
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
        merged_tool_bd: dict[str, dict[str, Any]] = {}
        for r in group:
            r_tools = r.get("breakdown_by_tool")
            if isinstance(r_tools, dict):
                for a_k, t_map in r_tools.items():
                    if isinstance(t_map, dict):
                        target_map = merged_tool_bd.setdefault(a_k, {})
                        for t_k, t_v in t_map.items():
                            if isinstance(t_v, dict):
                                target_map[t_k] = dict(t_v)

        for r in group:
            r["root_agent_name"] = r.get("root_agent_name") or session_root
            if r.get("agent_name") is None:
                r["agent_role"] = "root_rollup"
                r["parent_agent_name"] = None
                if not r.get("breakdown_by_tool") and merged_tool_bd:
                    r["breakdown_by_tool"] = merged_tool_bd
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
                "breakdown_by_tool": merged_tool_bd,
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

    # 1. Read local JSONL and CSV files across one or more comma-separated log_dir paths
    log_dirs = [p.strip() for p in str(log_dir).split(",") if p.strip()] if log_dir else []
    for d_str in log_dirs:
        base_path = Path(d_str)
        if base_path.exists():
            jsonl_files = sorted(base_path.rglob("*.jsonl"))
            csv_files = sorted(base_path.rglob("*.csv"))

            for jf in jsonl_files:
                try:
                    src_label = "http_ingest" if jf.name == "ingested_costs.jsonl" else f"jsonl:{jf.parent.name}/{jf.name}"
                    for line in jf.read_text(encoding="utf-8").splitlines():
                        if not line.strip():
                            continue
                        row = _normalize_row(json.loads(line), source=src_label)
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

    # 1b. Overlay any live rows pushed via POST /api/ingest in memory so source="http_ingest" is preserved
    if _INGESTED_ROWS:
        with _INGESTED_ROWS_LOCK:
            for k, v in _INGESTED_ROWS.items():
                deduped[k] = dict(v)
        if "http_ingest" not in sources_found:
            sources_found.append("http_ingest")

    # 2. Read live in-memory CostTracker sessions (real-time mid-run state)
    try:
        active_sessions = getattr(CostTracker, "_active_runs", None) or getattr(CostTracker, "_sessions", {})
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

                safe_bq = validate_bq_table_id(resolved_bq)
                parts = safe_bq.split(".")
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
                  breakdown_by_tool,
                  tags
                FROM `{safe_bq}`
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

      <!-- Tool Cost Attribution Table (Agent -> Tool -> Calls -> Cost) -->
      <div class="card overflow-hidden mb-6">
        <div class="px-4 py-3 border-b border-gray-800 flex items-center justify-between">
          <div>
            <h2 class="text-sm font-semibold text-white flex items-center gap-2">
              <span>🔧 Tool Cost Attribution (Agent ➔ Tool ➔ Calls ➔ Cost)</span>
            </h2>
            <p class="text-xs text-gray-400">Aggregated billable tool & grounding fees across the filtered sessions and agents.</p>
          </div>
          <span id="toolCountBadge" class="text-xs text-emerald-400 font-mono">0 tool calls ($0.0000)</span>
        </div>
        <div class="overflow-x-auto">
          <table class="w-full text-left border-collapse text-xs">
            <thead>
              <tr class="bg-gray-900/90 text-gray-400 border-b border-gray-800">
                <th class="py-2 px-4">🤖 Agent Name</th>
                <th class="py-2 px-4">🔧 Tool Name</th>
                <th class="py-2 px-4 text-right"># Calls</th>
                <th class="py-2 px-4 text-right">Avg Cost / Call</th>
                <th class="py-2 px-4 text-right">Total Tool Cost (USD)</th>
              </tr>
            </thead>
            <tbody id="toolBreakdownBody" class="divide-y divide-gray-800/70 font-mono"></tbody>
          </table>
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
                <th class="py-2.5 px-3 text-right">Tools (Calls / Fee)</th>
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

      function escapeHtml(val) {
        if (val === null || val === undefined) return '';
        return String(val)
          .replace(/&/g, '&amp;')
          .replace(/</g, '&lt;')
          .replace(/>/g, '&gt;')
          .replace(/"/g, '&quot;')
          .replace(/'/g, '&#39;');
      }

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

      function selectHierarchyFromEl(el) {
        const rootName = el.getAttribute('data-root') || 'ALL';
        const subName = el.getAttribute('data-sub') || 'ALL';
        selectHierarchyTarget(rootName, subName);
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
        sel.innerHTML = `<option value="ALL">${escapeHtml(allLabel)}</option>` +
          items.map(x => `<option value="${escapeHtml(x.value)}">${escapeHtml(x.label)}</option>`).join('');
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
        document.getElementById('kpiCostSub').innerText = `LLM: $${llmCost.toFixed(4)} • Tools: $${toolCost.toFixed(4)}`;
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
        renderToolBreakdownTable(chartSourceRows.length ? chartSourceRows : kpiRows);
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
                models: new Set(),
                tools: {}
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
            const bbt = r.breakdown_by_tool || {};
            const agentTools = bbt[cName] || {};
            Object.entries(agentTools).forEach(([tName, tStats]) => {
              if (!c.tools[tName]) c.tools[tName] = { calls: 0, cost: 0 };
              c.tools[tName].calls += (tStats && tStats.calls) || 0;
              c.tools[tName].cost += (tStats && tStats.total_cost_usd) || 0;
            });
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
            const toolPills = Object.entries(c.tools || {}).map(([tName, tStats]) =>
              `<span class="inline-flex items-center gap-1 px-1.5 py-0.5 rounded bg-emerald-950/80 text-emerald-300 border border-emerald-800/70 text-[10px] font-mono">🔧 ${escapeHtml(tName)} ×${Number(tStats.calls || 0)} ($${Number(tStats.cost || 0).toFixed(4)})</span>`
            ).join(' ');
            const modelsDisplay = escapeHtml([...c.models].join(', ') || 'default');
            return `
              <div data-root="${escapeHtml(rName)}" data-sub="${escapeHtml(c.name)}" onclick="selectHierarchyFromEl(this)" class="cursor-pointer rounded-lg border p-3 transition ${borderCls}">
                <div class="flex items-center justify-between gap-2 mb-1">
                  <div class="flex items-center gap-1.5 truncate">
                    ${roleBadge}
                    <span class="text-xs font-semibold text-white truncate">${escapeHtml(c.name)}</span>
                  </div>
                  <span class="text-xs font-bold text-white font-mono">$${c.cost.toFixed(5)}</span>
                </div>
                <div class="flex items-center justify-between text-[11px] text-gray-400 mb-1">
                  <span class="truncate">Model: <span class="text-gray-300 font-mono">${modelsDisplay}</span></span>
                  <span class="text-cyan-300 font-mono">${sharePct}% of parent</span>
                </div>
                <div class="text-[10px] text-gray-400 font-mono flex justify-between mb-1.5">
                  <span>LLM: $${c.llmCost.toFixed(4)}</span>
                  <span class="${c.toolCost > 0 ? 'text-emerald-300 font-semibold' : ''}">Tools: $${c.toolCost.toFixed(4)}</span>
                </div>
                <div class="w-full bg-gray-800 h-1.5 rounded-full overflow-hidden mb-1.5">
                  <div class="${barColor} h-1.5 rounded-full" style="width: ${Math.min(100, Math.max(2, parseFloat(sharePct)))}%"></div>
                </div>
                <div class="text-[10px] text-gray-400 font-mono flex justify-between">
                  <span>Tokens: ${c.tokens.toLocaleString()}</span>
                  <span>In:${c.prompt} Out:${c.completion} Think:${c.thoughts}</span>
                </div>
                ${toolPills ? `<div class="mt-2 pt-1.5 border-t border-gray-800/80 flex flex-wrap gap-1">${toolPills}</div>` : ''}
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
                  <span class="text-sm font-bold text-white font-mono">👑 ${escapeHtml(rName)}</span>
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
                  <button data-root="${escapeHtml(rName)}" data-sub="ALL" onclick="selectHierarchyFromEl(this)" class="text-xs px-3 py-1.5 rounded-lg font-medium ${isSelectedRoot ? 'bg-indigo-600 text-white' : 'bg-gray-800 hover:bg-gray-700 text-indigo-300 border border-gray-700'}">
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

      function renderToolBreakdownTable(rows) {
        const agg = {};
        let totalCalls = 0;
        let totalFee = 0;

        rows.forEach(r => {
          const bbt = r.breakdown_by_tool || {};
          Object.entries(bbt).forEach(([aName, tMap]) => {
            if (r.agent_name && r.agent_name !== aName) return;
            if (tMap && typeof tMap === 'object') {
              Object.entries(tMap).forEach(([tName, stats]) => {
                const key = `${aName}:::${tName}`;
                if (!agg[key]) agg[key] = { agent: aName, tool: tName, calls: 0, cost: 0 };
                const calls = (stats && stats.calls) || 0;
                const cost = (stats && stats.total_cost_usd) || 0;
                agg[key].calls += calls;
                agg[key].cost += cost;
                totalCalls += calls;
                totalFee += cost;
              });
            }
          });
        });

        document.getElementById('toolCountBadge').innerText = `${totalCalls} tool call(s) ($${totalFee.toFixed(4)})`;
        const tbody = document.getElementById('toolBreakdownBody');
        const entries = Object.values(agg).sort((a, b) => b.cost - a.cost);
        if (entries.length === 0) {
          tbody.innerHTML = `<tr><td colspan="5" class="py-3 px-4 text-gray-500 text-center">No billable tool or grounding calls in current filter scope.</td></tr>`;
          return;
        }
        tbody.innerHTML = entries.map(item => {
          const avg = item.calls > 0 ? item.cost / item.calls : 0;
          return `<tr class="hover:bg-gray-900/50">
            <td class="py-2 px-4 text-cyan-300 font-semibold">🤖 ${escapeHtml(item.agent)}</td>
            <td class="py-2 px-4 text-emerald-300">🔧 ${escapeHtml(item.tool)}</td>
            <td class="py-2 px-4 text-right text-gray-200">${Number(item.calls || 0)}</td>
            <td class="py-2 px-4 text-right text-gray-400">$${avg.toFixed(4)}</td>
            <td class="py-2 px-4 text-right font-bold text-amber-300">$${item.cost.toFixed(4)}</td>
          </tr>`;
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
          const outcomeLabel = escapeHtml((r.status || 'success').toUpperCase());
          const ts = escapeHtml((r.timestamp || '').replace('T', ' ').slice(0, 19));
          const safeRoot = escapeHtml(r.root_agent_name || 'root_agent');
          const safeAgent = escapeHtml(r.agent_name || '');
          const rootLabel = `<span class="text-indigo-300 font-semibold">👑 ${safeRoot}</span>`;
          let agentRoleCell = '';
          if (!r.agent_name) {
            agentRoleCell = `<span class="px-2 py-0.5 rounded bg-amber-950/90 text-amber-300 border border-amber-700/80 text-[11px] font-bold">∑ OVERALL PARENT TOTAL (${safeRoot})</span>`;
          } else if (r.agent_name === r.root_agent_name) {
            agentRoleCell = `<span class="text-indigo-200 font-semibold">👑 ${safeAgent} <span class="text-[10px] text-indigo-400">[ROOT DIRECT]</span></span>`;
          } else {
            agentRoleCell = `<span class="text-cyan-300">↳ 🤖 ${safeAgent} <span class="text-[10px] text-gray-400">[SUB-AGENT]</span></span>`;
          }
          const tCalls = Number(r.tool_calls_count || 0);
          const tCost = Number(r.tool_cost_usd || 0);
          const toolCell = tCalls > 0
            ? `<span class="text-emerald-300">${tCalls} call${tCalls === 1 ? '' : 's'} ($${tCost.toFixed(4)})</span>`
            : `<span class="text-gray-600">—</span>`;
          const safeTitle = escapeHtml(r.error || r.source || '');
          const safeSourceOrErr = r.error
            ? `<span class="text-rose-400">⚠️ ${escapeHtml(r.error)}</span>`
            : escapeHtml(r.source || '');
          return `<tr class="hover:bg-gray-900/50 ${!r.agent_name ? 'bg-amber-950/10' : ''}">
            <td class="py-2 px-3 text-gray-400">${ts}</td>
            <td class="py-2 px-3 text-indigo-300 font-medium">${escapeHtml(r.session_id)}</td>
            <td class="py-2 px-3">${rootLabel}</td>
            <td class="py-2 px-3">${agentRoleCell}</td>
            <td class="py-2 px-3 text-gray-300">${escapeHtml(r.model_name || 'ALL')}</td>
            <td class="py-2 px-3"><span class="px-2 py-0.5 rounded border text-[10px] font-semibold ${badgeClass}">${outcomeLabel}</span></td>
            <td class="py-2 px-3 text-right text-gray-300">${Number(r.total_tokens||0).toLocaleString()} <span class="text-gray-500">(${Number(r.prompt_tokens||0)}/${Number(r.completion_tokens||0)}/${Number(r.thoughts_tokens||0)})</span></td>
            <td class="py-2 px-3 text-right text-amber-400">${Number(r.cached_tokens||0).toLocaleString()}</td>
            <td class="py-2 px-3 text-right">${toolCell}</td>
            <td class="py-2 px-3 text-right font-semibold text-white">$${Number(r.total_cost_usd||0).toFixed(6)}</td>
            <td class="py-2 px-3 text-right text-emerald-400">${r.savings_usd > 0 ? '$' + Number(r.savings_usd).toFixed(6) : '—'}</td>
            <td class="py-2 px-3 text-gray-400 truncate max-w-xs" title="${safeTitle}">${safeSourceOrErr}</td>
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
    api_key: str | None = None,
) -> Any:
    """Creates a FastAPI application serving the near-live FinOps dashboard."""
    from fastapi import FastAPI, HTTPException, Request
    from fastapi.responses import HTMLResponse, JSONResponse

    globals()["Request"] = Request

    app = FastAPI(title="ADK FinOps Live Tracker", version="0.8.0")
    configured_api_key = (api_key or os.environ.get("ADK_FINOPS_DASHBOARD_API_KEY") or "").strip()

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
    async def api_ingest(request: Request) -> dict[str, Any]:
        """Allows remote agents across an organization to push telemetry rows directly to a central dashboard."""
        if configured_api_key:
            provided_key = (
                request.headers.get("X-FinOps-Key")
                or request.headers.get("Authorization", "").removeprefix("Bearer ").strip()
                or ""
            ).strip()
            if not provided_key or not hmac.compare_digest(provided_key, configured_api_key):
                raise HTTPException(status_code=401, detail="Invalid or missing X-FinOps-Key / Bearer token.")

        try:
            payload = await request.json()
        except Exception:
            raise HTTPException(status_code=400, detail="Invalid JSON payload.")
        if not isinstance(payload, dict):
            raise HTTPException(status_code=400, detail="JSON payload must be an object.")

        raw_rows_in = payload.get("rows", [payload] if "session_id" in payload else [])
        if not isinstance(raw_rows_in, list):
            raise HTTPException(status_code=400, detail="'rows' must be a list of telemetry dictionaries.")
        if len(raw_rows_in) > MAX_INGEST_BATCH_ROWS:
            raise HTTPException(
                status_code=413,
                detail=f"Batch size {len(raw_rows_in)} exceeds MAX_INGEST_BATCH_ROWS ({MAX_INGEST_BATCH_ROWS}).",
            )

        rows_in = [r for r in raw_rows_in if isinstance(r, dict)]
        count = 0
        for raw in rows_in:
            ingest_row_in_memory(raw, source="http_ingest")
            count += 1

        # Also persist normalized rows to local JSONL on the central dashboard server if log_dir is configured
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


def generate_finops_report(
    log_dir: str | Path | None = "logs",
    bigquery_table: str | None = None,
    include_bigquery: bool = True,
    root_agent_filter: str | None = None,
    agent_filter: str | None = None,
    status_filter: str = "all",
    output_format: str = "table",
    output_path: str | Path | None = None,
    print_report: bool = True,
) -> dict[str, Any]:
    """Generates a unified FinOps telemetry report from local JSONL/CSV files and/or BigQuery.

    Deduplicates records by `(session_id, scope, agent_name)` across local logs and BigQuery,
    computes executive KPIs, hierarchical Root -> Sub-Agent attribution, model breakdown,
    and tool fees, and renders as a Rich terminal table, GitHub Markdown, or JSON.
    """
    eff_log_dir = None if str(log_dir).strip().lower() in ("", "none", "null") else log_dir
    telemetry = collect_telemetry_rows(
        log_dir=eff_log_dir,
        include_bigquery=include_bigquery,
        bigquery_table=bigquery_table,
    )
    all_rows = telemetry.get("rows", [])

    # If a session has 'session'-scope rows, exclude its 'turn'-scope rows to prevent double-counting
    sessions_with_session_scope = {
        r.get("session_id") for r in all_rows if r.get("scope") == "session" and r.get("session_id")
    }
    scope_deduped = [
        r
        for r in all_rows
        if not (r.get("scope") == "turn" and r.get("session_id") in sessions_with_session_scope)
    ]

    # Apply optional filters
    filtered: list[dict[str, Any]] = []
    for r in scope_deduped:
        if root_agent_name := (r.get("root_agent_name") or ""):
            if root_agent_filter and root_agent_name != root_agent_filter:
                continue
        elif root_agent_filter:
            continue
        if agent_filter and (r.get("agent_name") or "") != agent_filter:
            continue
        if status_filter == "success" and r.get("is_failure"):
            continue
        if status_filter == "failed" and not r.get("is_failure"):
            continue
        filtered.append(r)

    # Separate session-level rollups (to avoid double-counting against per-agent rows) and per-agent rows
    rollup_rows = [r for r in filtered if r.get("agent_role") == "root_rollup" or not r.get("agent_name")]
    sub_agent_rows = [r for r in filtered if r.get("agent_role") in ("root_self", "sub_agent") and r.get("agent_name")]

    # Use sub_agent_rows when its total spend exceeds rollup_rows (e.g. when a generic session_id like 's1' is reused across different agents)
    rollup_cost = sum(float(r.get("total_cost_usd", 0.0)) for r in rollup_rows)
    sub_agent_cost = sum(float(r.get("total_cost_usd", 0.0)) for r in sub_agent_rows)
    if (agent_filter and not rollup_rows) or (sub_agent_rows and sub_agent_cost > rollup_cost + 1e-6):
        kpi_base = sub_agent_rows
    else:
        kpi_base = rollup_rows or sub_agent_rows

    unique_sessions = {r.get("session_id") for r in (rollup_rows + sub_agent_rows) if r.get("session_id")}
    failed_sessions = {r.get("session_id") for r in (rollup_rows + sub_agent_rows) if r.get("session_id") and r.get("is_failure")}
    success_sessions = unique_sessions - failed_sessions

    total_cost_usd = round(sum(float(r.get("total_cost_usd", 0.0)) for r in kpi_base), 6)
    llm_cost_usd = round(sum(float(r.get("llm_cost_usd", 0.0)) for r in kpi_base), 6)
    tool_cost_usd = round(sum(float(r.get("tool_cost_usd", 0.0)) for r in kpi_base), 6)
    gross_cost_usd = round(sum(float(r.get("gross_cost_usd", 0.0)) for r in kpi_base), 6)
    savings_usd = round(sum(float(r.get("savings_usd", 0.0)) for r in kpi_base), 6)
    savings_pct = round((savings_usd / gross_cost_usd) * 100.0, 1) if gross_cost_usd > 0 else 0.0

    wasted_cost_usd = round(
        sum(float(r.get("total_cost_usd", 0.0)) for r in kpi_base if r.get("is_failure")), 6
    )
    effective_cost_usd = round(max(0.0, total_cost_usd - wasted_cost_usd), 6)
    wasted_pct = round((wasted_cost_usd / total_cost_usd) * 100.0, 1) if total_cost_usd > 0 else 0.0

    prompt_tokens = sum(int(r.get("prompt_tokens", 0)) for r in kpi_base)
    completion_tokens = sum(int(r.get("completion_tokens", 0)) for r in kpi_base)
    thoughts_tokens = sum(int(r.get("thoughts_tokens", 0)) for r in kpi_base)
    cached_tokens = sum(int(r.get("cached_tokens", 0)) for r in kpi_base)
    total_tokens = sum(int(r.get("total_tokens", 0)) for r in kpi_base)

    # Per-Agent Breakdown (from sub_agent_rows, falling back to rollup_rows if only single-agent rows exist)
    agent_source_rows = sub_agent_rows if sub_agent_rows else rollup_rows
    by_agent: dict[str, dict[str, Any]] = {}
    for r in agent_source_rows:
        a_name = r.get("agent_name") or r.get("root_agent_name") or "root_agent"
        root_name = r.get("root_agent_name") or a_name
        entry = by_agent.setdefault(
            a_name,
            {
                "agent_name": a_name,
                "root_agent_name": root_name,
                "sessions": set(),
                "total_tokens": 0,
                "llm_cost_usd": 0.0,
                "tool_cost_usd": 0.0,
                "total_cost_usd": 0.0,
                "savings_usd": 0.0,
            },
        )
        if r.get("session_id"):
            entry["sessions"].add(r["session_id"])
        entry["total_tokens"] += int(r.get("total_tokens", 0))
        entry["llm_cost_usd"] = round(entry["llm_cost_usd"] + float(r.get("llm_cost_usd", 0.0)), 6)
        entry["tool_cost_usd"] = round(entry["tool_cost_usd"] + float(r.get("tool_cost_usd", 0.0)), 6)
        entry["total_cost_usd"] = round(entry["total_cost_usd"] + float(r.get("total_cost_usd", 0.0)), 6)
        entry["savings_usd"] = round(entry["savings_usd"] + float(r.get("savings_usd", 0.0)), 6)

    agent_denom_cost = sum(info["total_cost_usd"] for info in by_agent.values()) or total_cost_usd
    agent_list: list[dict[str, Any]] = []
    for a_name, info in sorted(by_agent.items(), key=lambda x: x[1]["total_cost_usd"], reverse=True):
        pct = round((info["total_cost_usd"] / agent_denom_cost) * 100.0, 1) if agent_denom_cost > 0 else 0.0
        agent_list.append(
            {
                "agent_name": a_name,
                "root_agent_name": info["root_agent_name"],
                "sessions_count": len(info["sessions"]),
                "total_tokens": info["total_tokens"],
                "llm_cost_usd": info["llm_cost_usd"],
                "tool_cost_usd": info["tool_cost_usd"],
                "total_cost_usd": info["total_cost_usd"],
                "savings_usd": info["savings_usd"],
                "share_pct": pct,
            }
        )

    # Per-Model Breakdown
    by_model: dict[str, dict[str, Any]] = {}
    for r in agent_source_rows:
        m_raw = r.get("model_name") or "unknown"
        for m_name in [m.strip() for m in str(m_raw).split(",") if m.strip()]:
            m_entry = by_model.setdefault(
                m_name,
                {
                    "model_name": m_name,
                    "rows": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thoughts_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "llm_cost_usd": 0.0,
                    "savings_usd": 0.0,
                },
            )
            m_entry["rows"] += 1
            m_entry["prompt_tokens"] += int(r.get("prompt_tokens", 0))
            m_entry["completion_tokens"] += int(r.get("completion_tokens", 0))
            m_entry["thoughts_tokens"] += int(r.get("thoughts_tokens", 0))
            m_entry["cached_tokens"] += int(r.get("cached_tokens", 0))
            m_entry["total_tokens"] += int(r.get("total_tokens", 0))
            m_entry["llm_cost_usd"] = round(m_entry["llm_cost_usd"] + float(r.get("llm_cost_usd", 0.0)), 6)
            m_entry["savings_usd"] = round(m_entry["savings_usd"] + float(r.get("savings_usd", 0.0)), 6)

    model_list = sorted(by_model.values(), key=lambda x: x["llm_cost_usd"], reverse=True)

    # Per-Tool Breakdown
    by_tool: dict[tuple[str, str], dict[str, Any]] = {}
    for r in agent_source_rows:
        tb = r.get("breakdown_by_tool")
        if isinstance(tb, dict):
            for ag_key, t_map in tb.items():
                if isinstance(t_map, dict):
                    for t_name, t_info in t_map.items():
                        calls = int(t_info.get("calls", 0)) if isinstance(t_info, dict) else int(t_info or 0)
                        cost = float(t_info.get("total_cost_usd", 0.0)) if isinstance(t_info, dict) else 0.0
                        item = by_tool.setdefault(
                            (ag_key, t_name),
                            {"agent_name": ag_key, "tool_name": t_name, "calls": 0, "total_cost_usd": 0.0},
                        )
                        item["calls"] += calls
                        item["total_cost_usd"] = round(item["total_cost_usd"] + cost, 6)

    tool_list = sorted(by_tool.values(), key=lambda x: x["total_cost_usd"], reverse=True)

    report_data: dict[str, Any] = {
        "sources": telemetry.get("sources", []),
        "log_dir": str(eff_log_dir) if eff_log_dir else None,
        "bigquery_table": telemetry.get("bigquery_table") if include_bigquery else None,
        "bigquery_error": telemetry.get("bigquery_error"),
        "kpis": {
            "total_sessions": len(unique_sessions),
            "success_sessions": len(success_sessions),
            "failed_sessions": len(failed_sessions),
            "total_cost_usd": total_cost_usd,
            "llm_cost_usd": llm_cost_usd,
            "tool_cost_usd": tool_cost_usd,
            "gross_cost_usd": gross_cost_usd,
            "savings_usd": savings_usd,
            "savings_pct": savings_pct,
            "effective_cost_usd": effective_cost_usd,
            "wasted_cost_usd": wasted_cost_usd,
            "wasted_pct": wasted_pct,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thoughts_tokens": thoughts_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
        },
        "by_agent": agent_list,
        "by_model": model_list,
        "by_tool": tool_list,
        "recent_sessions": [
            {
                "session_id": r.get("session_id"),
                "root_agent_name": r.get("root_agent_name") or r.get("agent_name") or "root_agent",
                "status": r.get("status", "success"),
                "total_tokens": int(r.get("total_tokens", 0)),
                "savings_usd": float(r.get("savings_usd", 0.0)),
                "total_cost_usd": float(r.get("total_cost_usd", 0.0)),
                "source": r.get("source", "local"),
            }
            for r in kpi_base[:15]
        ],
    }

    fmt = (output_format or "table").strip().lower()
    rendered_text = ""

    if fmt == "json":
        rendered_text = json.dumps(report_data, indent=2)
        if print_report:
            print(rendered_text)
    elif fmt == "markdown":
        k = report_data["kpis"]
        src_str = ", ".join(report_data["sources"]) or "none"
        lines = [
            "# 📊 ADK FinOps Telemetry Report",
            "",
            f"- **Data Sources Merged:** `{src_str}`",
        ]
        if report_data["bigquery_table"]:
            lines.append(f"- **BigQuery Table:** `{report_data['bigquery_table']}`")
        if report_data["log_dir"]:
            lines.append(f"- **Local Log Directory:** `{report_data['log_dir']}`")
        lines.extend(
            [
                "",
                "## 1. Executive KPIs",
                "",
                "| Metric | Value | Details |",
                "| :--- | :--- | :--- |",
                f"| **Total Net Spend** | **${k['total_cost_usd']:.6f}** | LLM: `${k['llm_cost_usd']:.6f}` • Tool Fees: `${k['tool_cost_usd']:.6f}` |",
                f"| **Context Caching Savings** | **${k['savings_usd']:.6f}** (`{k['savings_pct']:.1f}%`) | Gross without caching: `${k['gross_cost_usd']:.6f}` |",
                f"| **Effective vs. Wasted Spend** | **${k['effective_cost_usd']:.6f}** effective | Wasted on failed runs: `${k['wasted_cost_usd']:.6f}` (`{k['wasted_pct']:.1f}%`) |",
                f"| **Sessions Analyzed** | **{k['total_sessions']}** | ✅ Success: `{k['success_sessions']}` • ❌ Failed: `{k['failed_sessions']}` |",
                f"| **Total Tokens** | **{k['total_tokens']:,}** | In: `{k['prompt_tokens']:,}` (Cached: `{k['cached_tokens']:,}`) • Out: `{k['completion_tokens']:,}` (Thoughts: `{k['thoughts_tokens']:,}`) |",
                "",
                "## 2. Per-Agent Cost Attribution",
                "",
                "| Root Agent | Agent Name | Sessions | Tokens | LLM Cost | Tool Fees | Total Cost | Share % |",
                "| :--- | :--- | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for a in report_data["by_agent"]:
            lines.append(
                f"| `{a['root_agent_name']}` | **`{a['agent_name']}`** | {a['sessions_count']} | {a['total_tokens']:,} | ${a['llm_cost_usd']:.6f} | ${a['tool_cost_usd']:.6f} | **${a['total_cost_usd']:.6f}** | {a['share_pct']:.1f}% |"
            )
        lines.extend(
            [
                "",
                "## 3. Per-Model Breakdown",
                "",
                "| Model | Records | Prompt Tokens | Completion Tokens | Thoughts | Cached | LLM Cost | Savings |",
                "| :--- | ---: | ---: | ---: | ---: | ---: | ---: | ---: |",
            ]
        )
        for m in report_data["by_model"]:
            lines.append(
                f"| **`{m['model_name']}`** | {m['rows']} | {m['prompt_tokens']:,} | {m['completion_tokens']:,} | {m['thoughts_tokens']:,} | {m['cached_tokens']:,} | **${m['llm_cost_usd']:.6f}** | ${m['savings_usd']:.6f} |"
            )
        rendered_text = "\n".join(lines) + "\n"
        if print_report:
            print(rendered_text)
    else:
        # Rich terminal table output
        try:
            from rich.console import Console
            from rich.panel import Panel
            from rich.table import Table

            console = Console()
            k = report_data["kpis"]
            src_str = ", ".join(report_data["sources"]) or "none"
            header_lines = [
                f"[bold cyan]Sources Merged:[/bold cyan] {src_str}"
                + (f"  |  [bold cyan]BigQuery:[/bold cyan] {report_data['bigquery_table']}" if report_data["bigquery_table"] else ""),
                f"[bold green]Total Net Spend:[/bold green] ${k['total_cost_usd']:.6f} (LLM: ${k['llm_cost_usd']:.6f} | Tools: ${k['tool_cost_usd']:.6f})   "
                f"[bold yellow]Caching Savings:[/bold yellow] ${k['savings_usd']:.6f} ({k['savings_pct']:.1f}%)",
                f"[bold white]Sessions:[/bold white] {k['total_sessions']} (✅ {k['success_sessions']} success | ❌ {k['failed_sessions']} failed)   "
                f"[bold red]Wasted Spend:[/bold red] ${k['wasted_cost_usd']:.6f} ({k['wasted_pct']:.1f}%)   "
                f"[bold magenta]Total Tokens:[/bold magenta] {k['total_tokens']:,}",
            ]
            if report_data.get("bigquery_error"):
                header_lines.append(f"[yellow]⚠️ BigQuery Note: {report_data['bigquery_error']}[/yellow]")

            if print_report:
                console.print(Panel("\n".join(header_lines), title="📊 ADK FinOps Unified Telemetry Report (Local + BigQuery)", border_style="cyan"))

                ag_table = Table(title="🤖 Per-Agent Cost Attribution (Root ➔ Sub-Agent)", show_header=True, header_style="bold cyan")
                ag_table.add_column("Root Agent", style="dim")
                ag_table.add_column("Agent Name", style="bold white")
                ag_table.add_column("Sessions", justify="right")
                ag_table.add_column("Tokens", justify="right")
                ag_table.add_column("LLM Cost", justify="right")
                ag_table.add_column("Tool Fees", justify="right")
                ag_table.add_column("Total Cost", justify="right", style="bold green")
                ag_table.add_column("Share %", justify="right")
                for a in report_data["by_agent"]:
                    ag_table.add_row(
                        str(a["root_agent_name"]),
                        str(a["agent_name"]),
                        str(a["sessions_count"]),
                        f"{a['total_tokens']:,}",
                        f"${a['llm_cost_usd']:.6f}",
                        f"${a['tool_cost_usd']:.6f}",
                        f"${a['total_cost_usd']:.6f}",
                        f"{a['share_pct']:.1f}%",
                    )
                console.print(ag_table)

                m_table = Table(title="🧠 Per-Model Spend & Token Breakdown", show_header=True, header_style="bold magenta")
                m_table.add_column("Model", style="bold white")
                m_table.add_column("Records", justify="right")
                m_table.add_column("Prompt", justify="right")
                m_table.add_column("Completion", justify="right")
                m_table.add_column("Thoughts", justify="right")
                m_table.add_column("Cached", justify="right")
                m_table.add_column("LLM Cost", justify="right", style="bold green")
                m_table.add_column("Savings", justify="right", style="yellow")
                for m in report_data["by_model"]:
                    m_table.add_row(
                        str(m["model_name"]),
                        str(m["rows"]),
                        f"{m['prompt_tokens']:,}",
                        f"{m['completion_tokens']:,}",
                        f"{m['thoughts_tokens']:,}",
                        f"{m['cached_tokens']:,}",
                        f"${m['llm_cost_usd']:.6f}",
                        f"${m['savings_usd']:.6f}",
                    )
                console.print(m_table)
        except Exception:
            if print_report:
                print(json.dumps(report_data, indent=2))

    if output_path:
        out_p = Path(output_path)
        out_p.parent.mkdir(parents=True, exist_ok=True)
        out_content = rendered_text if rendered_text else json.dumps(report_data, indent=2)
        out_p.write_text(out_content, encoding="utf-8")
        if print_report:
            print(f"\n💾 Saved FinOps report to {out_p.resolve()}")

    return report_data


def cli_main() -> None:
    """CLI entrypoint for `adk-finops` (`dashboard`, `report`, `sync-rates`, `extract-pricing`)."""
    parser = argparse.ArgumentParser(
        prog="adk-finops",
        description="ADK FinOps CLI: Near-Live Dashboard, Unified Local/BigQuery Reporting, & Pricing Sync",
    )
    parser.add_argument(
        "command",
        nargs="?",
        default="dashboard",
        choices=["dashboard", "report", "sync-rates", "extract-pricing"],
        help="Command to execute: 'dashboard' (default), 'report', 'sync-rates', or 'extract-pricing'",
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
        help="Directory containing local timestamped JSONL/CSV FinOps logs (default: logs; pass 'none' for BigQuery-only)",
    )
    parser.add_argument(
        "--bigquery-table",
        type=str,
        default=os.getenv("ADK_FINOPS_BIGQUERY_TABLE"),
        help="Optional BigQuery table ID (project.dataset.table) for historical + cloud sync",
    )
    parser.add_argument(
        "--no-bigquery",
        action="store_true",
        help="Disable BigQuery querying in 'report' (use local JSONL/CSV logs only)",
    )
    parser.add_argument(
        "--format",
        type=str,
        default="table",
        choices=["table", "markdown", "json"],
        help="Output format for 'report': 'table' (Rich CLI), 'markdown' (GitHub PR table), or 'json'",
    )
    parser.add_argument(
        "--root-agent",
        type=str,
        default=None,
        help="Filter 'report' by Root Agent name",
    )
    parser.add_argument(
        "--agent",
        type=str,
        default=None,
        help="Filter 'report' by Sub-Agent name",
    )
    parser.add_argument(
        "--status",
        type=str,
        default="all",
        choices=["all", "success", "failed"],
        help="Filter 'report' by session outcome status",
    )
    parser.add_argument(
        "--api-key",
        type=str,
        default=os.getenv("ADK_FINOPS_DASHBOARD_API_KEY"),
        help="Optional shared secret API key to require X-FinOps-Key / Bearer auth on POST /api/ingest",
    )
    # Rate card sync & pricing extraction flags
    parser.add_argument(
        "--url",
        type=str,
        default=None,
        help="Remote rate card URL for 'sync-rates'",
    )
    parser.add_argument(
        "--cache-path",
        type=str,
        default=None,
        help="Local cache file path for 'sync-rates'",
    )
    parser.add_argument(
        "--force",
        action="store_true",
        help="Bypass local 24h cache TTL and force a fresh remote fetch in 'sync-rates'",
    )
    parser.add_argument(
        "--update-default",
        action="store_true",
        help="Update bundled src/adk_finops/rates/default_rates.json during 'extract-pricing'",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Output file path for 'report' or 'extract-pricing'",
    )
    parser.add_argument(
        "--include-new-models",
        action="store_true",
        help="Also discover and add new flagship models during 'extract-pricing'",
    )
    parser.add_argument(
        "--gcp-api-key",
        type=str,
        default=None,
        help="Optional Google Cloud Billing Catalog API key for 'extract-pricing'",
    )
    args = parser.parse_args()

    if args.command == "report":
        generate_finops_report(
            log_dir=args.log_dir,
            bigquery_table=args.bigquery_table,
            include_bigquery=not args.no_bigquery,
            root_agent_filter=args.root_agent,
            agent_filter=args.agent,
            status_filter=args.status,
            output_format=args.format,
            output_path=args.output,
            print_report=True,
        )
        return

    if args.command == "sync-rates":
        res = CostTracker.sync_remote_rate_card(
            url=args.url,
            cache_path=args.cache_path,
            force=args.force,
        )
        print("\n🔄 ADK FinOps — Dynamic Remote Rate Card Sync")
        print(f"   • Status:        {res.get('status')}")
        print(f"   • Source:        {res.get('source')}")
        if res.get("cache_path"):
            print(f"   • Local Cache:   {res.get('cache_path')}")
        print(f"   • Models Loaded: {res.get('models_loaded')}")
        if res.get("error"):
            print(f"   • Fallback Note: {res.get('error')}")
        print()
        return

    if args.command == "extract-pricing":
        from .pricing_extractor import extract_and_sync_pricing

        extract_and_sync_pricing(
            update_default=args.update_default,
            output_path=args.output,
            include_new_models=args.include_new_models,
            gcp_billing_api_key=args.gcp_api_key,
            print_report=True,
        )
        return

    import uvicorn

    app = create_dashboard_app(
        log_dir=args.log_dir,
        bigquery_table=args.bigquery_table,
        api_key=args.api_key,
    )
    print(f"\n📊 ADK FinOps Near-Live Dashboard running at: http://{args.host}:{args.port}")
    print(f"   • Watching local log directory: {Path(args.log_dir).resolve()}")
    if args.api_key:
        print("   • Ingest Auth (/api/ingest):    Enabled (X-FinOps-Key / Bearer token required)")
    if args.bigquery_table:
        print(f"   • BigQuery table configured: {args.bigquery_table}")
    print("   • Press Ctrl+C to stop.\n")
    uvicorn.run(app, host=args.host, port=args.port, log_level="warning")


if __name__ == "__main__":
    cli_main()
