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

"""Base exporter interface for adk-finops telemetry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from datetime import datetime, timezone
import json
from typing import Any


class BaseExporter(ABC):
    """Abstract base class for telemetry exporters (e.g. BigQuery, JSONL, CSV, OpenTelemetry)."""

    @abstractmethod
    def export_summary(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        """Exports a FinOps summary dictionary to the external sink."""
        pass

    def close(self) -> None:
        """Flushes and releases any exporter resources or background thread pools."""
        pass

    def _serialize_row(
        self,
        scope_data: dict[str, Any],
        scope_name: str,
        session_id: str,
        turn_id: str | None,
        budget_info: dict[str, Any] | None,
        agent_name: str | None = None,
        model_name: str | None = None,
        tags: dict[str, Any] | None = None,
        include_status_fields: bool = False,
    ) -> dict[str, Any]:
        """Maps a FinOps summary dictionary to a flat telemetry row."""
        now_utc = datetime.now(timezone.utc).isoformat()
        b_limit = None
        b_util = None
        b_exceeded = False
        if budget_info:
            b_limit = budget_info.get("budget_limit_usd")
            b_util = budget_info.get("utilization_pct")
            b_exceeded = bool(budget_info.get("exceeded", False))

        breakdown_by_agent = scope_data.get("breakdown_by_agent")
        breakdown_by_model = scope_data.get("breakdown_by_model")

        row: dict[str, Any] = {
            "timestamp": now_utc,
            "session_id": session_id,
            "turn_id": turn_id,
            "scope": scope_name,
            "agent_name": agent_name,
            "model_name": model_name,
            "prompt_tokens": int(scope_data.get("prompt_tokens", 0)),
            "completion_tokens": int(scope_data.get("completion_tokens", 0)),
            "thoughts_tokens": int(scope_data.get("thoughts_tokens", 0)),
            "cached_tokens": int(scope_data.get("cached_tokens", 0)),
            "total_tokens": int(scope_data.get("total_tokens", 0)),
            "llm_cost_usd": float(scope_data.get("llm_cost_usd", 0.0)),
            "tool_cost_usd": float(scope_data.get("tool_cost_usd", 0.0)),
            "total_cost_usd": float(scope_data.get("total_cost_usd", 0.0)),
            "gross_cost_usd": float(scope_data.get("gross_cost_usd", 0.0)),
            "savings_usd": float(scope_data.get("savings_usd", 0.0)),
            "savings_pct": float(scope_data.get("savings_pct", 0.0)),
            "tool_calls_count": int(
                scope_data.get("total_tool_calls", scope_data.get("tool_calls", 0))
            ),
            "budget_limit_usd": float(b_limit) if b_limit is not None else None,
            "budget_utilization_pct": float(b_util) if b_util is not None else None,
            "budget_exceeded": b_exceeded,
            "breakdown_by_agent": json.dumps(breakdown_by_agent) if breakdown_by_agent else None,
            "breakdown_by_model": json.dumps(breakdown_by_model) if breakdown_by_model else None,
            "tags": json.dumps(tags) if tags else None,
        }

        if include_status_fields:
            status = scope_data.get("status") or (tags or {}).get("status") or "success"
            is_failure = scope_data.get("is_failure")
            if is_failure is None:
                is_failure = bool((tags or {}).get("is_failure", False))
            error = scope_data.get("error") or (tags or {}).get("error")
            row["status"] = status
            row["is_failure"] = bool(is_failure)
            row["error"] = error

        return row

    def _prepare_rows(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
        include_status_fields: bool = False,
    ) -> list[dict[str, Any]]:
        """Prepares the list of rows to export based on scope ('turn', 'session', or 'both')."""
        rows: list[dict[str, Any]] = []
        sess_info = summary.get("session", summary)
        turn_info = summary.get("turn", {})
        budget_info = summary.get("budget", sess_info.get("budget", {}))

        session_id = sess_info.get("run_id") or summary.get("session_id") or "default_session"
        turn_id = turn_info.get("run_id") or summary.get("turn_id")

        scopes_to_export = []
        if scope in ("turn", "both") and turn_info:
            scopes_to_export.append(("turn", turn_info))
        if scope in ("session", "both") and sess_info:
            scopes_to_export.append(("session", sess_info))

        for scope_name, scope_data in scopes_to_export:
            models_used = list(scope_data.get("breakdown_by_model", {}).keys())
            scope_model = models_used[0] if len(models_used) == 1 else None

            rows.append(
                self._serialize_row(
                    scope_data=scope_data,
                    scope_name=scope_name,
                    session_id=session_id,
                    turn_id=turn_id,
                    budget_info=budget_info,
                    agent_name=None,
                    model_name=scope_model,
                    tags=tags,
                    include_status_fields=include_status_fields,
                )
            )

            agent_breakdown = scope_data.get("breakdown_by_agent", {})
            for a_name, a_data in agent_breakdown.items():
                rows.append(
                    self._serialize_row(
                        scope_data=a_data,
                        scope_name=scope_name,
                        session_id=session_id,
                        turn_id=turn_id,
                        budget_info=budget_info,
                        agent_name=a_name,
                        model_name=a_data.get("model_name"),
                        tags=tags,
                        include_status_fields=include_status_fields,
                    )
                )

            if not agent_breakdown and len(models_used) > 1:
                for m_name, m_data in scope_data.get("breakdown_by_model", {}).items():
                    rows.append(
                        self._serialize_row(
                            scope_data=m_data,
                            scope_name=scope_name,
                            session_id=session_id,
                            turn_id=turn_id,
                            budget_info=budget_info,
                            agent_name=None,
                            model_name=m_name,
                            tags=tags,
                            include_status_fields=include_status_fields,
                        )
                    )

        return rows

