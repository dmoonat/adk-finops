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

"""BigQuery telemetry exporter for adk-finops.

Streams turn, session, model, and multi-agent FinOps cost records
directly into a partitioned and clustered Google BigQuery table.
"""

from __future__ import annotations

import concurrent.futures
import json
import logging
import threading
from datetime import datetime, timezone
from typing import Any

from .base import BaseExporter

logger = logging.getLogger("adk_finops.exporters.bigquery")

try:
    from google.cloud import bigquery

    HAS_BIGQUERY = True
except ImportError:
    HAS_BIGQUERY = False
    bigquery = None  # type: ignore[assignment]


import re

_BQ_TABLE_ID_RE = re.compile(r"^[a-zA-Z0-9_:-]+\.[a-zA-Z0-9_]+\.[a-zA-Z0-9_$]+$")


def validate_bq_table_id(table_id: str) -> str:
    """Validates that a BigQuery table identifier strictly matches `project.dataset.table`."""
    cleaned = (table_id or "").strip().strip("`")
    if not _BQ_TABLE_ID_RE.match(cleaned):
        raise ValueError(
            f"Invalid BigQuery table_id '{table_id}'. Expected format 'project.dataset.table' with alphanumeric/underscore characters."
        )
    return cleaned


class BigQueryExporter(BaseExporter):
    """Streams FinOps telemetry into a partitioned & clustered Google BigQuery table."""

    SCHEMA = [
        ("timestamp", "TIMESTAMP", "REQUIRED", "Event timestamp in UTC (Partition Key)"),
        ("session_id", "STRING", "REQUIRED", "ADK Session ID (Clustering Key #1)"),
        ("turn_id", "STRING", "NULLABLE", "ADK Turn or Invocation ID"),
        ("scope", "STRING", "REQUIRED", "Record scope: 'turn' or 'session'"),
        ("agent_name", "STRING", "NULLABLE", "Attributed Agent name (Clustering Key #2)"),
        ("model_name", "STRING", "NULLABLE", "Model name or version (Clustering Key #3)"),
        ("prompt_tokens", "INTEGER", "REQUIRED", "Input prompt token count"),
        ("completion_tokens", "INTEGER", "REQUIRED", "Output candidate token count"),
        ("thoughts_tokens", "INTEGER", "REQUIRED", "Reasoning / thinking token count"),
        ("cached_tokens", "INTEGER", "REQUIRED", "Context cached token count"),
        ("total_tokens", "INTEGER", "REQUIRED", "Total billable tokens"),
        ("llm_cost_usd", "FLOAT", "REQUIRED", "Net LLM API cost in USD"),
        ("tool_cost_usd", "FLOAT", "REQUIRED", "Grounding / tool fees in USD"),
        ("total_cost_usd", "FLOAT", "REQUIRED", "Total net cost in USD"),
        ("gross_cost_usd", "FLOAT", "REQUIRED", "Gross cost before context caching savings"),
        ("savings_usd", "FLOAT", "REQUIRED", "Dollars saved via context caching"),
        ("savings_pct", "FLOAT", "REQUIRED", "Percentage saved via context caching"),
        ("tool_calls_count", "INTEGER", "REQUIRED", "Number of billable tool/grounding calls"),
        ("budget_limit_usd", "FLOAT", "NULLABLE", "Configured budget limit"),
        ("budget_utilization_pct", "FLOAT", "NULLABLE", "Budget percentage utilized"),
        ("budget_exceeded", "BOOLEAN", "REQUIRED", "Whether budget guard was tripped"),
        ("breakdown_by_agent", "JSON", "NULLABLE", "Multi-agent cost and token attribution"),
        ("breakdown_by_model", "JSON", "NULLABLE", "Model cost and token distribution"),
        ("breakdown_by_tool", "JSON", "NULLABLE", "Tool call count and cost attribution by agent and tool"),
        ("tags", "JSON", "NULLABLE", "Custom user-provided tags (e.g. env, tenant_id)"),
    ]

    def __init__(
        self,
        table_id: str,
        client: Any | None = None,
        auto_create_table: bool = True,
        background_workers: int = 2,
    ) -> None:
        if not HAS_BIGQUERY:
            raise ImportError(
                "google-cloud-bigquery is required to use BigQueryExporter. "
                "Install it with: pip install 'adk-finops[bigquery]'"
            )

        self.table_id = validate_bq_table_id(table_id)
        self._client = client
        self.auto_create_table = auto_create_table
        self._table_verified = False
        self._lock = threading.Lock()
        self._executor = concurrent.futures.ThreadPoolExecutor(
            max_workers=background_workers,
            thread_name_prefix="finops-bq",
        )

    @property
    def client(self) -> Any:
        """Lazy-loaded BigQuery client."""
        if self._client is None:
            self._client = bigquery.Client()
        return self._client

    def _build_bq_schema(self) -> list[Any]:
        """Converts internal schema definition to BigQuery SchemaField objects."""
        return [
            bigquery.SchemaField(
                name=name,
                field_type=field_type,
                mode=mode,
                description=desc,
            )
            for name, field_type, mode, desc in self.SCHEMA
        ]

    def ensure_table_exists(self) -> None:
        """Idempotently ensures the target dataset and table exist with partitioning & clustering."""
        if self._table_verified or not self.auto_create_table:
            return

        with self._lock:
            if self._table_verified:
                return

            client = self.client
            table_ref = bigquery.TableReference.from_string(self.table_id)
            dataset_ref = table_ref.dataset_id
            dataset = bigquery.Dataset(f"{table_ref.project}.{dataset_ref}")

            # 1. Ensure dataset exists
            try:
                client.get_dataset(dataset)
            except Exception:
                try:
                    dataset.location = "US"  # Default location; can be overridden in GCP
                    client.create_dataset(dataset, exists_ok=True)
                    logger.info(f"[FinOps BigQuery] Created dataset: {dataset.dataset_id}")
                except Exception as e:
                    logger.debug(f"[FinOps BigQuery] Dataset check/create notice: {e}")

            # 2. Ensure table exists (and auto-add any new NULLABLE schema columns such as breakdown_by_tool)
            try:
                existing_table = client.get_table(table_ref)
                existing_names = {f.name for f in existing_table.schema}
                missing_fields = [
                    f for f in self._build_bq_schema() if f.name not in existing_names
                ]
                if missing_fields:
                    try:
                        existing_table.schema = list(existing_table.schema) + missing_fields
                        client.update_table(existing_table, ["schema"])
                        logger.info(
                            f"[FinOps BigQuery] Upgraded table schema for {self.table_id} with {[f.name for f in missing_fields]}"
                        )
                    except Exception as schema_err:
                        logger.debug(
                            f"[FinOps BigQuery] Schema upgrade notice for {self.table_id}: {schema_err}"
                        )
                self._table_verified = True
            except Exception:
                table = bigquery.Table(table_ref, schema=self._build_bq_schema())
                # Partition by timestamp (DAY)
                table.time_partitioning = bigquery.TimePartitioning(
                    type_=bigquery.TimePartitioningType.DAY,
                    field="timestamp",
                )
                # Cluster by session_id, agent_name, model_name
                table.clustering_fields = ["session_id", "agent_name", "model_name"]

                try:
                    client.create_table(table, exists_ok=True)
                    logger.info(
                        f"[FinOps BigQuery] Created partitioned and clustered table: {self.table_id}"
                    )
                    self._table_verified = True
                except Exception as e:
                    logger.warning(
                        f"[FinOps BigQuery] Could not auto-create table {self.table_id}: {e}"
                    )

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
    ) -> dict[str, Any]:
        """Maps a FinOps summary dictionary to a BigQuery row."""
        now_utc = datetime.now(timezone.utc).isoformat()
        b_limit = None
        b_util = None
        b_exceeded = False
        if budget_info:
            b_limit = budget_info.get("budget_limit_usd")
            b_util = budget_info.get("utilization_pct")
            b_exceeded = bool(budget_info.get("exceeded", False))

        # Handle breakdowns
        breakdown_by_agent = scope_data.get("breakdown_by_agent")
        breakdown_by_model = scope_data.get("breakdown_by_model")
        breakdown_by_tool = scope_data.get("breakdown_by_tool")
        if not breakdown_by_tool and isinstance(scope_data.get("tools"), dict) and scope_data.get("tools"):
            breakdown_by_tool = {agent_name or "root_agent": scope_data["tools"]}

        return {
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
            "tool_calls_count": int(scope_data.get("total_tool_calls", scope_data.get("tool_calls", 0))),
            "budget_limit_usd": float(b_limit) if b_limit is not None else None,
            "budget_utilization_pct": float(b_util) if b_util is not None else None,
            "budget_exceeded": b_exceeded,
            "breakdown_by_agent": json.dumps(breakdown_by_agent) if breakdown_by_agent else None,
            "breakdown_by_model": json.dumps(breakdown_by_model) if breakdown_by_model else None,
            "breakdown_by_tool": json.dumps(breakdown_by_tool) if breakdown_by_tool else None,
            "tags": json.dumps(tags) if tags else None,
        }

    def _prepare_rows(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
    ) -> list[dict[str, Any]]:
        """Prepares the list of rows to insert based on scope ('turn', 'session', or 'both')."""
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
            # 1. Primary scope aggregate row
            # If all calls in this scope used a single model, populate model_name
            models_used = list(scope_data.get("breakdown_by_model", {}).keys())
            scope_model = models_used[0] if len(models_used) == 1 else None

            rows.append(
                self._serialize_row(
                    scope_data=scope_data,
                    scope_name=scope_name,
                    session_id=session_id,
                    turn_id=turn_id,
                    budget_info=budget_info,
                    agent_name=None,  # NULL signifies the overall aggregate for this scope
                    model_name=scope_model,
                    tags=tags,
                )
            )

            # 2. Per-Agent rows (for direct SQL filtering by agent and model)
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
                    )
                )

            # 3. If there are NO agents (e.g. standalone script without agent_name)
            # and multiple models were used, export per-model rows so model_name is never lost!
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
                        )
                    )

        return rows

    def _insert_rows(self, rows: list[dict[str, Any]]) -> None:
        """Internal synchronous insert into BigQuery."""
        if not rows:
            return

        try:
            self.ensure_table_exists()
            client = self.client
            errors = client.insert_rows_json(self.table_id, rows, ignore_unknown_values=True)
            if errors:
                logger.error(
                    f"[FinOps BigQuery] Errors occurred while streaming rows into {self.table_id}: {errors}"
                )
            else:
                msg = f"[FinOps BigQuery] Exported {len(rows)} row(s) to {self.table_id}"
                print(msg, flush=True)
                logger.info(msg)
        except Exception as e:
            logger.error(f"[FinOps BigQuery] Failed to stream rows to {self.table_id}: {e}")

    def export_summary(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        """Exports a FinOps summary dictionary to BigQuery.

        Args:
            summary: The FinOps summary dictionary from CostTracker.get_summary().
            scope: Export scope ('session', 'turn', or 'both'). Default 'session'.
            tags: Optional key-value dictionary of metadata (e.g. {'env': 'prod'}).
            blocking: If True, blocks until the insert completes. If False (default),
                      runs asynchronously in a background thread pool.
        """
        rows = self._prepare_rows(summary=summary, scope=scope, tags=tags)
        if not rows:
            return

        if blocking:
            self._insert_rows(rows)
        else:
            self._executor.submit(self._insert_rows, rows)

    def close(self) -> None:
        """Flushes and shuts down the background thread pool."""
        self._executor.shutdown(wait=True)
