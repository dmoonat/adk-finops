# Copyright 2026 Google LLC
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

"""Local File Exporters (JSONL & CSV) for adk-finops.

Provides zero-cloud-dependency telemetry exporters that persist session, turn,
and per-agent FinOps metrics (including task outcomes and wasted spend flags)
to local JSON Lines (.jsonl) or CSV (.csv) files for analysis with DuckDB,
Pandas, jq, or spreadsheets.
"""

from __future__ import annotations

from datetime import datetime
import csv
import json
import logging
from pathlib import Path
import threading
from typing import Any

from .base import BaseExporter

logger = logging.getLogger("adk_finops.exporters.local")

# Shared per-process run timestamp so JSONL and CSV files created in the same run
# are placed inside the exact same `<dir>/<YYYYMMDD_HHMMSS>/` folder.
_RUN_TIMESTAMP = datetime.now().strftime("%Y%m%d_%H%M%S")

CSV_FIELDNAMES = [
    "timestamp",
    "session_id",
    "turn_id",
    "scope",
    "agent_name",
    "model_name",
    "status",
    "is_failure",
    "error",
    "prompt_tokens",
    "completion_tokens",
    "thoughts_tokens",
    "cached_tokens",
    "total_tokens",
    "llm_cost_usd",
    "tool_cost_usd",
    "total_cost_usd",
    "gross_cost_usd",
    "savings_usd",
    "savings_pct",
    "tool_calls_count",
    "budget_limit_usd",
    "budget_utilization_pct",
    "budget_exceeded",
    "breakdown_by_agent",
    "breakdown_by_model",
    "tags",
]


def _resolve_timestamped_path(
    file_path: str | Path,
    default_filename: str,
    timestamp_dir: bool = True,
) -> Path:
    """Places the output file inside a `<parent>/<YYYYMMDD_HHMMSS>/<filename>` folder."""
    raw_path = Path(file_path)
    if raw_path.suffix == "":
        # User passed a directory path like "logs"
        base_dir = raw_path
        filename = default_filename
    else:
        base_dir = raw_path.parent if str(raw_path.parent) not in ("", ".") else Path("logs")
        filename = raw_path.name

    if timestamp_dir:
        return base_dir / _RUN_TIMESTAMP / filename
    return raw_path if raw_path.suffix != "" else (base_dir / filename)


class JSONLExporter(BaseExporter):
    """Exports FinOps cost and token telemetry to a local JSON Lines (.jsonl) file inside a timestamped run folder."""

    def __init__(
        self,
        file_path: str | Path = "logs/finops_costs.jsonl",
        timestamp_dir: bool = True,
    ):
        self.timestamp_dir = timestamp_dir
        self.file_path = _resolve_timestamped_path(
            file_path=file_path,
            default_filename="finops_costs.jsonl",
            timestamp_dir=timestamp_dir,
        )
        self._lock = threading.Lock()

    def export_summary(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        """Appends FinOps summary rows as JSON Lines to `self.file_path`."""
        rows = self._prepare_rows(
            summary=summary,
            scope=scope,
            tags=tags,
            include_status_fields=True,
        )
        if not rows:
            return

        try:
            with self._lock:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                with self.file_path.open("a", encoding="utf-8") as f:
                    for row in rows:
                        f.write(json.dumps(row, ensure_ascii=False) + "\n")
            msg = f"[FinOps JSONL] Saved {len(rows)} row(s) to {self.file_path}"
            print(msg, flush=True)
            logger.info(msg)
        except Exception as e:
            logger.warning(
                f"[FinOps JSONL] Failed to write telemetry to {self.file_path}: {e}"
            )


class CSVExporter(BaseExporter):
    """Exports FinOps cost and token telemetry to a local CSV (.csv) file inside a timestamped run folder."""

    def __init__(
        self,
        file_path: str | Path = "logs/finops_costs.csv",
        timestamp_dir: bool = True,
    ):
        self.timestamp_dir = timestamp_dir
        self.file_path = _resolve_timestamped_path(
            file_path=file_path,
            default_filename="finops_costs.csv",
            timestamp_dir=timestamp_dir,
        )
        self._lock = threading.Lock()

    def export_summary(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        """Appends FinOps summary rows to `self.file_path`, writing headers if new."""
        rows = self._prepare_rows(
            summary=summary,
            scope=scope,
            tags=tags,
            include_status_fields=True,
        )
        if not rows:
            return

        try:
            with self._lock:
                self.file_path.parent.mkdir(parents=True, exist_ok=True)
                write_header = not self.file_path.exists() or self.file_path.stat().st_size == 0
                with self.file_path.open("a", encoding="utf-8", newline="") as f:
                    writer = csv.DictWriter(
                        f,
                        fieldnames=CSV_FIELDNAMES,
                        extrasaction="ignore",
                    )
                    if write_header:
                        writer.writeheader()
                    for row in rows:
                        writer.writerow(row)
            msg = f"[FinOps CSV] Saved {len(rows)} row(s) to {self.file_path}"
            print(msg, flush=True)
            logger.info(msg)
        except Exception as e:
            logger.warning(
                f"[FinOps CSV] Failed to write telemetry to {self.file_path}: {e}"
            )
