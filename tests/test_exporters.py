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

"""Tests for local file (JSONL, CSV) and OpenTelemetry exporters."""

import csv
import json
from pathlib import Path

from adk_finops import (
    CostTracker,
    CSVExporter,
    FinOpsCostPlugin,
    JSONLExporter,
    OpenTelemetryExporter,
)


def _setup_sample_session(session_id: str, status: str = "success", error: str | None = None):
    CostTracker.start_turn(turn_id=f"{session_id}_t1", session_id=session_id)
    CostTracker.record_usage(
        run_id=f"{session_id}_t1",
        session_id=session_id,
        model_name="gemini-2.5-flash",
        prompt_tokens=1500,
        completion_tokens=300,
        thoughts_tokens=100,
        cached_tokens=500,
        agent_name="research_agent",
    )
    CostTracker.record_task_status(
        session_id=session_id,
        run_id=f"{session_id}_t1",
        status=status,
        error=error,
    )
    return CostTracker.get_summary(run_id=f"{session_id}_t1", session_id=session_id, pop=False)


def test_jsonl_exporter_writes_session_and_agent_rows(tmp_path: Path):
    CostTracker.reset()
    jsonl_file = tmp_path / "telemetry" / "costs.jsonl"
    exporter = JSONLExporter(file_path=jsonl_file)

    summary = _setup_sample_session("sess_jsonl_1", status="failed", error="Max retries exceeded")
    exporter.export_summary(summary, scope="session", tags={"env": "test"})

    # Verify timestamped subfolder was created automatically
    assert exporter.file_path.exists()
    assert exporter.file_path.parent != jsonl_file.parent
    assert exporter.file_path.parent.parent == jsonl_file.parent
    lines = [json.loads(line) for line in exporter.file_path.read_text(encoding="utf-8").splitlines()]
    # 1 aggregate session row + 1 per-agent row ("research_agent")
    assert len(lines) == 2

    agg_row = next(r for r in lines if r["agent_name"] is None)
    agent_row = next(r for r in lines if r["agent_name"] == "research_agent")

    assert agg_row["session_id"] == "sess_jsonl_1"
    assert agg_row["status"] == "failed"
    assert agg_row["is_failure"] is True
    assert agg_row["error"] == "Max retries exceeded"
    assert agg_row["total_tokens"] == 1900
    assert agg_row["total_cost_usd"] > 0

    assert agent_row["model_name"] == "gemini-2.5-flash"
    assert agent_row["prompt_tokens"] == 1500


def test_csv_exporter_writes_headers_and_rows(tmp_path: Path):
    CostTracker.reset()
    csv_file = tmp_path / "costs.csv"
    exporter = CSVExporter(file_path=csv_file)

    summary1 = _setup_sample_session("sess_csv_1", status="success")
    summary2 = _setup_sample_session("sess_csv_2", status="error", error="DB timeout")

    exporter.export_summary(summary1, scope="session")
    exporter.export_summary(summary2, scope="session")

    assert exporter.file_path.exists()
    assert exporter.file_path.parent != tmp_path
    with exporter.file_path.open("r", encoding="utf-8", newline="") as f:
        reader = list(csv.DictReader(f))

    # 2 sessions x (1 aggregate + 1 agent row) = 4 rows
    assert len(reader) == 4
    assert reader[0]["session_id"] == "sess_csv_1"
    assert reader[0]["status"] == "success"
    assert reader[0]["is_failure"] == "False"

    err_rows = [r for r in reader if r["session_id"] == "sess_csv_2" and not r["agent_name"]]
    assert len(err_rows) == 1
    assert err_rows[0]["status"] == "error"
    assert err_rows[0]["is_failure"] == "True"
    assert err_rows[0]["error"] == "DB timeout"


def test_opentelemetry_exporter_records_attributes():
    CostTracker.reset()
    otel_exporter = OpenTelemetryExporter(emit_child_spans=False, enrich_current_span=False)

    summary = _setup_sample_session("sess_otel_1", status="budget_exceeded", error="Over $0.05 cap")
    otel_exporter.export_summary(summary, scope="session")

    assert len(otel_exporter.recorded_attributes) == 2
    agg_attrs = next(
        a for a in otel_exporter.recorded_attributes if "gen_ai.agent.name" not in a
    )
    assert agg_attrs["gen_ai.finops.session_id"] == "sess_otel_1"
    assert agg_attrs["gen_ai.finops.task_outcome"] == "budget_exceeded"
    assert agg_attrs["gen_ai.finops.is_wasted_spend"] is True
    assert agg_attrs["gen_ai.usage.total_tokens"] == 1900
    assert agg_attrs["gen_ai.usage.cost_usd"] > 0


def test_plugin_multi_exporter_dispatch(tmp_path: Path):
    CostTracker.reset()
    jsonl_file = tmp_path / "plugin_costs.jsonl"
    csv_file = tmp_path / "plugin_costs.csv"

    plugin = FinOpsCostPlugin(
        jsonl_path=jsonl_file,
        csv_path=csv_file,
        enable_otel=True,
    )

    _setup_sample_session("sess_plugin_multi", status="pending")
    # Simulate an exception/failure that triggers record_task_status auto-export
    plugin.record_task_status(
        session_id="sess_plugin_multi",
        status="failed",
        error="Validation rejected after 3 loops",
    )

    jsonl_exp = next(e for e in plugin.exporters if isinstance(e, JSONLExporter))
    csv_exp = next(e for e in plugin.exporters if isinstance(e, CSVExporter))

    assert jsonl_exp.file_path.exists()
    assert csv_exp.file_path.exists()
    # Both JSONL and CSV should be placed inside the same timestamped subfolder
    assert jsonl_exp.file_path.parent == csv_exp.file_path.parent

    jsonl_rows = [json.loads(line) for line in jsonl_exp.file_path.read_text().splitlines()]
    assert any(r["session_id"] == "sess_plugin_multi" and r["status"] == "failed" for r in jsonl_rows)


def test_dashboard_collect_telemetry_rows(tmp_path: Path):
    from adk_finops.dashboard import collect_telemetry_rows

    CostTracker.reset()
    jsonl_file = tmp_path / "logs" / "finops_costs.jsonl"
    exporter = JSONLExporter(file_path=jsonl_file)

    CostTracker.register_agent_hierarchy("coordinator_agent", parent_agent_name=None, root_agent_name="coordinator_agent")
    CostTracker.register_agent_hierarchy("research_agent", parent_agent_name="coordinator_agent", root_agent_name="coordinator_agent")
    CostTracker.start_turn(turn_id="turn_h1", session_id="sess_dash_01", root_agent_name="coordinator_agent")
    CostTracker.record_usage(
        run_id="turn_h1",
        session_id="sess_dash_01",
        model_name="gemini-2.5-flash",
        prompt_tokens=400,
        completion_tokens=100,
        agent_name="coordinator_agent",
    )
    CostTracker.record_usage(
        run_id="turn_h1",
        session_id="sess_dash_01",
        model_name="gemini-2.5-pro",
        prompt_tokens=1200,
        completion_tokens=500,
        thoughts_tokens=200,
        agent_name="research_agent",
    )
    CostTracker.record_task_status(session_id="sess_dash_01", run_id="turn_h1", status="success")
    summary = CostTracker.get_summary(run_id="turn_h1", session_id="sess_dash_01", pop=False)
    exporter.export_summary(summary, scope="session")

    payload = collect_telemetry_rows(log_dir=tmp_path / "logs", include_bigquery=False)
    assert "local_jsonl" in payload["sources"]
    rows = [r for r in payload["rows"] if r["session_id"] == "sess_dash_01"]
    assert len(rows) == 3

    rollup_row = next(r for r in rows if r["agent_role"] == "root_rollup")
    root_self_row = next(r for r in rows if r["agent_role"] == "root_self")
    sub_row = next(r for r in rows if r["agent_role"] == "sub_agent")

    assert rollup_row["root_agent_name"] == "coordinator_agent"
    assert rollup_row["total_tokens"] == 2400
    assert root_self_row["agent_name"] == "coordinator_agent"
    assert root_self_row["root_agent_name"] == "coordinator_agent"
    assert sub_row["agent_name"] == "research_agent"
    assert sub_row["root_agent_name"] == "coordinator_agent"
    assert sub_row["parent_agent_name"] == "coordinator_agent"
