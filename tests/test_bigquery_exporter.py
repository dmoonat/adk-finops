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

"""Unit tests for the BigQuery Telemetry Exporter."""

import json
from unittest.mock import MagicMock, patch

from adk_finops import CostTracker, FinOpsCostPlugin
from adk_finops.exporters.bigquery import BigQueryExporter


def _reset_state():
    CostTracker.reset_budgets()
    with CostTracker._lock:
        CostTracker._active_runs.clear()


def test_missing_dependency_raises_import_error():
    """Verifies that an informative ImportError is raised when bigquery is absent."""
    with patch("adk_finops.exporters.bigquery.HAS_BIGQUERY", False):
        error_raised = False
        try:
            BigQueryExporter(table_id="my-proj.ds.table")
        except ImportError as e:
            error_raised = True
            assert "pip install 'adk-finops[bigquery]'" in str(e)
        assert error_raised, "Expected ImportError when bigquery is not installed"


def test_table_creation_with_partitioning_and_clustering():
    """Verifies that table creation sets day partitioning and clustering."""
    mock_client = MagicMock()
    # Mock table not existing on first get_table call
    mock_client.get_table.side_effect = Exception("Not found")

    exporter = BigQueryExporter(
        table_id="test-proj.finops_ds.agent_costs",
        client=mock_client,
        auto_create_table=True,
    )

    exporter.ensure_table_exists()

    assert mock_client.create_table.called
    created_table = mock_client.create_table.call_args[0][0]
    assert created_table.clustering_fields == ["session_id", "agent_name", "model_name"]
    assert created_table.time_partitioning.field == "timestamp"


def test_row_serialization_and_attribution():
    """Verifies complete row serialization matching the BigQuery schema."""
    _reset_state()
    mock_client = MagicMock()
    exporter = BigQueryExporter(
        table_id="test-proj.finops_ds.agent_costs",
        client=mock_client,
        auto_create_table=False,
    )

    CostTracker.start_turn("turn_1", "session_1")
    CostTracker.record_usage(
        run_id="turn_1",
        session_id="session_1",
        model_name="gemini-2.5-pro",
        prompt_tokens=10000,
        completion_tokens=500,
        cached_tokens=8000,
        agent_name="researcher",
    )
    CostTracker.record_tool_call(
        run_id="turn_1",
        session_id="session_1",
        tool_name="google_search",
        count=1,
        agent_name="researcher",
    )

    summary = CostTracker.get_summary(run_id="turn_1", session_id="session_1", pop=False)
    rows = exporter._prepare_rows(
        summary=summary,
        scope="session",
        tags={"env": "test", "tenant": "corp-1"},
    )

    # We expect 2 rows: 1 session aggregate row + 1 attributed agent row ("researcher")
    assert len(rows) == 2

    # 1. Aggregate row
    agg_row = rows[0]
    assert agg_row["session_id"] == "session_1"
    assert agg_row["scope"] == "session"
    assert agg_row["agent_name"] is None
    assert agg_row["cached_tokens"] == 8000
    assert agg_row["total_tokens"] == 10500
    assert agg_row["tool_calls_count"] == 1
    assert agg_row["savings_usd"] > 0
    assert json.loads(agg_row["tags"]) == {"env": "test", "tenant": "corp-1"}
    assert "researcher" in json.loads(agg_row["breakdown_by_agent"])
    agg_tools = json.loads(agg_row["breakdown_by_tool"])
    assert agg_tools["researcher"]["google_search"]["calls"] == 1
    assert agg_tools["researcher"]["google_search"]["total_cost_usd"] > 0

    # 2. Agent row
    agent_row = rows[1]
    assert agent_row["session_id"] == "session_1"
    assert agent_row["agent_name"] == "researcher"
    assert agent_row["total_tokens"] == 10500
    assert agent_row["tool_calls_count"] == 1
    agent_tools = json.loads(agent_row["breakdown_by_tool"])
    assert agent_tools["researcher"]["google_search"]["calls"] == 1
    assert agent_tools["researcher"]["google_search"]["total_cost_usd"] > 0


def test_scope_both_exports_turn_and_session_rows():
    """Verifies that scope='both' creates rows for both turn and session."""
    _reset_state()
    mock_client = MagicMock()
    exporter = BigQueryExporter(
        table_id="test-proj.finops_ds.agent_costs",
        client=mock_client,
        auto_create_table=False,
    )

    CostTracker.start_turn("turn_1", "session_1")
    CostTracker.record_usage(
        run_id="turn_1",
        session_id="session_1",
        model_name="gemini-2.5-flash",
        prompt_tokens=500,
        completion_tokens=100,
        agent_name="assistant",
    )

    summary = CostTracker.get_summary(run_id="turn_1", session_id="session_1", pop=False)
    rows = exporter._prepare_rows(summary=summary, scope="both")

    scopes = [r["scope"] for r in rows]
    assert "turn" in scopes
    assert "session" in scopes


def test_streaming_insert_error_resilience():
    """Verifies that insertion errors or client exceptions do not crash the application."""
    mock_client = MagicMock()
    # Simulate BigQuery streaming error return
    mock_client.insert_rows_json.return_value = [{"index": 0, "errors": ["Quota exceeded"]}]

    exporter = BigQueryExporter(
        table_id="test-proj.finops_ds.agent_costs",
        client=mock_client,
        auto_create_table=False,
    )

    # Should not raise exception
    exporter.export_summary(
        summary={"session": {"run_id": "sess_err", "total_tokens": 100}},
        blocking=True,
    )
    assert mock_client.insert_rows_json.called

    # Simulate client throwing network exception
    mock_client.insert_rows_json.side_effect = ConnectionError("Network unreachable")
    exporter.export_summary(
        summary={"session": {"run_id": "sess_err", "total_tokens": 100}},
        blocking=True,
    )


async def test_plugin_bigquery_integration():
    """Verifies that FinOpsCostPlugin automatically wires up BigQueryExporter."""
    _reset_state()
    mock_exporter = MagicMock()

    with patch("adk_finops.exporters.bigquery.BigQueryExporter", return_value=mock_exporter):
        plugin = FinOpsCostPlugin(
            name="bq_plugin",
            bigquery_table="my-proj.ds.table",
            bigquery_export_scope="both",
            bigquery_tags={"env": "staging"},
        )

        assert plugin.bq_exporter is mock_exporter

        # Simulate after_run_callback
        ctx = MagicMock()
        ctx.invocation_id = "turn_abc"
        ctx.session.id = "session_xyz"
        ctx.session_service = None
        ctx.session.state = {}

        CostTracker.start_turn("turn_abc", "session_xyz")
        CostTracker.record_usage(
            run_id="turn_abc",
            session_id="session_xyz",
            model_name="gemini-2.5-flash",
            prompt_tokens=200,
            completion_tokens=50,
        )

        await plugin.after_run_callback(invocation_context=ctx)

        assert mock_exporter.export_summary.called
        call_kwargs = mock_exporter.export_summary.call_args[1]
        assert call_kwargs["tags"]["env"] == "staging"
        assert call_kwargs["tags"]["status"] == "success"
        assert call_kwargs["blocking"] is False
