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

"""Unit tests for Latency, Streaming TTFT, and Token Throughput tracking (ADK Plugin & Standalone)."""

from __future__ import annotations

import asyncio
import time
from types import SimpleNamespace
from unittest.mock import MagicMock

from adk_finops import CostTracker, FinOpsCostPlugin
from adk_finops.display import format_plain_summary_box, format_summary_box
from adk_finops.exporters.base import BaseExporter
from adk_finops.exporters.bigquery import BigQueryExporter
from adk_finops.exporters.local import CSV_FIELDNAMES
from adk_finops.exporters.otel import OpenTelemetryExporter


def test_explicit_latency_and_throughput_aggregation():
    """Verifies explicit latency_ms, ttft_ms, P95, and tokens/sec across turn, session, model, agent, and tool."""
    CostTracker.reset()
    CostTracker.start_turn(turn_id="turn_lat_1", session_id="sess_lat_1", root_agent_name="orchestrator")

    # Call 1: 500ms latency, 100ms TTFT, 200 completion + 50 thoughts = 250 output tokens (500 tok/s output)
    # 1000 prompt + 250 output = 1250 total tokens (2500 tok/s total)
    rec1 = CostTracker.record_usage(
        run_id="turn_lat_1",
        session_id="sess_lat_1",
        model_name="gemini-2.5-pro",
        prompt_tokens=1000,
        completion_tokens=200,
        thoughts_tokens=50,
        latency_ms=500.0,
        ttft_ms=100.0,
        agent_name="orchestrator",
    )
    assert rec1["latency_ms"] == 500.0
    assert rec1["ttft_ms"] == 100.0
    assert rec1["output_tokens_per_sec"] == 500.0
    assert rec1["total_tokens_per_sec"] == 2500.0

    # Call 2: 1500ms latency, 300ms TTFT, 300 completion tokens (200 tok/s output)
    rec2 = CostTracker.record_usage(
        run_id="turn_lat_1",
        session_id="sess_lat_1",
        model_name="gemini-2.5-pro",
        prompt_tokens=1200,
        completion_tokens=300,
        thoughts_tokens=0,
        latency_ms=1500.0,
        ttft_ms=300.0,
        agent_name="orchestrator",
    )
    assert rec2["latency_ms"] == 1500.0
    assert rec2["ttft_ms"] == 300.0
    assert rec2["output_tokens_per_sec"] == 200.0

    # Tool Call: 250ms latency
    CostTracker.record_tool_call(
        run_id="turn_lat_1",
        session_id="sess_lat_1",
        tool_name="google_search",
        agent_name="orchestrator",
        latency_ms=250.0,
    )

    # Record Turn wall-clock latency: 2300ms
    CostTracker.record_turn_latency(run_id="turn_lat_1", session_id="sess_lat_1", latency_ms=2300.0)

    summary = CostTracker.get_summary(run_id="turn_lat_1", session_id="sess_lat_1", pop=False)
    assert summary is not None
    sess = summary["session"]
    turn = summary["turn"]

    # Cumulative LLM latency = 500 + 1500 = 2000ms (2.0s)
    # Total output tokens = 250 + 300 = 550 -> 550 / 2.0s = 275.0 tok/s
    # Total tokens = 1250 + 1500 = 2750 -> 2750 / 2.0s = 1375.0 tok/s
    assert sess["llm_latency_ms"] == 2000.0
    assert sess["tool_latency_ms"] == 250.0
    assert sess["turn_latency_ms"] == 2300.0
    assert sess["latency_ms"] == 2300.0
    assert sess["avg_llm_latency_ms"] == 1000.0
    assert sess["p95_llm_latency_ms"] == 1450.0
    assert sess["avg_ttft_ms"] == 200.0
    assert sess["output_tokens_per_sec"] == 275.0
    assert sess["total_tokens_per_sec"] == 1375.0
    assert turn["latency_ms"] == 2300.0

    # Ensure internal keys are stripped from get_summary()
    assert "_start_ts_ns" not in sess
    assert "_llm_latencies_ms" not in sess

    # Model breakdown
    m_bd = sess["breakdown_by_model"]["gemini-2.5-pro"]
    assert m_bd["llm_latency_ms"] == 2000.0
    assert m_bd["avg_llm_latency_ms"] == 1000.0
    assert m_bd["avg_ttft_ms"] == 200.0
    assert m_bd["output_tokens_per_sec"] == 275.0

    # Agent breakdown
    a_bd = sess["breakdown_by_agent"]["orchestrator"]
    assert a_bd["llm_latency_ms"] == 2000.0
    assert a_bd["tool_latency_ms"] == 250.0
    assert a_bd["latency_ms"] == 2250.0
    assert a_bd["output_tokens_per_sec"] == 275.0

    # Tool breakdown
    t_bd = sess["breakdown_by_tool"]["orchestrator"]["google_search"]
    assert t_bd["latency_ms"] == 250.0
    assert t_bd["avg_latency_ms"] == 250.0


def test_standalone_track_llm_call_and_preflight_auto_delta():
    """Verifies standalone `CostTracker.track_llm_call`, `track_tool_call`, `track_run`, and preflight delta timing."""
    CostTracker.reset()

    with CostTracker.track_run("standalone_run_1"):
        with CostTracker.track_llm_call(
            run_id="standalone_run_1",
            model_name="gemini-2.5-flash",
            agent_name="standalone_worker",
        ) as call:
            time.sleep(0.01)
            call.mark_first_token()
            time.sleep(0.01)
            mock_resp = SimpleNamespace(
                model_version="gemini-2.5-flash",
                usage_metadata=SimpleNamespace(
                    prompt_token_count=400,
                    candidates_token_count=100,
                    thoughts_token_count=20,
                    cached_content_token_count=0,
                ),
            )
            call.set_response(mock_resp)

        assert call.ttft_ms is not None and call.ttft_ms >= 5.0
        assert call.latency_ms >= call.ttft_ms
        assert call.record["output_tokens_per_sec"] > 0.0

        with CostTracker.track_tool_call(
            run_id="standalone_run_1",
            tool_name="paid_api",
            custom_cost_usd=0.02,
            agent_name="standalone_worker",
        ) as tc:
            time.sleep(0.005)

        assert tc.latency_ms >= 2.0
        assert tc.cost_usd == 0.02

    summary = CostTracker.get_summary("standalone_run_1", pop=False)
    assert summary is not None
    assert summary["turn_latency_ms"] >= summary["llm_latency_ms"]
    assert summary["llm_latency_ms"] > 0.0
    assert summary["tool_latency_ms"] > 0.0
    assert summary["avg_ttft_ms"] > 0.0

    # Also test automatic preflight -> record_usage latency delta when latency_ms is not passed
    CostTracker.start_run("preflight_auto_run")
    CostTracker.check_preflight_budget(
        model_name="gemini-2.5-flash",
        estimated_prompt_tokens=500,
        run_id="preflight_auto_run",
    )
    time.sleep(0.01)
    auto_rec = CostTracker.record_usage(
        run_id="preflight_auto_run",
        model_name="gemini-2.5-flash",
        prompt_tokens=500,
        completion_tokens=100,
    )
    assert auto_rec["latency_ms"] >= 5.0
    assert auto_rec["output_tokens_per_sec"] > 0.0


def test_plugin_streaming_ttft_tool_and_turn_latency(capsys):
    """Verifies FinOpsCostPlugin callbacks measure TTFT, LLM latency, tool latency, and turn latency."""
    CostTracker.reset()
    plugin = FinOpsCostPlugin(
        default_model="gemini-2.5-flash",
        tool_rates={"paid_search": 0.01},
        render_terminal_box=False,
    )

    inv_ctx = MagicMock()
    inv_ctx.invocation_id = "turn_plug_lat"
    inv_ctx.session.id = "sess_plug_lat"
    inv_ctx.session.state = {}
    inv_ctx.session_service = None
    inv_ctx.agent.name = "researcher"
    inv_ctx.agent.sub_agents = []

    cb_ctx = MagicMock()
    cb_ctx.invocation_id = "turn_plug_lat"
    cb_ctx.session.id = "sess_plug_lat"
    cb_ctx.agent.name = "researcher"
    cb_ctx.agent.model = "gemini-2.5-flash"

    llm_req = SimpleNamespace(
        model="gemini-2.5-flash",
        contents=[],
        config=None,
        tools_dict={},
    )

    async def _simulate_turn():
        await plugin.before_run_callback(invocation_context=inv_ctx)
        await plugin.before_model_callback(callback_context=cb_ctx, llm_request=llm_req)
        await asyncio.sleep(0.01)

        # First streamed partial chunk -> records TTFT
        partial_resp = SimpleNamespace(
            partial=True,
            model_version="gemini-2.5-flash",
            usage_metadata=SimpleNamespace(
                prompt_token_count=300,
                candidates_token_count=20,
                thoughts_token_count=0,
                cached_content_token_count=0,
            ),
        )
        await plugin.after_model_callback(callback_context=cb_ctx, llm_response=partial_resp)
        await asyncio.sleep(0.01)

        # Final non-partial response -> records total LLM latency + throughput
        final_resp = SimpleNamespace(
            partial=False,
            model_version="gemini-2.5-flash",
            usage_metadata=SimpleNamespace(
                prompt_token_count=300,
                candidates_token_count=120,
                thoughts_token_count=30,
                cached_content_token_count=0,
            ),
        )
        await plugin.after_model_callback(callback_context=cb_ctx, llm_response=final_resp)

        # Tool call
        class PaidSearchTool:
            name = "paid_search"

        tool_obj = PaidSearchTool()
        await plugin.before_tool_callback(tool=tool_obj, tool_args={"q": "adk"}, tool_context=cb_ctx)
        await asyncio.sleep(0.01)
        await plugin.after_tool_callback(
            tool=tool_obj,
            tool_args={"q": "adk"},
            tool_context=cb_ctx,
            result={"items": [1, 2]},
        )

        await plugin.after_run_callback(invocation_context=inv_ctx)

    asyncio.run(_simulate_turn())
    out = capsys.readouterr().out
    assert "latency=" in out
    assert "ttft=" in out
    assert "throughput=" in out
    assert "turn_latency=" in out

    summary = CostTracker.get_summary("turn_plug_lat", "sess_plug_lat", pop=False)
    assert summary is not None
    sess = summary["session"]
    assert sess["ttft_ms"] >= 5.0
    assert sess["llm_latency_ms"] >= sess["ttft_ms"]
    assert sess["tool_latency_ms"] >= 5.0
    assert sess["turn_latency_ms"] >= sess["llm_latency_ms"]
    assert sess["output_tokens_per_sec"] > 0.0

    # Verify display rendering (Rich & Plain) and Exporters serialization
    plain_box = format_plain_summary_box(summary)
    assert "Perf:" in plain_box
    assert "tok/s" in plain_box

    rich_box = format_summary_box(summary)
    assert "Latency" in rich_box
    assert "Throughput" in rich_box

    row = BaseExporter._serialize_row(
        None,  # type: ignore[arg-type]
        scope_data=sess,
        scope_name="session",
        session_id="sess_plug_lat",
        turn_id="turn_plug_lat",
        budget_info=None,
        agent_name="researcher",
        model_name="gemini-2.5-flash",
        tags=None,
    )
    for col in (
        "latency_ms",
        "llm_latency_ms",
        "tool_latency_ms",
        "ttft_ms",
        "output_tokens_per_sec",
        "total_tokens_per_sec",
    ):
        assert col in row and row[col] > 0.0
        assert col in CSV_FIELDNAMES

    bq_row = BigQueryExporter._serialize_row(
        None,  # type: ignore[arg-type]
        scope_data=sess,
        scope_name="session",
        session_id="sess_plug_lat",
        turn_id="turn_plug_lat",
        budget_info=None,
        agent_name="researcher",
        model_name="gemini-2.5-flash",
        tags=None,
    )
    assert bq_row["latency_ms"] == row["latency_ms"]
    assert bq_row["output_tokens_per_sec"] == row["output_tokens_per_sec"]

    otel_attrs = OpenTelemetryExporter.build_otel_attributes(row)
    assert otel_attrs["gen_ai.finops.latency_ms"] == row["latency_ms"]
    assert otel_attrs["gen_ai.finops.ttft_ms"] == row["ttft_ms"]
    assert otel_attrs["gen_ai.finops.output_tokens_per_sec"] == row["output_tokens_per_sec"]
