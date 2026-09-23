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

import asyncio
from unittest.mock import MagicMock

from adk_finops.plugin import FinOpsCostPlugin
from adk_finops.tracker import CostTracker


def test_tool_fee_discrimination():
    plugin = FinOpsCostPlugin()

    tool_context = MagicMock()
    tool_context.invocation_id = "test_turn_1"
    tool_context.session.id = "test_sess_1"

    CostTracker.start_turn("test_turn_1", "test_sess_1")

    # 1. MCP tool: search_documents should incur NO fee ($0.0)
    class MCPTool:
        name = "search_documents"
        __module__ = "google.adk.tools.mcp_tool.mcp_toolset"

    asyncio.run(
        plugin.after_tool_callback(
            tool=MCPTool(),
            tool_args={},
            tool_context=tool_context,
            result={},
        )
    )

    s1 = CostTracker.get_summary("test_turn_1", "test_sess_1", pop=False)
    assert s1["turn"]["total_tool_calls"] == 0
    assert s1["turn"]["tool_cost_usd"] == 0.0

    # 2. Google Search Grounding tool should incur $0.014 ($14 / 1,000 queries)
    class GoogleSearchTool:
        name = "google_search"
        __module__ = "google.adk.tools.grounding"

    asyncio.run(
        plugin.after_tool_callback(
            tool=GoogleSearchTool(),
            tool_args={},
            tool_context=tool_context,
            result={},
        )
    )

    s2 = CostTracker.get_summary("test_turn_1", "test_sess_1", pop=False)
    assert s2["turn"]["total_tool_calls"] == 1
    assert s2["turn"]["tool_cost_usd"] == 0.014


def test_explicit_billable_decorator_and_tool_rates():
    """Verifies @billable(fee=...), @billable(fee_fn=...), charge_on_error=False, and plugin tool_rates."""
    from adk_finops import billable

    CostTracker._active_runs.clear()
    plugin = FinOpsCostPlugin(
        tool_rates={"paid_mcp_terminal": 0.05},
    )

    tool_context = MagicMock()
    tool_context.invocation_id = "turn_billable_1"
    tool_context.session.id = "sess_billable_1"
    CostTracker.start_turn("turn_billable_1", "sess_billable_1")

    # 1. Fixed fee via @billable(fee=0.015) wrapped inside an ADK FunctionTool-like wrapper
    @billable(fee=0.015, provider="serpapi")
    def patent_search(query: str) -> dict:
        return {"hits": 10}

    class FunctionToolWrapper:
        name = "patent_search"
        func = staticmethod(patent_search)

    asyncio.run(
        plugin.after_tool_callback(
            tool=FunctionToolWrapper(),
            tool_args={"query": "quantum"},
            tool_context=tool_context,
            result={"hits": 10},
        )
    )
    s = CostTracker.get_summary("turn_billable_1", "sess_billable_1", pop=False)
    assert s["turn"]["total_tool_calls"] == 1
    assert s["turn"]["tool_cost_usd"] == 0.015

    # 2. Dynamic fee via @billable(fee_fn=...) + error guard (charge_on_error=False)
    @billable(
        fee_fn=lambda args, res: 0.002 * len(res.get("pages", [])),
        charge_on_error=False,
    )
    def ocr_pdf(uri: str) -> dict:
        return {}

    class OcrToolWrapper:
        name = "ocr_pdf"
        func = staticmethod(ocr_pdf)

    # 2a. Error response -> should NOT be charged ($0.00)
    asyncio.run(
        plugin.after_tool_callback(
            tool=OcrToolWrapper(),
            tool_args={"uri": "gs://bucket/bad.pdf"},
            tool_context=tool_context,
            result={"status": "error", "error": "Corrupt PDF"},
        )
    )
    s_err = CostTracker.get_summary("turn_billable_1", "sess_billable_1", pop=False)
    assert s_err["turn"]["total_tool_calls"] == 1
    assert s_err["turn"]["tool_cost_usd"] == 0.015

    # 2b. Successful 4-page OCR -> should charge 4 * $0.002 = $0.008 (cumulative $0.023)
    asyncio.run(
        plugin.after_tool_callback(
            tool=OcrToolWrapper(),
            tool_args={"uri": "gs://bucket/good.pdf"},
            tool_context=tool_context,
            result={"pages": ["p1", "p2", "p3", "p4"]},
        )
    )
    s_ok = CostTracker.get_summary("turn_billable_1", "sess_billable_1", pop=False)
    assert s_ok["turn"]["total_tool_calls"] == 2
    assert round(s_ok["turn"]["tool_cost_usd"], 6) == 0.023

    # 3. Paid MCP tool configured via FinOpsCostPlugin(tool_rates={"paid_mcp_terminal": 0.05})
    class PaidMCPTool:
        name = "paid_mcp_terminal"
        __module__ = "google.adk.tools.mcp_tool.mcp_toolset"

    asyncio.run(
        plugin.after_tool_callback(
            tool=PaidMCPTool(),
            tool_args={},
            tool_context=tool_context,
            result={"quote": 123.45},
        )
    )
    s_mcp = CostTracker.get_summary("turn_billable_1", "sess_billable_1", pop=False)
    assert s_mcp["turn"]["total_tool_calls"] == 3
    assert round(s_mcp["turn"]["tool_cost_usd"], 6) == 0.073

