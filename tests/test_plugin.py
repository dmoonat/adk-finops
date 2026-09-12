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

import asyncio
from unittest.mock import MagicMock

from adk_finops.plugin import FinOpsCostPlugin
from adk_finops.tracker import CostTracker


async def test_tool_fee_discrimination():
    plugin = FinOpsCostPlugin()

    tool_context = MagicMock()
    tool_context.invocation_id = "test_turn_1"
    tool_context.session.id = "test_sess_1"

    CostTracker.start_turn("test_turn_1", "test_sess_1")

    # 1. MCP tool: search_documents should incur NO fee ($0.0)
    class MCPTool:
        name = "search_documents"
        __module__ = "google.adk.tools.mcp_tool.mcp_toolset"

    await plugin.after_tool_callback(
        tool=MCPTool(),
        tool_args={},
        tool_context=tool_context,
        result={},
    )

    s1 = CostTracker.get_summary("test_turn_1", "test_sess_1", pop=False)
    assert s1["turn"]["total_tool_calls"] == 0
    assert s1["turn"]["tool_cost_usd"] == 0.0

    # 2. Google Search Grounding tool should incur $0.035
    class GoogleSearchTool:
        name = "google_search"
        __module__ = "google.adk.tools.grounding"

    await plugin.after_tool_callback(
        tool=GoogleSearchTool(),
        tool_args={},
        tool_context=tool_context,
        result={},
    )

    s2 = CostTracker.get_summary("test_turn_1", "test_sess_1", pop=False)
    assert s2["turn"]["total_tool_calls"] == 1
    assert s2["turn"]["tool_cost_usd"] == 0.035
