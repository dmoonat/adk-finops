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
from adk_finops.tracker import BudgetExceededError, CostTracker


def test_multi_agent_cost_attribution_tracker():
    CostTracker.set_enabled(True)
    turn_id = "turn_multi_1"
    session_id = "sess_multi_1"

    CostTracker.start_turn(turn_id, session_id)

    # 1. Supervisor agent plans the task
    CostTracker.record_usage(
        run_id=turn_id,
        session_id=session_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=1000,
        completion_tokens=200,
        agent_name="supervisor",
    )

    # 2. Researcher agent runs search and calls Gemini Flash
    CostTracker.record_usage(
        run_id=turn_id,
        session_id=session_id,
        model_name="gemini-2.5-flash",
        prompt_tokens=4000,
        completion_tokens=500,
        cached_tokens=2000,
        agent_name="researcher",
    )
    CostTracker.record_tool_call(
        run_id=turn_id,
        session_id=session_id,
        tool_name="vertex_grounding_google_search",
        count=2,
        agent_name="researcher",
    )

    # 3. Coder agent writes code using CodeMender
    CostTracker.record_usage(
        run_id=turn_id,
        session_id=session_id,
        model_name="codemender",
        prompt_tokens=8000,
        completion_tokens=1500,
        thoughts_tokens=300,
        agent_name="coder",
    )

    summary = CostTracker.get_summary(run_id=turn_id, session_id=session_id, pop=False)
    assert summary is not None

    agent_breakdown = summary["session"]["breakdown_by_agent"]
    assert "supervisor" in agent_breakdown
    assert "researcher" in agent_breakdown
    assert "coder" in agent_breakdown

    # Verify supervisor
    sup = agent_breakdown["supervisor"]
    assert sup["calls"] == 1
    assert sup["prompt_tokens"] == 1000
    assert sup["completion_tokens"] == 200
    assert sup["tool_cost_usd"] == 0.0
    assert sup["total_cost_usd"] > 0

    # Verify researcher
    res = agent_breakdown["researcher"]
    assert res["calls"] == 1
    assert res["prompt_tokens"] == 4000
    assert res["cached_tokens"] == 2000
    assert res["savings_usd"] > 0
    assert res["tool_cost_usd"] == 0.028  # 2 queries * $0.014
    assert res["total_cost_usd"] == round(res["llm_cost_usd"] + 0.028, 7)

    # Verify coder
    cdr = agent_breakdown["coder"]
    assert cdr["calls"] == 1
    assert cdr["prompt_tokens"] == 8000
    assert cdr["completion_tokens"] == 1500
    assert cdr["thoughts_tokens"] == 300
    assert cdr["total_tokens"] == 9800  # 8000 + 1500 + 300


def test_agent_budget_guardrails():
    CostTracker.set_enabled(True)
    CostTracker.reset_budgets()
    turn_id = "turn_agent_budget"
    session_id = "sess_agent_budget"

    # Set budget limit for "researcher" to $0.02
    CostTracker.set_budget(
        session_id=session_id,
        agent_limits_usd={"researcher": 0.02, "coder": 1.00},
    )

    CostTracker.start_turn(turn_id, session_id)

    # Coder runs first - cost is small (~$0.005), well under $1.00
    CostTracker.record_usage(
        run_id=turn_id,
        session_id=session_id,
        model_name="gemini-2.5-flash",
        prompt_tokens=5000,
        completion_tokens=500,
        agent_name="coder",
    )
    exceeded, _, _ = CostTracker.check_budget(turn_id, session_id, agent_name="coder")
    assert not exceeded

    # Researcher runs 2 search queries ($0.028) -> exceeds $0.02 budget limit!
    CostTracker.record_tool_call(
        run_id=turn_id,
        session_id=session_id,
        tool_name="vertex_grounding_google_search",
        count=2,
        agent_name="researcher",
    )
    exceeded, reason, status = CostTracker.check_budget(turn_id, session_id, agent_name="researcher")
    assert exceeded
    assert "researcher" in reason
    assert status["scope"] == "agent"
    assert status["current_cost_usd"] >= 0.028

    # Coder should still NOT be exceeded
    exceeded_coder, _, _ = CostTracker.check_budget(turn_id, session_id, agent_name="coder")
    assert not exceeded_coder


async def test_plugin_multi_agent_attribution_and_halt():
    CostTracker.reset_budgets()
    plugin = FinOpsCostPlugin(
        agent_budgets={"runaway_agent": 0.001},
        on_budget_exceeded="halt",
    )

    callback_context = MagicMock()
    callback_context.invocation_id = "turn_plugin_agent"
    callback_context.session.id = "sess_plugin_agent"
    callback_context.agent_name = "runaway_agent"

    CostTracker.start_turn("turn_plugin_agent", "sess_plugin_agent")

    llm_response = MagicMock()
    llm_response.model_version = "gemini-2.5-pro"
    llm_response.usage_metadata.prompt_token_count = 5000
    llm_response.usage_metadata.candidates_token_count = 1000
    llm_response.usage_metadata.thoughts_token_count = 0
    llm_response.usage_metadata.cached_content_token_count = 0

    # Running gemini-2.5-pro with 5k prompt / 1k completion will cost ~$0.019, exceeding $0.001
    try:
        await plugin.after_model_callback(
            callback_context=callback_context,
            llm_response=llm_response,
        )
        assert False, "Should have raised BudgetExceededError"
    except BudgetExceededError as e:
        assert "runaway_agent" in str(e)
        assert e.scope == "agent"
