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


def test_tracker_budget_checks():
    CostTracker.set_enabled(True)
    sess_id = "test_budget_sess_1"
    turn_id = "test_budget_turn_1"

    # Set session budget of $0.10 and turn budget of $0.005
    CostTracker.set_budget(session_id=sess_id, session_limit_usd=0.10, turn_limit_usd=0.005)
    CostTracker.start_turn(turn_id=turn_id, session_id=sess_id)

    # 1. Under budget call: $0.002
    CostTracker.record_usage(
        run_id=turn_id,
        session_id=sess_id,
        model_name="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=200,
    )
    is_exceeded, reason, status = CostTracker.check_budget(run_id=turn_id, session_id=sess_id)
    assert not is_exceeded
    assert status["utilization_pct"] > 0
    assert not status["exceeded"]

    # 2. Exceed turn budget ($0.005): Add a large call in this turn
    CostTracker.record_usage(
        run_id=turn_id,
        session_id=sess_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=5000,
        completion_tokens=1000,
    )
    is_exceeded, reason, status = CostTracker.check_budget(run_id=turn_id, session_id=sess_id)
    assert is_exceeded
    assert status["scope"] == "turn"
    assert "Turn budget limit" in reason


def test_summary_includes_budget():
    sess_id = "test_budget_sess_2"
    CostTracker.set_budget(session_id=sess_id, session_limit_usd=0.50)
    CostTracker.start_run(run_id=sess_id)

    summary = CostTracker.get_summary(session_id=sess_id, pop=False)
    assert summary is not None
    assert "budget" in summary
    assert summary["budget"]["budget_limit_usd"] == 0.50
    assert summary["budget"]["exceeded"] is False


async def test_plugin_budget_halt_on_model():
    plugin = FinOpsCostPlugin(
        budget_limit_usd=0.001,  # very tight budget ($0.001)
        on_budget_exceeded="halt",
    )

    ctx = MagicMock()
    ctx.session.id = "halt_sess_1"
    ctx.invocation_id = "halt_turn_1"
    ctx.agent_name = "test_agent"

    CostTracker.start_turn(turn_id="halt_turn_1", session_id="halt_sess_1")

    llm_resp = MagicMock()
    llm_resp.model_version = "gemini-2.5-pro"
    llm_resp.usage_metadata.prompt_token_count = 2000
    llm_resp.usage_metadata.candidates_token_count = 500
    llm_resp.usage_metadata.thoughts_token_count = 0
    llm_resp.usage_metadata.cached_content_token_count = 0

    # after_model_callback should raise BudgetExceededError when limit ($0.001) is breached
    try:
        await plugin.after_model_callback(callback_context=ctx, llm_response=llm_resp)
        assert False, "Should have raised BudgetExceededError"
    except BudgetExceededError as e:
        assert e.budget_limit_usd == 0.001
        assert e.scope == "session"


async def test_plugin_budget_halt_on_before_run():
    sess_id = "halt_before_sess"
    plugin = FinOpsCostPlugin(
        budget_limit_usd=0.005,
        on_budget_exceeded="halt",
    )

    # Pre-exhaust the session budget
    CostTracker.start_run(sess_id)
    CostTracker.record_usage(
        run_id=sess_id,
        session_id=sess_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=10000,
        completion_tokens=2000,
    )

    inv_ctx = MagicMock()
    inv_ctx.session.id = sess_id
    inv_ctx.invocation_id = "new_turn_attempt"

    # before_run_callback should intercept and return Content to block the turn
    content = await plugin.before_run_callback(invocation_context=inv_ctx)
    assert content is not None
    assert len(content.parts) > 0
    assert "halted to prevent unexpected charges" in content.parts[0].text


async def test_plugin_budget_downgrade():
    sess_id = "downgrade_sess"
    plugin = FinOpsCostPlugin(
        budget_limit_usd=0.005,
        on_budget_exceeded="downgrade",
        fallback_model="gemini-2.5-flash",
    )

    # Pre-exhaust budget
    CostTracker.start_run(sess_id)
    CostTracker.record_usage(
        run_id=sess_id,
        session_id=sess_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=10000,
        completion_tokens=2000,
    )

    inv_ctx = MagicMock()
    inv_ctx.session.id = sess_id
    inv_ctx.invocation_id = "downgrade_turn"
    inv_ctx.agent.model = "gemini-2.5-pro"

    await plugin.before_run_callback(invocation_context=inv_ctx)
    # Agent model should be downgraded to gemini-2.5-flash!
    assert inv_ctx.agent.model == "gemini-2.5-flash"
