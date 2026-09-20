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

"""Unit tests for the Automated FinOps Optimization Advisor."""

from __future__ import annotations

from adk_finops import (
    CostTracker,
    FinOpsCostPlugin,
    format_summary_box,
)
from adk_finops.display import format_plain_summary_box


def setup_function() -> None:
    CostTracker._active_runs.clear()
    CostTracker.set_enabled(True)
    CostTracker.set_optimization_advisor_enabled(True)


def test_advisor_all_three_rules_detected() -> None:
    """Verifies Context Caching, Thinking Token Alert, and Model Right-Sizing."""
    sess_id = "sess_advisor_demo"

    # 1. Context Caching Opportunity:
    # 'researcher' sends >10k uncached prompt tokens across 4 turns
    for i in range(4):
        CostTracker.record_usage(
            run_id=f"turn_r_{i}",
            session_id=sess_id,
            model_name="gemini-2.5-flash",
            prompt_tokens=3200,
            completion_tokens=400,
            cached_tokens=0,
            agent_name="researcher",
        )

    # 2. Thinking Token Alert:
    # 'router_agent' spends 4,200 thinking tokens vs 900 completion tokens (~82% of output cost)
    CostTracker.record_usage(
        run_id="turn_router",
        session_id=sess_id,
        model_name="gemini-2.5-flash",
        prompt_tokens=800,
        completion_tokens=900,
        thoughts_tokens=4200,
        cached_tokens=0,
        agent_name="router_agent",
    )

    # 3. Model Right-Sizing:
    # 'formatter' uses gemini-2.5-pro for <150 output tokens with 0 tool calls -> gemini-2.5-flash
    CostTracker.record_usage(
        run_id="turn_fmt",
        session_id=sess_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=1500,
        completion_tokens=110,
        thoughts_tokens=0,
        cached_tokens=0,
        agent_name="formatter",
    )

    summary = CostTracker.get_summary(session_id=sess_id, pop=False)
    assert summary is not None

    insights = summary["optimization_insights"]
    categories = {item["category"]: item for item in insights}

    # Verify Rule 1: Context Caching
    assert "context_caching" in categories
    cc = categories["context_caching"]
    assert cc["agent_name"] == "researcher"
    assert cc["potential_savings_usd"] > 0
    assert "Context Caching Opportunity" in cc["message"]
    assert "researcher" in cc["message"]
    assert "4 turns" in cc["message"]

    # Verify Rule 2: Thinking Token Alert
    assert "thinking_budget" in categories
    tb = categories["thinking_budget"]
    assert tb["agent_name"] == "router_agent"
    assert tb["potential_savings_usd"] > 0
    assert "Thinking Token Alert" in tb["message"]
    assert "4,200" in tb["message"]
    assert "thinking_budget=0" in tb["message"]

    # Verify Rule 3: Model Right-Sizing
    assert "model_right_sizing" in categories
    mr = categories["model_right_sizing"]
    assert mr["agent_name"] == "formatter"
    assert mr["target_model"] == "gemini-2.5-flash"
    assert mr["potential_savings_usd"] > 0
    assert "Model Right-Sizing" in mr["message"]
    assert "gemini-2.5-pro" in mr["message"]
    assert "gemini-2.5-flash" in mr["message"]

    # Verify rendering and disclaimer in both Rich and Plain summary boxes
    rich_box = format_summary_box(summary)
    plain_box = format_plain_summary_box(summary)
    assert "Optimization Insights" in rich_box
    assert "Disclaimer" in rich_box
    assert "Optimization Insights" in plain_box
    assert "Disclaimer" in plain_box


def test_plugin_enable_optimization_advisor_flag() -> None:
    """Verifies FinOpsCostPlugin(enable_optimization_advisor=False) disables advisor insights."""
    plugin_off = FinOpsCostPlugin(
        enable_optimization_advisor=False,
        render_terminal_box=False,
    )
    assert plugin_off.enable_optimization_advisor is False
    assert CostTracker.is_optimization_advisor_enabled() is False

    sess_id = "sess_advisor_off"
    CostTracker.record_usage(
        run_id="t1",
        session_id=sess_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=2000,
        completion_tokens=80,
        agent_name="formatter",
    )

    summary = CostTracker.get_summary(session_id=sess_id, pop=False)
    assert summary is not None
    assert summary["optimization_insights"] == []
    assert summary["potential_savings_usd"] == 0.0

    plain_box = format_plain_summary_box(summary)
    assert "Optimization Insights" not in plain_box

    # Re-enable via plugin init
    plugin_on = FinOpsCostPlugin(
        enable_optimization_advisor=True,
        render_terminal_box=False,
    )
    assert plugin_on.enable_optimization_advisor is True
    assert CostTracker.is_optimization_advisor_enabled() is True

    summary_on = CostTracker.get_summary(session_id=sess_id, pop=False)
    assert summary_on is not None
    assert len(summary_on["optimization_insights"]) == 1
    assert summary_on["optimization_insights"][0]["category"] == "model_right_sizing"
