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

import io
from adk_finops.display import format_plain_summary_box, format_summary_box, print_summary
from adk_finops.tracker import CostTracker


def test_plain_summary_box_formatting():
    summary = {
        "total_calls": 2,
        "total_tokens": 15000,
        "total_cost_usd": 0.0450,
        "gross_cost_usd": 0.0600,
        "savings_usd": 0.0150,
        "savings_pct": 25.0,
        "turn": {
            "total_calls": 2,
            "total_tokens": 15000,
            "total_cost_usd": 0.0450,
            "llm_cost_usd": 0.0450,
            "tool_cost_usd": 0.0,
        },
        "session": {
            "total_calls": 2,
            "total_tokens": 15000,
            "total_cost_usd": 0.0450,
            "llm_cost_usd": 0.0450,
            "tool_cost_usd": 0.0,
            "savings_usd": 0.0150,
            "savings_pct": 25.0,
            "breakdown_by_model": {
                "gemini-2.5-pro": {
                    "calls": 2,
                    "prompt_tokens": 12000,
                    "completion_tokens": 3000,
                    "thoughts_tokens": 0,
                    "cached_tokens": 5000,
                    "total_cost_usd": 0.0450,
                }
            },
            "breakdown_by_agent": {
                "researcher": {
                    "calls": 2,
                    "total_tokens": 15000,
                    "llm_cost_usd": 0.0450,
                    "tool_cost_usd": 0.0,
                    "total_cost_usd": 0.0450,
                }
            },
        },
        "budget": {
            "budget_limit_usd": 1.00,
            "current_cost_usd": 0.0450,
            "utilization_pct": 4.5,
            "exceeded": False,
        },
    }

    box = format_plain_summary_box(summary)
    assert "ADK FinOps Cost Summary" in box
    assert "Turn Cost:" in box
    assert "$0.0450" in box
    assert "Caching Savings" in box
    assert "researcher" in box
    assert "gemini-2.5-pro" in box
    assert "Budget Guard" in box


def test_rich_summary_box_export():
    summary = {
        "total_calls": 1,
        "total_tokens": 5000,
        "total_cost_usd": 0.0120,
        "gross_cost_usd": 0.0120,
        "savings_usd": 0.0,
        "savings_pct": 0.0,
        "turn": {
            "total_calls": 1,
            "total_tokens": 5000,
            "total_cost_usd": 0.0120,
            "llm_cost_usd": 0.0120,
            "tool_cost_usd": 0.0,
        },
        "session": {
            "total_calls": 1,
            "total_tokens": 5000,
            "total_cost_usd": 0.0120,
            "llm_cost_usd": 0.0120,
            "tool_cost_usd": 0.0,
            "savings_usd": 0.0,
            "breakdown_by_model": {
                "gemini-2.5-flash": {
                    "calls": 1,
                    "prompt_tokens": 4000,
                    "completion_tokens": 1000,
                    "thoughts_tokens": 0,
                    "cached_tokens": 0,
                    "total_cost_usd": 0.0120,
                }
            },
            "breakdown_by_agent": {
                "coder": {
                    "calls": 1,
                    "total_tokens": 5000,
                    "llm_cost_usd": 0.0120,
                    "tool_cost_usd": 0.0,
                    "total_cost_usd": 0.0120,
                }
            },
        },
    }

    formatted = format_summary_box(summary)
    assert "ADK FinOps Cost Summary" in formatted
    assert "coder" in formatted


def test_tracker_print_and_format_methods():
    CostTracker.set_enabled(True)
    CostTracker.start_turn("test_disp_turn", "test_disp_sess")
    CostTracker.record_usage(
        run_id="test_disp_turn",
        session_id="test_disp_sess",
        model_name="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=200,
        agent_name="tester",
    )

    # Test format_summary_box on CostTracker
    box_str = CostTracker.format_summary_box("test_disp_turn", "test_disp_sess")
    assert "ADK FinOps Cost Summary" in box_str
    assert "tester" in box_str

    # Test print_summary on CostTracker (should not raise)
    CostTracker.print_summary("test_disp_turn", "test_disp_sess")
