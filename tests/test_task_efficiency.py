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

from adk_finops.tracker import CostTracker
from adk_finops.display import (
    render_task_efficiency_table,
    format_plain_task_efficiency,
    format_summary_box,
)


def test_task_status_and_efficiency_metrics():
    CostTracker.reset()
    CostTracker.set_enabled(True)

    # Task 1: Success
    CostTracker.start_turn(turn_id="turn_s1", session_id="sess_s1")
    CostTracker.record_usage(
        run_id="turn_s1",
        session_id="sess_s1",
        model_name="gemini-2.5-flash",
        prompt_tokens=100,
        completion_tokens=50,
    )
    CostTracker.record_task_status(session_id="sess_s1", status="success")

    # Task 2: Failed Loop (burned multiple calls)
    CostTracker.start_turn(turn_id="turn_f1", session_id="sess_f1")
    CostTracker.record_usage(
        run_id="turn_f1",
        session_id="sess_f1",
        model_name="gemini-2.5-flash",
        prompt_tokens=300,
        completion_tokens=150,
    )
    CostTracker.record_usage(
        run_id="turn_f1",
        session_id="sess_f1",
        model_name="gemini-2.5-flash",
        prompt_tokens=400,
        completion_tokens=200,
    )
    CostTracker.record_task_status(
        session_id="sess_f1",
        status="failed",
        error="Max validation retries exceeded",
    )

    metrics = CostTracker.get_task_efficiency_metrics()

    assert metrics["total_tasks"] == 2
    assert metrics["successful_tasks"] == 1
    assert metrics["failed_tasks"] == 1
    assert metrics["successful_spend_usd"] > 0.0
    assert metrics["wasted_spend_usd"] > 0.0
    assert metrics["wasted_spend_usd"] > metrics["successful_spend_usd"]
    assert metrics["wasted_spend_pct"] > 50.0

    # Test display rendering
    panel = render_task_efficiency_table(metrics)
    assert panel is not None

    plain_box = format_plain_task_efficiency(metrics)
    assert "FinOps Task Efficiency & Wasted Spend Analysis" in plain_box
    assert "Successful: 1" in plain_box
    assert "Failed: 1" in plain_box
    assert "Capital Loss:" in plain_box

    # Test summary box status inclusion
    s_summary = CostTracker.get_summary(session_id="sess_s1", pop=False)
    s_box = format_summary_box(s_summary)
    assert "SUCCESS" in s_box

    f_summary = CostTracker.get_summary(session_id="sess_f1", pop=False)
    f_box = format_summary_box(f_summary)
    assert "FAILED" in f_box
    assert "Wasted Spend" in f_box
