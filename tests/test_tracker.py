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


def test_dual_turn_and_session_tracking():
    CostTracker.set_enabled(True)

    # Turn 1
    CostTracker.start_turn(turn_id="turn_1", session_id="session_A")
    CostTracker.record_usage(
        run_id="turn_1",
        session_id="session_A",
        model_name="gemini-2.5-pro",
        prompt_tokens=2395,
        completion_tokens=31,
        thoughts_tokens=36,
    )
    s1 = CostTracker.get_summary(run_id="turn_1", session_id="session_A", pop=False)
    assert s1["turn"]["total_tokens"] == 2462
    assert s1["session"]["total_tokens"] == 2462

    # Turn 2
    CostTracker.start_turn(turn_id="turn_2", session_id="session_A")
    # Call 1
    CostTracker.record_usage(
        run_id="turn_2",
        session_id="session_A",
        model_name="gemini-2.5-pro",
        prompt_tokens=2473,
        completion_tokens=13,
        thoughts_tokens=428,
    )
    # Call 2
    CostTracker.record_usage(
        run_id="turn_2",
        session_id="session_A",
        model_name="gemini-2.5-pro",
        prompt_tokens=11749,
        completion_tokens=1066,
        thoughts_tokens=480,
    )

    s2 = CostTracker.get_summary(run_id="turn_2", session_id="session_A", pop=False)

    # Turn 2 metrics (Call 1 + Call 2)
    assert s2["turn"]["prompt_tokens"] == 14222
    assert s2["turn"]["completion_tokens"] == 1079
    assert s2["turn"]["thoughts_tokens"] == 908
    assert s2["turn"]["total_tokens"] == 16209

    # Session metrics (Turn 1 + Turn 2)
    assert s2["session"]["prompt_tokens"] == 16617
    assert s2["session"]["completion_tokens"] == 1110
    assert s2["session"]["thoughts_tokens"] == 944
    assert s2["session"]["total_tokens"] == 18671
    assert s2["total_tokens"] == 18671


def test_thinking_tokens_billing():
    CostTracker.set_enabled(True)
    # gemini-2.5-flash: output is $2.50 per 1M ($0.0000025 per token)
    # prompt: 1000 tokens ($0.00030)
    # completion: 100 tokens ($0.00025)
    # thoughts: 200 tokens ($0.00050)
    # total expected = 0.00030 + 0.00025 + 0.00050 = 0.00105
    cost = CostTracker.calculate_call_cost(
        model_name="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=300,  # completion + thoughts
        cached_tokens=0,
    )
    assert round(cost, 5) == 0.00105
