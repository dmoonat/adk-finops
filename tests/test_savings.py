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


def test_zero_caching_savings():
    CostTracker.set_enabled(True)
    net, gross, savings = CostTracker.calculate_call_cost_and_savings(
        model_name="gemini-2.5-flash",
        prompt_tokens=1000,
        completion_tokens=200,
        cached_tokens=0,
    )
    assert net == gross
    assert savings == 0.0


def test_context_caching_savings_roi():
    CostTracker.set_enabled(True)
    # gemini-2.5-flash: input=$0.30/1M, cached=$0.03/1M (90% discount!), output=$2.50/1M
    # Prompt: 10,000 tokens, 8,000 cached, 2,000 non-cached. Completion: 1,000 tokens.
    # Gross prompt cost: 10,000 * 0.30 / 1e6 = 0.00300
    # Net prompt cost: 2,000 * 0.30 / 1e6 ($0.0006) + 8,000 * 0.03 / 1e6 ($0.00024) = 0.00084
    # Output cost: 1,000 * 2.50 / 1e6 = 0.0025
    # Gross total = 0.00300 + 0.0025 = 0.00550
    # Net total = 0.00084 + 0.0025 = 0.00334
    # Savings = 0.00550 - 0.00334 = 0.00216

    net, gross, savings = CostTracker.calculate_call_cost_and_savings(
        model_name="gemini-2.5-flash",
        prompt_tokens=10000,
        completion_tokens=1000,
        cached_tokens=8000,
    )
    assert round(gross, 5) == 0.00550
    assert round(net, 5) == 0.00334
    assert round(savings, 5) == 0.00216


def test_savings_accumulation_in_summary():
    sess_id = "savings_sess_1"
    turn_id = "savings_turn_1"
    CostTracker.start_turn(turn_id=turn_id, session_id=sess_id)

    record = CostTracker.record_usage(
        run_id=turn_id,
        session_id=sess_id,
        model_name="gemini-2.5-pro",
        prompt_tokens=20000,
        completion_tokens=2000,
        cached_tokens=15000,
    )
    assert record["savings_usd"] > 0
    assert record["savings_pct"] > 0

    summary = CostTracker.get_summary(run_id=turn_id, session_id=sess_id, pop=False)
    assert summary is not None
    assert summary["savings_usd"] > 0
    assert summary["gross_cost_usd"] > summary["total_cost_usd"]
    assert summary["savings_pct"] > 0
    assert "gemini-2.5-pro" in summary["breakdown_by_model"]
    assert summary["breakdown_by_model"]["gemini-2.5-pro"]["savings_usd"] > 0
