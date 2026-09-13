#!/usr/bin/env python3
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

"""Basic Standalone Example for adk-finops (Without Google ADK)."""

from adk_finops import CostTracker


def main() -> None:
    print("=" * 70)
    print("💸 adk-finops: Basic Standalone Usage (Without ADK)")
    print("=" * 70)

    CostTracker.reset()
    request_id = "req_fastapi_001"

    print(f"\n[1/3] Wrapping execution block with CostTracker.track_run('{request_id}')...")

    with CostTracker.track_run(request_id):
        # Call 1: Fast initial prompt with Gemini 2.5 Flash + thoughts tokens
        print("  ▶ Call 1: gemini-2.5-flash (with reasoning/thinking tokens)...")
        CostTracker.record_usage(
            run_id=request_id,
            model_name="gemini-2.5-flash",
            prompt_tokens=1_200,
            completion_tokens=350,
            thoughts_tokens=150,
            cached_tokens=0,
        )

        # Call 2: Google Search Grounding tool call ($0.014 / query)
        print("  ▶ Call 2: Google Search Grounding tool call...")
        CostTracker.record_tool_call(
            run_id=request_id,
            tool_name="google_search",
            count=1,
        )

        # Call 3: Gemini 2.5 Pro with Context Caching (30k cached tokens)
        print("  ▶ Call 3: gemini-2.5-pro (with 30,000 cached tokens)...")
        CostTracker.record_usage(
            run_id=request_id,
            model_name="gemini-2.5-pro",
            prompt_tokens=35_000,
            completion_tokens=1_800,
            thoughts_tokens=400,
            cached_tokens=30_000,
        )

    # Render Rich Terminal Summary Box
    print("\n[2/3] Rendering Rich Terminal Summary Box:")
    CostTracker.print_summary(run_id=request_id, pop=False)

    # Programmatic access to summary
    print("\n[3/3] Inspecting Structured Summary Data:")
    summary = CostTracker.get_summary(run_id=request_id, pop=True)
    if summary:
        print(f"  • Total Billable Tokens: {summary['total_tokens']:,}")
        print(f"  • Total Spend (USD): ${summary['total_cost_usd']:.6f}")
        print(f"  • Caching Savings (USD): ${summary['savings_usd']:.6f} ({summary['savings_pct']}%)")

    print("\n✅ Basic standalone tracking completed successfully!")


if __name__ == "__main__":
    main()
