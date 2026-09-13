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

"""Advanced Standalone Pipeline Example for adk-finops (Without Google ADK).

Demonstrates:
  1. Multi-turn session tracking (turn + session scopes)
  2. Multi-agent / worker cost attribution in custom orchestrators
  3. Enterprise negotiated discounts (e.g., 15%)
  4. Dynamic budget guard & circuit breaker
  5. One-line BigQuery streaming exporter
"""

from unittest.mock import MagicMock
from adk_finops import BigQueryExporter, CostTracker


def main() -> None:
    print("=" * 75)
    print("🚀 adk-finops: Advanced Standalone Pipeline with BigQuery Exporter")
    print("=" * 75)

    CostTracker.reset()

    # 1. Enterprise Negotiated Discount (15%)
    print("\n[Step 1] Applying 15% Enterprise Discount...")
    CostTracker.set_discount(15.0)

    # 2. Configure Dynamic Budgets
    session_id = "session_pipeline_2026"
    print(f"\n[Step 2] Setting Budgets for {session_id}...")
    CostTracker.set_budget(
        session_id=session_id,
        session_budget_usd=1.00,
        turn_budget_usd=0.50,
        agent_budgets={"doc_parser": 0.20, "lead_architect": 0.50},
    )

    # 3. Initialize BigQuery Exporter (Mock client for demo without GCP credentials)
    print("\n[Step 3] Initializing BigQuery Exporter...")
    mock_bq_client = MagicMock()
    mock_bq_client.insert_rows_json.return_value = []
    bq_exporter = BigQueryExporter(
        table_id="my-enterprise-project.finops_analytics.pipeline_costs",
        client=mock_bq_client,
        auto_create_table=False,
    )

    # 4. Turn 1: Ingestion & Planning
    turn_1 = "turn_001_ingest"
    print(f"\n--- [Turn 1: Ingestion & Planning] ---")
    CostTracker.record_usage(
        run_id=turn_1,
        session_id=session_id,
        agent_name="doc_parser",
        model_name="gemini-2.5-flash",
        prompt_tokens=4_500,
        completion_tokens=600,
        thoughts_tokens=120,
    )
    CostTracker.record_tool_call(
        run_id=turn_1,
        session_id=session_id,
        agent_name="doc_parser",
        tool_name="google_search",
        count=1,
    )
    CostTracker.record_usage(
        run_id=turn_1,
        session_id=session_id,
        agent_name="lead_architect",
        model_name="gemini-2.5-pro",
        prompt_tokens=28_000,
        completion_tokens=2_200,
        thoughts_tokens=650,
        cached_tokens=22_000,
    )

    # Turn 1 Summary Box & BQ Export
    CostTracker.print_summary(run_id=turn_1, session_id=session_id, pop=False)
    summary_1 = CostTracker.get_summary(run_id=turn_1, session_id=session_id, pop=True)
    if summary_1:
        bq_exporter.export_summary(
            summary=summary_1,
            scope="both",
            tags={"env": "production", "pipeline": "custom_rag"},
            blocking=True,
        )

    # 5. Turn 2: Code Generation
    turn_2 = "turn_002_codegen"
    print(f"\n--- [Turn 2: Code Generation] ---")
    CostTracker.record_usage(
        run_id=turn_2,
        session_id=session_id,
        agent_name="code_generator",
        model_name="gemini-2.5-flash",
        prompt_tokens=6_200,
        completion_tokens=1_950,
        thoughts_tokens=300,
    )

    CostTracker.print_summary(run_id=turn_2, session_id=session_id, pop=False)
    summary_2 = CostTracker.get_summary(run_id=turn_2, session_id=session_id, pop=True)
    if summary_2:
        bq_exporter.export_summary(
            summary=summary_2,
            scope="both",
            tags={"env": "production", "pipeline": "custom_rag"},
            blocking=True,
        )

    # 6. Verify BQ Export
    total_rows = sum(len(c[0][1]) for c in mock_bq_client.insert_rows_json.call_args_list)
    print(f"\n✅ Total BigQuery rows streamed: {total_rows}")

    bq_exporter.close()
    print("\n🎉 Advanced standalone pipeline test completed successfully!")


if __name__ == "__main__":
    main()
