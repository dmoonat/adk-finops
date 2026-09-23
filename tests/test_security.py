# Copyright 2026
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     https://www.apache.org/licenses/LICENSE-2.0

"""Security verification tests covering SEC-01 through SEC-08."""

from __future__ import annotations

import csv
from pathlib import Path
import tempfile
import unittest

from adk_finops.dashboard import (
    DASHBOARD_HTML,
    MAX_INGEST_BATCH_ROWS,
    MAX_INGESTED_ROWS,
    _INGESTED_ROWS,
    _INGESTED_ROWS_LOCK,
    ingest_row_in_memory,
)
from adk_finops.exporters.bigquery import validate_bq_table_id
from adk_finops.exporters.local import CSVExporter, HTTPExporter, _sanitize_csv_cell
from adk_finops.pricing_extractor import (
    GCP_BILLING_SKUS_URL_TEMPLATE,
    _validate_http_url as validate_pricing_url,
)
from adk_finops.rate_card import RateCardRegistry, _safe_non_negative_float, _validate_http_url
from adk_finops.tracker import CostTracker


class TestSecurityRemediations(unittest.TestCase):
    """Verifies all 8 security controls (SEC-01 through SEC-08)."""

    def setUp(self) -> None:
        CostTracker.reset()
        with _INGESTED_ROWS_LOCK:
            _INGESTED_ROWS.clear()

    def test_sec01_dashboard_xss_sanitization_and_data_attributes(self) -> None:
        """SEC-01: Verify DASHBOARD_HTML defines escapeHtml and avoids inline JS string interpolation."""
        self.assertIn("function escapeHtml(val)", DASHBOARD_HTML)
        self.assertIn("function selectHierarchyFromEl(el)", DASHBOARD_HTML)
        self.assertIn("data-root=", DASHBOARD_HTML)
        # Ensure vulnerable inline onclick interpolation is absent
        self.assertNotIn("onclick=\"selectHierarchy('${", DASHBOARD_HTML)

    def test_sec02_bounded_ingested_rows_lru_eviction(self) -> None:
        """SEC-02: Verify _INGESTED_ROWS enforces MAX_INGESTED_ROWS LRU cap."""
        self.assertEqual(MAX_INGESTED_ROWS, 5_000)
        self.assertEqual(MAX_INGEST_BATCH_ROWS, 250)

        for i in range(MAX_INGESTED_ROWS + 25):
            ingest_row_in_memory({"session_id": f"sess-{i}", "total_cost_usd": 0.01})

        with _INGESTED_ROWS_LOCK:
            self.assertEqual(len(_INGESTED_ROWS), MAX_INGESTED_ROWS)
            session_ids = {r["session_id"] for r in _INGESTED_ROWS.values()}
            # First 25 sessions should have been evicted
            self.assertNotIn("sess-0", session_ids)
            self.assertIn(f"sess-{MAX_INGESTED_ROWS + 24}", session_ids)

    def test_sec03_bigquery_table_id_validation(self) -> None:
        """SEC-03: Verify strict regex validation on BigQuery table identifiers."""
        valid_id = "my-gcp-project.finops_dataset.agent_costs_2026"
        self.assertEqual(validate_bq_table_id(valid_id), valid_id)

        malicious_inputs = [
            "proj.dataset.table` UNION ALL SELECT * FROM secrets --",
            "proj.dataset.table; DROP TABLE users;",
            "invalid_table_without_dots",
            "proj.dataset.table/../../etc/passwd",
        ]
        for bad in malicious_inputs:
            with self.assertRaises(ValueError):
                validate_bq_table_id(bad)

    def test_sec04_ssrf_scheme_and_metadata_blocking_plus_rate_validation(self) -> None:
        """SEC-04: Block file://, ftp://, cloud metadata IPs, and negative/NaN rate cards."""
        for validator in (_validate_http_url, validate_pricing_url):
            self.assertEqual(
                validator("https://example.com/rates.json"),
                "https://example.com/rates.json",
            )
            with self.assertRaises(ValueError):
                validator("file:///etc/passwd")
            with self.assertRaises(ValueError):
                validator("ftp://example.com/rates.json")
            with self.assertRaises(ValueError):
                validator("http://169.254.169.254/latest/meta-data/")
            with self.assertRaises(ValueError):
                validator("http://metadata.google.internal/computeMetadata/v1/")

        with self.assertRaises(ValueError):
            HTTPExporter(endpoint="file:///etc/passwd")
        with self.assertRaises(ValueError):
            HTTPExporter(endpoint="http://169.254.169.254")

        # Rate validation blocks negative, NaN, and Inf values
        self.assertEqual(_safe_non_negative_float(2.5, "input_rate"), 2.5)
        for bad_num in (-1.0, float("nan"), float("inf"), "-5.0"):
            with self.assertRaises(ValueError):
                _safe_non_negative_float(bad_num, "input_rate")

        reg = RateCardRegistry()
        with self.assertRaises(ValueError):
            reg.register_model("bad-model", {"input_rate": -10.0, "output_rate": 5.0})

    def test_sec05_csv_formula_injection_protection(self) -> None:
        """SEC-05: Prefix dangerous formula characters in CSV cells while preserving negative floats."""
        self.assertEqual(_sanitize_csv_cell("=1+1"), "'=1+1")
        self.assertEqual(_sanitize_csv_cell("+cmd|' /C calc'!A0"), "'+cmd|' /C calc'!A0")
        self.assertEqual(_sanitize_csv_cell("@SUM(A1:A10)"), "'@SUM(A1:A10)")
        self.assertEqual(_sanitize_csv_cell("-1+1"), "'-1+1")
        # Valid negative numbers should remain untouched
        self.assertEqual(_sanitize_csv_cell("-0.015"), "-0.015")
        self.assertEqual(_sanitize_csv_cell(-0.015), -0.015)

        with tempfile.TemporaryDirectory() as tmpdir:
            csv_path = Path(tmpdir) / "export.csv"
            exporter = CSVExporter(csv_path, timestamp_dir=False)
            exporter.export_summary(
                {
                    "session": {
                        "run_id": "=cmd|' /C calc'!A0",
                        "error": "@formula_error",
                        "total_cost_usd": 0.05,
                        "breakdown_by_agent": {
                            "+sub_agent": {"total_cost_usd": 0.05, "total_tokens": 100}
                        },
                    },
                },
                scope="session",
            )
            with open(csv_path, encoding="utf-8") as f:
                rows = list(csv.DictReader(f))
            self.assertEqual(len(rows), 2)
            self.assertTrue(rows[0]["session_id"].startswith("'=cmd"))
            self.assertTrue(rows[0]["error"].startswith("'@formula_error"))
            self.assertTrue(rows[1]["agent_name"].startswith("'+sub_agent"))

    def test_sec06_gcp_api_key_removed_from_url_query_string(self) -> None:
        """SEC-06: Ensure GCP Billing API key is not interpolated in URL query parameters."""
        self.assertNotIn("key=", GCP_BILLING_SKUS_URL_TEMPLATE)

    def test_sec07_tracker_memory_bounds(self) -> None:
        """SEC-07: Ensure CostTracker._active_runs and _task_history enforce upper bounds."""
        orig_max_sessions = CostTracker.MAX_ACTIVE_SESSIONS
        orig_max_history = CostTracker.MAX_TASK_HISTORY
        try:
            CostTracker.MAX_ACTIVE_SESSIONS = 10
            CostTracker.MAX_TASK_HISTORY = 10

            for i in range(15):
                CostTracker.start_run(f"run-{i}")
                CostTracker.record_task_status(f"run-{i}", status="success")

            self.assertLessEqual(len(CostTracker._active_runs), 10)
            self.assertLessEqual(len(CostTracker._task_history), 10)
        finally:
            CostTracker.MAX_ACTIVE_SESSIONS = orig_max_sessions
            CostTracker.MAX_TASK_HISTORY = orig_max_history
            CostTracker.reset()

    def test_sec08_github_workflow_uses_env_variable_for_inputs(self) -> None:
        """SEC-08: Verify sync_pricing.yml passes workflow_dispatch inputs via env block."""
        workflow_path = (
            Path(__file__).resolve().parents[1] / ".github" / "workflows" / "sync_pricing.yml"
        )
        content = workflow_path.read_text(encoding="utf-8")
        self.assertIn("INCLUDE_NEW_MODELS: ${{ github.event.inputs.include_new_models", content)
        self.assertIn('if [ "$INCLUDE_NEW_MODELS" = "true" ]', content)


if __name__ == "__main__":
    unittest.main()
