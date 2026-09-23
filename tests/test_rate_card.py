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

import json
import tempfile
from pathlib import Path

from adk_finops.rate_card import ModelRate, RateCardRegistry


def test_default_rates_loaded():
    registry = RateCardRegistry()
    assert "gemini-2.5-flash" in registry.registered_models
    assert "gemini-2.5-pro" in registry.registered_models
    assert "gpt-4o" in registry.registered_models

    flash = registry.resolve_model("gemini-2.5-flash")
    assert flash.input_per_1m == 0.30
    assert flash.output_per_1m == 2.50


def test_custom_file_override():
    custom_data = {
        "models": {
            "custom-llm-1": {
                "provider": "custom",
                "input_per_1m": 0.50,
                "output_per_1m": 1.50,
                "cached_input_per_1m": 0.05,
            }
        },
        "tools": {
            "custom_search_api": 0.01
        }
    }

    with tempfile.NamedTemporaryFile("w", suffix=".json", delete=False) as f:
        json.dump(custom_data, f)
        temp_path = f.name

    try:
        registry = RateCardRegistry(load_defaults=False)
        registry.load_from_file(temp_path)

        rate = registry.resolve_model("custom-llm-1")
        assert rate.input_per_1m == 0.50
        assert rate.output_per_1m == 1.50
        assert registry.get_tool_fee("custom_search_api") == 0.01
    finally:
        Path(temp_path).unlink(missing_ok=True)


def test_enterprise_discounts():
    registry = RateCardRegistry()

    # 1. Global discount (10%)
    registry.set_global_discount(10.0)
    assert registry.get_effective_discount("google") == 0.90
    assert registry.get_effective_discount("openai") == 0.90

    # 2. Provider-specific discount (additional 20% on Google -> 0.90 * 0.80 = 0.72)
    registry.set_provider_discount("google", 20.0)
    assert round(registry.get_effective_discount("google"), 4) == 0.72
    assert registry.get_effective_discount("openai") == 0.90


def test_dynamic_registration():
    registry = RateCardRegistry(load_defaults=False)
    registry.register_model("fine-tuned-model", {"input_per_1m": 2.0, "output_per_1m": 6.0})
    rate = registry.resolve_model("fine-tuned-model")
    assert rate.input_per_1m == 2.0
    assert rate.output_per_1m == 6.0


def test_gemini_3_7_flash_global_nonglobal_and_2027_pricing():
    registry = RateCardRegistry()

    # 1. Global 2026
    registry.set_region("global")
    registry.set_effective_date("2026-09-21")
    r_g26 = registry.resolve_model("gemini-3.7-flash")
    assert r_g26.input_per_1m == 0.75
    assert r_g26.output_per_1m == 3.75
    assert r_g26.cached_input_per_1m == 0.075

    # 2. Non-Global (Regional) 2026
    registry.set_region("non_global")
    registry.set_effective_date("2026-09-21")
    r_ng26 = registry.resolve_model("gemini-3.7-flash")
    assert r_ng26.input_per_1m == 0.825
    assert r_ng26.output_per_1m == 4.125
    assert r_ng26.cached_input_per_1m == 0.0825

    # 3. Global 2027 (Standard Pricing)
    registry.set_region("global")
    registry.set_effective_date("2027-01-01")
    r_g27 = registry.resolve_model("gemini-3.7-flash")
    assert r_g27.input_per_1m == 1.50
    assert r_g27.output_per_1m == 7.50
    assert r_g27.cached_input_per_1m == 0.15

    # 4. Non-Global (Regional) 2027
    registry.set_region("us-central1")
    registry.set_effective_date("2027-01-01")
    r_ng27 = registry.resolve_model("gemini-3.7-flash")
    assert r_ng27.input_per_1m == 1.65
    assert r_ng27.output_per_1m == 8.25
    assert r_ng27.cached_input_per_1m == 0.165


def test_region_env_precedence_google_cloud_location(monkeypatch):
    """Verifies priority: explicit region > GOOGLE_CLOUD_LOCATION > ADK_FINOPS_REGION > 'global'."""
    # 1. Neither env var set -> defaults to 'global' ($0.75 input)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)
    monkeypatch.delenv("ADK_FINOPS_REGION", raising=False)
    reg = RateCardRegistry(effective_date="2026-09-21")
    assert reg.region == "global"
    assert reg.resolve_model("gemini-3.7-flash").input_per_1m == 0.75

    # 2. ADK_FINOPS_REGION set -> uses ADK_FINOPS_REGION ('europe-west1' -> non_global $0.825)
    monkeypatch.setenv("ADK_FINOPS_REGION", "europe-west1")
    assert reg.region == "europe-west1"
    assert reg.resolve_model("gemini-3.7-flash").input_per_1m == 0.825

    # 3. GOOGLE_CLOUD_LOCATION set -> takes precedence over ADK_FINOPS_REGION
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    assert reg.region == "global"
    assert reg.resolve_model("gemini-3.7-flash").input_per_1m == 0.75

    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "us-central1")
    assert reg.region == "us-central1"
    assert reg.resolve_model("gemini-3.7-flash").input_per_1m == 0.825


def test_gt_200k_rate_card_and_cost_calculation() -> None:
    """Verifies _gt_200k rate card fields and >200k context window pricing calculation."""
    from adk_finops import CostTracker

    reg = RateCardRegistry(region="global", effective_date="2026-09-21")
    g31_pro = reg.resolve_model("gemini-3.1-pro-preview")
    assert g31_pro.input_per_1m == 2.00
    assert g31_pro.output_per_1m == 12.00
    assert g31_pro.cached_input_per_1m == 0.20
    assert g31_pro.input_per_1m_gt_200k == 4.00
    assert g31_pro.output_per_1m_gt_200k == 18.00
    assert g31_pro.cached_input_per_1m_gt_200k == 0.40

    g25_pro = reg.resolve_model("gemini-2.5-pro")
    assert g25_pro.input_per_1m == 1.25
    assert g25_pro.input_per_1m_gt_200k == 2.50
    assert g25_pro.output_per_1m_gt_200k == 15.00
    assert g25_pro.cached_input_per_1m_gt_200k == 0.25

    # Verify CostTracker uses <= 200k rate at 150,000 tokens ($2.00/1M -> $0.30)
    CostTracker.set_region("global")
    net_under, _, _ = CostTracker.calculate_call_cost_and_savings("gemini-3.1-pro-preview", 150_000, 0, 0)
    assert net_under == 0.30

    # Verify CostTracker uses > 200k rate at 250,000 tokens ($4.00/1M -> $1.00)
    net_over, _, _ = CostTracker.calculate_call_cost_and_savings("gemini-3.1-pro-preview", 250_000, 0, 0)
    assert net_over == 1.00


def test_unrecognized_model_logs_warning_and_marks_fallback_rate(capsys) -> None:
    """Verifies Issue #1: unrecognized models log a warning requesting CostTracker.register_rate_card and set is_fallback_rate=True."""
    from adk_finops import CostTracker

    reg = RateCardRegistry()
    rate = reg.resolve_model("acme-llm-7b")
    assert rate.is_fallback is True
    captured = capsys.readouterr().out
    assert "Model 'acme-llm-7b' is not in the default rate card" in captured
    assert "CostTracker.register_rate_card('acme-llm-7b'" in captured

    rec = CostTracker.record_usage(
        run_id="run-acme",
        model_name="acme-llm-7b",
        prompt_tokens=1_000_000,
        completion_tokens=100_000,
    )
    assert rec["is_fallback_rate"] is True

    # Once registered via CostTracker.register_rate_card, is_fallback_rate becomes False
    CostTracker.register_rate_card(
        "acme-llm-7b",
        {"provider": "acme", "input_per_1m": 1.0, "output_per_1m": 5.0, "cached_input_per_1m": 0.1},
    )
    rec_registered = CostTracker.record_usage(
        run_id="run-acme-2",
        model_name="acme-llm-7b",
        prompt_tokens=1_000_000,
        completion_tokens=100_000,
    )
    assert rec_registered["is_fallback_rate"] is False
    assert rec_registered["cost_usd"] == 1.50


def test_remote_rate_card_sync_with_cache_and_offline_fallback(tmp_path: Path, monkeypatch) -> None:
    """Verifies dynamic remote rate card syncing, 24h local disk cache TTL, and offline fallback."""
    cache_file = tmp_path / "remote_rates.json"
    reg = RateCardRegistry()

    fake_remote_payload = {
        "models": {
            "gemini-4.0-ultra-preview": {
                "provider": "google",
                "input_per_1m": 3.50,
                "output_per_1m": 14.00,
                "cached_input_per_1m": 0.35,
            }
        },
        "tools": {"custom_search_tool": 0.005},
    }

    def mock_load_from_url(url: str, timeout_seconds: float = 5.0):
        reg.load_from_dict(fake_remote_payload)
        return fake_remote_payload

    monkeypatch.setattr(reg, "load_from_url", mock_load_from_url)

    # 1. Initial forced sync fetches from remote URL and populates local cache
    status_sync = reg.sync_remote_rate_card(
        url="https://example.com/rates.json",
        cache_path=cache_file,
        cache_ttl_seconds=86400,
        force=True,
    )
    assert status_sync["status"] == "synced"
    assert cache_file.exists()
    rate = reg.resolve_model("gemini-4.0-ultra-preview")
    assert rate.input_per_1m == 3.50
    assert rate.is_fallback is False

    # 2. Second sync within TTL loads from local disk cache without hitting load_from_url
    def fail_if_called(url: str, timeout_seconds: float = 5.0):
        raise RuntimeError("Should not hit network when local cache is fresh!")

    monkeypatch.setattr(reg, "load_from_url", fail_if_called)
    status_cached = reg.sync_remote_rate_card(
        url="https://example.com/rates.json",
        cache_path=cache_file,
        cache_ttl_seconds=86400,
        force=False,
    )
    assert status_cached["status"] == "cached"
    assert status_cached["models_loaded"] == 1

    # 3. Forced sync when offline falls back gracefully to stale_cache_fallback
    status_offline = reg.sync_remote_rate_card(
        url="https://example.com/rates.json",
        cache_path=cache_file,
        cache_ttl_seconds=86400,
        force=True,
    )
    assert status_offline["status"] == "stale_cache_fallback"


def test_pricing_extractor_merges_and_preserves_curated_fields(monkeypatch) -> None:
    """Verifies GoogleCloudPricingExtractor merges upstream feeds while preserving non_global, 2027, and _gt_200k."""
    from adk_finops.pricing_extractor import GoogleCloudPricingExtractor

    extractor = GoogleCloudPricingExtractor(include_new_models=True)
    # Mock network calls so unit test runs offline deterministically
    monkeypatch.setattr(
        extractor,
        "extract_from_litellm_registry",
        lambda: (
            {
                "gpt-4o": {
                    "provider": "openai",
                    "input_per_1m": 2.50,
                    "output_per_1m": 10.00,
                    "cached_input_per_1m": 1.25,
                },
                "gpt-99-ultra-test": {
                    "provider": "openai",
                    "input_per_1m": 2.00,
                    "output_per_1m": 8.00,
                    "cached_input_per_1m": 0.50,
                },
            },
            True,
        ),
    )
    monkeypatch.setattr(
        extractor,
        "extract_from_google_pricing_pages",
        lambda *args, **kwargs: ({}, ["gemini_enterprise_pricing"]),
    )

    report = extractor.build_merged_rate_card()
    assert "gpt-99-ultra-test" in report.added_models
    assert len(report.merged_rate_card["source_urls"]) == 2
    source_names = [s["name"] for s in report.merged_rate_card["sources"]]
    assert source_names == ["gemini_enterprise_pricing", "litellm_registry"]
    models = report.merged_rate_card["models"]
    assert "non_global" in models["gemini-3.7-flash"]
    assert "standard_pricing_2027" in models["gemini-3.7-flash"]
    assert models["gemini-3.1-pro-preview"]["input_per_1m_gt_200k"] == 4.00





