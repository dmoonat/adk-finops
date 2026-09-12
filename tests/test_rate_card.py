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
