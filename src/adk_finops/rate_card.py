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

"""Decoupled Rate Card Management for LLM & Tool Pricing.

Allows organizations and users to customize pricing without modifying code:
1. JSON / YAML files (local or network path).
2. Remote HTTP/HTTPS URL (e.g., enterprise pricing service or GCS bucket).
3. Environment variables (ADK_FINOPS_RATE_CARD_PATH, ADK_FINOPS_DISCOUNT_PERCENT).
4. Programmatic registration for custom or fine-tuned models.
"""

from __future__ import annotations

import json
import logging
import os
import threading
import urllib.request
from dataclasses import asdict, dataclass, field
from pathlib import Path
from typing import Any

from .rates import DEFAULT_RATES_FILE

logger = logging.getLogger("adk_finops.rate_card")


@dataclass
class ModelRate:
    """Pricing rate card for an individual LLM model."""

    provider: str = "unknown"
    input_per_1m: float = 0.30
    output_per_1m: float = 2.50
    cached_input_per_1m: float = 0.03
    input_per_1m_gt_128k: float | None = None
    output_per_1m_gt_128k: float | None = None
    cached_input_per_1m_gt_128k: float | None = None

    @classmethod
    def from_dict(cls, data: dict[str, Any]) -> ModelRate:
        """Creates a ModelRate from a raw dictionary."""
        return cls(
            provider=data.get("provider", "unknown"),
            input_per_1m=float(data.get("input_per_1m", 0.30)),
            output_per_1m=float(data.get("output_per_1m", 2.50)),
            cached_input_per_1m=float(data.get("cached_input_per_1m", 0.03)),
            input_per_1m_gt_128k=(
                float(data["input_per_1m_gt_128k"])
                if "input_per_1m_gt_128k" in data and data["input_per_1m_gt_128k"] is not None
                else None
            ),
            output_per_1m_gt_128k=(
                float(data["output_per_1m_gt_128k"])
                if "output_per_1m_gt_128k" in data and data["output_per_1m_gt_128k"] is not None
                else None
            ),
            cached_input_per_1m_gt_128k=(
                float(data["cached_input_per_1m_gt_128k"])
                if "cached_input_per_1m_gt_128k" in data and data["cached_input_per_1m_gt_128k"] is not None
                else None
            ),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes ModelRate to a dictionary."""
        return asdict(self)


class RateCardRegistry:
    """Thread-safe registry for model and tool pricing rate cards."""

    def __init__(self, load_defaults: bool = True) -> None:
        self._lock = threading.RLock()
        self._models: dict[str, ModelRate] = {}
        self._tools: dict[str, float] = {}
        self._fallback: ModelRate = ModelRate()
        self._provider_discounts: dict[str, float] = {}  # provider -> discount multiplier (e.g. 0.85)
        self._global_discount: float = 1.0  # multiplier (1.0 = no discount)

        if load_defaults:
            self.load_defaults()

        # Check environment variable for auto-configuration
        env_path = os.environ.get("ADK_FINOPS_RATE_CARD_PATH")
        if env_path:
            try:
                self.load_from_file(env_path)
            except Exception as e:
                logger.warning(f"Failed to load rate card from ADK_FINOPS_RATE_CARD_PATH ({env_path}): {e}")

        # Check environment variable for global discount
        env_discount = os.environ.get("ADK_FINOPS_DISCOUNT_PERCENT")
        if env_discount:
            try:
                self.set_global_discount(float(env_discount))
            except ValueError:
                pass

    def load_defaults(self) -> None:
        """Loads default bundled rate card from default_rates.json."""
        if DEFAULT_RATES_FILE.exists():
            with open(DEFAULT_RATES_FILE, encoding="utf-8") as f:
                data = json.load(f)
                self.load_from_dict(data)

    def load_from_dict(self, data: dict[str, Any]) -> None:
        """Loads or updates rate cards from a dictionary."""
        with self._lock:
            # Models
            models_data = data.get("models", {})
            for name, mdata in models_data.items():
                self._models[name.strip().lower()] = ModelRate.from_dict(mdata)

            # Tools
            tools_data = data.get("tools", {})
            for name, fee in tools_data.items():
                self._tools[name.strip().lower()] = float(fee)

            # Fallback
            if "fallback" in data:
                self._fallback = ModelRate.from_dict(data["fallback"])

    def load_from_file(self, file_path: str | Path) -> None:
        """Loads rate cards from a JSON file path."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Rate card file not found: {file_path}")

        with open(p, encoding="utf-8") as f:
            data = json.load(f)
            self.load_from_dict(data)
            logger.info(f"Loaded custom rate card from {file_path}")

    def load_from_url(self, url: str, timeout_seconds: float = 5.0) -> None:
        """Fetches and loads rate cards from a remote HTTP/HTTPS URL."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": "adk-finops/0.1.0"},
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds) as response:
            content = response.read().decode("utf-8")
            data = json.loads(content)
            self.load_from_dict(data)
            logger.info(f"Loaded custom rate card from URL: {url}")

    def register_model(self, model_name: str, rate: ModelRate | dict[str, Any]) -> None:
        """Registers or overrides a single model rate card."""
        with self._lock:
            if isinstance(rate, dict):
                rate = ModelRate.from_dict(rate)
            self._models[model_name.strip().lower()] = rate

    def register_tool(self, tool_name: str, cost_per_call_usd: float) -> None:
        """Registers or overrides a fixed fee per tool call."""
        with self._lock:
            self._tools[tool_name.strip().lower()] = float(cost_per_call_usd)

    def set_global_discount(self, discount_percent: float) -> None:
        """Sets a global discount percentage applied to all models and tools.

        Example:
            registry.set_global_discount(15.0)  # 15% discount -> multiplier 0.85
        """
        with self._lock:
            self._global_discount = max(0.0, 1.0 - (discount_percent / 100.0))
            logger.info(f"Set global discount: {discount_percent}% (multiplier={self._global_discount:.4f})")

    def set_provider_discount(self, provider: str, discount_percent: float) -> None:
        """Sets an enterprise discount percentage for a specific provider (e.g. 'google').

        Example:
            registry.set_provider_discount('google', 20.0)  # 20% discount on all Google models
        """
        with self._lock:
            mult = max(0.0, 1.0 - (discount_percent / 100.0))
            self._provider_discounts[provider.strip().lower()] = mult
            logger.info(f"Set provider '{provider}' discount: {discount_percent}% (multiplier={mult:.4f})")

    def resolve_model(self, model_name: str | None) -> ModelRate:
        """Resolves the best matching ModelRate for a model identifier."""
        if not model_name:
            return self._fallback

        clean = model_name.strip().lower()
        with self._lock:
            # 1. Exact match
            if clean in self._models:
                return self._models[clean]

            # 2. Prefix / substring match (e.g. 'gemini-2.5-pro-001' -> 'gemini-2.5-pro')
            for key, rate in self._models.items():
                if clean.startswith(key) or key in clean:
                    return rate

            return self._fallback

    def get_effective_discount(self, provider: str) -> float:
        """Calculates effective multiplier after combining global and provider discounts."""
        with self._lock:
            provider_mult = self._provider_discounts.get(provider.strip().lower(), 1.0)
            return self._global_discount * provider_mult

    def get_tool_fee(self, tool_name: str) -> float:
        """Retrieves fee for a tool after applying any global discount."""
        clean = tool_name.strip().lower()
        with self._lock:
            base_fee = self._tools.get(clean, 0.0)
            return round(base_fee * self._global_discount, 7)

    @property
    def registered_models(self) -> list[str]:
        with self._lock:
            return list(self._models.keys())

    @property
    def registered_tools(self) -> list[str]:
        with self._lock:
            return list(self._tools.keys())
