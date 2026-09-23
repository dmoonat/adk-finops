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
import ssl
import threading
import time
import urllib.request
from dataclasses import asdict, dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from ._version import __version__
from .rates import DEFAULT_RATES_FILE

logger = logging.getLogger("adk_finops.rate_card")

DEFAULT_REMOTE_RATE_CARD_URL = (
    "https://raw.githubusercontent.com/dmoonat/adk-finops/main/src/adk_finops/rates/default_rates.json"
)
DEFAULT_REMOTE_CACHE_TTL_SECONDS = 86400  # 24 hours
DEFAULT_REMOTE_CACHE_PATH = Path.home() / ".cache" / "adk-finops" / "remote_rates.json"


def _create_ssl_context() -> ssl.SSLContext:
    """Creates an SSL context using certifi CA bundle if available."""
    try:
        import certifi

        return ssl.create_default_context(cafile=certifi.where())
    except Exception:
        return ssl.create_default_context()


def _resolve_region(explicit_region: str | None = None) -> str:
    """Resolves pricing region in priority order:
    1. Explicit region argument (if provided)
    2. GOOGLE_CLOUD_LOCATION environment variable (standard for Google ADK / GenAI)
    3. ADK_FINOPS_REGION environment variable
    4. Default: 'global'
    """
    val = (
        explicit_region
        or os.environ.get("GOOGLE_CLOUD_LOCATION")
        or os.environ.get("ADK_FINOPS_REGION")
        or "global"
    )
    return val.strip().lower()


@dataclass
class ModelRate:
    """Pricing rate card for an individual LLM model."""

    provider: str = "unknown"
    input_per_1m: float = 0.30
    output_per_1m: float = 2.50
    cached_input_per_1m: float = 0.03
    input_per_1m_gt_200k: float | None = None
    output_per_1m_gt_200k: float | None = None
    cached_input_per_1m_gt_200k: float | None = None
    input_per_1m_gt_128k: float | None = None
    output_per_1m_gt_128k: float | None = None
    cached_input_per_1m_gt_128k: float | None = None
    is_fallback: bool = False

    @classmethod
    def from_dict(
        cls,
        data: dict[str, Any],
        region: str | None = None,
        effective_date: Any = None,
        is_fallback: bool = False,
    ) -> ModelRate:
        """Creates a ModelRate from a raw dictionary, respecting region and transition dates."""
        from datetime import date

        eff_date: date
        if isinstance(effective_date, date):
            eff_date = effective_date
        elif isinstance(effective_date, str) and effective_date.strip():
            eff_date = date.fromisoformat(effective_date.strip()[:10])
        else:
            env_dt = os.environ.get("ADK_FINOPS_EFFECTIVE_DATE", "").strip()
            eff_date = date.fromisoformat(env_dt[:10]) if env_dt else date.today()

        clean_region = _resolve_region(region)
        is_non_global = clean_region not in ("global", "", "default")

        effective_data = dict(data)
        # 1. Apply regional (non-global) base rates if in non-global region
        if is_non_global and isinstance(effective_data.get("non_global"), dict):
            effective_data.update(effective_data["non_global"])

        # 2. Automatically apply standard pricing starting January 1, 2027
        if "standard_pricing_2027" in data and eff_date >= date(2027, 1, 1):
            p2027 = data["standard_pricing_2027"]
            if isinstance(p2027, dict):
                effective_data.update(
                    {k: v for k, v in p2027.items() if k != "non_global"}
                )
                if is_non_global:
                    if isinstance(p2027.get("non_global"), dict):
                        effective_data.update(p2027["non_global"])
                    elif isinstance(data.get("non_global"), dict):
                        # Derive proportional regional multiplier if non_global is only at root
                        base_in = float(data.get("input_per_1m", 0.75) or 0.75)
                        ng_in = float(data["non_global"].get("input_per_1m", base_in))
                        mult = (ng_in / base_in) if base_in > 0 else 1.1
                        effective_data["input_per_1m"] = round(float(effective_data["input_per_1m"]) * mult, 6)
                        effective_data["output_per_1m"] = round(float(effective_data["output_per_1m"]) * mult, 6)
                        effective_data["cached_input_per_1m"] = round(float(effective_data["cached_input_per_1m"]) * mult, 6)

        in_gt_200k = effective_data.get("input_per_1m_gt_200k", effective_data.get("input_per_1m_gt_128k"))
        out_gt_200k = effective_data.get("output_per_1m_gt_200k", effective_data.get("output_per_1m_gt_128k"))
        cached_gt_200k = effective_data.get("cached_input_per_1m_gt_200k", effective_data.get("cached_input_per_1m_gt_128k"))

        in_gt_128k = effective_data.get("input_per_1m_gt_128k", in_gt_200k)
        out_gt_128k = effective_data.get("output_per_1m_gt_128k", out_gt_200k)
        cached_gt_128k = effective_data.get("cached_input_per_1m_gt_128k", cached_gt_200k)

        return cls(
            provider=effective_data.get("provider", "unknown"),
            input_per_1m=float(effective_data.get("input_per_1m", 0.30)),
            output_per_1m=float(effective_data.get("output_per_1m", 2.50)),
            cached_input_per_1m=float(effective_data.get("cached_input_per_1m", 0.03)),
            input_per_1m_gt_200k=float(in_gt_200k) if in_gt_200k is not None else None,
            output_per_1m_gt_200k=float(out_gt_200k) if out_gt_200k is not None else None,
            cached_input_per_1m_gt_200k=float(cached_gt_200k) if cached_gt_200k is not None else None,
            input_per_1m_gt_128k=float(in_gt_128k) if in_gt_128k is not None else None,
            output_per_1m_gt_128k=float(out_gt_128k) if out_gt_128k is not None else None,
            cached_input_per_1m_gt_128k=float(cached_gt_128k) if cached_gt_128k is not None else None,
            is_fallback=bool(effective_data.get("is_fallback", is_fallback)),
        )

    def to_dict(self) -> dict[str, Any]:
        """Serializes ModelRate to a dictionary."""
        return asdict(self)


class RateCardRegistry:
    """Thread-safe registry for model and tool pricing rate cards."""

    def __init__(
        self,
        load_defaults: bool = True,
        region: str | None = None,
        effective_date: Any = None,
        sync_remote_rates: bool | None = None,
        remote_url: str | None = None,
        remote_cache_ttl_seconds: int | None = None,
        remote_cache_path: str | Path | None = None,
    ) -> None:
        self._lock = threading.RLock()
        self._raw_models: dict[str, dict[str, Any]] = {}
        self._models: dict[str, ModelRate] = {}
        self._tools: dict[str, float] = {}
        self._fallback: ModelRate = ModelRate(is_fallback=True)
        self._warned_fallback_models: set[str] = set()
        self._provider_discounts: dict[str, float] = {}  # provider -> discount multiplier (e.g. 0.85)
        self._global_discount: float = 1.0  # multiplier (1.0 = no discount)
        self._explicit_region: str | None = region.strip().lower() if region else None
        self._region: str = _resolve_region(self._explicit_region)
        self._effective_date: Any = effective_date or os.environ.get("ADK_FINOPS_EFFECTIVE_DATE")
        self._last_sync_status: dict[str, Any] = {"status": "bundled_default"}

        if load_defaults:
            self.load_defaults()

        # Opt-in dynamic remote rate card syncing (via parameter or ADK_FINOPS_SYNC_REMOTE_RATES=1)
        env_sync = os.environ.get("ADK_FINOPS_SYNC_REMOTE_RATES", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        env_remote_url = os.environ.get("ADK_FINOPS_REMOTE_RATE_CARD_URL", "").strip()
        should_sync = bool(sync_remote_rates) or env_sync or bool(remote_url) or bool(env_remote_url)
        if should_sync:
            self.sync_remote_rate_card(
                url=remote_url or (env_remote_url if env_remote_url else None),
                cache_path=remote_cache_path,
                cache_ttl_seconds=remote_cache_ttl_seconds,
            )

        # Check environment variable for auto-configuration
        env_path = os.environ.get("ADK_FINOPS_RATE_CARD_PATH")
        if env_path:
            try:
                if env_path.startswith(("http://", "https://")):
                    self.load_from_url(env_path)
                else:
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

    def _rebuild_models(self) -> None:
        """Re-evaluates raw model dictionaries with the current region and effective_date."""
        with self._lock:
            for name, mdata in self._raw_models.items():
                self._models[name] = ModelRate.from_dict(
                    mdata,
                    region=self._region,
                    effective_date=self._effective_date,
                )

    def set_region(self, region: str | None) -> None:
        """Sets the pricing region ('global' or 'non_global' / specific GCP location like 'us-central1').
        Pass None to reset to environment-variable auto-detection (GOOGLE_CLOUD_LOCATION -> ADK_FINOPS_REGION -> 'global').
        """
        with self._lock:
            self._explicit_region = region.strip().lower() if region else None
            self._region = _resolve_region(self._explicit_region)
            self._rebuild_models()

    @property
    def region(self) -> str:
        if self._explicit_region is None:
            env_reg = _resolve_region(None)
            if env_reg != self._region:
                self._region = env_reg
                self._rebuild_models()
        return self._region

    def set_effective_date(self, effective_date: Any) -> None:
        """Sets the effective date (e.g. '2026-09-21' or '2027-01-01') for date-tiered pricing."""
        with self._lock:
            self._effective_date = effective_date
            self._rebuild_models()

    @property
    def effective_date(self) -> Any:
        return self._effective_date

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
                clean_name = name.strip().lower()
                self._raw_models[clean_name] = dict(mdata)
                self._models[clean_name] = ModelRate.from_dict(
                    mdata,
                    region=self._region,
                    effective_date=self._effective_date,
                )
                self._warned_fallback_models.discard(clean_name)

            # Tools
            tools_data = data.get("tools", {})
            for name, fee in tools_data.items():
                self._tools[name.strip().lower()] = float(fee)

            # Fallback
            if "fallback" in data:
                self._fallback = ModelRate.from_dict(
                    data["fallback"],
                    region=self._region,
                    effective_date=self._effective_date,
                    is_fallback=True,
                )

    def load_from_file(self, file_path: str | Path) -> None:
        """Loads rate cards from a JSON file path."""
        p = Path(file_path)
        if not p.exists():
            raise FileNotFoundError(f"Rate card file not found: {file_path}")

        with open(p, encoding="utf-8") as f:
            data = json.load(f)
            self.load_from_dict(data)
            logger.info(f"Loaded custom rate card from {file_path}")

    def load_from_url(self, url: str, timeout_seconds: float = 5.0) -> dict[str, Any]:
        """Fetches and loads rate cards from a remote HTTP/HTTPS URL."""
        req = urllib.request.Request(
            url,
            headers={"User-Agent": f"adk-finops/{__version__}"},
        )
        with urllib.request.urlopen(req, timeout=timeout_seconds, context=_create_ssl_context()) as response:
            content = response.read().decode("utf-8")
            data = json.loads(content)
            self.load_from_dict(data)
            logger.info(f"Loaded custom rate card from URL: {url}")
            return data

    def sync_remote_rate_card(
        self,
        url: str | None = None,
        cache_path: str | Path | None = None,
        cache_ttl_seconds: int | None = None,
        force: bool = False,
        timeout_seconds: float = 4.0,
        background: bool = False,
    ) -> dict[str, Any]:
        """Dynamically syncs the rate card from a remote JSON endpoint with local disk caching (24h TTL) and offline fallback.

        Args:
            url: Remote JSON URL (defaults to ADK_FINOPS_REMOTE_RATE_CARD_URL or canonical GitHub raw default_rates.json).
            cache_path: Local disk cache path (defaults to ~/.cache/adk-finops/remote_rates.json).
            cache_ttl_seconds: Cache freshness TTL in seconds (defaults to 86400s / 24h).
            force: If True, bypasses the local cache TTL and fetches fresh rates from the remote URL.
            timeout_seconds: HTTP request timeout in seconds.
            background: If True, performs the remote sync asynchronously in a daemon thread.

        Returns:
            Status dictionary describing whether rates were loaded from 'cached', 'synced', 'stale_cache_fallback', or 'bundled_fallback'.
        """
        target_url = (
            url
            or os.environ.get("ADK_FINOPS_REMOTE_RATE_CARD_URL", "").strip()
            or DEFAULT_REMOTE_RATE_CARD_URL
        )
        target_cache = Path(
            cache_path
            or os.environ.get("ADK_FINOPS_REMOTE_CACHE_PATH", "").strip()
            or DEFAULT_REMOTE_CACHE_PATH
        )
        ttl = (
            cache_ttl_seconds
            if cache_ttl_seconds is not None
            else int(os.environ.get("ADK_FINOPS_REMOTE_CACHE_TTL", str(DEFAULT_REMOTE_CACHE_TTL_SECONDS)))
        )

        def _do_sync() -> dict[str, Any]:
            now = time.time()
            # 1. Check fresh local cache if not forced
            if not force and target_cache.exists():
                try:
                    with open(target_cache, encoding="utf-8") as f:
                        cached_data = json.load(f)
                    meta = cached_data.get("_sync_metadata", {})
                    fetched_at = float(meta.get("fetched_at_epoch") or target_cache.stat().st_mtime)
                    age = now - fetched_at
                    if age < ttl and isinstance(cached_data.get("models"), dict):
                        self.load_from_dict(cached_data)
                        status = {
                            "status": "cached",
                            "source": str(target_cache),
                            "url": target_url,
                            "models_loaded": len(cached_data.get("models", {})),
                            "tools_loaded": len(cached_data.get("tools", {})),
                            "age_seconds": round(age, 1),
                            "ttl_seconds": ttl,
                        }
                        self._last_sync_status = status
                        logger.info(
                            f"[FinOps Rate Card] Loaded {status['models_loaded']} models from fresh local cache ({target_cache}, age={status['age_seconds']}s)"
                        )
                        return status
                except Exception as e:
                    logger.debug(f"[FinOps Rate Card] Could not read cache file {target_cache}: {e}")

            # 2. Fetch from remote URL
            try:
                data = self.load_from_url(target_url, timeout_seconds=timeout_seconds)
                if not isinstance(data.get("models"), dict):
                    raise ValueError(f"Remote rate card at {target_url} missing 'models' dictionary")

                # Save to local disk cache with sync metadata
                try:
                    target_cache.parent.mkdir(parents=True, exist_ok=True)
                    to_cache = dict(data)
                    to_cache["_sync_metadata"] = {
                        "fetched_at_epoch": now,
                        "fetched_at_iso": datetime.now(timezone.utc).isoformat(),
                        "source_url": target_url,
                        "adk_finops_version": __version__,
                    }
                    with open(target_cache, "w", encoding="utf-8") as f:
                        json.dump(to_cache, f, indent=2)
                except Exception as cache_err:
                    logger.debug(f"[FinOps Rate Card] Could not write cache to {target_cache}: {cache_err}")

                status = {
                    "status": "synced",
                    "source": target_url,
                    "cache_path": str(target_cache),
                    "models_loaded": len(data.get("models", {})),
                    "tools_loaded": len(data.get("tools", {})),
                }
                self._last_sync_status = status
                logger.info(
                    f"[FinOps Rate Card] Synced {status['models_loaded']} models from remote rate card ({target_url})"
                )
                return status
            except Exception as net_err:
                # 3. Graceful fallback to stale cache if available, else bundled default_rates.json
                if target_cache.exists():
                    try:
                        with open(target_cache, encoding="utf-8") as f:
                            stale_data = json.load(f)
                        if isinstance(stale_data.get("models"), dict):
                            self.load_from_dict(stale_data)
                            status = {
                                "status": "stale_cache_fallback",
                                "source": str(target_cache),
                                "url": target_url,
                                "models_loaded": len(stale_data.get("models", {})),
                                "error": str(net_err),
                            }
                            self._last_sync_status = status
                            logger.warning(
                                f"[FinOps Rate Card] Remote sync failed ({net_err}); using cached rate card at {target_cache}"
                            )
                            return status
                    except Exception:
                        pass

                status = {
                    "status": "bundled_fallback",
                    "source": str(DEFAULT_RATES_FILE),
                    "url": target_url,
                    "models_loaded": len(self._models),
                    "error": str(net_err),
                }
                self._last_sync_status = status
                logger.warning(
                    f"[FinOps Rate Card] Remote sync failed ({net_err}); falling back to bundled default_rates.json"
                )
                return status

        if background:
            t = threading.Thread(target=_do_sync, daemon=True, name="adk-finops-rate-sync")
            t.start()
            return {
                "status": "scheduled_background",
                "url": target_url,
                "cache_path": str(target_cache),
            }

        return _do_sync()

    @property
    def last_sync_status(self) -> dict[str, Any]:
        return dict(self._last_sync_status)

    def register_model(self, model_name: str, rate: ModelRate | dict[str, Any]) -> None:
        """Registers or overrides a single model rate card."""
        with self._lock:
            clean = model_name.strip().lower()
            if isinstance(rate, dict):
                self._raw_models[clean] = dict(rate)
                rate = ModelRate.from_dict(
                    rate,
                    region=self._region,
                    effective_date=self._effective_date,
                )
            self._models[clean] = rate
            self._warned_fallback_models.discard(clean)

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
            # Ensure late-loaded GOOGLE_CLOUD_LOCATION / ADK_FINOPS_REGION env vars are reflected
            _ = self.region
            # 1. Exact match
            if clean in self._models:
                return self._models[clean]

            # 2. Prefix / substring match (e.g. 'gemini-2.5-pro-001' -> 'gemini-2.5-pro')
            for key, rate in self._models.items():
                if clean.startswith(key) or key in clean:
                    return rate

            # 3. Unrecognized model -> warn once and return fallback rate card
            if clean not in self._warned_fallback_models:
                self._warned_fallback_models.add(clean)
                warn_msg = (
                    f"Model '{model_name}' is not in the default rate card; applying fallback pricing "
                    f"(input=${self._fallback.input_per_1m}/1M, output=${self._fallback.output_per_1m}/1M, "
                    f"cached=${self._fallback.cached_input_per_1m}/1M). "
                    f"Please register exact rates for this model via: "
                    f"CostTracker.register_rate_card('{model_name}', "
                    f"{{'provider': '...', 'input_per_1m': ..., 'output_per_1m': ..., 'cached_input_per_1m': ...}})"
                )
                logger.warning(warn_msg)
                print(f"⚠️ [FinOps Rate Card] {warn_msg}", flush=True)

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
