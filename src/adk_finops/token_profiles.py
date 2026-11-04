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

"""Decoupled Provider Tokenization Profiles for Pre-Flight Estimation.

Allows users and organizations to register new provider/model tokenization
profiles or override existing ones (just like `rate_card.py` for pricing):
1. Programmatically via `register_token_profile()` / `update_token_profile()`
   or `CostTracker.register_token_profile()`.
2. Via `FinOpsCostPlugin(token_profiles={...}, token_profiles_path="...")`.
3. Via JSON file (`ADK_FINOPS_TOKEN_PROFILES_PATH`).
"""

from __future__ import annotations

import copy
import json
import logging
import os
import threading
from pathlib import Path
from typing import Any

logger = logging.getLogger("adk_finops.token_profiles")

# Official Google Cloud Gemini Multimodal Constants
# Ref: https://cloud.google.com/vertex-ai/generative-ai/docs/multimodal/get-token-count
GEMINI_IMAGE_TILE_TOKENS = 258
GEMINI_PDF_PAGE_TOKENS = 258
DEFAULT_PDF_BYTES_PER_PAGE = 35_000

# Default Built-In Provider Tokenization Profiles
DEFAULT_PROVIDER_TOKEN_PROFILES: dict[str, dict[str, Any]] = {
    "google": {
        "ascii_chars_per_token": 4.0,       # SentencePiece 256k vocabulary
        "non_ascii_chars_per_token": 1.5,
        "turn_framing_tokens": 4,           # <start_of_turn>role ... <end_of_turn>
        "tool_envelope_tokens": 8,          # function_call / function_response JSON wrapper
        "tool_System_preamble_tokens": 0,   # No extra hidden system prompt
        "tool_schema_overhead_tokens": 36,  # OpenAPI JSON Schema boilerplate per tool
        "image_tokens": GEMINI_IMAGE_TILE_TOKENS,
        "pdf_page_tokens": GEMINI_PDF_PAGE_TOKENS,
        "min_cacheable_tokens": {
            "pro": 2048,
            "flash": 1024,
            "legacy_1_5": 32768,
            "default": 1024,
        },
    },
    "openai": {
        "ascii_chars_per_token": 4.4,       # Optimized for o200k_base (gpt-4o) / 4.0 fallback for cl100k_base
        "non_ascii_chars_per_token": 1.45,
        "turn_framing_tokens": 4,           # <|im_start|>role\n ... <|im_end|>\n
        "tool_envelope_tokens": 12,         # tool_call_id + function name/arguments wrapper
        "tool_System_preamble_tokens": 16,  # TypeScript namespace declaration header
        "tool_schema_overhead_tokens": 42,  # Per-function schema wrapper
        "image_tokens": 425,                # 85 base + 2x 170 (512x512 tiles) standard vision average
        "pdf_page_tokens": 800,             # Extracted page text + vision tile
        "min_cacheable_tokens": {
            "default": 1024,
        },
    },
    "anthropic": {
        "ascii_chars_per_token": 3.6,       # Claude 65k BPE vocab splits words into ~10% more tokens
        "non_ascii_chars_per_token": 1.35,
        "turn_framing_tokens": 5,           # Human/Assistant turn role framing
        "tool_envelope_tokens": 16,         # <tool_use> / <tool_result> XML/JSON blocks
        # Preamble is model and tool_choice dependent.
        # Ref: https://platform.claude.com/docs/en/agents-and-tools/tool-use/overview
        "tool_System_preamble_tokens": {
            "opus_5.5_auto_or_none": 286,
            "opus_5_auto_or_none": 286,
            "opus_5_forced_tool": 406,
            "sonnet_5_auto_or_none": 354,
            "sonnet_5_forced_tool": 474,
            "haiku_auto_or_none": 264,
            "haiku_forced_tool": 340,
        },
        "tool_schema_overhead_tokens": 45,
        "image_tokens": 1334,               # (1024 * 1024) / 750 per Anthropic Vision docs
        "pdf_page_tokens": 2250,            # Adjusted average: Text (1500-3000) + dual image extraction overhead
        "min_cacheable_tokens": {
            "haiku": 2048,
            "sonnet": 1024,
            "opus": 1024,
            "default": 1024,
        },
    },
    "deepseek": {
        "ascii_chars_per_token": 3.7,       # DeepSeek V3 / R1 128k BPE tokenizer
        "non_ascii_chars_per_token": 1.4,
        "turn_framing_tokens": 4,
        "tool_envelope_tokens": 10,
        "tool_System_preamble_tokens": 12,
        "tool_schema_overhead_tokens": 40,
        "image_tokens": 512,
        "pdf_page_tokens": 512,
        "min_cacheable_tokens": {
            "default": 64,
        },
    },
}

# Active mutable dictionary referenced across the estimator and registry
PROVIDER_TOKEN_PROFILES: dict[str, dict[str, Any]] = copy.deepcopy(
    DEFAULT_PROVIDER_TOKEN_PROFILES
)


class TokenProfileRegistry:
    """Thread-safe registry for managing and customizing provider/model token profiles."""

    _lock = threading.RLock()

    @classmethod
    def register(
        cls,
        provider_or_model: str,
        profile: dict[str, Any],
        *,
        merge: bool = True,
        base_provider: str = "google",
    ) -> dict[str, Any]:
        """Registers a new provider/model token profile or updates an existing one.

        Args:
            provider_or_model: Canonical provider key (e.g. 'anthropic', 'mistral', 'meta')
                or a specific model identifier (e.g. 'claude-opus-5.5').
            profile: Dictionary of tokenization parameters to set or override.
            merge: If True (default), merges partial keys into the existing profile
                (or `base_provider` template if registering a new provider).
                If False, replaces the entry entirely (filling any missing required keys
                from `base_provider`).
            base_provider: Fallback profile template to inherit missing fields from when
                registering a brand-new provider.

        Returns:
            The resulting registered profile dictionary.
        """
        if not provider_or_model or not isinstance(profile, dict):
            raise ValueError("provider_or_model must be a non-empty string and profile must be a dict.")

        key = provider_or_model.strip().lower()
        with cls._lock:
            fallback_template = copy.deepcopy(
                PROVIDER_TOKEN_PROFILES.get(
                    base_provider.strip().lower(),
                    DEFAULT_PROVIDER_TOKEN_PROFILES["google"],
                )
            )

            if merge and key in PROVIDER_TOKEN_PROFILES:
                target = copy.deepcopy(PROVIDER_TOKEN_PROFILES[key])
            else:
                target = fallback_template

            for k, v in profile.items():
                if (
                    merge
                    and isinstance(v, dict)
                    and isinstance(target.get(k), dict)
                ):
                    merged_sub = dict(target[k])
                    merged_sub.update(v)
                    target[k] = merged_sub
                else:
                    target[k] = copy.deepcopy(v)

            PROVIDER_TOKEN_PROFILES[key] = target
            return copy.deepcopy(target)

    @classmethod
    def update(cls, provider_or_model: str, updates: dict[str, Any]) -> dict[str, Any]:
        """Updates specific keys in an existing provider/model token profile (or creates it by merging)."""
        return cls.register(provider_or_model, updates, merge=True)

    @classmethod
    def get(cls, provider_or_model: str, default_provider: str = "google") -> dict[str, Any]:
        """Returns the active token profile for a provider or model key."""
        key = (provider_or_model or "").strip().lower()
        with cls._lock:
            if key in PROVIDER_TOKEN_PROFILES:
                return PROVIDER_TOKEN_PROFILES[key]
            return PROVIDER_TOKEN_PROFILES.get(
                default_provider, DEFAULT_PROVIDER_TOKEN_PROFILES["google"]
            )

    @classmethod
    def list_profiles(cls) -> dict[str, dict[str, Any]]:
        """Returns a deep copy of all registered provider/model token profiles."""
        with cls._lock:
            return copy.deepcopy(PROVIDER_TOKEN_PROFILES)

    @classmethod
    def load_from_file(cls, file_path: str | Path, *, merge: bool = True) -> None:
        """Loads and merges provider token profiles from a local JSON file."""
        path = Path(file_path).expanduser().resolve()
        if not path.exists():
            raise FileNotFoundError(f"Token profiles file not found: {path}")

        with path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        profiles_map = data.get("profiles", data) if isinstance(data, dict) else {}
        if not isinstance(profiles_map, dict):
            raise ValueError(f"Invalid token profiles format in {path}; expected a JSON object.")

        with cls._lock:
            for key, prof in profiles_map.items():
                if isinstance(prof, dict):
                    cls.register(str(key), prof, merge=merge)

    @classmethod
    def reset(cls) -> None:
        """Resets `PROVIDER_TOKEN_PROFILES` in-place back to the built-in defaults."""
        with cls._lock:
            PROVIDER_TOKEN_PROFILES.clear()
            PROVIDER_TOKEN_PROFILES.update(copy.deepcopy(DEFAULT_PROVIDER_TOKEN_PROFILES))


def register_token_profile(
    provider_or_model: str,
    profile: dict[str, Any],
    *,
    merge: bool = True,
    base_provider: str = "google",
) -> dict[str, Any]:
    """Registers a new provider/model token profile or merges overrides into an existing one."""
    return TokenProfileRegistry.register(
        provider_or_model, profile, merge=merge, base_provider=base_provider
    )


def update_token_profile(provider_or_model: str, updates: dict[str, Any]) -> dict[str, Any]:
    """Updates specific fields on an existing provider/model token profile."""
    return TokenProfileRegistry.update(provider_or_model, updates)


def get_token_profile(provider_or_model: str, default_provider: str = "google") -> dict[str, Any]:
    """Retrieves a provider/model token profile."""
    return TokenProfileRegistry.get(provider_or_model, default_provider=default_provider)


def load_token_profiles_file(file_path: str | Path, *, merge: bool = True) -> None:
    """Loads provider/model token profile overrides from a JSON file."""
    TokenProfileRegistry.load_from_file(file_path, merge=merge)


def reset_token_profiles() -> None:
    """Resets all provider token profiles back to built-in defaults."""
    TokenProfileRegistry.reset()


def get_min_cacheable_tokens(
    model_name: str | None = None,
    provider: str | None = None,
    rate: Any = None,
) -> int | None:
    """Resolves the minimum prefix token floor required for Context Caching on a given model.

    Resolution order:
    1. Explicit `rate.min_cacheable_tokens` if configured on `ModelRate`.
    2. If `rate` has no caching discount (`cached_input_per_1m >= input_per_1m`), returns `None`.
    3. Exact model or provider `min_cacheable_tokens` from `PROVIDER_TOKEN_PROFILES`.
    4. Provider/model family rules (e.g. 2048 for Gemini Pro & Claude Haiku, 1024 for Flash/OpenAI/Sonnet/Opus, 64 for DeepSeek).
    """
    if rate is not None:
        explicit_floor = getattr(rate, "min_cacheable_tokens", None)
        if isinstance(explicit_floor, int) and explicit_floor > 0:
            return explicit_floor
        in_rate = getattr(rate, "input_per_1m", None)
        cached_rate = getattr(rate, "cached_input_per_1m", None)
        if in_rate is not None and cached_rate is not None and float(cached_rate) >= float(in_rate):
            return None

    m_clean = (model_name or "").strip().lower()
    if "/" in m_clean:
        m_clean = m_clean.split("/")[-1]

    p_clean = (provider or "").strip().lower()
    if not p_clean:
        if m_clean.startswith(("gemini-", "palm-", "learnlm-")):
            p_clean = "google"
        elif m_clean.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-")):
            p_clean = "openai"
        elif m_clean.startswith("claude-"):
            p_clean = "anthropic"
        elif m_clean.startswith("deepseek-"):
            p_clean = "deepseek"
        elif rate is not None and getattr(rate, "provider", None):
            p_clean = str(getattr(rate, "provider", "")).strip().lower()
        else:
            p_clean = "google"

    prof = get_token_profile(m_clean if m_clean in PROVIDER_TOKEN_PROFILES else p_clean, default_provider="google")
    floor_cfg = prof.get("min_cacheable_tokens")
    if isinstance(floor_cfg, int):
        return floor_cfg if floor_cfg > 0 else None
    if isinstance(floor_cfg, dict):
        if m_clean in floor_cfg:
            return int(floor_cfg[m_clean])
        if p_clean == "google":
            if "1.5" in m_clean:
                return int(floor_cfg.get("legacy_1_5", 32768))
            if "pro" in m_clean:
                return int(floor_cfg.get("pro", 2048))
            if "flash" in m_clean:
                return int(floor_cfg.get("flash", 1024))
        elif p_clean == "anthropic":
            if "haiku" in m_clean:
                return int(floor_cfg.get("haiku", 2048))
            if "opus" in m_clean:
                return int(floor_cfg.get("opus", 1024))
            if "sonnet" in m_clean:
                return int(floor_cfg.get("sonnet", 1024))
        if "default" in floor_cfg:
            return int(floor_cfg["default"])

    return 1024


# Automatically load custom token profiles if ADK_FINOPS_TOKEN_PROFILES_PATH is set
_env_profiles_path = os.environ.get("ADK_FINOPS_TOKEN_PROFILES_PATH")
if _env_profiles_path:
    try:
        load_token_profiles_file(_env_profiles_path)
    except Exception as exc:
        logger.warning(f"Failed to load token profiles from ADK_FINOPS_TOKEN_PROFILES_PATH: {exc}")

