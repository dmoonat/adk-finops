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

"""Provider-Aware Pre-Flight Token & Cost Estimator for Google ADK and Multi-Provider LLMs.

Supports provider-specific tokenization profiles for:
- Google Gemini (`google`): 256k SentencePiece vocab (~4.0 chars/tok), 258 tokens/image tile & PDF page.
- OpenAI (`openai`): o200k_base / cl100k_base (~4.0 chars/tok, or exact `tiktoken` if installed),
  85 base + 170/tile image tokens (~425 avg), 800 tokens/PDF page.
- Anthropic Claude (`anthropic`): 65k BPE vocab (~3.6 chars/tok), ~1,334 tokens/image ((w*h)/750),
  ~1,600 tokens/PDF page, and +310 system tool-use preamble when tools are present.
- DeepSeek (`deepseek`): 128k BPE vocab (~3.7 chars/tok, ~1.4 CJK chars/tok).
"""

from __future__ import annotations

import inspect
import json
import re
from typing import Any

from .token_profiles import (
    DEFAULT_PDF_BYTES_PER_PAGE,
    DEFAULT_PROVIDER_TOKEN_PROFILES,
    GEMINI_IMAGE_TILE_TOKENS,
    GEMINI_PDF_PAGE_TOKENS,
    PROVIDER_TOKEN_PROFILES,
    TokenProfileRegistry,
    get_token_profile,
    load_token_profiles_file,
    register_token_profile,
    reset_token_profiles,
    update_token_profile,
)

_PDF_PAGE_REGEX = re.compile(rb"/Type\s*/Page\b")


def resolve_provider_from_model(model_name: str | None = None, provider: str | None = None) -> str:
    """Resolves the canonical provider or custom profile key from model_name or provider."""
    if provider:
        p_clean = provider.strip().lower()
        if p_clean in PROVIDER_TOKEN_PROFILES:
            return p_clean
        if "vertex" in p_clean or "gemini" in p_clean:
            return "google"
        if "claude" in p_clean:
            return "anthropic"

    if not model_name:
        return "google"

    m_clean = str(model_name).strip().lower()
    if m_clean in PROVIDER_TOKEN_PROFILES:
        return m_clean

    # Strip LiteLLM / ADK provider prefixes like "openai/gpt-4o" or "mistral/mistral-large"
    if "/" in m_clean:
        prefix, rest = m_clean.split("/", 1)
        if rest in PROVIDER_TOKEN_PROFILES:
            return rest
        if prefix in PROVIDER_TOKEN_PROFILES:
            return prefix
        m_clean = rest

    # Check if a registered custom profile key matches as a prefix (e.g., "mistral" -> "mistral-large")
    for custom_key in PROVIDER_TOKEN_PROFILES:
        if custom_key not in ("google", "openai", "anthropic", "deepseek") and m_clean.startswith(custom_key):
            return custom_key

    if m_clean.startswith(("gpt-", "o1", "o3", "o4", "chatgpt-", "text-embedding-")):
        return "openai"
    if m_clean.startswith("claude-"):
        return "anthropic"
    if m_clean.startswith("deepseek-"):
        return "deepseek"
    if m_clean.startswith(("gemini-", "codemender", "palm-", "learnlm-")):
        return "google"

    # Check RateCardRegistry if available
    try:
        from .tracker import CostTracker

        card = CostTracker.get_registry().resolve_model(m_clean)
        card_prov = getattr(card, "provider", "").lower()
        if card_prov in PROVIDER_TOKEN_PROFILES:
            return card_prov
    except Exception:
        pass

    return "google"


def get_provider_profile(model_name: str | None = None, provider: str | None = None) -> dict[str, Any]:
    """Returns the tokenization profile dictionary for the resolved provider or custom model."""
    resolved = resolve_provider_from_model(model_name=model_name, provider=provider)
    return PROVIDER_TOKEN_PROFILES.get(resolved, PROVIDER_TOKEN_PROFILES["google"])


def estimate_text_tokens(
    text: str | None,
    model_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Estimates token count for a string using provider-specific character and symbol density heuristics.

    - Google Gemini (`google`): ~4.0 ASCII chars/token, ~1.5 non-ASCII chars/token.
    - OpenAI (`openai`): Uses `tiktoken` if installed and string <= 64KB, otherwise ~4.0 ASCII chars/token.
    - Anthropic (`anthropic`): ~3.6 ASCII chars/token, ~1.35 non-ASCII chars/token.
    - DeepSeek (`deepseek`): ~3.7 ASCII chars/token, ~1.4 non-ASCII chars/token.
    """
    if not text:
        return 0

    length = len(text)
    if length == 0:
        return 0

    resolved_provider = resolve_provider_from_model(model_name=model_name, provider=provider)
    m_lower = (model_name or "").strip().lower()
    is_legacy_cl100k = resolved_provider == "openai" and (
        "gpt-3.5" in m_lower or "gpt-4-turbo" in m_lower or "gpt-4-0" in m_lower
    )

    # Optional exact tiktoken path for OpenAI models when tiktoken is installed and text is moderate size
    if resolved_provider == "openai" and length <= 65_536:
        try:
            import tiktoken  # type: ignore[import-not-found]

            enc_name = "cl100k_base" if is_legacy_cl100k else "o200k_base"
            enc = tiktoken.get_encoding(enc_name)
            return max(1, len(enc.encode(text, disallowed_special=())))
        except Exception:
            pass

    profile = PROVIDER_TOKEN_PROFILES.get(resolved_provider, PROVIDER_TOKEN_PROFILES["google"])
    ascii_cpt = 4.0 if is_legacy_cl100k else float(profile["ascii_chars_per_token"])  # type: ignore[arg-type]
    non_ascii_cpt = float(profile["non_ascii_chars_per_token"])  # type: ignore[arg-type]

    # Fast sample for very large strings (>32 KB) to stay sub-millisecond
    sample = text if length <= 32_768 else text[:16_384] + text[-16_384:]
    sample_len = len(sample)

    non_ascii = sum(1 for ch in sample if ord(ch) > 127)
    structural = sum(1 for ch in sample if ch in "{}[]():,;\"'\n\t")

    if sample_len > 0 and length > sample_len:
        scale = length / sample_len
        non_ascii = int(non_ascii * scale)
        structural = int(structural * scale)

    ascii_chars = max(0, length - non_ascii)
    estimated = (ascii_chars / ascii_cpt) + (structural * 0.12) + (non_ascii / non_ascii_cpt)
    return max(1, int(round(estimated)))


def _safe_serialize(obj: Any) -> str:
    """Safely serializes a dict, list, or object into a compact string for token estimation."""
    if obj is None:
        return ""
    if isinstance(obj, str):
        return obj
    if isinstance(obj, (bytes, bytearray)):
        return ""
    try:
        return json.dumps(obj, default=str, ensure_ascii=False)
    except Exception:
        return str(obj)


def _count_pdf_pages(raw_bytes: bytes) -> int:
    """Counts exact pages in an unencrypted PDF via `/Type /Page` markers, falling back to byte size."""
    if not raw_bytes:
        return 1
    matches = len(_PDF_PAGE_REGEX.findall(raw_bytes))
    if matches > 0:
        return matches
    return max(1, len(raw_bytes) // DEFAULT_PDF_BYTES_PER_PAGE)


def _estimate_inline_data_tokens(
    inline_data: Any,
    profile: dict[str, float | int],
) -> int:
    """Estimates tokens for inline binary media (images, PDFs, audio) using the provider's multimodal rules."""
    mime_type = ""
    raw_bytes = b""
    if isinstance(inline_data, dict):
        mime_type = str(inline_data.get("mime_type") or "").lower()
        data = inline_data.get("data")
        if isinstance(data, (bytes, bytearray)):
            raw_bytes = bytes(data)
        elif isinstance(data, str):
            raw_bytes = data.encode("utf-8", errors="ignore")
    else:
        mime_type = str(getattr(inline_data, "mime_type", "") or "").lower()
        data = getattr(inline_data, "data", None)
        if isinstance(data, (bytes, bytearray)):
            raw_bytes = bytes(data)
        elif isinstance(data, str):
            raw_bytes = data.encode("utf-8", errors="ignore")

    byte_len = len(raw_bytes)
    image_tokens = int(profile["image_tokens"])
    pdf_page_tokens = int(profile["pdf_page_tokens"])

    if mime_type.startswith("image/"):
        return image_tokens
    if mime_type == "application/pdf":
        pages = _count_pdf_pages(raw_bytes)
        return pages * pdf_page_tokens
    if byte_len > 0:
        return max(64, (byte_len // 1024) * 32)
    return image_tokens


def _estimate_part_tokens(
    part: Any,
    profile: dict[str, float | int],
    model_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Estimates tokens for a single ADK / google.genai / OpenAI / Anthropic Part object, dict, or string."""
    if part is None:
        return 0
    if isinstance(part, str):
        return estimate_text_tokens(part, model_name=model_name, provider=provider)

    tool_envelope = int(profile["tool_envelope_tokens"])
    image_tokens = int(profile["image_tokens"])

    if isinstance(part, dict):
        tokens = 0
        if "text" in part and part["text"]:
            tokens += estimate_text_tokens(str(part["text"]), model_name=model_name, provider=provider)
        if "function_call" in part and part["function_call"]:
            tokens += tool_envelope + estimate_text_tokens(
                _safe_serialize(part["function_call"]), model_name=model_name, provider=provider
            )
        if "function_response" in part and part["function_response"]:
            tokens += tool_envelope + estimate_text_tokens(
                _safe_serialize(part["function_response"]), model_name=model_name, provider=provider
            )
        if "inline_data" in part and part["inline_data"]:
            tokens += _estimate_inline_data_tokens(part["inline_data"], profile)
        return tokens

    tokens = 0
    # 1. Text content
    text_val = getattr(part, "text", None)
    if isinstance(text_val, str) and text_val:
        tokens += estimate_text_tokens(text_val, model_name=model_name, provider=provider)

    # 2. Function call (model -> tool invocation in history)
    fn_call = getattr(part, "function_call", None)
    if fn_call is not None:
        fn_name = getattr(fn_call, "name", "") or ""
        fn_args = getattr(fn_call, "args", None)
        tokens += (
            tool_envelope
            + estimate_text_tokens(str(fn_name), model_name=model_name, provider=provider)
            + estimate_text_tokens(_safe_serialize(fn_args), model_name=model_name, provider=provider)
        )

    # 3. Function response (tool output sent back to model)
    fn_resp = getattr(part, "function_response", None)
    if fn_resp is not None:
        resp_name = getattr(fn_resp, "name", "") or ""
        resp_payload = getattr(fn_resp, "response", None)
        tokens += (
            tool_envelope
            + estimate_text_tokens(str(resp_name), model_name=model_name, provider=provider)
            + estimate_text_tokens(_safe_serialize(resp_payload), model_name=model_name, provider=provider)
        )

    # 4. Multimodal inline_data (images, PDFs, audio, blobs)
    inline_data = getattr(part, "inline_data", None)
    if inline_data is not None:
        tokens += _estimate_inline_data_tokens(inline_data, profile)

    # 5. File URI reference (file_data)
    file_data = getattr(part, "file_data", None)
    if file_data is not None:
        tokens += image_tokens

    return tokens


def _estimate_content_tokens(
    content: Any,
    profile: dict[str, float | int],
    model_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Estimates tokens for a Content object, list, dict, or string."""
    if content is None:
        return 0
    if isinstance(content, str):
        return estimate_text_tokens(content, model_name=model_name, provider=provider)
    if isinstance(content, (list, tuple)):
        return sum(
            _estimate_content_tokens(item, profile, model_name=model_name, provider=provider)
            for item in content
        )

    turn_framing = int(profile["turn_framing_tokens"])

    if isinstance(content, dict):
        parts = content.get("parts")
        if isinstance(parts, list):
            return turn_framing + sum(
                _estimate_part_tokens(p, profile, model_name=model_name, provider=provider)
                for p in parts
            )
        if "text" in content:
            return estimate_text_tokens(str(content["text"]), model_name=model_name, provider=provider)
        return estimate_text_tokens(_safe_serialize(content), model_name=model_name, provider=provider)

    parts = getattr(content, "parts", None)
    if isinstance(parts, (list, tuple)):
        return turn_framing + sum(
            _estimate_part_tokens(p, profile, model_name=model_name, provider=provider)
            for p in parts
        )

    return _estimate_part_tokens(content, profile, model_name=model_name, provider=provider)


def _is_forced_tool_choice(llm_request: Any) -> bool:
    """Detects whether the request forces tool execution ('any', 'required', or a specific 'tool')."""
    # 1. Check direct tool_choice attribute (LiteLLM / Anthropic / OpenAI format)
    tool_choice = getattr(llm_request, "tool_choice", None)
    if tool_choice is None:
        config = getattr(llm_request, "config", None)
        tool_choice = getattr(config, "tool_choice", None) if config is not None else None

    if isinstance(tool_choice, str):
        return tool_choice.strip().lower() in ("any", "required", "tool")
    if isinstance(tool_choice, dict):
        tc_type = str(tool_choice.get("type", "")).strip().lower()
        return tc_type in ("any", "required", "tool", "function")

    # 2. Check Google GenAI / ADK config.tool_config.function_calling_config.mode
    config = getattr(llm_request, "config", None)
    tool_cfg = getattr(config, "tool_config", None) if config is not None else None
    fn_cfg = getattr(tool_cfg, "function_calling_config", None) if tool_cfg is not None else None
    mode = getattr(fn_cfg, "mode", None) if fn_cfg is not None else None
    if mode is not None:
        mode_str = str(getattr(mode, "name", mode)).strip().upper()
        if "ANY" in mode_str or "REQUIRED" in mode_str:
            return True

    return False


def _resolve_tool_preamble_tokens(
    profile: dict[str, Any],
    model_name: str | None,
    llm_request: Any,
) -> int:
    """Resolves the hidden tool-use system prompt token count (supports flat int or model/tool_choice dict)."""
    raw_preamble = profile.get("tool_System_preamble_tokens", 0)
    if isinstance(raw_preamble, (int, float)):
        return int(raw_preamble)

    if isinstance(raw_preamble, dict):
        m_lower = (model_name or "").strip().lower()
        forced = _is_forced_tool_choice(llm_request)

        if "opus" in m_lower:
            if forced:
                return int(raw_preamble.get("opus_5_forced_tool", 406))
            if "5.5" in m_lower or "4.5" in m_lower:
                return int(raw_preamble.get("opus_5.5_auto_or_none", 286))
            return int(raw_preamble.get("opus_5_auto_or_none", 286))

        if "haiku" in m_lower:
            return int(
                raw_preamble.get("haiku_forced_tool", 340)
                if forced
                else raw_preamble.get("haiku_auto_or_none", 264)
            )

        # Default Anthropic family: Sonnet
        return int(
            raw_preamble.get("sonnet_5_forced_tool", 474)
            if forced
            else raw_preamble.get("sonnet_5_auto_or_none", 354)
        )

    return 0


def _estimate_tools_tokens(
    llm_request: Any,
    profile: dict[str, Any],
    model_name: str | None = None,
    provider: str | None = None,
) -> int:
    """Estimates token overhead from declared tool schemas in LlmRequest using provider rules."""
    total_tool_tokens = 0
    seen_tools: set[str] = set()
    schema_overhead = int(profile["tool_schema_overhead_tokens"])
    system_preamble = _resolve_tool_preamble_tokens(profile, model_name, llm_request)

    # 1. Inspect tools_dict on ADK LlmRequest
    tools_dict = getattr(llm_request, "tools_dict", None)
    if isinstance(tools_dict, dict):
        for t_name, tool_obj in tools_dict.items():
            seen_tools.add(str(t_name))
            desc = (
                getattr(tool_obj, "description", None)
                or getattr(tool_obj, "__doc__", None)
                or ""
            )
            func = getattr(tool_obj, "func", None) or getattr(tool_obj, "_func", None)
            sig_str = ""
            if callable(func):
                try:
                    sig_str = str(inspect.signature(func))
                except Exception:
                    sig_str = ""
            total_tool_tokens += (
                schema_overhead
                + estimate_text_tokens(str(t_name), model_name=model_name, provider=provider)
                + estimate_text_tokens(str(desc), model_name=model_name, provider=provider)
                + estimate_text_tokens(sig_str, model_name=model_name, provider=provider)
            )

    # 2. Inspect config.tools if tools_dict wasn't populated
    config = getattr(llm_request, "config", None)
    config_tools = getattr(config, "tools", None) if config is not None else None
    if isinstance(config_tools, (list, tuple)) and not seen_tools:
        for tool_decl in config_tools:
            fn_decls = getattr(tool_decl, "function_declarations", None)
            if isinstance(fn_decls, (list, tuple)):
                for fd in fn_decls:
                    fd_name = getattr(fd, "name", "") or ""
                    fd_desc = getattr(fd, "description", "") or ""
                    fd_params = getattr(fd, "parameters", None)
                    seen_tools.add(str(fd_name) or "tool")
                    total_tool_tokens += (
                        schema_overhead
                        + estimate_text_tokens(str(fd_name), model_name=model_name, provider=provider)
                        + estimate_text_tokens(str(fd_desc), model_name=model_name, provider=provider)
                        + estimate_text_tokens(_safe_serialize(fd_params), model_name=model_name, provider=provider)
                    )
            else:
                seen_tools.add("tool")
                total_tool_tokens += 24

    if seen_tools and system_preamble > 0:
        total_tool_tokens += system_preamble

    return total_tool_tokens


def estimate_request_tokens(
    llm_request: Any,
    model_name: str | None = None,
    provider: str | None = None,
) -> dict[str, Any]:
    """Estimates the full input prompt token breakdown for an outgoing LlmRequest across any provider.

    Automatically resolves the model/provider ('google', 'openai', 'anthropic', 'deepseek')
    from `model_name`, `provider`, or `llm_request.model` and applies that provider's
    character density, multimodal image/PDF token rules, and tool-schema preamble overhead.
    """
    eff_model = model_name or getattr(llm_request, "model", None)
    resolved_provider = resolve_provider_from_model(model_name=eff_model, provider=provider)
    profile = get_provider_profile(model_name=eff_model, provider=resolved_provider)

    if llm_request is None:
        return {
            "provider": resolved_provider,
            "prompt_tokens": 0,
            "system_tokens": 0,
            "contents_tokens": 0,
            "tools_tokens": 0,
            "cached_tokens": 0,
            "uncached_prompt_tokens": 0,
        }

    # Allow passing a raw string or list of Contents directly
    if isinstance(llm_request, (str, list, tuple)):
        c_tokens = _estimate_content_tokens(
            llm_request, profile, model_name=eff_model, provider=resolved_provider
        )
        return {
            "provider": resolved_provider,
            "prompt_tokens": c_tokens,
            "system_tokens": 0,
            "contents_tokens": c_tokens,
            "tools_tokens": 0,
            "cached_tokens": 0,
            "uncached_prompt_tokens": c_tokens,
        }

    config = getattr(llm_request, "config", None)
    sys_inst = getattr(config, "system_instruction", None) if config is not None else None
    if sys_inst is None:
        sys_inst = getattr(llm_request, "system_instruction", None)

    system_tokens = _estimate_content_tokens(
        sys_inst, profile, model_name=eff_model, provider=resolved_provider
    )
    contents_tokens = _estimate_content_tokens(
        getattr(llm_request, "contents", None),
        profile,
        model_name=eff_model,
        provider=resolved_provider,
    )
    tools_tokens = _estimate_tools_tokens(
        llm_request, profile, model_name=eff_model, provider=resolved_provider
    )

    total_prompt = system_tokens + contents_tokens + tools_tokens

    raw_cached = getattr(llm_request, "cacheable_contents_token_count", None)
    cached_tokens = int(raw_cached) if isinstance(raw_cached, (int, float)) and raw_cached > 0 else 0
    cached_tokens = min(cached_tokens, total_prompt)

    return {
        "provider": resolved_provider,
        "prompt_tokens": total_prompt,
        "system_tokens": system_tokens,
        "contents_tokens": contents_tokens,
        "tools_tokens": tools_tokens,
        "cached_tokens": cached_tokens,
        "uncached_prompt_tokens": max(0, total_prompt - cached_tokens),
    }
