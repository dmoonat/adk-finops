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

"""Unit tests for v0.8.0 Pre-Flight Budget Guards (before_model_callback & estimator)."""

import asyncio
from types import SimpleNamespace
from unittest.mock import MagicMock

from adk_finops import CostTracker, FinOpsCostPlugin, estimate_request_tokens


def _make_mock_request(
    model: str,
    user_text: str,
    system_instruction: str = "You are a helpful assistant.",
    tools_dict: dict | None = None,
    cached_tokens: int = 0,
):
    part = SimpleNamespace(
        text=user_text,
        function_call=None,
        function_response=None,
        inline_data=None,
        file_data=None,
    )
    content = SimpleNamespace(parts=[part], role="user")
    config = SimpleNamespace(system_instruction=system_instruction, tools=None)
    return SimpleNamespace(
        model=model,
        contents=[content],
        config=config,
        tools_dict=tools_dict or {},
        cacheable_contents_token_count=cached_tokens,
    )


def test_estimate_request_tokens_accuracy():
    # ~40,000 chars -> ~10,000 tokens
    large_text = "word " * 8000
    req = _make_mock_request(
        model="gemini-2.5-pro",
        user_text=large_text,
        system_instruction="Analyze this document carefully.",
        cached_tokens=2000,
    )
    est = estimate_request_tokens(req)
    assert 9500 <= est["prompt_tokens"] <= 11000
    assert est["cached_tokens"] == 2000
    assert est["uncached_prompt_tokens"] == est["prompt_tokens"] - 2000


def test_preflight_budget_guard_halts_massive_200k_request():
    CostTracker.reset()
    plugin = FinOpsCostPlugin(
        default_model="gemini-2.5-pro",
        budget_limit_usd=0.25,
        on_budget_exceeded="halt",
        preflight_budget_guard=True,
    )

    ctx = MagicMock()
    ctx.invocation_id = "turn_massive_1"
    ctx.session.id = "sess_massive_1"
    ctx.agent.name = "research_agent"
    ctx.agent.model = "gemini-2.5-pro"

    CostTracker.start_turn("turn_massive_1", "sess_massive_1")

    # 900,000 characters (~225,000 input tokens) on gemini-2.5-pro (>200K tier @ $2.50/1M = ~$0.5625)
    massive_prompt = "A" * 900_000
    llm_req = _make_mock_request(model="gemini-2.5-pro", user_text=massive_prompt)

    response = asyncio.run(
        plugin.before_model_callback(callback_context=ctx, llm_request=llm_req)
    )

    # Should return a short-circuit LlmResponse (blocking the network call!)
    assert response is not None
    assert hasattr(response, "content")

    # Verify $0.00 actual cloud spend, while recording the pre-flight block & avoided cost
    summary = CostTracker.get_summary("turn_massive_1", "sess_massive_1", pop=False)
    assert summary is not None
    assert summary["total_cost_usd"] == 0.0
    assert summary["status"] == "budget_exceeded"
    assert summary["preflight_blocks_count"] == 1
    assert summary["preflight_avoided_tokens"] >= 200_000
    assert summary["preflight_avoided_cost_usd"] >= 0.50


def test_preflight_budget_guard_downgrades_model_in_place():
    CostTracker.reset()
    plugin = FinOpsCostPlugin(
        default_model="gemini-2.5-pro",
        turn_budget_limit_usd=0.05,
        on_budget_exceeded="downgrade",
        fallback_model="gemini-2.5-flash",
        preflight_budget_guard=True,
    )

    ctx = MagicMock()
    ctx.invocation_id = "turn_dg_1"
    ctx.session.id = "sess_dg_1"
    ctx.agent.name = "analyst_agent"
    ctx.agent.model = "gemini-2.5-pro"

    CostTracker.start_turn("turn_dg_1", "sess_dg_1")

    # 240,000 chars (~60,000 tokens):
    # On gemini-2.5-pro ($1.25/1M) -> $0.0750 (> $0.05 turn budget!)
    # On gemini-2.5-flash ($0.30/1M) -> $0.0180 (saves $0.0570!)
    prompt_60k_tokens = "data " * 48_000
    llm_req = _make_mock_request(model="gemini-2.5-pro", user_text=prompt_60k_tokens)

    response = asyncio.run(
        plugin.before_model_callback(callback_context=ctx, llm_request=llm_req)
    )

    # Should allow request to proceed (response is None) after mutating llm_req.model in-place
    assert response is None
    assert llm_req.model == "gemini-2.5-flash"
    assert ctx.agent.model == "gemini-2.5-flash"

    summary = CostTracker.get_summary("turn_dg_1", "sess_dg_1", pop=False)
    assert summary is not None
    assert summary["preflight_downgrades_count"] == 1
    assert summary["preflight_avoided_cost_usd"] > 0.04


def test_preflight_max_prompt_tokens_ceiling():
    CostTracker.reset()
    plugin = FinOpsCostPlugin(
        default_model="gemini-2.5-flash",
        budget_limit_usd=10.0,  # High USD budget
        max_prompt_tokens=200_000,  # Hard cap to avoid >200K pricing tier
        on_budget_exceeded="downgrade",
    )

    ctx = MagicMock()
    ctx.invocation_id = "turn_cap_1"
    ctx.session.id = "sess_cap_1"
    ctx.agent.name = "reader_agent"

    CostTracker.start_turn("turn_cap_1", "sess_cap_1")

    # ~210,000 tokens (> 200,000 max_prompt_tokens ceiling)
    llm_req = _make_mock_request(model="gemini-2.5-flash", user_text="X" * 840_000)

    response = asyncio.run(
        plugin.before_model_callback(callback_context=ctx, llm_request=llm_req)
    )

    # Even in downgrade mode, breaching max_prompt_tokens halts the request before execution
    assert response is not None
    summary = CostTracker.get_summary("turn_cap_1", "sess_cap_1", pop=False)
    assert summary["preflight_blocks_count"] == 1
    assert summary["total_cost_usd"] == 0.0


def test_provider_specific_token_estimation_profiles():
    text_3600_chars = "a" * 3600
    tools = {"search_db": SimpleNamespace(name="search_db", description="Search SQL database")}

    req_gemini = _make_mock_request("gemini-2.5-pro", text_3600_chars, tools_dict=tools)
    req_openai = _make_mock_request("gpt-4o", text_3600_chars, tools_dict=tools)
    req_claude_sonnet_auto = _make_mock_request("claude-sonnet-4-5", text_3600_chars, tools_dict=tools)
    req_claude_sonnet_forced = _make_mock_request("claude-sonnet-4-5", text_3600_chars, tools_dict=tools)
    req_claude_sonnet_forced.tool_choice = "any"
    req_claude_opus_auto = _make_mock_request("claude-opus-4-6", text_3600_chars, tools_dict=tools)
    req_claude_opus_forced = _make_mock_request("claude-opus-4-6", text_3600_chars, tools_dict=tools)
    req_claude_opus_forced.tool_choice = {"type": "any"}
    req_deepseek = _make_mock_request("deepseek-reasoner", text_3600_chars, tools_dict=tools)

    est_gemini = estimate_request_tokens(req_gemini)
    est_openai = estimate_request_tokens(req_openai)
    est_sonnet_auto = estimate_request_tokens(req_claude_sonnet_auto)
    est_sonnet_forced = estimate_request_tokens(req_claude_sonnet_forced)
    est_opus_auto = estimate_request_tokens(req_claude_opus_auto)
    est_opus_forced = estimate_request_tokens(req_claude_opus_forced)
    est_deepseek = estimate_request_tokens(req_deepseek)

    assert est_gemini["provider"] == "google"
    assert est_openai["provider"] == "openai"
    assert est_sonnet_auto["provider"] == "anthropic"
    assert est_deepseek["provider"] == "deepseek"

    # Verify dynamic Anthropic tool preamble:
    # - Sonnet auto (354) vs forced (474) -> +120 tokens
    # - Opus auto (286) vs forced (406) -> +120 tokens
    assert est_sonnet_forced["tools_tokens"] - est_sonnet_auto["tools_tokens"] == 120
    assert est_opus_forced["tools_tokens"] - est_opus_auto["tools_tokens"] == 120
    assert est_sonnet_auto["tools_tokens"] - est_opus_auto["tools_tokens"] == 68  # 354 - 286

    # Claude has a 65k BPE vocab (3.6 chars/token vs 4.0/4.4)
    assert est_sonnet_auto["prompt_tokens"] > est_gemini["prompt_tokens"]
    assert est_deepseek["prompt_tokens"] > est_gemini["prompt_tokens"]


def test_pdf_exact_page_counting_across_providers():
    # Synthetic 3-page PDF byte stream containing three /Type /Page object markers
    fake_3page_pdf = (
        b"%PDF-1.7\n"
        b"1 0 obj << /Type /Catalog /Pages 2 0 R >> endobj\n"
        b"3 0 obj << /Type /Page /Parent 2 0 R >> endobj\n"
        b"4 0 obj << /Type /Page /Parent 2 0 R >> endobj\n"
        b"5 0 obj << /Type /Page /Parent 2 0 R >> endobj\n"
    )
    pdf_part = SimpleNamespace(
        text=None,
        function_call=None,
        function_response=None,
        inline_data=SimpleNamespace(mime_type="application/pdf", data=fake_3page_pdf),
        file_data=None,
    )
    content = SimpleNamespace(parts=[pdf_part], role="user")
    config = SimpleNamespace(system_instruction=None, tools=None)

    # Gemini: 3 pages * 258 tokens/page + 4 turn framing = 778 contents tokens
    req_gemini = SimpleNamespace(model="gemini-2.5-pro", contents=[content], config=config, tools_dict={})
    est_gemini = estimate_request_tokens(req_gemini)
    assert est_gemini["contents_tokens"] == 3 * 258 + 4

    # Claude: 3 pages * 2250 tokens/page + 5 turn framing = 6755 contents tokens
    req_claude = SimpleNamespace(model="claude-3-7-sonnet", contents=[content], config=config, tools_dict={})
    est_claude = estimate_request_tokens(req_claude)
    assert est_claude["contents_tokens"] == 3 * 2250 + 5


def test_custom_token_profile_registration_and_update():
    from adk_finops import register_token_profile, reset_token_profiles, update_token_profile

    try:
        # 1. Register a brand-new provider ("mistral")
        register_token_profile(
            "mistral",
            {
                "ascii_chars_per_token": 3.5,
                "turn_framing_tokens": 6,
                "tool_System_preamble_tokens": 50,
                "pdf_page_tokens": 900,
            },
        )
        req_mistral = _make_mock_request(
            "mistral/mistral-large-latest",
            "a" * 3500,
            tools_dict={"my_tool": SimpleNamespace(name="my_tool", description="Test")},
        )
        est_mistral = estimate_request_tokens(req_mistral)
        assert est_mistral["provider"] == "mistral"
        assert est_mistral["tools_tokens"] >= 50 + 36

        # 2. Update an existing provider ("anthropic") nested preamble dict & pdf_page_tokens
        update_token_profile(
            "anthropic",
            {
                "pdf_page_tokens": 2500,
                "tool_System_preamble_tokens": {"opus_5.5_auto_or_none": 300},
            },
        )
        prof = CostTracker.get_token_profile("anthropic")
        assert prof["pdf_page_tokens"] == 2500
        assert prof["tool_System_preamble_tokens"]["opus_5.5_auto_or_none"] == 300
        # Ensure other nested keys were preserved during merge
        assert prof["tool_System_preamble_tokens"]["opus_5_forced_tool"] == 406
    finally:
        reset_token_profiles()


