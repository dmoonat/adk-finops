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

"""Automated FinOps Optimization Advisor (Deterministic, Zero-LLM Overhead).

Analyzes completed session telemetry and agent-level token breakdowns against
the active RateCardRegistry to calculate concrete USD and percentage savings:
1. Context Caching Opportunity
2. Thinking Token Alert (thinking_budget right-sizing)
3. Model Right-Sizing (Pro/Flagship -> Flash/Mini downgrade)
"""

from __future__ import annotations

from typing import Any

from .rate_card import RateCardRegistry

ADVISOR_DISCLAIMER: str = (
    "Note: Optimization insights are generated using deterministic heuristics and should be treated "
    "as directional hints rather than prescriptive actions. Production changes require deeper analysis "
    "of your specific use-case, evaluation datasets, quality metrics, and business requirements."
)

# Flagship/reasoning models mapped to their cost-effective right-sized targets
RIGHT_SIZE_MODEL_MAP: dict[str, str] = {
    "gemini-3.5-flash": "gemini-3.5-flash-lite",
    "gemini-2.5-pro": "gemini-2.5-flash",
    "gpt-4o": "gpt-4o-mini",
    "claude-3-5-sonnet": "claude-3-5-haiku",
    "claude-3-opus": "claude-3-5-haiku",
}


def _resolve_target_model(model_name: str) -> str | None:
    """Returns a right-sized Flash/Mini model if `model_name` is a heavyweight model."""
    clean = (model_name or "").strip().lower()
    if "/" in clean:
        clean = clean.split("/")[-1]
    if clean in RIGHT_SIZE_MODEL_MAP:
        return RIGHT_SIZE_MODEL_MAP[clean]
    for prefix, target in RIGHT_SIZE_MODEL_MAP.items():
        if clean.startswith(prefix):
            return target
    return None


def generate_optimization_insights(
    session_data: dict[str, Any] | None,
    registry: RateCardRegistry | None = None,
    min_caching_uncached_tokens: int = 2000,
    min_caching_turns: int = 2,
    min_thinking_tokens: int = 500,
    min_thinking_cost_ratio: float = 0.60,
    max_right_size_avg_output_tokens: int = 200,
) -> dict[str, Any]:
    """Deterministically analyzes session telemetry to generate actionable cost savings insights.

    Runs in <1ms with zero LLM calls or external network dependencies.

    Returns:
        dict with:
          - "insights": list of structured insight dicts
          - "potential_savings_usd": float total estimated savings across all insights
          - "potential_savings_pct": float percentage of total session LLM cost
    """
    if not session_data:
        return {
            "insights": [],
            "potential_savings_usd": 0.0,
            "potential_savings_pct": 0.0,
        }

    reg = registry or RateCardRegistry()
    sess = session_data.get("session", session_data)
    agents: dict[str, dict[str, Any]] = sess.get("breakdown_by_agent", {}) or {}

    # Fallback: if breakdown_by_agent is empty, synthesize a default entry from session totals
    if not agents and sess.get("total_calls", 0) > 0:
        models_used = list((sess.get("breakdown_by_model") or {}).keys())
        primary_model = models_used[0] if models_used else "gemini-2.5-flash"
        agents = {
            sess.get("root_agent_name") or "default_agent": {
                "calls": sess.get("total_calls", 0),
                "prompt_tokens": sess.get("total_prompt_tokens", 0),
                "completion_tokens": sess.get("total_completion_tokens", 0),
                "thoughts_tokens": sess.get("total_thoughts_tokens", 0),
                "cached_tokens": sess.get("total_cached_tokens", 0),
                "total_tokens": sess.get("total_tokens", 0),
                "tool_calls": sess.get("total_tool_calls", 0),
                "llm_cost_usd": sess.get("total_llm_cost_usd", 0.0),
                "total_cost_usd": sess.get("total_cost_usd", 0.0),
                "models": models_used,
                "model_name": primary_model,
            }
        }

    insights: list[dict[str, Any]] = []
    total_potential_savings_usd = 0.0

    for agent_name, adata in agents.items():
        calls = int(adata.get("calls", 0) or 0)
        if calls <= 0:
            continue

        prompt_tokens = int(adata.get("prompt_tokens", 0) or 0)
        cached_tokens = int(adata.get("cached_tokens", 0) or 0)
        uncached_prompt = max(0, prompt_tokens - cached_tokens)
        completion_tokens = int(adata.get("completion_tokens", 0) or 0)
        thoughts_tokens = int(adata.get("thoughts_tokens", 0) or 0)
        tool_calls = int(adata.get("tool_calls", 0) or 0)
        llm_cost_usd = float(adata.get("llm_cost_usd", 0.0) or 0.0)

        models = adata.get("models") or []
        model_name = (
            adata.get("model_name")
            or (models[0] if models else None)
            or "gemini-2.5-flash"
        )
        avg_prompt_per_call = int(prompt_tokens / calls) if calls > 0 else prompt_tokens
        rate = reg.resolve_model(model_name)

        # ------------------------------------------------------------------
        # 1. Context Caching Opportunity
        # ------------------------------------------------------------------
        if calls >= min_caching_turns and uncached_prompt >= min_caching_uncached_tokens:
            input_rate = rate.input_per_1m
            cached_rate = rate.cached_input_per_1m
            if input_rate > cached_rate:
                # Subsequent turns (calls - 1) can reuse cached system/context prefix
                cacheable_tokens = int(uncached_prompt * ((calls - 1) / calls))
                savings_usd = round(
                    (cacheable_tokens / 1_000_000) * (input_rate - cached_rate), 6
                )
                discount_pct = round(((input_rate - cached_rate) / input_rate) * 100, 0)
                if savings_usd > 0:
                    token_label = (
                        f">{uncached_prompt // 1000}k"
                        if uncached_prompt >= 10000
                        else f"{uncached_prompt:,}"
                    )
                    msg = (
                        f"Context Caching Opportunity: Agent '{agent_name}' sent "
                        f"{token_label} uncached prompt tokens across {calls} turns. "
                        f"Enabling Context Caching would save ${savings_usd:.4f} ({int(discount_pct)}%)."
                    )
                    insights.append(
                        {
                            "category": "context_caching",
                            "agent_name": agent_name,
                            "model_name": model_name,
                            "potential_savings_usd": savings_usd,
                            "potential_savings_pct": float(discount_pct),
                            "message": msg,
                        }
                    )
                    total_potential_savings_usd += savings_usd

        # ------------------------------------------------------------------
        # 2. Thinking Token Alert (thinking_budget optimization)
        # ------------------------------------------------------------------
        total_output_tokens = completion_tokens + thoughts_tokens
        if thoughts_tokens >= min_thinking_tokens and total_output_tokens > 0:
            thinking_share = thoughts_tokens / total_output_tokens
            if thinking_share >= min_thinking_cost_ratio:
                thinking_cost_usd = round(
                    (thoughts_tokens / 1_000_000) * rate.output_per_1m, 6
                )
                share_pct = int(round(thinking_share * 100, 0))
                if thinking_cost_usd > 0:
                    msg = (
                        f"Thinking Token Alert: Thinking tokens ({thoughts_tokens:,}) "
                        f"were {share_pct}% of '{agent_name}' output cost (${thinking_cost_usd:.4f}); "
                        f"consider setting thinking_budget=0."
                    )
                    insights.append(
                        {
                            "category": "thinking_budget",
                            "agent_name": agent_name,
                            "model_name": model_name,
                            "potential_savings_usd": thinking_cost_usd,
                            "potential_savings_pct": float(share_pct),
                            "message": msg,
                        }
                    )
                    total_potential_savings_usd += thinking_cost_usd

        # ------------------------------------------------------------------
        # 3. Model Right-Sizing (Pro/Flagship -> Flash/Mini)
        # ------------------------------------------------------------------
        target_model = _resolve_target_model(model_name)
        avg_visible_output = completion_tokens / calls if calls > 0 else completion_tokens
        if (
            target_model
            and avg_visible_output < max_right_size_avg_output_tokens
            and tool_calls == 0
        ):
            target_rate = reg.resolve_model(target_model)
            # Compute current vs target LLM cost using exact session token counts
            current_est_cost = (
                llm_cost_usd
                if llm_cost_usd > 0
                else (
                    (uncached_prompt / 1_000_000) * rate.input_per_1m
                    + (cached_tokens / 1_000_000) * rate.cached_input_per_1m
                    + (total_output_tokens / 1_000_000) * rate.output_per_1m
                )
            )
            target_est_cost = (
                (uncached_prompt / 1_000_000) * target_rate.input_per_1m
                + (cached_tokens / 1_000_000) * target_rate.cached_input_per_1m
                + (total_output_tokens / 1_000_000) * target_rate.output_per_1m
            )
            savings_usd = round(max(0.0, current_est_cost - target_est_cost), 6)
            if current_est_cost > 0 and savings_usd > 0:
                savings_pct = int(round((savings_usd / current_est_cost) * 100, 0))
                out_cap = (
                    150
                    if avg_visible_output < 150
                    else int(round(avg_visible_output + 10, -1))
                )
                msg = (
                    f"Model Right-Sizing: Agent '{agent_name}' used {model_name} for "
                    f"<{out_cap} output tokens with 0 tool calls; switching to "
                    f"{target_model} saves {savings_pct}% (${savings_usd:.4f})."
                )
                insights.append(
                    {
                        "category": "model_right_sizing",
                        "agent_name": agent_name,
                        "model_name": model_name,
                        "target_model": target_model,
                        "potential_savings_usd": savings_usd,
                        "potential_savings_pct": float(savings_pct),
                        "message": msg,
                    }
                )
                total_potential_savings_usd += savings_usd

    session_llm_cost = float(
        sess.get("total_llm_cost_usd") or sess.get("total_cost_usd") or 0.0
    )
    # Cap aggregate potential savings at session LLM cost so overlapping advisors never exceed 100%
    capped_savings_usd = round(
        min(total_potential_savings_usd, session_llm_cost)
        if session_llm_cost > 0
        else total_potential_savings_usd,
        6,
    )
    overall_savings_pct = (
        round((capped_savings_usd / session_llm_cost) * 100, 1)
        if session_llm_cost > 0
        else 0.0
    )

    return {
        "insights": insights,
        "potential_savings_usd": capped_savings_usd,
        "potential_savings_pct": overall_savings_pct,
    }
