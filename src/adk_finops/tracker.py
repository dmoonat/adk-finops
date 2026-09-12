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

"""Universal LLM Token Usage & FinOps Cost Tracking Engine.

Zero external dependencies (pure Python standard library).
Thread-safe, async-safe, multi-provider, with decoupled rate cards,
context caching, thinking tokens billing, and dual turn/session tracking.
"""

from __future__ import annotations

import contextlib
import copy
import logging
import os
import threading
from collections.abc import Generator
from typing import Any, ClassVar

from .rate_card import ModelRate, RateCardRegistry

logger = logging.getLogger("adk_finops.tracker")


class CostTracker:
    """Universal LLM & Tool Cost Tracking Engine."""

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _active_runs: ClassVar[dict[str, dict[str, Any]]] = {}
    _registry: ClassVar[RateCardRegistry] = RateCardRegistry()
    _enabled: ClassVar[bool] = True

    @classmethod
    def get_registry(cls) -> RateCardRegistry:
        """Returns the active RateCardRegistry instance."""
        return cls._registry

    @classmethod
    def set_registry(cls, registry: RateCardRegistry) -> None:
        """Replaces the active RateCardRegistry instance."""
        with cls._lock:
            cls._registry = registry

    @classmethod
    def is_enabled(cls) -> bool:
        """Checks if cost tracking is globally active."""
        return cls._enabled

    @classmethod
    def set_enabled(cls, enabled: bool) -> None:
        """Dynamically enable or disable cost tracking system-wide."""
        cls._enabled = bool(enabled)
        logger.info(f"CostTracker active status changed: enabled={cls._enabled}")

    @classmethod
    def load_rate_card_file(cls, file_path: str) -> None:
        """Loads rate cards from a JSON file into the registry."""
        cls._registry.load_from_file(file_path)

    @classmethod
    def load_rate_card_url(cls, url: str) -> None:
        """Loads rate cards from a remote URL into the registry."""
        cls._registry.load_from_url(url)

    @classmethod
    def register_rate_card(cls, model_name: str, rate_card: dict[str, Any] | ModelRate) -> None:
        """Registers or overrides a pricing rate card for a model."""
        cls._registry.register_model(model_name, rate_card)

    @classmethod
    def register_tool_rate(cls, tool_name: str, cost_per_call_usd: float) -> None:
        """Registers a fixed fee per invocation for a specific tool or grounding service."""
        cls._registry.register_tool(tool_name, cost_per_call_usd)

    @classmethod
    def set_discount(cls, discount_percent: float, provider: str | None = None) -> None:
        """Sets enterprise discount percentage (globally or for a specific provider)."""
        if provider:
            cls._registry.set_provider_discount(provider, discount_percent)
        else:
            cls._registry.set_global_discount(discount_percent)

    @classmethod
    def calculate_call_cost(
        cls,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculates exact call cost in USD given token counts and applicable tier rules."""
        card = cls._registry.resolve_model(model_name)
        discount_mult = cls._registry.get_effective_discount(card.provider)

        non_cached_prompt = max(0, prompt_tokens - cached_tokens)

        # Evaluate > 128k context tier if supported
        if prompt_tokens > 128_000 and card.input_per_1m_gt_128k is not None:
            input_rate = card.input_per_1m_gt_128k
            output_rate = card.output_per_1m_gt_128k if card.output_per_1m_gt_128k is not None else card.output_per_1m
            cached_rate = card.cached_input_per_1m_gt_128k if card.cached_input_per_1m_gt_128k is not None else card.cached_input_per_1m
        else:
            input_rate = card.input_per_1m
            output_rate = card.output_per_1m
            cached_rate = card.cached_input_per_1m

        cost_prompt = (non_cached_prompt / 1_000_000.0) * input_rate * discount_mult
        cost_cached = (cached_tokens / 1_000_000.0) * cached_rate * discount_mult
        cost_output = (completion_tokens / 1_000_000.0) * output_rate * discount_mult

        return round(cost_prompt + cost_cached + cost_output, 7)

    @classmethod
    def _create_empty_run(cls, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "total_calls": 0,
            "total_tool_calls": 0,
            "prompt_tokens": 0,
            "completion_tokens": 0,
            "thoughts_tokens": 0,
            "cached_tokens": 0,
            "total_tokens": 0,
            "llm_cost_usd": 0.0,
            "tool_cost_usd": 0.0,
            "total_cost_usd": 0.0,
            "currency": "USD",
            "breakdown_by_model": {},
            "breakdown_by_task": {},
        }

    @classmethod
    def _apply_usage_to_run(
        cls,
        run: dict[str, Any],
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        thoughts_tokens: int,
        cached_tokens: int,
        total_tokens: int,
        call_cost: float,
        task_name: str | None,
    ) -> None:
        run["total_calls"] += 1
        run["prompt_tokens"] += prompt_tokens
        run["completion_tokens"] += completion_tokens
        run["thoughts_tokens"] = run.get("thoughts_tokens", 0) + thoughts_tokens
        run["cached_tokens"] += cached_tokens
        run["total_tokens"] += total_tokens
        run["llm_cost_usd"] = round(run["llm_cost_usd"] + call_cost, 7)
        run["total_cost_usd"] = round(run["total_cost_usd"] + call_cost, 7)

        # Model breakdown
        m = run["breakdown_by_model"].setdefault(
            model_name,
            {
                "calls": 0,
                "prompt_tokens": 0,
                "completion_tokens": 0,
                "thoughts_tokens": 0,
                "cached_tokens": 0,
                "total_cost_usd": 0.0,
            },
        )
        m["calls"] += 1
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["thoughts_tokens"] = m.get("thoughts_tokens", 0) + thoughts_tokens
        m["cached_tokens"] += cached_tokens
        m["total_cost_usd"] = round(m["total_cost_usd"] + call_cost, 7)

        # Task breakdown
        if task_name:
            t = run["breakdown_by_task"].setdefault(
                task_name,
                {"calls": 0, "total_tokens": 0, "total_cost_usd": 0.0},
            )
            t["calls"] += 1
            t["total_tokens"] += total_tokens
            t["total_cost_usd"] = round(t["total_cost_usd"] + call_cost, 7)

    @classmethod
    def _apply_tool_to_run(
        cls,
        run: dict[str, Any],
        count: int,
        total_tool_fee: float,
        task_name: str | None,
    ) -> None:
        run["total_tool_calls"] += count
        run["tool_cost_usd"] = round(run["tool_cost_usd"] + total_tool_fee, 7)
        run["total_cost_usd"] = round(run["total_cost_usd"] + total_tool_fee, 7)
        if task_name:
            t = run["breakdown_by_task"].setdefault(
                task_name,
                {"calls": 0, "total_tokens": 0, "total_cost_usd": 0.0},
            )
            t["total_cost_usd"] = round(t["total_cost_usd"] + total_tool_fee, 7)

    @classmethod
    def start_run(cls, run_id: str, reset: bool = True) -> None:
        """Initializes a new tracking session for a run/request/session ID."""
        if not cls._enabled:
            return

        with cls._lock:
            if not reset and run_id in cls._active_runs:
                return
            cls._active_runs[run_id] = cls._create_empty_run(run_id)

    @classmethod
    def start_turn(cls, turn_id: str, session_id: str | None = None) -> None:
        """Initializes a tracking run for a turn (resetting existing turn counts)
        and ensures the session is tracked without resetting previous turns.
        """
        if not cls._enabled:
            return

        with cls._lock:
            cls._active_runs[turn_id] = cls._create_empty_run(turn_id)
            if session_id and session_id != turn_id:
                if session_id not in cls._active_runs:
                    cls._active_runs[session_id] = cls._create_empty_run(session_id)

    @classmethod
    def record_usage(
        cls,
        run_id: str | None,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
        thoughts_tokens: int = 0,
        task_name: str | None = None,
        session_id: str | None = None,
    ) -> dict[str, Any]:
        """Records a single LLM invocation. Thread-safe and fast-bypassed when disabled.
        If both run_id (turn) and session_id (session) are provided, records to both.
        """
        if not cls._enabled:
            return {}

        total_output_tokens = completion_tokens + thoughts_tokens
        call_cost = cls.calculate_call_cost(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=total_output_tokens,
            cached_tokens=cached_tokens,
        )
        total_tokens = prompt_tokens + total_output_tokens
        effective_id = run_id or "default_session"

        with cls._lock:
            if effective_id not in cls._active_runs:
                cls._active_runs[effective_id] = cls._create_empty_run(effective_id)
            cls._apply_usage_to_run(
                cls._active_runs[effective_id],
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thoughts_tokens=thoughts_tokens,
                cached_tokens=cached_tokens,
                total_tokens=total_tokens,
                call_cost=call_cost,
                task_name=task_name,
            )

            if session_id and session_id != effective_id:
                if session_id not in cls._active_runs:
                    cls._active_runs[session_id] = cls._create_empty_run(session_id)
                cls._apply_usage_to_run(
                    cls._active_runs[session_id],
                    model_name=model_name,
                    prompt_tokens=prompt_tokens,
                    completion_tokens=completion_tokens,
                    thoughts_tokens=thoughts_tokens,
                    cached_tokens=cached_tokens,
                    total_tokens=total_tokens,
                    call_cost=call_cost,
                    task_name=task_name,
                )

        return {
            "model": model_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thoughts_tokens": thoughts_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "cost_usd": call_cost,
        }

    @classmethod
    def record_tool_call(
        cls,
        run_id: str | None,
        tool_name: str,
        count: int = 1,
        task_name: str | None = None,
        custom_cost_usd: float | None = None,
        session_id: str | None = None,
    ) -> float:
        """Records fixed fees for tools, web search, or grounding calls.
        If both run_id (turn) and session_id (session) are provided, records to both.
        """
        if not cls._enabled:
            return 0.0

        clean_tool = tool_name.strip().lower()
        if custom_cost_usd is not None:
            unit_rate = float(custom_cost_usd)
        else:
            unit_rate = cls._registry.get_tool_fee(clean_tool)

        total_tool_fee = round(unit_rate * count, 7)
        effective_id = run_id or "default_session"

        with cls._lock:
            if effective_id not in cls._active_runs:
                cls._active_runs[effective_id] = cls._create_empty_run(effective_id)
            cls._apply_tool_to_run(
                cls._active_runs[effective_id],
                count=count,
                total_tool_fee=total_tool_fee,
                task_name=task_name,
            )

            if session_id and session_id != effective_id:
                if session_id not in cls._active_runs:
                    cls._active_runs[session_id] = cls._create_empty_run(session_id)
                cls._apply_tool_to_run(
                    cls._active_runs[session_id],
                    count=count,
                    total_tool_fee=total_tool_fee,
                    task_name=task_name,
                )

        return total_tool_fee

    @classmethod
    def get_summary(
        cls,
        run_id: str | None = None,
        session_id: str | None = None,
        pop: bool = True,
    ) -> dict[str, Any] | None:
        """Retrieves the accumulated summary.

        If both run_id (turn) and session_id are provided:
          Returns a dictionary containing:
            - Top-level: Cumulative session totals (matching Session Summary)
            - 'turn': Metrics for the current turn (matching Call 1 + Call 2)
            - 'session': Cumulative metrics for the session
        """
        if not cls._enabled:
            return None

        effective_turn = run_id
        effective_session = session_id or effective_turn

        with cls._lock:
            turn_run = None
            if effective_turn:
                if pop and effective_turn != effective_session:
                    turn_run = cls._active_runs.pop(effective_turn, None)
                else:
                    turn_run = cls._active_runs.get(effective_turn, None)

            session_run = None
            if effective_session:
                if pop and effective_turn is None:
                    session_run = cls._active_runs.pop(effective_session, None)
                else:
                    session_run = cls._active_runs.get(effective_session, None)

        if not turn_run and not session_run:
            return None

        turn_dict = (
            copy.deepcopy(turn_run)
            if turn_run
            else cls._create_empty_run(effective_turn or "turn")
        )
        session_dict = (
            copy.deepcopy(session_run)
            if session_run
            else copy.deepcopy(turn_dict)
        )

        result = dict(session_dict)
        result["turn"] = turn_dict
        result["session"] = session_dict
        return result

    @classmethod
    @contextlib.contextmanager
    def track_run(cls, run_id: str) -> Generator[dict[str, Any], None, None]:
        """Convenience context manager for wrapping a code block or request."""
        cls.start_run(run_id)
        try:
            yield cls._active_runs.get(run_id, {})
        finally:
            pass
