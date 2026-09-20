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


class BudgetExceededError(Exception):
    """Raised when an agent turn or session exceeds its configured USD budget limit."""

    def __init__(
        self,
        message: str,
        current_cost_usd: float,
        budget_limit_usd: float,
        scope: str = "session",
        session_id: str | None = None,
    ) -> None:
        super().__init__(message)
        self.current_cost_usd = current_cost_usd
        self.budget_limit_usd = budget_limit_usd
        self.scope = scope
        self.session_id = session_id


class CostTracker:
    """Universal LLM & Tool Cost Tracking Engine with Budget Guardrails."""

    _lock: ClassVar[threading.RLock] = threading.RLock()
    _active_runs: ClassVar[dict[str, dict[str, Any]]] = {}
    _registry: ClassVar[RateCardRegistry] = RateCardRegistry()
    _enabled: ClassVar[bool] = True
    _budgets: ClassVar[dict[str, dict[str, float]]] = {}
    _global_budget: ClassVar[dict[str, float]] = {}
    _task_history: ClassVar[list[dict[str, Any]]] = []
    _agent_hierarchy: ClassVar[dict[str, dict[str, str | None]]] = {}

    @classmethod
    def register_agent_hierarchy(
        cls,
        agent_name: str,
        parent_agent_name: str | None = None,
        root_agent_name: str | None = None,
    ) -> None:
        """Registers the parent and root agent relationship for a given agent."""
        if not agent_name:
            return
        with cls._lock:
            eff_root = root_agent_name or parent_agent_name or agent_name
            eff_parent = parent_agent_name if parent_agent_name != agent_name else None
            cls._agent_hierarchy[agent_name] = {
                "parent_agent_name": eff_parent,
                "root_agent_name": eff_root,
            }

    @classmethod
    def get_agent_hierarchy(
        cls,
        agent_name: str | None,
        session_id: str | None = None,
    ) -> tuple[str | None, str | None]:
        """Returns (parent_agent_name, root_agent_name) for an agent within a session."""
        with cls._lock:
            sess_root = None
            if session_id and session_id in cls._active_runs:
                sess_root = cls._active_runs[session_id].get("root_agent_name")
            if agent_name and agent_name in cls._agent_hierarchy:
                info = cls._agent_hierarchy[agent_name]
                root = info.get("root_agent_name") or sess_root or agent_name
                parent = info.get("parent_agent_name")
                if parent is None and root and agent_name != root:
                    parent = root
                return parent, root
            if agent_name:
                root = sess_root or agent_name
                parent = root if agent_name != root else None
                return parent, root
            return None, sess_root

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
    def reset(cls) -> None:
        """Clears all active runs, budgets, and resets global state."""
        with cls._lock:
            cls._active_runs.clear()
            cls._budgets.clear()
            cls._global_budget.clear()
            cls._task_history.clear()
            cls._agent_hierarchy.clear()

    @classmethod
    def reset_budgets(cls) -> None:
        """Clears all configured session, turn, and agent budgets."""
        with cls._lock:
            cls._budgets.clear()
            cls._global_budget.clear()

    @classmethod
    def set_budget(
        cls,
        session_id: str | None = None,
        session_limit_usd: float | None = None,
        turn_limit_usd: float | None = None,
        agent_limits_usd: dict[str, float] | None = None,
        session_budget_usd: float | None = None,
        turn_budget_usd: float | None = None,
        agent_budgets: dict[str, float] | None = None,
    ) -> None:
        """Configures budget limits in USD globally or for a specific session/agent."""
        eff_session_limit = session_limit_usd if session_limit_usd is not None else session_budget_usd
        eff_turn_limit = turn_limit_usd if turn_limit_usd is not None else turn_budget_usd
        eff_agent_limits = agent_limits_usd if agent_limits_usd is not None else agent_budgets

        with cls._lock:
            target = cls._budgets.setdefault(session_id, {}) if session_id else cls._global_budget
            if eff_session_limit is not None:
                target["session"] = float(eff_session_limit)
            if eff_turn_limit is not None:
                target["turn"] = float(eff_turn_limit)
            if eff_agent_limits is not None:
                current_agents = target.setdefault("agents", {})
                for a_name, a_limit in eff_agent_limits.items():
                    current_agents[a_name] = float(a_limit)

    @classmethod
    def get_budget(cls, session_id: str | None = None) -> dict[str, Any]:
        """Retrieves configured budget limits for a session or global defaults."""
        with cls._lock:
            sess_budget = cls._budgets.get(session_id, {}) if session_id else {}
            global_agents = cls._global_budget.get("agents", {})
            sess_agents = sess_budget.get("agents", {})
            merged_agents = {**global_agents, **sess_agents}
            return {
                "session": sess_budget.get("session", cls._global_budget.get("session")),
                "turn": sess_budget.get("turn", cls._global_budget.get("turn")),
                "agents": merged_agents,
            }

    @classmethod
    def check_budget(
        cls,
        run_id: str | None = None,
        session_id: str | None = None,
        agent_name: str | None = None,
    ) -> tuple[bool, str | None, dict[str, Any]]:
        """Checks if either turn, session, or specific agent has exceeded its configured budget.

        Returns: (is_exceeded, reason_str, budget_status_dict)
        """
        budget_info = cls.get_budget(session_id)
        session_limit = budget_info.get("session")
        turn_limit = budget_info.get("turn")
        agent_limits: dict[str, float] = budget_info.get("agents", {})

        with cls._lock:
            sess_cost = 0.0
            agent_cost = 0.0
            if session_id and session_id in cls._active_runs:
                sess_run = cls._active_runs[session_id]
                sess_cost = sess_run.get("total_cost_usd", 0.0)
                if agent_name and "breakdown_by_agent" in sess_run:
                    agent_cost = sess_run["breakdown_by_agent"].get(agent_name, {}).get("total_cost_usd", 0.0)

            turn_cost = 0.0
            if run_id and run_id in cls._active_runs:
                turn_run = cls._active_runs[run_id]
                turn_cost = turn_run.get("total_cost_usd", 0.0)
                if agent_name and not agent_cost and "breakdown_by_agent" in turn_run:
                    agent_cost = turn_run["breakdown_by_agent"].get(agent_name, {}).get("total_cost_usd", 0.0)

        # Check agent limit first if agent_name has a specific limit configured
        if agent_name and agent_name in agent_limits:
            agent_limit = agent_limits[agent_name]
            if agent_cost >= agent_limit:
                reason = f"Agent '{agent_name}' budget limit of ${agent_limit:.4f} exceeded (current: ${agent_cost:.4f})"
                status = {
                    "exceeded": True,
                    "scope": "agent",
                    "agent_name": agent_name,
                    "current_cost_usd": agent_cost,
                    "budget_limit_usd": agent_limit,
                    "utilization_pct": round((agent_cost / agent_limit) * 100, 1) if agent_limit > 0 else 100.0,
                }
                return True, reason, status

        # Check session limit
        if session_limit is not None and sess_cost >= session_limit:
            reason = f"Session budget limit of ${session_limit:.4f} exceeded (current: ${sess_cost:.4f})"
            status = {
                "exceeded": True,
                "scope": "session",
                "current_cost_usd": sess_cost,
                "budget_limit_usd": session_limit,
                "utilization_pct": round((sess_cost / session_limit) * 100, 1) if session_limit > 0 else 100.0,
            }
            return True, reason, status

        # Check turn limit
        if turn_limit is not None and turn_cost >= turn_limit:
            reason = f"Turn budget limit of ${turn_limit:.4f} exceeded (current: ${turn_cost:.4f})"
            status = {
                "exceeded": True,
                "scope": "turn",
                "current_cost_usd": turn_cost,
                "budget_limit_usd": turn_limit,
                "utilization_pct": round((turn_cost / turn_limit) * 100, 1) if turn_limit > 0 else 100.0,
            }
            return True, reason, status

        # Under budget
        utilization = 0.0
        if agent_name and agent_name in agent_limits and agent_limits[agent_name] > 0:
            utilization = round((agent_cost / agent_limits[agent_name]) * 100, 1)
        elif session_limit and session_limit > 0:
            utilization = round((sess_cost / session_limit) * 100, 1)
        elif turn_limit and turn_limit > 0:
            utilization = round((turn_cost / turn_limit) * 100, 1)

        status = {
            "exceeded": False,
            "scope": None,
            "current_cost_usd": sess_cost if session_limit else turn_cost,
            "budget_limit_usd": session_limit or turn_limit,
            "utilization_pct": utilization,
        }
        return False, None, status

    @classmethod
    def calculate_call_cost(
        cls,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> float:
        """Calculates exact call cost in USD given token counts and applicable tier rules."""
        net_cost, _, _ = cls.calculate_call_cost_and_savings(
            model_name=model_name,
            prompt_tokens=prompt_tokens,
            completion_tokens=completion_tokens,
            cached_tokens=cached_tokens,
        )
        return net_cost

    @classmethod
    def calculate_call_cost_and_savings(
        cls,
        model_name: str,
        prompt_tokens: int,
        completion_tokens: int,
        cached_tokens: int = 0,
    ) -> tuple[float, float, float]:
        """Calculates (net_cost_usd, gross_cost_usd, savings_usd).

        gross_cost_usd is the cost if cached_tokens were billed at the regular input rate.
        savings_usd is (gross_cost_usd - net_cost_usd).
        """
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

        net_cost = round(cost_prompt + cost_cached + cost_output, 7)

        # Gross cost: prompt tokens charged at standard input_rate
        gross_prompt = (prompt_tokens / 1_000_000.0) * input_rate * discount_mult
        gross_cost = round(gross_prompt + cost_output, 7)
        savings = max(0.0, round(gross_cost - net_cost, 7))

        return net_cost, gross_cost, savings

    @classmethod
    def _create_empty_run(cls, run_id: str) -> dict[str, Any]:
        return {
            "run_id": run_id,
            "status": "pending",
            "is_failure": False,
            "error": None,
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
            "gross_cost_usd": 0.0,
            "savings_usd": 0.0,
            "savings_pct": 0.0,
            "currency": "USD",
            "breakdown_by_model": {},
            "breakdown_by_agent": {},
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
        gross_cost: float,
        savings: float,
        task_name: str | None,
        agent_name: str | None = None,
    ) -> None:
        run["total_calls"] += 1
        run["prompt_tokens"] += prompt_tokens
        run["completion_tokens"] += completion_tokens
        run["thoughts_tokens"] = run.get("thoughts_tokens", 0) + thoughts_tokens
        run["cached_tokens"] += cached_tokens
        run["total_tokens"] += total_tokens
        run["llm_cost_usd"] = round(run["llm_cost_usd"] + call_cost, 7)
        run["total_cost_usd"] = round(run["total_cost_usd"] + call_cost, 7)
        run["gross_cost_usd"] = round(run.get("gross_cost_usd", 0.0) + gross_cost, 7)
        run["savings_usd"] = round(run.get("savings_usd", 0.0) + savings, 7)
        if run["gross_cost_usd"] > 0:
            run["savings_pct"] = round((run["savings_usd"] / run["gross_cost_usd"]) * 100, 1)

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
                "gross_cost_usd": 0.0,
                "savings_usd": 0.0,
                "savings_pct": 0.0,
            },
        )
        m["calls"] += 1
        m["prompt_tokens"] += prompt_tokens
        m["completion_tokens"] += completion_tokens
        m["thoughts_tokens"] = m.get("thoughts_tokens", 0) + thoughts_tokens
        m["cached_tokens"] += cached_tokens
        m["total_cost_usd"] = round(m["total_cost_usd"] + call_cost, 7)
        m["gross_cost_usd"] = round(m.get("gross_cost_usd", 0.0) + gross_cost, 7)
        m["savings_usd"] = round(m.get("savings_usd", 0.0) + savings, 7)
        if m["gross_cost_usd"] > 0:
            m["savings_pct"] = round((m["savings_usd"] / m["gross_cost_usd"]) * 100, 1)

        # Agent breakdown
        if agent_name:
            if not run.get("root_agent_name"):
                _, inferred_root = cls.get_agent_hierarchy(agent_name, session_id=run.get("run_id"))
                run["root_agent_name"] = inferred_root or agent_name
            eff_root = run.get("root_agent_name") or agent_name
            parent_from_reg, _ = cls.get_agent_hierarchy(agent_name, session_id=run.get("run_id"))
            eff_parent = None if agent_name == eff_root else (parent_from_reg or eff_root)

            a = run["breakdown_by_agent"].setdefault(
                agent_name,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thoughts_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "llm_cost_usd": 0.0,
                    "tool_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                    "gross_cost_usd": 0.0,
                    "savings_usd": 0.0,
                    "savings_pct": 0.0,
                    "models": [],
                    "model_name": None,
                    "root_agent_name": eff_root,
                    "parent_agent_name": eff_parent,
                    "agent_role": "root_self" if agent_name == eff_root else "sub_agent",
                },
            )
            a["root_agent_name"] = eff_root
            a["parent_agent_name"] = eff_parent
            a["agent_role"] = "root_self" if agent_name == eff_root else "sub_agent"
            a["calls"] += 1
            if "models" not in a:
                a["models"] = []
            if model_name not in a["models"]:
                a["models"].append(model_name)
            a["model_name"] = a["models"][0] if len(a["models"]) == 1 else ", ".join(a["models"])
            a["prompt_tokens"] += prompt_tokens
            a["completion_tokens"] += completion_tokens
            a["thoughts_tokens"] = a.get("thoughts_tokens", 0) + thoughts_tokens
            a["cached_tokens"] += cached_tokens
            a["total_tokens"] += total_tokens
            a["llm_cost_usd"] = round(a["llm_cost_usd"] + call_cost, 7)
            a["total_cost_usd"] = round(a["total_cost_usd"] + call_cost, 7)
            a["gross_cost_usd"] = round(a.get("gross_cost_usd", 0.0) + gross_cost, 7)
            a["savings_usd"] = round(a.get("savings_usd", 0.0) + savings, 7)
            if a["gross_cost_usd"] > 0:
                a["savings_pct"] = round((a["savings_usd"] / a["gross_cost_usd"]) * 100, 1)

        # Task breakdown
        if task_name:
            t = run["breakdown_by_task"].setdefault(
                task_name,
                {"calls": 0, "total_tokens": 0, "total_cost_usd": 0.0, "savings_usd": 0.0},
            )
            t["calls"] += 1
            t["total_tokens"] += total_tokens
            t["total_cost_usd"] = round(t["total_cost_usd"] + call_cost, 7)
            t["savings_usd"] = round(t.get("savings_usd", 0.0) + savings, 7)

    @classmethod
    def _apply_tool_to_run(
        cls,
        run: dict[str, Any],
        count: int,
        total_tool_fee: float,
        task_name: str | None,
        agent_name: str | None = None,
    ) -> None:
        run["total_tool_calls"] += count
        run["tool_cost_usd"] = round(run["tool_cost_usd"] + total_tool_fee, 7)
        run["total_cost_usd"] = round(run["total_cost_usd"] + total_tool_fee, 7)
        run["gross_cost_usd"] = round(run.get("gross_cost_usd", 0.0) + total_tool_fee, 7)

        if agent_name:
            if not run.get("root_agent_name"):
                _, inferred_root = cls.get_agent_hierarchy(agent_name, session_id=run.get("run_id"))
                run["root_agent_name"] = inferred_root or agent_name
            eff_root = run.get("root_agent_name") or agent_name
            parent_from_reg, _ = cls.get_agent_hierarchy(agent_name, session_id=run.get("run_id"))
            eff_parent = None if agent_name == eff_root else (parent_from_reg or eff_root)

            a = run["breakdown_by_agent"].setdefault(
                agent_name,
                {
                    "calls": 0,
                    "prompt_tokens": 0,
                    "completion_tokens": 0,
                    "thoughts_tokens": 0,
                    "cached_tokens": 0,
                    "total_tokens": 0,
                    "llm_cost_usd": 0.0,
                    "tool_cost_usd": 0.0,
                    "total_cost_usd": 0.0,
                    "gross_cost_usd": 0.0,
                    "savings_usd": 0.0,
                    "savings_pct": 0.0,
                    "root_agent_name": eff_root,
                    "parent_agent_name": eff_parent,
                    "agent_role": "root_self" if agent_name == eff_root else "sub_agent",
                },
            )
            a["root_agent_name"] = eff_root
            a["parent_agent_name"] = eff_parent
            a["agent_role"] = "root_self" if agent_name == eff_root else "sub_agent"
            a["tool_calls"] = a.get("tool_calls", 0) + count
            a["tool_cost_usd"] = round(a["tool_cost_usd"] + total_tool_fee, 7)
            a["total_cost_usd"] = round(a["total_cost_usd"] + total_tool_fee, 7)
            a["gross_cost_usd"] = round(a.get("gross_cost_usd", 0.0) + total_tool_fee, 7)

        if task_name:
            t = run["breakdown_by_task"].setdefault(
                task_name,
                {"calls": 0, "total_tokens": 0, "total_cost_usd": 0.0, "savings_usd": 0.0},
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
    def start_turn(
        cls,
        turn_id: str,
        session_id: str | None = None,
        root_agent_name: str | None = None,
    ) -> None:
        """Initializes a tracking run for a turn (resetting existing turn counts)
        and ensures the session is tracked without resetting previous turns.
        """
        if not cls._enabled:
            return

        with cls._lock:
            turn_run = cls._create_empty_run(turn_id)
            if root_agent_name:
                turn_run["root_agent_name"] = root_agent_name
            cls._active_runs[turn_id] = turn_run
            if session_id and session_id != turn_id:
                if session_id not in cls._active_runs:
                    cls._active_runs[session_id] = cls._create_empty_run(session_id)
                if root_agent_name and not cls._active_runs[session_id].get("root_agent_name"):
                    cls._active_runs[session_id]["root_agent_name"] = root_agent_name

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
        agent_name: str | None = None,
    ) -> dict[str, Any]:
        """Records a single LLM invocation. Thread-safe and fast-bypassed when disabled.
        If both run_id (turn) and session_id (session) are provided, records to both.
        """
        if not cls._enabled:
            return {}

        total_output_tokens = completion_tokens + thoughts_tokens
        call_cost, gross_cost, savings = cls.calculate_call_cost_and_savings(
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
                gross_cost=gross_cost,
                savings=savings,
                task_name=task_name,
                agent_name=agent_name,
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
                    gross_cost=gross_cost,
                    savings=savings,
                    task_name=task_name,
                    agent_name=agent_name,
                )

        savings_pct = round((savings / gross_cost) * 100, 1) if gross_cost > 0 else 0.0
        return {
            "model": model_name,
            "agent": agent_name,
            "prompt_tokens": prompt_tokens,
            "completion_tokens": completion_tokens,
            "thoughts_tokens": thoughts_tokens,
            "cached_tokens": cached_tokens,
            "total_tokens": total_tokens,
            "cost_usd": call_cost,
            "gross_cost_usd": gross_cost,
            "savings_usd": savings,
            "savings_pct": savings_pct,
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
        agent_name: str | None = None,
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
                agent_name=agent_name,
            )

            if session_id and session_id != effective_id:
                if session_id not in cls._active_runs:
                    cls._active_runs[session_id] = cls._create_empty_run(session_id)
                cls._apply_tool_to_run(
                    cls._active_runs[session_id],
                    count=count,
                    total_tool_fee=total_tool_fee,
                    task_name=task_name,
                    agent_name=agent_name,
                )

        return total_tool_fee

    @classmethod
    def record_task_status(
        cls,
        session_id: str,
        run_id: str | None = None,
        status: str = "success",
        error: str | None = None,
    ) -> dict[str, Any]:
        """Records the outcome of a task (success or failure) and logs its efficiency metrics.

        Args:
            session_id: The session ID for the task.
            run_id: Optional turn or invocation ID.
            status: 'success', 'failed', 'error', 'aborted', or 'budget_exceeded'.
            error: Optional error message or failure reason.
        """
        clean_status = status.lower().strip()
        is_failure = clean_status in ("failed", "failure", "error", "aborted", "budget_exceeded")
        target_ids = [session_id]
        if run_id and run_id != session_id:
            target_ids.append(run_id)

        with cls._lock:
            record_info: dict[str, Any] = {}
            for tid in target_ids:
                if tid in cls._active_runs:
                    run = cls._active_runs[tid]
                    run["status"] = clean_status
                    run["is_failure"] = is_failure
                    run["error"] = error
                    record_info = {
                        "session_id": session_id,
                        "run_id": run_id,
                        "status": clean_status,
                        "is_failure": is_failure,
                        "error": error,
                        "total_tokens": run.get("total_tokens", 0),
                        "total_cost_usd": run.get("total_cost_usd", 0.0),
                        "llm_calls": run.get("total_calls", 0),
                        "tool_calls": run.get("total_tool_calls", 0),
                        "agent_breakdown": copy.deepcopy(run.get("breakdown_by_agent", {})),
                    }

            # If not in active runs, check if already in task history to update
            if not record_info:
                for item in cls._task_history:
                    if item.get("session_id") == session_id:
                        item["status"] = clean_status
                        item["is_failure"] = is_failure
                        item["error"] = error
                        record_info = item
                        break

            if record_info:
                existing = next((item for item in cls._task_history if item.get("session_id") == session_id), None)
                if existing and existing is not record_info:
                    existing.update(record_info)
                elif not existing:
                    cls._task_history.append(record_info)

            return record_info


    @classmethod
    def get_task_efficiency_metrics(cls) -> dict[str, Any]:
        """Computes comparative metrics between successful tasks and failed/wasted agent loops."""
        with cls._lock:
            history = list(cls._task_history)

        successes = [t for t in history if not t.get("is_failure", False)]
        failures = [t for t in history if t.get("is_failure", False)]

        success_count = len(successes)
        failed_count = len(failures)
        total_tasks = success_count + failed_count

        success_spend = sum(t.get("total_cost_usd", 0.0) for t in successes)
        wasted_spend = sum(t.get("total_cost_usd", 0.0) for t in failures)
        total_spend = success_spend + wasted_spend

        success_tokens = sum(t.get("total_tokens", 0) for t in successes)
        wasted_tokens = sum(t.get("total_tokens", 0) for t in failures)
        total_tokens = success_tokens + wasted_tokens

        avg_success_spend = (success_spend / success_count) if success_count > 0 else 0.0
        avg_wasted_spend = (wasted_spend / failed_count) if failed_count > 0 else 0.0

        wasted_pct = round((wasted_spend / total_spend * 100), 1) if total_spend > 0 else 0.0

        budget_limit = cls._global_budget.get("session")
        budget_util_pct = round((total_spend / budget_limit * 100), 2) if budget_limit and budget_limit > 0 else None
        wasted_of_budget_limit_pct = round((wasted_spend / budget_limit * 100), 2) if budget_limit and budget_limit > 0 else None

        return {
            "total_tasks": total_tasks,
            "successful_tasks": success_count,
            "failed_tasks": failed_count,
            "total_spend_usd": round(total_spend, 6),
            "successful_spend_usd": round(success_spend, 6),
            "wasted_spend_usd": round(wasted_spend, 6),
            "avg_successful_spend_usd": round(avg_success_spend, 6),
            "avg_wasted_spend_usd": round(avg_wasted_spend, 6),
            "wasted_spend_pct": wasted_pct,
            "budget_limit_usd": budget_limit,
            "budget_utilization_pct": budget_util_pct,
            "wasted_of_budget_limit_pct": wasted_of_budget_limit_pct,
            "total_tokens": total_tokens,
            "successful_tokens": success_tokens,
            "wasted_tokens": wasted_tokens,
            "history": history,
        }

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
        result["status"] = session_dict.get("status", "pending")
        result["is_failure"] = session_dict.get("is_failure", False)
        result["error"] = session_dict.get("error")

        # Evaluate budget status
        _, _, budget_status = cls.check_budget(run_id=effective_turn, session_id=effective_session)
        result["budget"] = budget_status
        result["session"]["budget"] = budget_status
        result["turn"]["budget"] = {
            "limit_usd": cls.get_budget(effective_session).get("turn"),
            "current_cost_usd": turn_dict.get("total_cost_usd", 0.0),
            "exceeded": (
                cls.get_budget(effective_session).get("turn") is not None
                and turn_dict.get("total_cost_usd", 0.0) >= (cls.get_budget(effective_session).get("turn") or 0.0)
            ),
        }
        return result

    @classmethod
    def print_summary(
        cls,
        run_id: str | None = None,
        session_id: str | None = None,
        pop: bool = False,
    ) -> None:
        """Prints a beautiful Rich or Unicode summary box to stdout."""
        from .display import print_summary as _print_summary

        summary = cls.get_summary(run_id=run_id, session_id=session_id, pop=pop)
        _print_summary(summary)

    @classmethod
    def format_summary_box(
        cls,
        run_id: str | None = None,
        session_id: str | None = None,
        pop: bool = False,
    ) -> str:
        """Formats the summary as a string box (Rich ANSI or Unicode plain text)."""
        from .display import format_summary_box as _format_summary_box

        summary = cls.get_summary(run_id=run_id, session_id=session_id, pop=pop)
        if not summary:
            return ""
        return _format_summary_box(summary)

    @classmethod
    @contextlib.contextmanager
    def track_run(cls, run_id: str) -> Generator[dict[str, Any], None, None]:
        """Convenience context manager for wrapping a code block or request."""
        cls.start_run(run_id)
        try:
            yield cls._active_runs.get(run_id, {})
        finally:
            pass
