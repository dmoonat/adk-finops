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

"""Google Agent Development Kit (ADK) Native FinOps Plugin.

Hooks directly into the Google ADK BasePlugin lifecycle to transparently track:
1. LLM Generation: Prompt tokens, completion tokens, thoughts tokens, cached tokens, and USD cost.
2. Grounding Fees: Actual Vertex AI Grounding and Google Search Grounding fees.
3. Dual Scopes: Real-time turn-level and cumulative session-level tracking.
4. Session Telemetry: Saves metrics into native ADK session state (Web UI) and terminal stdout.
"""

from __future__ import annotations

import logging
import os
from pathlib import Path
from typing import Any

try:
    from google.adk.agents.invocation_context import InvocationContext
    from google.adk.models.llm_request import LlmRequest
    from google.adk.models.llm_response import LlmResponse
    from google.adk.plugins import BasePlugin
    from google.adk.plugins.base_plugin import CallbackContext
    from google.adk.tools import BaseTool
    from google.adk.tools.tool_context import ToolContext
    from google.genai import types
except ImportError:
    InvocationContext = Any  # type: ignore[misc,assignment]
    LlmRequest = Any  # type: ignore[misc,assignment]
    CallbackContext = Any  # type: ignore[misc,assignment]
    ToolContext = Any  # type: ignore[misc,assignment]

    class LlmResponse:  # type: ignore[no-redef]
        """Fallback shim when google-adk is not installed."""

        def __init__(self, content: Any = None, **kwargs: Any) -> None:
            self.content = content
            for k, v in kwargs.items():
                setattr(self, k, v)

    class BasePlugin:  # type: ignore[no-redef]
        """Fallback shim when google-adk is not installed."""

        def __init__(self, name: str = "finops_cost_tracker") -> None:
            self.name = name

    class BaseTool:  # type: ignore[no-redef]
        """Fallback shim when google-adk is not installed."""

    class _FallbackTypes:
        class Part(dict):
            def __init__(self, text: str = "") -> None:
                super().__init__(text=text)
                self.text = text

            @classmethod
            def from_text(cls, *, text: str) -> "_FallbackTypes.Part":
                return cls(text=text)

        class Content:
            def __init__(
                self,
                parts: list[Any] | None = None,
                role: str | None = None,
            ) -> None:
                self.parts = parts or []
                self.role = role

    types = _FallbackTypes()  # type: ignore[assignment]

from .estimator import estimate_request_tokens
from .tracker import BudgetExceededError, CostTracker

logger = logging.getLogger("adk_finops.plugin")


class FinOpsCostPlugin(BasePlugin):
    """Native Google ADK Plugin that intercepts model calls and tool executions

    to track tokens, context caching, thinking tokens, and grounding fees,
    with built-in budget guardrails and circuit breakers.
    """

    def __init__(
        self,
        name: str = "finops_cost_tracker",
        default_model: str = "gemini-2.5-flash",
        rate_card_path: str | Path | None = None,
        rate_card: dict[str, Any] | None = None,
        sync_remote_rates: bool = False,
        remote_rate_card_url: str | None = None,
        remote_cache_ttl_seconds: int | None = None,
        discount_percent: float | None = None,
        region: str | None = None,
        effective_date: str | None = None,
        budget_limit_usd: float | None = None,
        turn_budget_limit_usd: float | None = None,
        agent_budgets: dict[str, float] | None = None,
        max_prompt_tokens: int | None = None,
        preflight_budget_guard: bool = True,
        on_budget_exceeded: str = "halt",  # "halt", "warn", or "downgrade"
        fallback_model: str = "gemini-2.5-flash",
        render_terminal_box: bool = True,
        enable_optimization_advisor: bool = True,
        bigquery_table: str | None = None,
        bigquery_export_scope: str = "session",  # "session", "turn", or "both"
        bigquery_tags: dict[str, Any] | None = None,
        bigquery_auto_create_table: bool = True,
        jsonl_path: str | Path | None = None,
        csv_path: str | Path | None = None,
        enable_otel: bool = False,
        otlp_endpoint: str | None = None,
        enable_dashboard: bool = False,
        dashboard_port: int = 8088,
        dashboard_endpoint: str | None = None,
        dashboard_api_key: str | None = None,
        exporters: list[Any] | None = None,
        export_scope: str | None = None,
        export_tags: dict[str, Any] | None = None,
        tool_rates: dict[str, float] | None = None,
        token_profiles: dict[str, dict[str, Any]] | None = None,
        token_profiles_path: str | Path | None = None,
    ):
        super().__init__(name=name)
        try:
            import certifi

            os.environ.setdefault("SSL_CERT_FILE", certifi.where())
            os.environ.setdefault("REQUESTS_CA_BUNDLE", certifi.where())
        except ImportError:
            pass
        self.default_model = default_model
        self.budget_limit_usd = budget_limit_usd
        self.turn_budget_limit_usd = turn_budget_limit_usd
        self.agent_budgets = agent_budgets
        env_max_tok = os.getenv("ADK_FINOPS_MAX_PROMPT_TOKENS", "").strip()
        if max_prompt_tokens is not None:
            self.max_prompt_tokens: int | None = int(max_prompt_tokens)
        elif env_max_tok.isdigit():
            self.max_prompt_tokens = int(env_max_tok)
        else:
            self.max_prompt_tokens = None

        env_preflight = os.getenv("ADK_FINOPS_PREFLIGHT_GUARD", "").strip().lower()
        if env_preflight in ("0", "false", "no", "off"):
            self.preflight_budget_guard = False
        elif env_preflight in ("1", "true", "yes", "on"):
            self.preflight_budget_guard = True
        else:
            self.preflight_budget_guard = bool(preflight_budget_guard)

        self.on_budget_exceeded = on_budget_exceeded.lower()
        self.fallback_model = fallback_model
        self.render_terminal_box = render_terminal_box
        self.tool_rates: dict[str, float] = {}
        if tool_rates:
            for t_name, t_fee in tool_rates.items():
                clean_t = t_name.strip().lower()
                self.tool_rates[clean_t] = float(t_fee)
                CostTracker.register_tool_rate(clean_t, float(t_fee))
        if token_profiles_path:
            CostTracker.load_token_profiles_file(token_profiles_path)
        if token_profiles:
            for p_key, p_cfg in token_profiles.items():
                if isinstance(p_cfg, dict):
                    CostTracker.register_token_profile(p_key, p_cfg, merge=True)
        env_advisor = os.getenv("ADK_FINOPS_OPTIMIZATION_ADVISOR", "").strip().lower()
        if env_advisor in ("0", "false", "no", "off"):
            self.enable_optimization_advisor = False
        elif env_advisor in ("1", "true", "yes", "on"):
            self.enable_optimization_advisor = True
        else:
            self.enable_optimization_advisor = bool(enable_optimization_advisor)
        CostTracker.set_optimization_advisor_enabled(self.enable_optimization_advisor)
        self.bigquery_table = bigquery_table or os.getenv("ADK_FINOPS_BIGQUERY_TABLE")
        self.export_scope = (export_scope or bigquery_export_scope).lower()
        self.export_tags = export_tags if export_tags is not None else bigquery_tags
        # Backward-compatible attribute aliases
        self.bigquery_export_scope = self.export_scope
        self.bigquery_tags = self.export_tags
        self.bigquery_auto_create_table = bigquery_auto_create_table
        self.bq_exporter = None
        self.exporters: list[Any] = list(exporters) if exporters else []
        self._exported_sessions: set[str] = set()
        self._last_partial_usage: dict[str, tuple[Any, str | None]] = {}

        if self.bigquery_table:
            from .exporters.bigquery import BigQueryExporter

            self.bq_exporter = BigQueryExporter(
                table_id=self.bigquery_table,
                auto_create_table=self.bigquery_auto_create_table,
            )
            self.exporters.append(self.bq_exporter)

        resolved_jsonl = jsonl_path or os.getenv("ADK_FINOPS_JSONL_PATH")
        if resolved_jsonl:
            from .exporters.local import JSONLExporter

            self.exporters.append(JSONLExporter(file_path=resolved_jsonl))

        resolved_csv = csv_path or os.getenv("ADK_FINOPS_CSV_PATH")
        if resolved_csv:
            from .exporters.local import CSVExporter

            self.exporters.append(CSVExporter(file_path=resolved_csv))

        resolved_http = dashboard_endpoint or os.getenv("ADK_FINOPS_DASHBOARD_ENDPOINT")
        if resolved_http:
            from .exporters.local import HTTPExporter

            self.exporters.append(HTTPExporter(endpoint=resolved_http, api_key=dashboard_api_key))

        env_otel = os.getenv("ADK_FINOPS_ENABLE_OTEL", "").strip().lower() in ("1", "true", "yes")
        resolved_otlp = otlp_endpoint or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT") or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        if enable_otel or env_otel or resolved_otlp:
            from .exporters.otel import OpenTelemetryExporter

            self.exporters.append(OpenTelemetryExporter(otlp_endpoint=otlp_endpoint))

        env_dash = os.getenv("ADK_FINOPS_ENABLE_DASHBOARD", "").strip().lower() in ("1", "true", "yes")
        self.dashboard_url: str | None = None
        if enable_dashboard or env_dash:
            from .dashboard import start_background_dashboard

            local_path = resolved_jsonl or resolved_csv
            if local_path:
                log_dir: str | Path | None = (
                    Path(local_path).parent if Path(local_path).suffix else local_path
                )
            else:
                log_dir = None
            self.dashboard_url = start_background_dashboard(
                port=dashboard_port,
                log_dir=log_dir,
                bigquery_table=self.bigquery_table,
            )

        # Configure budgets in CostTracker if specified
        if (
            budget_limit_usd is not None
            or turn_budget_limit_usd is not None
            or agent_budgets is not None
            or self.max_prompt_tokens is not None
        ):
            CostTracker.set_budget(
                session_limit_usd=budget_limit_usd,
                turn_limit_usd=turn_budget_limit_usd,
                agent_limits_usd=agent_budgets,
                max_prompt_tokens=self.max_prompt_tokens,
            )

        # Apply custom rate card configuration & region auto-detection (GOOGLE_CLOUD_LOCATION -> ADK_FINOPS_REGION -> global)
        CostTracker.set_region(region)
        if effective_date is not None:
            CostTracker.set_effective_date(effective_date)

        env_sync_rates = os.getenv("ADK_FINOPS_SYNC_REMOTE_RATES", "").strip().lower() in (
            "1",
            "true",
            "yes",
            "on",
        )
        env_remote_url = os.getenv("ADK_FINOPS_REMOTE_RATE_CARD_URL", "").strip()
        if sync_remote_rates or env_sync_rates or remote_rate_card_url or env_remote_url:
            CostTracker.sync_remote_rate_card(
                url=remote_rate_card_url or (env_remote_url if env_remote_url else None),
                cache_ttl_seconds=remote_cache_ttl_seconds,
            )

        if rate_card_path:
            if str(rate_card_path).startswith(("http://", "https://")):
                CostTracker.load_rate_card_url(str(rate_card_path))
            else:
                CostTracker.load_rate_card_file(str(rate_card_path))
        elif rate_card:
            CostTracker.get_registry().load_from_dict(rate_card)

        if discount_percent is not None:
            CostTracker.set_discount(discount_percent)

    def _extract_ids(self, ctx: Any) -> tuple[str, str]:
        """Extracts (turn_id, session_id) from any ADK invocation/callback/tool context."""
        session_id = None
        if hasattr(ctx, "session") and hasattr(ctx.session, "id"):
            session_id = ctx.session.id

        turn_id = getattr(ctx, "invocation_id", None) or session_id or "default_turn"
        session_id = session_id or turn_id
        return turn_id, session_id

    def _register_agent_tree(
        self,
        agent: Any,
        root_name: str | None = None,
        parent_name: str | None = None,
    ) -> str | None:
        """Recursively discovers and registers ADK root -> sub-agent hierarchy."""
        if agent is None:
            return root_name
        name = getattr(agent, "name", None)
        if not isinstance(name, str) or not name:
            return root_name

        actual_root = root_name
        root_obj = getattr(agent, "root_agent", None)
        if root_obj is not None and isinstance(getattr(root_obj, "name", None), str):
            actual_root = getattr(root_obj, "name")
        if actual_root is None:
            actual_root = name

        actual_parent = parent_name
        parent_obj = getattr(agent, "parent_agent", None)
        if parent_obj is not None and isinstance(getattr(parent_obj, "name", None), str):
            actual_parent = getattr(parent_obj, "name")

        CostTracker.register_agent_hierarchy(
            agent_name=name,
            parent_agent_name=actual_parent if actual_parent != name else None,
            root_agent_name=actual_root,
        )
        for sub in getattr(agent, "sub_agents", None) or []:
            self._register_agent_tree(sub, root_name=actual_root, parent_name=name)
        return actual_root

    def _extract_agent_name(self, ctx: Any) -> str:
        """Extracts the agent name and registers any discoverable agent hierarchy from context."""
        agent_obj = getattr(ctx, "agent", None)
        if agent_obj is not None:
            self._register_agent_tree(agent_obj)

        inv_ctx = getattr(ctx, "invocation_context", None) or getattr(ctx, "_invocation_context", None)
        if inv_ctx is not None and getattr(inv_ctx, "agent", None) is not None:
            self._register_agent_tree(getattr(inv_ctx, "agent", None))

        agent_name = getattr(ctx, "agent_name", None)
        if isinstance(agent_name, str) and agent_name:
            return agent_name

        if agent_obj is not None:
            name = getattr(agent_obj, "name", None)
            if isinstance(name, str) and name:
                return name

        if inv_ctx is not None:
            agent = getattr(inv_ctx, "agent", None)
            if agent is not None:
                name = getattr(agent, "name", None)
                if isinstance(name, str) and name:
                    return name

        return "root_agent"

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        """Initializes a FinOps tracking run and evaluates budget guardrails."""
        for exporter in self.exporters:
            if hasattr(exporter, "ensure_provider_ready"):
                exporter.ensure_provider_ready()

        turn_id, session_id = self._extract_ids(invocation_context)
        root_name = self._register_agent_tree(getattr(invocation_context, "agent", None))
        CostTracker.start_turn(turn_id=turn_id, session_id=session_id, root_agent_name=root_name)
        msg = f"[FinOps] Initialized cost tracking: turn={turn_id[:8]} session={session_id[:8]}"
        print(msg, flush=True)
        logger.info(msg)

        # Pre-flight Budget Guard check: Halt or downgrade if session is already over budget
        is_exceeded, reason, status = CostTracker.check_budget(run_id=turn_id, session_id=session_id)
        if is_exceeded:
            if self.on_budget_exceeded == "halt":
                halt_msg = f"⚠️ [FinOps Budget Guard] Turn blocked: {reason}."
                print(halt_msg, flush=True)
                logger.warning(halt_msg)
                return types.Content(
                    parts=[
                        types.Part.from_text(
                            text=f"⚠️ {reason}. Turn was halted to prevent unexpected charges."
                        )
                    ]
                )
            elif self.on_budget_exceeded == "downgrade":
                if hasattr(invocation_context, "agent") and self.fallback_model:
                    invocation_context.agent.model = self.fallback_model
                    msg = f"⚠️ [FinOps Budget Guard] Downgraded agent model to {self.fallback_model}: {reason}."
                    print(msg, flush=True)
                    logger.warning(msg)
            else:  # "warn"
                msg = f"⚠️ [FinOps Budget Guard] Warning: {reason}."
                print(msg, flush=True)
                logger.warning(msg)

        return None

    async def before_model_callback(
        self, *, callback_context: CallbackContext, llm_request: LlmRequest
    ) -> LlmResponse | None:
        """Pre-flight budget guard that estimates input tokens and USD cost BEFORE the LLM network call.

        Blocks massive prompts (e.g., >200K tokens) or projected budget breaches at $0.00 cloud cost
        by returning an early LlmResponse ("halt") or downgrading llm_request.model in-place ("downgrade").
        """
        if not self.preflight_budget_guard:
            return None

        turn_id, session_id = self._extract_ids(callback_context)
        agent_name = self._extract_agent_name(callback_context)
        target_model = (
            getattr(llm_request, "model", None)
            or getattr(getattr(callback_context, "agent", None), "model", None)
            or self.default_model
        )
        if not isinstance(target_model, str) or not target_model:
            target_model = self.default_model

        est = estimate_request_tokens(llm_request, model_name=target_model)
        est_prompt_tokens = est["prompt_tokens"]
        est_cached_tokens = est["cached_tokens"]

        is_exceeded, reason, preflight_status = CostTracker.check_preflight_budget(
            model_name=target_model,
            estimated_prompt_tokens=est_prompt_tokens,
            estimated_cached_tokens=est_cached_tokens,
            run_id=turn_id,
            session_id=session_id,
            agent_name=agent_name,
            max_prompt_tokens_override=self.max_prompt_tokens,
        )

        if not is_exceeded:
            return None

        est_call_cost = preflight_status.get("estimated_call_cost_usd", 0.0)
        scope = preflight_status.get("scope")

        # 1. Downgrade mode: switch llm_request.model to fallback_model before the network call
        if (
            self.on_budget_exceeded == "downgrade"
            and scope != "max_prompt_tokens"
            and self.fallback_model
            and target_model != self.fallback_model
        ):
            downgraded_cost = CostTracker.calculate_call_cost(
                model_name=self.fallback_model,
                prompt_tokens=est_prompt_tokens,
                completion_tokens=0,
                cached_tokens=est_cached_tokens,
            )
            avoided_cost = max(0.0, round(est_call_cost - downgraded_cost, 7))
            if hasattr(llm_request, "model"):
                llm_request.model = self.fallback_model
            inv_ctx = getattr(callback_context, "invocation_context", None) or getattr(
                callback_context, "_invocation_context", None
            )
            if inv_ctx is not None and hasattr(inv_ctx, "agent") and inv_ctx.agent is not None:
                inv_ctx.agent.model = self.fallback_model
            if hasattr(callback_context, "agent") and callback_context.agent is not None:
                callback_context.agent.model = self.fallback_model

            CostTracker.record_preflight_event(
                run_id=turn_id,
                session_id=session_id,
                action="downgraded",
                estimated_prompt_tokens=est_prompt_tokens,
                avoided_cost_usd=avoided_cost,
            )
            msg = (
                f"⚠️ [FinOps Pre-Flight Guard] Downgraded outgoing request model from {target_model} "
                f"to {self.fallback_model} before execution: {reason} "
                f"(new est. call cost: ${downgraded_cost:.6f}, saved ${avoided_cost:.6f})"
            )
            print(msg, flush=True)
            logger.warning(msg)
            return None

        # 2. Halt mode: short-circuit before the network request is sent ($0.00 cloud cost)
        if self.on_budget_exceeded == "halt" or scope == "max_prompt_tokens":
            CostTracker.record_preflight_event(
                run_id=turn_id,
                session_id=session_id,
                action="blocked",
                estimated_prompt_tokens=est_prompt_tokens,
                avoided_cost_usd=est_call_cost,
            )
            CostTracker.record_task_status(
                session_id=session_id,
                run_id=turn_id,
                status="budget_exceeded",
                error=reason,
            )
            halt_msg = (
                f"🛑 [FinOps Pre-Flight Guard] Request blocked before LLM call "
                f"($0.00 spent, avoided ~${est_call_cost:.4f} / {est_prompt_tokens:,} input tokens): {reason}"
            )
            print(halt_msg, flush=True)
            logger.warning(halt_msg)
            return LlmResponse(
                content=types.Content(
                    role="model",
                    parts=[
                        types.Part.from_text(
                            text=f"⚠️ [FinOps Pre-Flight Guard] Request blocked before execution: {reason}"
                        )
                    ],
                )
            )

        # 3. Warn mode: log pre-flight warning and let request proceed
        warn_msg = f"⚠️ [FinOps Pre-Flight Guard] Warning: {reason}"
        print(warn_msg, flush=True)
        logger.warning(warn_msg)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Intercepts Gemini LLM responses to record token usage, cost, and budget limits."""
        turn_id, session_id = self._extract_ids(callback_context)
        agent_name = self._extract_agent_name(callback_context)
        partial_key = f"{turn_id}:{agent_name}"

        usage = getattr(llm_response, "usage_metadata", None)
        model_version = getattr(llm_response, "model_version", None)

        # In SSE / live streaming mode, ADK calls after_model_callback once per streamed
        # partial chunk (llm_response.partial == True) with cumulative usage_metadata,
        # followed by one final aggregated LlmResponse (partial == False / None).
        # Skip partial chunks so a streamed model call is recorded only once.
        if getattr(llm_response, "partial", None) is True:
            if usage is not None:
                self._last_partial_usage[partial_key] = (usage, model_version)
            return None

        partial_fallback = self._last_partial_usage.pop(partial_key, None)
        if usage is None and partial_fallback is not None:
            usage, fallback_model_ver = partial_fallback
            if not model_version:
                model_version = fallback_model_ver

        model_name = model_version or self.default_model

        if usage:
            prompt_tokens = getattr(usage, "prompt_token_count", 0) or 0
            completion_tokens = getattr(usage, "candidates_token_count", 0) or 0
            thoughts_tokens = getattr(usage, "thoughts_token_count", 0) or 0
            cached_tokens = getattr(usage, "cached_content_token_count", 0) or 0

            record = CostTracker.record_usage(
                run_id=turn_id,
                session_id=session_id,
                model_name=model_name,
                prompt_tokens=prompt_tokens,
                completion_tokens=completion_tokens,
                thoughts_tokens=thoughts_tokens,
                cached_tokens=cached_tokens,
                task_name=agent_name,
                agent_name=agent_name,
            )
            savings_str = ""
            if record.get("savings_usd", 0.0) > 0:
                savings_str = f" | 💰 Saved ${record['savings_usd']:.6f} ({record['savings_pct']}%) via Context Caching"

            # Enrich ADK's active OpenTelemetry span (e.g. when running with --trace_to_cloud)
            try:
                from opentelemetry import trace

                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    current_span.set_attribute("gen_ai.finops.agent_name", agent_name)
                    current_span.set_attribute("gen_ai.finops.model", model_name)
                    current_span.set_attribute("gen_ai.finops.cost_usd", float(record.get("cost_usd", 0.0)))
                    current_span.set_attribute("gen_ai.finops.savings_usd", float(record.get("savings_usd", 0.0)))
                    current_span.set_attribute("gen_ai.finops.savings_pct", float(record.get("savings_pct", 0.0)))
                    current_span.set_attribute("gen_ai.usage.thoughts_tokens", int(thoughts_tokens))
                    current_span.set_attribute("gen_ai.usage.cached_tokens", int(cached_tokens))
            except Exception:
                pass

            fallback_str = (
                " | ⚠️ [FALLBACK RATE - register via CostTracker.register_rate_card()]"
                if record.get("is_fallback_rate")
                else ""
            )
            msg = (
                f"[FinOps LLM] turn={turn_id[:8]} session={session_id[:8]} agent={agent_name} model={model_name} "
                f"tokens={record.get('total_tokens')} (prompt={prompt_tokens}, completion={completion_tokens}, thoughts={thoughts_tokens}, cached={cached_tokens}) "
                f"cost=${record.get('cost_usd', 0.0):.6f}{savings_str}{fallback_str}"
            )
            print(msg, flush=True)
            logger.info(msg)

            # Post-model Budget Guard check
            is_exceeded, reason, status = CostTracker.check_budget(
                run_id=turn_id, session_id=session_id, agent_name=agent_name
            )
            if is_exceeded:
                if self.on_budget_exceeded == "halt":
                    msg = f"⚠️ [FinOps Budget Guard] {reason}"
                    print(msg, flush=True)
                    logger.warning(msg)
                    raise BudgetExceededError(
                        message=reason or "Budget limit exceeded",
                        current_cost_usd=status.get("current_cost_usd", 0.0),
                        budget_limit_usd=status.get("budget_limit_usd", 0.0),
                        scope=status.get("scope", "session"),
                        session_id=session_id,
                    )
                elif self.on_budget_exceeded == "downgrade":
                    if hasattr(callback_context, "agent") and self.fallback_model:
                        callback_context.agent.model = self.fallback_model
                    msg = f"⚠️ [FinOps Budget Guard] Budget limit reached; subsequent model calls downgraded to {self.fallback_model}."
                    print(msg, flush=True)
                    logger.warning(msg)
                else:  # "warn"
                    msg = f"⚠️ [FinOps Budget Guard] Warning: {reason}"
                    print(msg, flush=True)
                    logger.warning(msg)

        return None

    async def on_event_callback(
        self, *, invocation_context: InvocationContext, event: Any
    ) -> Any | None:
        """Attaches FinOps cost metrics (both turn and session) to event state_delta so ADK Web UI updates session state in real-time."""
        if getattr(event, "partial", None) is True:
            return event

        turn_id, session_id = self._extract_ids(invocation_context)

        summary = CostTracker.get_summary(run_id=turn_id, session_id=session_id, pop=False)
        if summary and hasattr(event, "actions") and event.actions is not None:
            if event.actions.state_delta is None:
                event.actions.state_delta = {}
            event.actions.state_delta["finops_cost"] = summary

        return event

    async def after_tool_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        result: dict,
    ) -> dict | None:
        """Intercepts tool executions to record explicit @billable, Search, and Grounding fees for turn and session."""
        from .billable import extract_billable_spec, is_error_tool_result

        turn_id, session_id = self._extract_ids(tool_context)
        agent_name = self._extract_agent_name(tool_context)
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
        clean_name = tool_name.strip().lower()
        tool_class_name = tool.__class__.__name__.lower()
        tool_module = getattr(tool.__class__, "__module__", "").lower()

        custom_cost_usd: float | None = None
        provider: str | None = None
        tool_type: str | None = None

        # 1. Inspect explicit @billable decorator or ADK tool custom_metadata
        billable_spec = extract_billable_spec(tool)
        if billable_spec is not None:
            computed_fee = billable_spec.compute_fee(tool_args=tool_args, result=result)
            if computed_fee <= 0.0:
                return None
            tool_type = (billable_spec.tool_name or clean_name).strip().lower()
            custom_cost_usd = computed_fee
            provider = billable_spec.provider
        # 2. Check plugin-level tool_rates (overrides MCP skip so paid MCP tools in tool_rates are billed)
        elif clean_name in self.tool_rates:
            if is_error_tool_result(result):
                return None
            tool_type = clean_name
            custom_cost_usd = self.tool_rates[clean_name]
        # 3. Check if tool is explicitly registered with a custom fee in CostTracker
        elif clean_name in CostTracker.get_registry().registered_tools:
            if is_error_tool_result(result):
                return None
            tool_type = clean_name
        # 4. Skip un-decorated MCP tools, local function tools, and database tools
        elif "mcp" in tool_module or "mcp" in tool_class_name:
            return None
        # 5. Check for Google Search / Web Grounding tools (by name, class, or ADK grounding config attributes)
        elif (
            "google_search" in clean_name
            or "googlesearch" in tool_class_name
            or "enterprisewebsearch" in tool_class_name
            or hasattr(tool, "google_search")
        ):
            tool_type = "vertex_grounding_google_search"
        # 6. Check for Vertex AI Search / Datastore Grounding tools
        elif (
            any(k in clean_name for k in ("vertex_search", "vertex_ai_search", "vertex_grounding"))
            or any(k in tool_class_name for k in ("vertexsearch", "vertexaisearch", "groundingtool"))
            or hasattr(tool, "vertex_ai_search")
            or hasattr(tool, "data_store_id")
        ):
            tool_type = "vertex_grounding_private"

        if tool_type:
            cost = CostTracker.record_tool_call(
                run_id=turn_id,
                session_id=session_id,
                tool_name=tool_type,
                count=1,
                task_name=tool_name,
                custom_cost_usd=custom_cost_usd,
                agent_name=agent_name,
                provider=provider,
            )
            try:
                from opentelemetry import trace

                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    current_span.set_attribute("gen_ai.finops.tool_name", tool_name)
                    current_span.set_attribute("gen_ai.finops.tool_fee_usd", float(cost))
            except Exception:
                pass
            is_grounding = tool_type in (
                "vertex_grounding_google_search",
                "vertex_grounding_private",
                "google_search",
                "google_maps_grounding",
                "web_grounding",
                "enterprise_web_search",
                "vertex_search",
                "vertex_ai_search",
            )
            log_prefix = "[FinOps Grounding]" if is_grounding else "[FinOps Tool]"
            msg = f"{log_prefix} turn={turn_id[:8]} session={session_id[:8]} agent={agent_name} tool={tool_name} fee=${cost:.6f}"
            print(msg, flush=True)
            logger.info(msg)

        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Gathers cumulative FinOps summary and persists it into ADK session state."""
        turn_id, session_id = self._extract_ids(invocation_context)

        # Flush any leftover partial streaming usage if a stream terminated without a final non-partial chunk
        prefix = f"{turn_id}:"
        for k in [k for k in self._last_partial_usage if k.startswith(prefix)]:
            usage, model_version = self._last_partial_usage.pop(k)
            agent_name = k[len(prefix) :] or "root_agent"
            CostTracker.record_usage(
                run_id=turn_id,
                session_id=session_id,
                model_name=model_version or self.default_model,
                prompt_tokens=getattr(usage, "prompt_token_count", 0) or 0,
                completion_tokens=getattr(usage, "candidates_token_count", 0) or 0,
                thoughts_tokens=getattr(usage, "thoughts_token_count", 0) or 0,
                cached_tokens=getattr(usage, "cached_content_token_count", 0) or 0,
                task_name=agent_name,
                agent_name=agent_name,
            )

        # Auto-detect task status from session state if set by workflow nodes/critics
        if hasattr(invocation_context, "session") and invocation_context.session:
            sess_state = getattr(invocation_context.session, "state", {}) or {}
            if sess_state.get("failed") or sess_state.get("is_failure"):
                CostTracker.record_task_status(
                    session_id=session_id,
                    run_id=turn_id,
                    status=sess_state.get("status", "failed"),
                    error=sess_state.get("error_reason") or sess_state.get("error"),
                )
            elif sess_state.get("status"):
                CostTracker.record_task_status(
                    session_id=session_id,
                    run_id=turn_id,
                    status=sess_state.get("status"),
                    error=sess_state.get("error_reason") or sess_state.get("error"),
                )
            else:
                # If no status set yet, mark as success
                CostTracker.record_task_status(
                    session_id=session_id,
                    run_id=turn_id,
                    status="success",
                )

        summary = CostTracker.get_summary(run_id=turn_id, session_id=session_id, pop=False)
        if summary:
            if hasattr(invocation_context, "session") and invocation_context.session:
                if hasattr(invocation_context.session, "state"):
                    invocation_context.session.state["finops_cost"] = summary

                if (
                    hasattr(invocation_context, "session_service")
                    and invocation_context.session_service
                ):
                    try:
                        from google.adk.events import Event, EventActions

                        event = Event(
                            author="system",
                            actions=EventActions(state_delta={"finops_cost": summary}),
                        )
                        await invocation_context.session_service.append_event(
                            session=invocation_context.session, event=event
                        )
                    except Exception as e:
                        logger.debug(
                            f"[FinOps] Could not append state event to session_service: {e}"
                        )

            turn_info = summary.get("turn", {})
            sess_info = summary.get("session", {})
            savings_info = ""
            if sess_info.get("savings_usd", 0.0) > 0:
                savings_info = f" | 💰 Total Saved: ${sess_info['savings_usd']:.6f} ({sess_info.get('savings_pct', 0.0)}%) via Caching"

            msg = (
                f"[FinOps Summary] Turn tokens={turn_info.get('total_tokens', 0)} cost=${turn_info.get('total_cost_usd', 0.0):.6f} | "
                f"Session tokens={sess_info.get('total_tokens', 0)} cost=${sess_info.get('total_cost_usd', 0.0):.6f}{savings_info}"
            )
            print(msg, flush=True)
            logger.info(msg)

            agent_breakdown = sess_info.get("breakdown_by_agent", {})
            if agent_breakdown:
                agent_parts = [
                    f"{name}: ${data.get('total_cost_usd', 0.0):.4f} ({data.get('total_tokens', 0)} tok)"
                    for name, data in agent_breakdown.items()
                ]
                agents_msg = f"[FinOps Agents] " + " | ".join(agent_parts)
                print(agents_msg, flush=True)
                logger.info(agents_msg)

            tool_breakdown = sess_info.get("breakdown_by_tool", {})
            if tool_breakdown:
                tool_parts = [
                    f"{a_name} -> {t_name}: {t_data.get('calls', 0)} call(s) (${t_data.get('total_cost_usd', 0.0):.4f})"
                    for a_name, t_map in tool_breakdown.items()
                    if isinstance(t_map, dict)
                    for t_name, t_data in t_map.items()
                    if isinstance(t_data, dict)
                ]
                if tool_parts:
                    tools_msg = "[FinOps Tools] " + " | ".join(tool_parts)
                    print(tools_msg, flush=True)
                    logger.info(tools_msg)

            if self.render_terminal_box:
                from .display import print_summary

                print_summary(
                    summary,
                    show_optimization_insights=self.enable_optimization_advisor,
                )

            if self.exporters:
                tags = dict(self.export_tags or {})
                task_status = sess_info.get("status") or summary.get("status")
                is_fail = sess_info.get("is_failure") if sess_info.get("is_failure") is not None else summary.get("is_failure")
                if task_status:
                    tags["status"] = task_status
                if is_fail is not None:
                    tags["is_failure"] = is_fail
                if sess_info.get("error"):
                    tags["error"] = sess_info.get("error")

                for exporter in self.exporters:
                    exporter.export_summary(
                        summary=summary,
                        scope=self.export_scope,
                        tags=tags if tags else None,
                        blocking=False,
                    )
                self._exported_sessions.add(session_id)
                self._exported_sessions.add((session_id, str(task_status or "success").lower()))

    async def on_model_error_callback(
        self,
        *,
        callback_context: CallbackContext,
        llm_request: LlmRequest,
        error: Exception,
    ) -> LlmResponse | None:
        """Captures unhandled LLM model execution errors and marks the turn/session as error."""
        turn_id, session_id = self._extract_ids(callback_context)
        err_status = "budget_exceeded" if isinstance(error, BudgetExceededError) else "error"
        err_msg = str(error) if isinstance(error, BudgetExceededError) else f"{type(error).__name__}: {error}"
        CostTracker.record_task_status(
            session_id=session_id,
            run_id=turn_id,
            status=err_status,
            error=err_msg,
        )
        return None

    async def on_tool_error_callback(
        self,
        *,
        tool: BaseTool,
        tool_args: dict[str, Any],
        tool_context: ToolContext,
        error: Exception,
    ) -> dict[str, Any] | None:
        """Captures unhandled tool execution errors.

        Respects `@billable(charge_on_error=True)` if configured; otherwise skips charging
        and records the error reason if the run later aborts.
        """
        from .billable import extract_billable_spec

        turn_id, session_id = self._extract_ids(tool_context)
        agent_name = self._extract_agent_name(tool_context)
        tool_name = getattr(tool, "name", None) or getattr(tool, "__name__", "tool")
        billable_spec = extract_billable_spec(tool)
        if billable_spec is not None and billable_spec.charge_on_error:
            computed_fee = billable_spec.compute_fee(tool_args=tool_args, result=error)
            if computed_fee > 0.0:
                tool_type = (billable_spec.tool_name or tool_name).strip().lower()
                CostTracker.record_tool_call(
                    run_id=turn_id,
                    session_id=session_id,
                    tool_name=tool_type,
                    count=1,
                    task_name=tool_name,
                    custom_cost_usd=computed_fee,
                    agent_name=agent_name,
                    provider=billable_spec.provider,
                )
        return None

    async def on_agent_error_callback(
        self,
        *,
        agent: BaseAgent,
        callback_context: CallbackContext,
        error: Exception,
    ) -> None:
        """Captures unhandled exceptions escaping an ADK agent."""
        turn_id, session_id = self._extract_ids(callback_context)
        err_status = "budget_exceeded" if isinstance(error, BudgetExceededError) else "error"
        err_msg = str(error) if isinstance(error, BudgetExceededError) else f"{type(error).__name__}: {error}"
        CostTracker.record_task_status(
            session_id=session_id,
            run_id=turn_id,
            status=err_status,
            error=err_msg,
        )

    async def on_run_error_callback(
        self,
        *,
        invocation_context: InvocationContext,
        error: Exception,
    ) -> None:
        """Automatically records wasted spend and flushes telemetry when an unhandled exception escapes the ADK Runner."""
        turn_id, session_id = self._extract_ids(invocation_context)
        err_status = "budget_exceeded" if isinstance(error, BudgetExceededError) else "error"
        err_msg = str(error) if isinstance(error, BudgetExceededError) else f"{type(error).__name__}: {error}"
        self.record_task_status(
            session_id=session_id,
            run_id=turn_id,
            status=err_status,
            error=err_msg,
            auto_export=True,
        )

    def export_session(
        self,
        session_id: str,
        run_id: str | None = None,
        status: str | None = None,
        error: str | None = None,
        tags: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        """Exports a session's FinOps summary to all configured exporters (BigQuery, JSONL, CSV, OTEL).

        Useful when an unhandled exception or workflow error prevented
        after_run_callback from executing automatically.
        """
        if status:
            CostTracker.record_task_status(session_id=session_id, run_id=run_id, status=status, error=error)

        summary = CostTracker.get_summary(run_id=run_id, session_id=session_id, pop=False)
        if not summary or not self.exporters:
            return

        export_tags = dict(self.export_tags or {})
        if tags:
            export_tags.update(tags)

        sess_info = summary.get("session", {})
        task_status = status or sess_info.get("status") or summary.get("status")
        is_fail = sess_info.get("is_failure") if sess_info.get("is_failure") is not None else summary.get("is_failure")
        if task_status:
            export_tags["status"] = task_status
        if is_fail is not None:
            export_tags["is_failure"] = is_fail
        if error or sess_info.get("error"):
            export_tags["error"] = error or sess_info.get("error")

        for exporter in self.exporters:
            exporter.export_summary(
                summary=summary,
                scope=self.export_scope,
                tags=export_tags if export_tags else None,
                blocking=blocking,
            )
        self._exported_sessions.add(session_id)
        self._exported_sessions.add((session_id, str(task_status or "success").lower()))

    def record_task_status(
        self,
        session_id: str,
        run_id: str | None = None,
        status: str = "success",
        error: str | None = None,
        auto_export: bool = True,
        export_to_bigquery: bool | None = None,
    ) -> dict[str, Any]:
        """Records task outcome (success vs failure) and updates efficiency metrics.

        If `auto_export` is True and the session with this status was not already exported,
        it automatically exports the record to all configured exporters (BigQuery, JSONL, CSV, OpenTelemetry).

        Args:
            session_id: The session ID for the task.
            run_id: Optional turn or invocation ID.
            status: 'success', 'failed', 'error', 'aborted', or 'budget_exceeded'.
            error: Optional error message or failure reason.
            auto_export: Whether to automatically export to all configured exporters if not yet exported.
            export_to_bigquery: Deprecated alias for `auto_export` kept for backward compatibility.
        """
        res = CostTracker.record_task_status(
            session_id=session_id,
            run_id=run_id,
            status=status,
            error=error,
        )

        norm_status = str(status or "success").lower()
        should_export = export_to_bigquery if export_to_bigquery is not None else auto_export
        already_exported = (
            (session_id, norm_status) in self._exported_sessions
            or (norm_status == "success" and session_id in self._exported_sessions)
        )
        if should_export and self.exporters and not already_exported:
            self.export_session(
                session_id=session_id,
                run_id=run_id,
                status=status,
                error=error,
                blocking=False,
            )

        return res

    def close(self) -> None:
        """Flushes and shuts down exporter background workers."""
        for exporter in self.exporters:
            if hasattr(exporter, "close"):
                exporter.close()

