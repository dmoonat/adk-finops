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

from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.plugins.base_plugin import CallbackContext
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

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
        discount_percent: float | None = None,
        budget_limit_usd: float | None = None,
        turn_budget_limit_usd: float | None = None,
        agent_budgets: dict[str, float] | None = None,
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
        exporters: list[Any] | None = None,
        export_scope: str | None = None,
        export_tags: dict[str, Any] | None = None,
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
        self.on_budget_exceeded = on_budget_exceeded.lower()
        self.fallback_model = fallback_model
        self.render_terminal_box = render_terminal_box
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

            self.exporters.append(HTTPExporter(endpoint=resolved_http))

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
        if budget_limit_usd is not None or turn_budget_limit_usd is not None or agent_budgets is not None:
            CostTracker.set_budget(
                session_limit_usd=budget_limit_usd,
                turn_limit_usd=turn_budget_limit_usd,
                agent_limits_usd=agent_budgets,
            )

        # Apply custom rate card configuration if provided
        if rate_card_path:
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

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Intercepts Gemini LLM responses to record token usage, cost, and budget limits."""
        turn_id, session_id = self._extract_ids(callback_context)
        agent_name = self._extract_agent_name(callback_context)

        usage = getattr(llm_response, "usage_metadata", None)
        model_name = getattr(llm_response, "model_version", None) or self.default_model

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

            msg = (
                f"[FinOps LLM] turn={turn_id[:8]} session={session_id[:8]} agent={agent_name} model={model_name} "
                f"tokens={record.get('total_tokens')} (prompt={prompt_tokens}, completion={completion_tokens}, thoughts={thoughts_tokens}, cached={cached_tokens}) "
                f"cost=${record.get('cost_usd', 0.0):.6f}{savings_str}"
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
        """Intercepts tool executions to record Search and Grounding fees for turn and session."""
        turn_id, session_id = self._extract_ids(tool_context)
        agent_name = self._extract_agent_name(tool_context)
        tool_name = getattr(tool, "name", "tool")
        clean_name = tool_name.strip().lower()
        tool_class_name = tool.__class__.__name__.lower()
        tool_module = getattr(tool.__class__, "__module__", "").lower()

        # 1. Check if tool is explicitly registered with a custom fee in CostTracker
        tool_type = None
        if clean_name in CostTracker.get_registry().registered_tools:
            tool_type = clean_name
        # 2. Skip MCP tools, local function tools, and database tools
        elif "mcp" in tool_module or "mcp" in tool_class_name:
            return None
        # 3. Check for actual Google Search Grounding tools
        elif "google_search" in clean_name or "googlesearch" in tool_class_name:
            tool_type = "vertex_grounding_google_search"
        # 4. Check for actual Vertex AI Search / Datastore Grounding tools
        elif (
            any(k in clean_name for k in ("vertex_search", "vertex_ai_search", "vertex_grounding"))
            or any(k in tool_class_name for k in ("vertexsearch", "vertexaisearch", "groundingtool"))
        ):
            tool_type = "vertex_grounding_private"

        if tool_type:
            cost = CostTracker.record_tool_call(
                run_id=turn_id,
                session_id=session_id,
                tool_name=tool_type,
                count=1,
                task_name=tool_name,
                agent_name=agent_name,
            )
            try:
                from opentelemetry import trace

                current_span = trace.get_current_span()
                if current_span and current_span.is_recording():
                    current_span.set_attribute("gen_ai.finops.tool_name", tool_name)
                    current_span.set_attribute("gen_ai.finops.tool_fee_usd", float(cost))
            except Exception:
                pass
            msg = f"[FinOps Grounding] turn={turn_id[:8]} session={session_id[:8]} agent={agent_name} tool={tool_name} fee=${cost:.6f}"
            print(msg, flush=True)
            logger.info(msg)

        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Gathers cumulative FinOps summary and persists it into ADK session state."""
        turn_id, session_id = self._extract_ids(invocation_context)

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

        If `auto_export` is True and the session was not already exported (e.g. because
        an exception prevented `after_run_callback` from executing), it automatically exports
        the final record to all configured exporters (BigQuery, JSONL, CSV, OpenTelemetry).

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

        should_export = export_to_bigquery if export_to_bigquery is not None else auto_export
        if should_export and self.exporters and session_id not in self._exported_sessions:
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

