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

"""Google Agent Development Kit (ADK) Native FinOps Plugin.

Hooks directly into the Google ADK BasePlugin lifecycle to transparently track:
1. LLM Generation: Prompt tokens, completion tokens, thoughts tokens, cached tokens, and USD cost.
2. Grounding Fees: Actual Vertex AI Grounding and Google Search Grounding fees.
3. Dual Scopes: Real-time turn-level and cumulative session-level tracking.
4. Session Telemetry: Saves metrics into native ADK session state (Web UI) and terminal stdout.
"""

from __future__ import annotations

import logging
from pathlib import Path
from typing import Any

from google.adk.agents.invocation_context import InvocationContext
from google.adk.models.llm_response import LlmResponse
from google.adk.plugins import BasePlugin
from google.adk.plugins.base_plugin import CallbackContext
from google.adk.tools import BaseTool
from google.adk.tools.tool_context import ToolContext
from google.genai import types

from .tracker import CostTracker

logger = logging.getLogger("adk_finops.plugin")


class FinOpsCostPlugin(BasePlugin):
    """Native Google ADK Plugin that intercepts model calls and tool executions

    to track tokens, context caching, thinking tokens, and grounding fees.
    """

    def __init__(
        self,
        name: str = "finops_cost_tracker",
        default_model: str = "gemini-2.5-flash",
        rate_card_path: str | Path | None = None,
        rate_card: dict[str, Any] | None = None,
        discount_percent: float | None = None,
    ):
        super().__init__(name=name)
        self.default_model = default_model

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

    async def before_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> types.Content | None:
        """Initializes a FinOps tracking run for the current turn and session."""
        turn_id, session_id = self._extract_ids(invocation_context)
        CostTracker.start_turn(turn_id=turn_id, session_id=session_id)
        msg = f"[FinOps] Initialized cost tracking: turn={turn_id[:8]} session={session_id[:8]}"
        print(msg, flush=True)
        logger.info(msg)
        return None

    async def after_model_callback(
        self, *, callback_context: CallbackContext, llm_response: LlmResponse
    ) -> LlmResponse | None:
        """Intercepts Gemini LLM responses to record token usage and cost for turn and session."""
        turn_id, session_id = self._extract_ids(callback_context)

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
                task_name=getattr(callback_context, "agent_name", "adk_agent"),
            )
            msg = (
                f"[FinOps LLM] turn={turn_id[:8]} session={session_id[:8]} model={model_name} "
                f"tokens={record.get('total_tokens')} (prompt={prompt_tokens}, completion={completion_tokens}, thoughts={thoughts_tokens}) "
                f"cost=${record.get('cost_usd', 0.0):.6f}"
            )
            print(msg, flush=True)
            logger.info(msg)

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
            )
            msg = f"[FinOps Grounding] turn={turn_id[:8]} session={session_id[:8]} tool={tool_name} fee=${cost:.6f}"
            print(msg, flush=True)
            logger.info(msg)

        return None

    async def after_run_callback(
        self, *, invocation_context: InvocationContext
    ) -> None:
        """Gathers cumulative FinOps summary and persists it into ADK session state."""
        turn_id, session_id = self._extract_ids(invocation_context)

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
            msg = (
                f"[FinOps Summary] Turn tokens={turn_info.get('total_tokens', 0)} cost=${turn_info.get('total_cost_usd', 0.0):.6f} | "
                f"Session tokens={sess_info.get('total_tokens', 0)} cost=${sess_info.get('total_cost_usd', 0.0):.6f}"
            )
            print(msg, flush=True)
            logger.info(msg)
