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

"""OpenTelemetry Span Exporter & Attribute Enricher for adk-finops.

Attaches OpenTelemetry GenAI semantic conventions and FinOps cost/waste
attributes (`gen_ai.usage.cost_usd`, `gen_ai.finops.task_outcome`,
`gen_ai.finops.is_wasted_spend`, etc.) to active spans and/or emits
dedicated FinOps telemetry spans compatible with Jaeger, Datadog,
Arize Phoenix, Honeycomb, and Cloud Trace.
"""

from __future__ import annotations

import logging
import os
from typing import Any

from .base import BaseExporter

logger = logging.getLogger("adk_finops.exporters.otel")


class OpenTelemetryExporter(BaseExporter):
    """Enriches OpenTelemetry spans and exports traces to Arize Phoenix, OTLP Collectors, or Google Cloud Trace."""

    def __init__(
        self,
        project_id: str | None = None,
        otlp_endpoint: str | None = None,
        tracer_name: str = "adk_finops",
        tracer: Any | None = None,
        emit_child_spans: bool = True,
        enrich_current_span: bool = True,
        auto_setup_cloud_trace: bool = True,
    ):
        self.project_id = (
            project_id
            or os.getenv("GOOGLE_CLOUD_PROJECT")
            or os.getenv("GCP_PROJECT")
        )
        self.otlp_endpoint = (
            otlp_endpoint
            or os.getenv("OTEL_EXPORTER_OTLP_TRACES_ENDPOINT")
            or os.getenv("OTEL_EXPORTER_OTLP_ENDPOINT")
            or os.getenv("PHOENIX_COLLECTOR_ENDPOINT")
        )
        self.tracer_name = tracer_name
        self._tracer = tracer
        self.emit_child_spans = emit_child_spans
        self.enrich_current_span = enrich_current_span
        self.auto_setup_cloud_trace = auto_setup_cloud_trace
        self.cloud_trace_enabled = False
        self._provider_checked = False
        self.recorded_attributes: list[dict[str, Any]] = []

    def ensure_provider_ready(self) -> None:
        """Lazily ensures an OpenTelemetry TracerProvider is active.

        Runs lazily at runtime rather than import time so third-party initializers
        like `phoenix.otel.register()` (Arize Phoenix), Jaeger, or Datadog called
        in `main()` are detected and never overwritten.
        """
        if self._provider_checked or self._tracer is not None:
            return
        self._provider_checked = True

        try:
            from opentelemetry import trace
            from opentelemetry.sdk.trace import TracerProvider
            from opentelemetry.sdk.trace.export import BatchSpanProcessor

            current_provider = trace.get_tracer_provider()
            # 1. If a TracerProvider already has active span processors (e.g., from Arize Phoenix
            #    `phoenix.otel.register()`, `adk web --trace_to_cloud`, Datadog, or Jaeger),
            #    reuse it directly without adding a conflicting exporter!
            if isinstance(current_provider, TracerProvider):
                multi_proc = getattr(current_provider, "_active_span_processor", None)
                procs = getattr(multi_proc, "_span_processors", ())
                if procs:
                    self.cloud_trace_enabled = True
                    return

            provider = (
                current_provider
                if isinstance(current_provider, TracerProvider)
                else TracerProvider()
            )

            # 2. If an OTLP or Arize Phoenix endpoint is configured via args or env vars, use OTLPSpanExporter
            if self.otlp_endpoint:
                try:
                    from opentelemetry.exporter.otlp.proto.http.trace_exporter import (
                        OTLPSpanExporter,
                    )

                    endpoint = self.otlp_endpoint
                    if not endpoint.endswith("/v1/traces"):
                        endpoint = endpoint.rstrip("/") + "/v1/traces"
                    otlp_exporter = OTLPSpanExporter(endpoint=endpoint)
                    provider.add_span_processor(BatchSpanProcessor(otlp_exporter))
                    if not isinstance(current_provider, TracerProvider):
                        trace.set_tracer_provider(provider)
                    self.cloud_trace_enabled = True
                    msg = f"[FinOps OTEL] Enabled OTLP / Arize Phoenix trace exporter (endpoint={endpoint})"
                    print(msg, flush=True)
                    logger.info(msg)
                    return
                except ImportError:
                    logger.debug(
                        "[FinOps OTEL] opentelemetry-exporter-otlp-proto-http not installed."
                    )

            # 3. Fallback: If auto_setup_cloud_trace is enabled, attach Google CloudTraceSpanExporter
            if self.auto_setup_cloud_trace:
                from opentelemetry.exporter.cloud_trace import CloudTraceSpanExporter

                exporter = (
                    CloudTraceSpanExporter(project_id=self.project_id)
                    if self.project_id
                    else CloudTraceSpanExporter()
                )
                provider.add_span_processor(BatchSpanProcessor(exporter))
                if not isinstance(current_provider, TracerProvider):
                    trace.set_tracer_provider(provider)
                self.cloud_trace_enabled = True
                resolved_proj = getattr(exporter, "project_id", self.project_id)
                msg = f"[FinOps OTEL] Enabled Google Cloud Trace exporter (project={resolved_proj})"
                print(msg, flush=True)
                logger.info(msg)
        except ImportError:
            logger.debug(
                "[FinOps OTEL] Cloud Trace / OTLP exporter packages not installed; using active TracerProvider only."
            )
        except Exception as e:
            logger.debug(f"[FinOps OTEL] Trace provider auto-setup skipped: {e}")

    def _get_tracer(self) -> Any | None:
        self.ensure_provider_ready()
        if self._tracer is not None:
            return self._tracer
        try:
            from opentelemetry import trace

            return trace.get_tracer(self.tracer_name)
        except ImportError:
            return None

    @staticmethod
    def build_otel_attributes(row: dict[str, Any]) -> dict[str, Any]:
        """Converts a serialized FinOps row into OpenTelemetry GenAI & OpenInference (Arize Phoenix) attributes."""
        prompt_tok = int(row.get("prompt_tokens", 0))
        completion_tok = int(row.get("completion_tokens", 0))
        total_tok = int(row.get("total_tokens", 0))
        total_cost = float(row.get("total_cost_usd", 0.0))
        llm_cost = float(row.get("llm_cost_usd", 0.0))

        attrs: dict[str, Any] = {
            # Standard OpenTelemetry GenAI + FinOps attributes
            "gen_ai.system": "google_adk",
            "gen_ai.finops.session_id": str(row.get("session_id") or ""),
            "gen_ai.finops.scope": str(row.get("scope") or "session"),
            "gen_ai.usage.input_tokens": prompt_tok,
            "gen_ai.usage.output_tokens": completion_tok,
            "gen_ai.usage.thoughts_tokens": int(row.get("thoughts_tokens", 0)),
            "gen_ai.usage.cached_tokens": int(row.get("cached_tokens", 0)),
            "gen_ai.usage.total_tokens": total_tok,
            "gen_ai.usage.cost_usd": total_cost,
            "gen_ai.usage.llm_cost_usd": llm_cost,
            "gen_ai.usage.tool_cost_usd": float(row.get("tool_cost_usd", 0.0)),
            "gen_ai.usage.savings_usd": float(row.get("savings_usd", 0.0)),
            "gen_ai.usage.savings_pct": float(row.get("savings_pct", 0.0)),
            "gen_ai.finops.task_outcome": str(row.get("status") or "success"),
            "gen_ai.finops.is_wasted_spend": bool(row.get("is_failure", False)),
            "gen_ai.finops.budget_exceeded": bool(row.get("budget_exceeded", False)),
            # OpenInference semantic conventions (for Arize Phoenix native Cost & Token UI columns)
            "openinference.span.kind": "LLM" if row.get("agent_name") else "CHAIN",
            "llm.token_count.prompt": prompt_tok,
            "llm.token_count.completion": completion_tok,
            "llm.token_count.total": total_tok,
            "llm.cost.total": total_cost,
        }

        if row.get("turn_id"):
            attrs["gen_ai.finops.turn_id"] = str(row["turn_id"])
        if row.get("agent_name"):
            attrs["gen_ai.agent.name"] = str(row["agent_name"])
        if row.get("model_name"):
            attrs["gen_ai.request.model"] = str(row["model_name"])
        if row.get("budget_utilization_pct") is not None:
            attrs["gen_ai.finops.budget_utilization_pct"] = float(
                row["budget_utilization_pct"]
            )
        if row.get("error"):
            attrs["gen_ai.finops.error"] = str(row["error"])

        return attrs

    def export_summary(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        """Attaches FinOps attributes to the current OTEL span and/or emits FinOps spans."""
        rows = self._prepare_rows(
            summary=summary,
            scope=scope,
            tags=tags,
            include_status_fields=True,
        )
        if not rows:
            return

        for row in rows:
            attrs = self.build_otel_attributes(row)
            self.recorded_attributes.append(attrs)

            # 1. Enrich the currently active span with aggregate scope metrics (when agent_name is None)
            if self.enrich_current_span and row.get("agent_name") is None:
                try:
                    from opentelemetry import trace

                    current_span = trace.get_current_span()
                    if current_span and current_span.is_recording():
                        for k, v in attrs.items():
                            current_span.set_attribute(k, v)
                except ImportError:
                    pass
                except Exception as e:
                    logger.debug(f"[FinOps OTEL] Could not enrich current span: {e}")

            # 2. Emit dedicated child spans for FinOps cost accounting
            if self.emit_child_spans:
                tracer = self._get_tracer()
                if tracer is not None:
                    span_suffix = row.get("agent_name") or row.get("scope") or "session"
                    span_name = f"gen_ai.finops.{span_suffix}"
                    try:
                        with tracer.start_as_current_span(span_name) as span:
                            if hasattr(span, "set_attribute"):
                                for k, v in attrs.items():
                                    span.set_attribute(k, v)
                    except Exception as e:
                        logger.debug(f"[FinOps OTEL] Could not emit child span: {e}")

        if blocking:
            self.close()

    def close(self) -> None:
        """Flushes pending OpenTelemetry spans to Cloud Trace / configured span processors."""
        try:
            from opentelemetry import trace

            provider = trace.get_tracer_provider()
            if hasattr(provider, "force_flush"):
                provider.force_flush()
        except Exception as e:
            logger.debug(f"[FinOps OTEL] Could not flush TracerProvider: {e}")
