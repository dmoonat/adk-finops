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

"""Explicit tool billing decorators (`@billable`) and ADK tool metadata inspection for adk-finops."""

from __future__ import annotations

import logging
from dataclasses import dataclass
from typing import Any

logger = logging.getLogger("adk_finops.billable")


def is_error_tool_result(result: Any) -> bool:
    """Returns True if a tool result indicates an execution error or MCP tool failure."""
    if isinstance(result, BaseException):
        return True
    if isinstance(result, dict):
        if result.get("isError") is True or result.get("is_error") is True:
            return True
        if result.get("error"):
            return True
        status = str(result.get("status", "")).strip().lower()
        if status in ("error", "failed", "failure"):
            return True
    return False


@dataclass
class BillableToolSpec:
    """Explicit billing specification attached to a tool via `@billable` or tool metadata."""

    fee: float = 0.0
    fee_fn: Any | None = None
    provider: str = "custom_tool"
    tool_name: str | None = None
    charge_on_error: bool = False

    def compute_fee(self, tool_args: dict[str, Any] | None = None, result: Any = None) -> float:
        """Evaluates the effective USD fee for a single tool invocation."""
        if not self.charge_on_error and is_error_tool_result(result):
            return 0.0
        if callable(self.fee_fn):
            try:
                return max(0.0, float(self.fee_fn(tool_args or {}, result)))
            except Exception as e:
                logger.warning(f"[FinOps @billable] fee_fn evaluation failed for '{self.tool_name}': {e}")
                return max(0.0, float(self.fee))
        return max(0.0, float(self.fee))


def extract_billable_spec(tool: Any) -> BillableToolSpec | None:
    """Inspects an ADK BaseTool, FunctionTool, MCPTool, or Python callable for explicit billing metadata."""
    if tool is None:
        return None

    # 1. Direct attribute on tool or callable (unwrapping ADK FunctionTool wrappers)
    for candidate in (
        tool,
        getattr(tool, "func", None),
        getattr(tool, "_func", None),
        getattr(tool, "callable", None),
    ):
        if candidate is not None:
            spec = getattr(candidate, "__finops_billing__", None)
            if isinstance(spec, BillableToolSpec):
                return spec

    # 2. Check ADK tool metadata / custom_metadata dicts (e.g., {"finops_fee": 0.01} or {"billable": {...}})
    for meta_attr in ("custom_metadata", "metadata"):
        meta = getattr(tool, meta_attr, None)
        if isinstance(meta, dict):
            if "finops_fee" in meta or "billable_fee" in meta:
                raw_fee = meta.get("finops_fee", meta.get("billable_fee", 0.0))
                return BillableToolSpec(
                    fee=float(raw_fee),
                    provider=str(meta.get("provider", "custom_tool")),
                    tool_name=getattr(tool, "name", None),
                    charge_on_error=bool(meta.get("charge_on_error", False)),
                )
            if isinstance(meta.get("billable"), dict):
                bdict = meta["billable"]
                return BillableToolSpec(
                    fee=float(bdict.get("fee", 0.0)),
                    fee_fn=bdict.get("fee_fn"),
                    provider=str(bdict.get("provider", "custom_tool")),
                    tool_name=getattr(tool, "name", None),
                    charge_on_error=bool(bdict.get("charge_on_error", False)),
                )

    return None


def billable(
    fee: float | Any = 0.0,
    *,
    fee_fn: Any | None = None,
    provider: str = "custom_tool",
    tool_name: str | None = None,
    charge_on_error: bool = False,
) -> Any:
    """Decorator to mark a tool function or BaseTool class with explicit FinOps billing metadata.

    Supports:
      - Fixed per-call fee: `@billable(fee=0.015)` or `@billable(0.015)`
      - Dynamic per-call fee: `@billable(fee_fn=lambda args, result: 0.002 * len(result.get("items", [])))`
      - Error guard: `charge_on_error=False` (default) skips charging when the tool returns an error dict.
    """
    # Handle bare `@billable` without parentheses
    if callable(fee) and fee_fn is None:
        target_fn = fee
        return billable(
            0.0,
            provider=provider,
            tool_name=tool_name,
            charge_on_error=charge_on_error,
        )(target_fn)

    numeric_fee = float(fee) if fee is not None else 0.0

    def _decorator(target: Any) -> Any:
        resolved_name = (tool_name or getattr(target, "name", None) or getattr(target, "__name__", "tool")).strip()
        spec = BillableToolSpec(
            fee=numeric_fee,
            fee_fn=fee_fn,
            provider=provider,
            tool_name=resolved_name,
            charge_on_error=charge_on_error,
        )
        setattr(target, "__finops_billing__", spec)

        # Also register static base rate in CostTracker's default registry for visibility
        try:
            from .tracker import CostTracker

            if numeric_fee > 0:
                CostTracker.register_tool_rate(resolved_name.lower(), numeric_fee)
        except Exception:
            pass

        return target

    return _decorator
