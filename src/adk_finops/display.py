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

"""Rich Terminal Summary Box for adk-finops.

Provides gorgeous, color-coded, border-styled terminal summary boxes for:
- Turn & Session token counts and costs
- Context Caching savings & ROI
- Multi-Agent cost attribution
- Model breakdowns (including thinking tokens)
- Budget utilization & circuit breaker status

Supports the 'rich' library with a pure-Python Unicode box fallback.
"""

from __future__ import annotations

import sys
from typing import Any

try:
    from rich.box import ROUNDED, SIMPLE
    from rich.console import Console, Group
    from rich.panel import Panel
    from rich.table import Table
    from rich.text import Text

    HAS_RICH = True
except ImportError:
    HAS_RICH = False


def _format_tokens(num: int) -> str:
    """Formats integer token counts with commas."""
    return f"{num:,}"


def _format_usd(amount: float) -> str:
    """Formats USD currency to 4 decimal places, or 6 if very small."""
    if amount == 0.0:
        return "$0.00"
    if amount < 0.0001:
        return f"${amount:.6f}"
    return f"${amount:.4f}"


def render_rich_summary(summary: dict[str, Any], console: Console | None = None) -> Panel:
    """Creates a Rich Panel containing formatted tables for the FinOps summary."""
    sess_info = summary.get("session", summary)
    turn_info = summary.get("turn", {})
    budget_info = summary.get("budget", sess_info.get("budget", {}))

    # --- 1. Scope Overview Table ---
    scope_table = Table(box=SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False)
    scope_table.add_column("Scope", style="bold white", no_wrap=True)
    scope_table.add_column("Calls", justify="right", no_wrap=True)
    scope_table.add_column("Tokens", justify="right", no_wrap=True)
    scope_table.add_column("LLM Cost", justify="right", no_wrap=True)
    scope_table.add_column("Tool Fees", justify="right", no_wrap=True)
    scope_table.add_column("Total Cost", justify="right", style="bold yellow", no_wrap=True)

    # Turn row
    if turn_info:
        turn_tok = turn_info.get("total_tokens", 0)
        scope_table.add_row(
            "Current Turn",
            str(turn_info.get("total_calls", 0)),
            _format_tokens(turn_tok),
            _format_usd(turn_info.get("llm_cost_usd", 0.0)),
            _format_usd(turn_info.get("tool_cost_usd", 0.0)),
            _format_usd(turn_info.get("total_cost_usd", 0.0)),
        )

    # Session row
    sess_tok = sess_info.get("total_tokens", 0)
    scope_table.add_row(
        "Session Total",
        str(sess_info.get("total_calls", 0)),
        _format_tokens(sess_tok),
        _format_usd(sess_info.get("llm_cost_usd", 0.0)),
        _format_usd(sess_info.get("tool_cost_usd", 0.0)),
        _format_usd(sess_info.get("total_cost_usd", 0.0)),
        style="bold",
    )

    render_items = [scope_table]

    # --- 2. Context Caching ROI Banner ---
    savings_usd = sess_info.get("savings_usd", 0.0)
    savings_pct = sess_info.get("savings_pct", 0.0)
    gross_usd = sess_info.get("gross_cost_usd", 0.0)
    if savings_usd > 0:
        caching_text = Text()
        caching_text.append(" 💰 Context Caching Savings: ", style="bold green")
        caching_text.append(f"{_format_usd(savings_usd)} saved ", style="bold underline green")
        caching_text.append(f"({savings_pct:.1f}% reduction from {_format_usd(gross_usd)} gross)", style="green")
        render_items.append(caching_text)

    # --- 3. Model Breakdown Table ---
    models = sess_info.get("breakdown_by_model", {})
    if models:
        model_table = Table(box=SIMPLE, show_header=True, header_style="bold magenta", pad_edge=False)
        model_table.add_column("Model", style="cyan", no_wrap=True)
        model_table.add_column("Calls", justify="right", no_wrap=True)
        model_table.add_column("Tokens (In / Out)", justify="right", no_wrap=True)
        model_table.add_column("Cost (USD)", justify="right", style="bold yellow", no_wrap=True)
        model_table.add_column("Cache Savings", justify="right", style="green", no_wrap=True)

        for m_name, m_data in models.items():
            m_sav = m_data.get("savings_usd", 0.0)
            sav_str = f"{_format_usd(m_sav)} ({m_data.get('savings_pct', 0.0)}%)" if m_sav > 0 else "—"
            prompt_tok = _format_tokens(m_data.get("prompt_tokens", 0))
            out_tok = m_data.get("completion_tokens", 0) + m_data.get("thoughts_tokens", 0)
            th_tok = m_data.get("thoughts_tokens", 0)
            out_str = f"{_format_tokens(out_tok)} ({th_tok} th)" if th_tok > 0 else _format_tokens(out_tok)
            model_table.add_row(
                m_name,
                str(m_data.get("calls", 0)),
                f"{prompt_tok} / {out_str}",
                _format_usd(m_data.get("total_cost_usd", 0.0)),
                sav_str,
            )
        render_items.append(model_table)

    # --- 4. Multi-Agent Breakdown Table ---
    agents = sess_info.get("breakdown_by_agent", {})
    if len(agents) > 0:
        agent_table = Table(box=SIMPLE, show_header=True, header_style="bold blue", pad_edge=False)
        agent_table.add_column("Agent", style="bold white", no_wrap=True)
        agent_table.add_column("Calls", justify="right", no_wrap=True)
        agent_table.add_column("Tokens", justify="right", no_wrap=True)
        agent_table.add_column("LLM Cost", justify="right", no_wrap=True)
        agent_table.add_column("Tool Fees", justify="right", no_wrap=True)
        agent_table.add_column("Total Cost", justify="right", style="bold yellow", no_wrap=True)

        for a_name, a_data in agents.items():
            agent_table.add_row(
                f"🤖 {a_name}",
                str(a_data.get("calls", 0)),
                _format_tokens(a_data.get("total_tokens", 0)),
                _format_usd(a_data.get("llm_cost_usd", 0.0)),
                _format_usd(a_data.get("tool_cost_usd", 0.0)),
                _format_usd(a_data.get("total_cost_usd", 0.0)),
            )
        render_items.append(agent_table)

    # --- 5. Budget Status Footer ---
    if budget_info and budget_info.get("budget_limit_usd"):
        b_limit = budget_info.get("budget_limit_usd", 0.0)
        b_curr = budget_info.get("current_cost_usd", 0.0)
        b_pct = budget_info.get("utilization_pct", 0.0)
        b_exceeded = budget_info.get("exceeded", False)

        b_text = Text()
        if b_exceeded:
            b_text.append(" ⚠️  BUDGET EXCEEDED: ", style="bold red")
            b_text.append(f"{_format_usd(b_curr)} / {_format_usd(b_limit)} ({b_pct:.1f}%)", style="bold underline red")
        else:
            b_color = "yellow" if b_pct > 80.0 else "green"
            b_text.append(" 🛡️  Budget Guard: ", style="bold " + b_color)
            b_text.append(f"{_format_usd(b_curr)} / {_format_usd(b_limit)} ({b_pct:.1f}% utilized)", style=b_color)
        render_items.append(b_text)

    panel = Panel(
        Group(*render_items),
        title="[bold cyan]💸 ADK FinOps Cost Summary[/bold cyan]",
        subtitle="[dim]adk-finops • Universal Token & Cost Engine[/dim]",
        box=ROUNDED,
        border_style="cyan",
    )
    return panel


def format_plain_summary_box(summary: dict[str, Any]) -> str:
    """Pure-Python Unicode box fallback when 'rich' is not available."""
    sess_info = summary.get("session", summary)
    turn_info = summary.get("turn", {})
    budget_info = summary.get("budget", sess_info.get("budget", {}))

    lines: list[str] = []
    width = 76
    lines.append("╭" + "─" * (width - 2) + "╮")
    lines.append(f"│ {'💸 ADK FinOps Cost Summary':^{width - 4}} │")
    lines.append("├" + "─" * (width - 2) + "┤")

    # Metrics
    turn_cost = turn_info.get("total_cost_usd", 0.0)
    turn_tok = turn_info.get("total_tokens", 0)
    sess_cost = sess_info.get("total_cost_usd", 0.0)
    sess_tok = sess_info.get("total_tokens", 0)

    lines.append(f"│  Turn Cost:    {_format_usd(turn_cost):<12}  (Tokens: {_format_tokens(turn_tok):<8} | Calls: {turn_info.get('total_calls', 0)}){' ' * 16}│")
    lines.append(f"│  Session Cost: {_format_usd(sess_cost):<12}  (Tokens: {_format_tokens(sess_tok):<8} | Calls: {sess_info.get('total_calls', 0)}){' ' * 16}│")

    # Savings
    savings = sess_info.get("savings_usd", 0.0)
    if savings > 0:
        pct = sess_info.get("savings_pct", 0.0)
        lines.append(f"│  💰 Caching Savings: {_format_usd(savings)} ({pct:.1f}% saved via Context Caching){' ' * 12}│")

    # Budget
    if budget_info and budget_info.get("budget_limit_usd"):
        b_limit = budget_info["budget_limit_usd"]
        b_curr = budget_info.get("current_cost_usd", sess_cost)
        b_pct = budget_info.get("utilization_pct", 0.0)
        tag = "⚠️ EXCEEDED" if budget_info.get("exceeded") else "🛡️ Active"
        lines.append(f"│  Budget Guard: {tag} ({_format_usd(b_curr)} / {_format_usd(b_limit)} - {b_pct:.1f}%){' ' * 14}│")

    # Agents
    agents = sess_info.get("breakdown_by_agent", {})
    if agents:
        lines.append("├" + "─" * (width - 2) + "┤")
        lines.append(f"│  {'Agents Breakdown:':<72}│")
        for a_name, a_data in agents.items():
            a_cost = _format_usd(a_data.get("total_cost_usd", 0.0))
            a_tok = _format_tokens(a_data.get("total_tokens", 0))
            line_content = f"   • {a_name}: {a_cost} ({a_tok} tokens, {a_data.get('calls', 0)} calls)"
            lines.append(f"│  {line_content:<72}│")

    # Models
    models = sess_info.get("breakdown_by_model", {})
    if models:
        lines.append("├" + "─" * (width - 2) + "┤")
        lines.append(f"│  {'Models Breakdown:':<72}│")
        for m_name, m_data in models.items():
            m_cost = _format_usd(m_data.get("total_cost_usd", 0.0))
            m_tok = _format_tokens(m_data.get("prompt_tokens", 0) + m_data.get("completion_tokens", 0) + m_data.get("thoughts_tokens", 0))
            line_content = f"   • {m_name}: {m_cost} ({m_tok} tokens, {m_data.get('calls', 0)} calls)"
            lines.append(f"│  {line_content:<72}│")

    lines.append("╰" + "─" * (width - 2) + "╯")
    return "\n".join(lines)


def format_summary_box(summary: dict[str, Any]) -> str:
    """Formats the summary as a string box (Rich ANSI or Unicode plain text)."""
    if HAS_RICH:
        console = Console(record=True, width=80)
        panel = render_rich_summary(summary, console=console)
        console.print(panel)
        return console.export_text(clear=True)
    return format_plain_summary_box(summary)


def print_summary(
    summary: dict[str, Any] | None = None,
    run_id: str | None = None,
    session_id: str | None = None,
    console: Console | None = None,
) -> None:
    """Prints a beautiful summary box to the terminal using Rich or Unicode fallback."""
    if summary is None:
        from .tracker import CostTracker
        summary = CostTracker.get_summary(run_id=run_id, session_id=session_id, pop=False)

    if not summary:
        print("[FinOps] No cost data recorded.")
        return

    if HAS_RICH:
        c = console or Console()
        panel = render_rich_summary(summary, console=c)
        c.print(panel)
    else:
        print(format_plain_summary_box(summary))
