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

    # --- 1b. Task Outcome & Wasted Spend Status ---
    task_status = sess_info.get("status") or summary.get("status")
    is_failure = sess_info.get("is_failure") or summary.get("is_failure", False)
    task_error = sess_info.get("error") or summary.get("error")
    if task_status:
        status_text = Text()
        if is_failure:
            status_text.append(" ❌ TASK OUTCOME: ", style="bold red")
            status_text.append(f"FAILED ({task_status.upper()})", style="bold underline red")
            status_text.append(f" | 💸 Wasted Spend: {_format_usd(sess_info.get('total_cost_usd', 0.0))}", style="bold red")
            if task_error:
                status_text.append(f" ({task_error})", style="italic red")
        else:
            status_text.append(" ✅ TASK OUTCOME: ", style="bold green")
            status_text.append(f"SUCCESS", style="bold underline green")
            status_text.append(f" | 💰 Effective Spend: {_format_usd(sess_info.get('total_cost_usd', 0.0))}", style="green")
        render_items.append(status_text)

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
        model_table = Table(box=SIMPLE, show_header=True, header_style="bold magenta", pad_edge=False, expand=False)
        model_table.add_column("Model", style="cyan", no_wrap=True)
        model_table.add_column("Calls", justify="right", no_wrap=True)
        model_table.add_column("Tokens (In/Out)", justify="right", no_wrap=True)
        model_table.add_column("Cost (USD)", justify="right", style="bold yellow", no_wrap=True)
        model_table.add_column("Savings", justify="right", style="green", no_wrap=True)

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


def render_task_efficiency_table(metrics: dict[str, Any], console: Console | None = None) -> Panel:
    """Creates a Rich Panel comparing successful tasks vs failed/wasted agent loops."""
    table = Table(box=SIMPLE, show_header=True, header_style="bold cyan", pad_edge=False)
    table.add_column("Metric", style="bold white", no_wrap=True)
    table.add_column("Successful Tasks", justify="right", style="bold green", no_wrap=True)
    table.add_column("Failed Loops (Wasted)", justify="right", style="bold red", no_wrap=True)
    table.add_column("Total / Impact", justify="right", style="bold yellow", no_wrap=True)

    succ_count = metrics.get("successful_tasks", 0)
    fail_count = metrics.get("failed_tasks", 0)
    total_tasks = metrics.get("total_tasks", 0)
    fail_rate = (fail_count / total_tasks * 100) if total_tasks > 0 else 0.0

    table.add_row(
        "Task Count",
        str(succ_count),
        str(fail_count),
        f"{total_tasks} ({fail_rate:.1f}% fail rate)",
    )

    succ_spend = metrics.get("successful_spend_usd", 0.0)
    wasted_spend = metrics.get("wasted_spend_usd", 0.0)
    total_spend = metrics.get("total_spend_usd", 0.0)
    wasted_pct = metrics.get("wasted_spend_pct", 0.0)

    table.add_row(
        "Total Spend",
        _format_usd(succ_spend),
        _format_usd(wasted_spend),
        f"{_format_usd(total_spend)} ({wasted_pct:.1f}% wasted)",
        style="bold",
    )

    avg_succ = metrics.get("avg_successful_spend_usd", 0.0)
    avg_wasted = metrics.get("avg_wasted_spend_usd", 0.0)
    ratio_str = f"Wasted is {avg_wasted / avg_succ:.1f}x avg success" if avg_succ > 0 and avg_wasted > 0 else "—"

    table.add_row(
        "Avg Spend / Task",
        _format_usd(avg_succ),
        _format_usd(avg_wasted),
        ratio_str,
    )

    succ_tok = metrics.get("successful_tokens", 0)
    wasted_tok = metrics.get("wasted_tokens", 0)
    total_tok = metrics.get("total_tokens", 0)

    table.add_row(
        "Total Tokens",
        _format_tokens(succ_tok),
        _format_tokens(wasted_tok),
        _format_tokens(total_tok),
    )

    render_items = [table]

    # Add efficiency summary text
    eff_text = Text()
    if wasted_spend > 0:
        eff_text.append(" ⚠️  Capital Loss: ", style="bold red")
        eff_text.append(f"{_format_usd(wasted_spend)} ({wasted_pct:.1f}% of total spend) ", style="bold underline red")
        eff_text.append("was burned on uncompleted or failed tasks.", style="red")
        b_limit = metrics.get("budget_limit_usd")
        if b_limit:
            w_b_pct = metrics.get("wasted_of_budget_limit_pct", 0.0)
            u_pct = metrics.get("budget_utilization_pct", 0.0)
            eff_text.append(
                f"\n 🛡️  Budget Context: Total spend is {_format_usd(total_spend)} / {_format_usd(b_limit)} "
                f"({u_pct:.1f}% limit used; waste is {w_b_pct:.2f}% of budget limit).",
                style="dim",
            )
    else:
        eff_text.append(" ✨ 100% Capital Efficiency: No wasted spend recorded across all tasks.", style="bold green")
    render_items.append(eff_text)

    panel = Panel(
        Group(*render_items),
        title="[bold cyan]📊 FinOps Task Efficiency & Wasted Spend Analysis[/bold cyan]",
        subtitle="[dim]adk-finops • Successful vs. Wasted Agent Loops[/dim]",
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

    # Task Outcome
    task_status = sess_info.get("status") or summary.get("status")
    is_failure = sess_info.get("is_failure") or summary.get("is_failure", False)
    task_error = sess_info.get("error") or summary.get("error")
    if task_status:
        if is_failure:
            lines.append(f"│  ❌ TASK FAILED (WASTED SPEND): {_format_usd(sess_cost)} wasted{' ' * 20}│")
            if task_error:
                err_line = f"   Reason: {task_error}"[:72]
                lines.append(f"│  {err_line:<72}│")
        else:
            lines.append(f"│  ✅ TASK STATUS: SUCCESS ({_format_usd(sess_cost)} spent){' ' * 28}│")

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


def format_plain_task_efficiency(metrics: dict[str, Any]) -> str:
    """Pure-Python Unicode box fallback for Task Efficiency Analysis."""
    lines: list[str] = []
    width = 76
    lines.append("╭" + "─" * (width - 2) + "╮")
    lines.append(f"│ {'📊 FinOps Task Efficiency & Wasted Spend Analysis':^{width - 4}} │")
    lines.append("├" + "─" * (width - 2) + "┤")

    succ_count = metrics.get("successful_tasks", 0)
    fail_count = metrics.get("failed_tasks", 0)
    total_tasks = metrics.get("total_tasks", 0)
    fail_rate = (fail_count / total_tasks * 100) if total_tasks > 0 else 0.0

    lines.append(f"│  Tasks:        Successful: {succ_count:<4} | Failed: {fail_count:<4} | Total: {total_tasks} ({fail_rate:.1f}% fail){' ' * 6}│")

    succ_spend = metrics.get("successful_spend_usd", 0.0)
    wasted_spend = metrics.get("wasted_spend_usd", 0.0)
    total_spend = metrics.get("total_spend_usd", 0.0)
    wasted_pct = metrics.get("wasted_spend_pct", 0.0)

    lines.append(f"│  Total Spend:  Success: {_format_usd(succ_spend):<10} | Wasted: {_format_usd(wasted_spend):<10} ({wasted_pct:.1f}% wasted){' ' * 5}│")

    avg_succ = metrics.get("avg_successful_spend_usd", 0.0)
    avg_wasted = metrics.get("avg_wasted_spend_usd", 0.0)
    ratio_str = f"{avg_wasted / avg_succ:.1f}x" if avg_succ > 0 and avg_wasted > 0 else "—"

    lines.append(f"│  Avg / Task:   Success: {_format_usd(avg_succ):<10} | Wasted: {_format_usd(avg_wasted):<10} (Wasted is {ratio_str}){' ' * 4}│")

    succ_tok = metrics.get("successful_tokens", 0)
    wasted_tok = metrics.get("wasted_tokens", 0)
    lines.append(f"│  Tokens:       Success: {_format_tokens(succ_tok):<10} | Wasted: {_format_tokens(wasted_tok):<10}{' ' * 20}│")

    lines.append("├" + "─" * (width - 2) + "┤")
    if wasted_spend > 0:
        lines.append(f"│  ⚠️ Capital Loss: {_format_usd(wasted_spend)} ({wasted_pct:.1f}% of total spend) burned on failed tasks{' ' * 4}│")
        b_limit = metrics.get("budget_limit_usd")
        if b_limit:
            w_b_pct = metrics.get("wasted_of_budget_limit_pct", 0.0)
            u_pct = metrics.get("budget_utilization_pct", 0.0)
            b_line = f"   Budget Context: Waste is {w_b_pct:.2f}% of {_format_usd(b_limit)} limit ({u_pct:.1f}% used)"
            lines.append(f"│  {b_line:<72}│")
    else:
        lines.append(f"│  ✨ 100% Capital Efficiency: No wasted spend recorded.{' ' * 19}│")

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


def print_task_efficiency_summary(
    metrics: dict[str, Any] | None = None,
    console: Console | None = None,
) -> None:
    """Prints a comparative table of successful tasks vs. wasted/failed agent loops."""
    if metrics is None:
        from .tracker import CostTracker
        metrics = CostTracker.get_task_efficiency_metrics()

    if not metrics or metrics.get("total_tasks", 0) == 0:
        print("[FinOps] No task history recorded for efficiency analysis.")
        return

    if HAS_RICH:
        c = console or Console()
        panel = render_task_efficiency_table(metrics, console=c)
        c.print(panel)
    else:
        print(format_plain_task_efficiency(metrics))

