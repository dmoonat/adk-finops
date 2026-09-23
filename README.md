# adk-finops

[![PyPI](https://img.shields.io/pypi/v/adk-finops.svg)](https://pypi.org/project/adk-finops/)
[![Downloads](https://static.pepy.tech/badge/adk-finops)](https://pepy.tech/project/adk-finops)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-green.svg)](src/adk_finops/py.typed)

**Universal FinOps cost, token usage, and grounding fee tracking for the Google Agent Development Kit (ADK) and LLM workflows.**

![adk-finops Near-Live FinOps Web Dashboard](https://raw.githubusercontent.com/dmoonat/adk-finops/main/img/filter_dashboard.png)


---

## Table of Contents

- [Why adk-finops?](#why-adk-finops)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quickstart with Google ADK](#quickstart-with-google-adk)
- [Dual-Scope Telemetry: Turn vs. Session](#dual-scope-telemetry-turn-vs-session)
- [Budget Guards & Circuit Breakers](#budget-guards--circuit-breakers)
- [Task Outcome & Wasted Spend Analytics (Success vs. Failure)](#task-outcome--wasted-spend-analytics-success-vs-failure)
- [Context Caching Savings Analytics (ROI Tracker)](#context-caching-savings-analytics-roi-tracker)
- [Multi-Agent Cost Attribution & Delegation Tracking](#multi-agent-cost-attribution--delegation-tracking)
- [Automated FinOps Optimization Advisor](#automated-finops-optimization-advisor)
- [Rich Terminal Summary Box](#rich-terminal-summary-box)
  - [Automatic & On-Demand Integration](#automatic--on-demand-integration)
- [Decoupled Rate Cards (Custom & Enterprise Pricing)](#decoupled-rate-cards-custom--enterprise-pricing)
  - [1. Custom JSON Rate Card](#1-custom-json-rate-card)
  - [2. Environment Variable](#2-environment-variable)
  - [3. Remote URL / Enterprise Pricing Endpoint](#3-remote-url--enterprise-pricing-endpoint)
  - [4. Enterprise Negotiated Discounts](#4-enterprise-negotiated-discounts)
  - [5. Programmatic Model & Tool Registration](#5-programmatic-model--tool-registration)
- [Smart Tool Classification (MCP vs. Grounding)](#smart-tool-classification-mcp-vs-grounding)
- [Gemini 2.5 Thinking Tokens Billing](#gemini-25-thinking-tokens-billing)
- [Telemetry State Schema](#telemetry-state-schema)
- [Standalone Usage (Without ADK)](#standalone-usage-without-adk)
- [Configuration Reference](#configuration-reference)
- [Built-In Model Rate Cards](#built-in-model-rate-cards)
- [Flexible Exporters (Local, Cloud & OpenTelemetry)](#one-line-bigquery-exporter)
  - [Near-Live FinOps Web Dashboard (`adk-finops dashboard`)](#near-live-finops-web-dashboard-adk-finops-dashboard)
  - [Zero-Boilerplate Schema Management](#zero-boilerplate-schema-management)
  - [Standalone Exporter Usage](#standalone-exporter-usage)
  - [Sample SQL Queries for Looker Studio](#sample-sql-queries-for-looker-studio)
- [License](#license)

---

## Why adk-finops?

Building production AI agents with Google ADK involves multi-step tool-calling loops, reasoning models, and external search tools. However, monitoring real financial costs in production presents several challenges:

1. **Missing Cumulative Turn Tokens**: In a tool-calling loop (e.g., Call 1 decides to call a tool $\rightarrow$ Tool runs $\rightarrow$ Call 2 generates final response), default event telemetry only displays the token count of *Call 2*, hiding the cost of earlier reasoning calls.
2. **Thinking / Reasoning Tokens**: Models like Gemini 2.5 Pro separate `thoughtsTokenCount` from `candidatesTokenCount`. If thinking tokens are not explicitly extracted, your token usage will not match your Google Cloud bill.
3. **MCP vs. Paid Grounding**: Generic keyword matching often misclassifies free local/MCP tools (like `search_documents`) as paid Vertex AI Grounding queries (\$35/1k or \$2.50/1k).
4. **Hardcoded Pricing**: Model prices change frequently, and enterprises negotiate custom volume discounts. Hardcoded rates in application code require constant code updates.

`adk-finops` solves all of these problems with a single lightweight, decoupled plugin.

---

## Key Features

- **Native Google ADK Integration**: Intercepts model and tool invocations via the ADK `BasePlugin` lifecycle (`before_run`, `after_model`, `on_event`, `after_tool`, `after_run`).
- **Dual-Scope Accounting**: Simultaneously tracks metrics for both the **active turn** (all calls within a user message) and the **cumulative session** (entire conversation history).
- **Budget Guards & Circuit Breakers**: Set hard session and turn USD spending limits. Prevent runaway bills by halting execution, emitting warnings, or automatically downgrading expensive models (e.g. Gemini 2.5 Pro → Flash) when limits are breached.
- **Automated FinOps Optimization Advisor**: Zero-LLM, deterministic rule engine that analyzes completed session telemetry in `< 1ms` and calculates concrete `$` and `%` savings across **Context Caching Opportunities**, **Thinking Token Alerts** (`thinking_budget=0`), and **Model Right-Sizing** (`gemini-3.5-flash` ➔ `gemini-3.5-flash-lite`, `gemini-2.5-pro` ➔ `gemini-2.5-flash`, `gpt-4o` ➔ `gpt-4o-mini`). Easily toggled on/off via `enable_optimization_advisor=True/False`.
- **Task Outcome & Wasted Spend Analytics**: Distinguishes productive spend (`status="success"`) from wasted capital burned on failed retry loops or runtime exceptions (`status="failed"`, `"error"`). Computes average spend per successful task vs. average wasted spend per failed loop, capital loss percentage, and auto-exports crashed sessions to BigQuery.
- **Context Caching Savings ROI**: Demonstrates financial value by tracking gross cost (cost without caching) vs. actual net cost, reporting exact dollars and percentage saved (e.g. up to 90% savings via Gemini Context Caching).
- **Decoupled Rate Cards**: Pricing data is stored in clean JSON. Override rates via local file, remote URL, environment variable, or code without modifying the engine.
- **Enterprise Volume Discounts**: Configure global or provider-specific discount multipliers (e.g., 15% Google Cloud negotiated discount).
- **Accurate Thinking Tokens**: Automatically captures and bills `thoughts_token_count` at the output rate while displaying thinking tokens separately in reports.
- **Smart Tool Discrimination**: Automatically excludes `MCPTool`, `McpToolset`, BigQuery, and local function tools (\$0.00 fee) while accurately billing Google Search Grounding (\$0.014/query) and Vertex AI Search (\$0.0025/prompt).
- **Real-Time UI Streaming**: Streams cost metrics to `event.actions.state_delta["finops_cost"]` for live updates in the ADK Web UI, with formatted terminal stdout logging.
- **Zero Heavy Dependencies**: Pure Python standard library for the core tracker.

---

## Installation

### From PyPI
```bash
pip install adk-finops
```

### With Rich Terminal Summary
```bash
pip install "adk-finops[rich]"
```

### With BigQuery Exporter
```bash
pip install "adk-finops[bigquery]"
```

### With Google ADK
```bash
pip install "adk-finops[adk]"
```

### Full Enterprise Suite (ADK + Rich + BigQuery)
```bash
pip install "adk-finops[all]"
```

### From GitHub (Direct Git Dependency)
```bash
pip install git+https://github.com/dmoonat/adk-finops.git
```

Or in `requirements.txt`:
```text
adk-finops @ git+https://github.com/dmoonat/adk-finops.git@main
```

---

## Quickstart with Google ADK

Add `FinOpsCostPlugin` to your `App` in `agent.py`:

```python
from google.adk.agents import Agent
from google.adk.apps import App
from adk_finops import FinOpsCostPlugin

# 1. Define your agent
root_agent = Agent(
    model="gemini-2.5-pro",
    name="my_agent",
    instruction="You are a helpful assistant.",
    tools=[...]
)

# 2. Instantiate the FinOps plugin
finops_plugin = FinOpsCostPlugin(
    name="finops_cost_tracker",
    default_model="gemini-2.5-pro",
)

# 3. Attach plugin to your App
app = App(
    name="my_agent",
    root_agent=root_agent,
    plugins=[finops_plugin],
)
```

> **⚠️ Note:** `FinOpsCostPlugin` is a native `App`/`Runner`-level `BasePlugin`. Register it **only** via `App(plugins=[finops_plugin])` (or `Runner(plugins=[finops_plugin])`) — do not also pass `plugin.after_model_callback` to `Agent(after_model_callback=...)`, or ADK will invoke the callback twice.

Run your agent with `adk web` or `adk run`. Telemetry will log directly to the terminal and appear in the Web UI session state under `finops_cost`.

---

## Dual-Scope Telemetry: Turn vs. Session

`adk-finops` resolves the mismatch between single-event inspection and session-level totals by reporting both scopes concurrently:

| Scope | Description | Matches in ADK Web UI |
| :--- | :--- | :--- |
| **`turn`** | Cumulative metrics for the current user interaction (Call 1 + Tool Execution + Call 2) | Full cost of the current turn |
| **`session`** | Cumulative metrics across all turns in the chat session (Turn 1 + Turn 2 + ...) | ADK "Usage Summary for Session" |

### State Delta Example
```json
{
  "total_tokens": 18671,
  "prompt_tokens": 16617,
  "completion_tokens": 1110,
  "thoughts_tokens": 944,
  "total_cost_usd": 0.0135785,
  "currency": "USD",

  "turn": {
    "total_calls": 2,
    "total_tool_calls": 0,
    "prompt_tokens": 14222,
    "completion_tokens": 1079,
    "thoughts_tokens": 908,
    "total_tokens": 16209,
    "llm_cost_usd": 0.0092341,
    "tool_cost_usd": 0.0,
    "total_cost_usd": 0.0092341,
    "breakdown_by_model": {
      "gemini-2.5-pro": {
        "calls": 2,
        "prompt_tokens": 14222,
        "completion_tokens": 1079,
        "thoughts_tokens": 908,
        "total_cost_usd": 0.0092341
      }
    }
  },

  "session": {
    "total_calls": 3,
    "total_tool_calls": 0,
    "prompt_tokens": 16617,
    "completion_tokens": 1110,
    "thoughts_tokens": 944,
    "total_tokens": 18671,
    "llm_cost_usd": 0.0135785,
    "tool_cost_usd": 0.0,
    "total_cost_usd": 0.0135785
  }
}
```

---

## Budget Guards & Circuit Breakers

Prevent runaway agent loops and surprise bills with proactive budget enforcement. Set hard USD spending limits per session or per turn:

```python
from adk_finops import FinOpsCostPlugin

finops_plugin = FinOpsCostPlugin(
    default_model="gemini-2.5-pro",
    budget_limit_usd=1.00,             # Hard stop at $1.00 per chat session
    turn_budget_limit_usd=0.25,        # Maximum $0.25 on any single turn
    on_budget_exceeded="halt",         # Action: "halt", "warn", or "downgrade"
    fallback_model="gemini-2.5-flash", # Target model when using "downgrade"
)
```

### Enforcement Modes

| Mode | Behavior | Best Used For |
| :--- | :--- | :--- |
| **`"halt"`** *(default)* | Halts execution immediately, raises `BudgetExceededError` or returns a safe warning message in ADK to block further model calls. | Production safeguards, preventing runaway costs. |
| **`"downgrade"`** | Automatically downgrades the agent's model to `fallback_model` (e.g. Gemini 2.5 Pro → Flash) once the budget threshold is reached. | Graceful service degradation with zero user downtime. |
| **`"warn"`** | Logs a warning and marks `exceeded: True` in session state (`finops_cost.budget`) without interrupting the user. | Soft monitoring and alerting. |

---

## Task Outcome & Wasted Spend Analytics (Success vs. Failure)

In production AI systems, a failed agent loop (e.g., an agent repeatedly failing schema validation across 3 retries, or crashing due to a downstream API outage after generating a complex plan) often burns **5x–20x more tokens** than a successful task while delivering **zero business value**.

`adk-finops` tracks every LLM call in real time and classifies outcomes into **Productive Spend** vs. **Wasted Capital**, allowing teams to compare the average spend of a successful task against the wasted spend of failed loops.

### Understanding `status` vs. `is_failure`

| Field | Type | Values | Purpose |
| :--- | :--- | :--- | :--- |
| **`status`** | `str` | `"success"`, `"pending"`, `"failed"`, `"error"`, `"aborted"`, `"budget_exceeded"` | **Operational Root Cause**: Identifies *how* the task ended (e.g., validation loop exhausted `"failed"`, downstream tool crash `"error"`, or circuit breaker halt `"budget_exceeded"`). |
| **`is_failure`** | `bool` | `True` or `False` | **Financial Bucket**: Binary flag (`True` when status is `failed`, `error`, `aborted`, or `budget_exceeded`) used to separate **Wasted Spend (`True`)** from **Productive Spend (`False`)**. |

### 1. Automatic Workflow State Detection (Google ADK)

If any ADK workflow node or critic sets `ctx.state["failed"] = True` and `ctx.state["error_reason"] = "..."`, `FinOpsCostPlugin` automatically detects the failure in `after_run_callback`, tags the root cause, and marks the run's cost as wasted spend:

```python
def strict_validator(node_input, ctx):
    attempts = ctx.state.get("attempts", 0) + 1
    ctx.state["attempts"] = attempts
    if attempts >= 3:
        ctx.state["failed"] = True
        ctx.state["error_reason"] = "Exhausted 3 retry attempts without passing validation"
        return Event(output="Aborted", actions=EventActions(route="abort"))
```

### 2. Explicit Recording & Crash Recovery (Auto-Export to BigQuery)

When an unhandled exception crashes an agent run, ADK skips `after_run_callback`. Calling `finops_plugin.record_task_status()` inside your `except` block ensures the wasted tokens are logged and **automatically streamed to BigQuery**:

```python
from adk_finops import CostTracker, print_summary, print_task_efficiency_summary

try:
    async for event in runner.run_async(user_id="u1", session_id=session_id, new_message=msg):
        ...
except Exception as e:
    # Logs error outcome AND automatically flushes the crashed session to BigQuery
    finops_plugin.record_task_status(
        session_id=session_id,
        status="error",
        error=f"{type(e).__name__}: {str(e)}",
    )
    print_summary(session_id=session_id)

# Render comparative efficiency report across all tasks
print_task_efficiency_summary()
```

### Comparative Task Efficiency Report

```text
╭───────────── 📊 FinOps Task Efficiency & Wasted Spend Analysis ──────────────╮
│                                                                              │
│  Metric         Successful Tasks   Failed Loops (Wasted)      Total / Impact │
│  ──────────────────────────────────────────────────────────────────────────  │
│  Task Count                    2                       2 4 (50.0% fail rate) │
│  Total Spend             $0.0021                 $0.0080 $0.0101 (79.6% was) │
│  Avg Spend / Task        $0.0010                 $0.0040 Wasted is 3.9x avg  │
│  Total Tokens                864                   3,383               4,247 │
│                                                                              │
│  ⚠️  Capital Loss: $0.0080 (79.6% of total spend) was burned on uncompleted  │
│ or failed tasks.                                                             │
│  🛡️  Budget Context: Total spend is $0.0101 / $5.0000 (0.2% limit used;      │
│ waste is 0.16% of budget limit).                                             │
╰─────────────── adk-finops • Successful vs. Wasted Agent Loops ───────────────╯
```

---

## Context Caching Savings Analytics (ROI Tracker)

Context caching can reduce prompt token costs by up to 90%. `adk-finops` automatically calculates **Gross Cost** (what you would have paid without caching), **Actual Net Cost**, and **Total Savings**:

```json
{
  "total_cost_usd": 0.00334,
  "gross_cost_usd": 0.00550,
  "savings_usd": 0.00216,
  "savings_pct": 39.3,
  "cached_tokens": 8000
}
```

Real-time stdout log highlight:
```text
[FinOps LLM] turn=turn_1 model=gemini-2.5-flash tokens=11000 cost=$0.003340 | 💰 Saved $0.002160 (39.3%) via Context Caching
[FinOps Summary] Turn cost=$0.003340 | Session cost=$0.003340 | 💰 Total Saved: $0.002160 (39.3%) via Caching
```

---

## Multi-Agent Cost Attribution & Delegation Tracking

In hierarchical multi-agent architectures (e.g., a `supervisor` delegating subtasks to a `researcher` and a `coder`), each agent makes distinct model calls, executes different tools, and consumes different context windows. Without granular attribution, teams cannot identify which sub-agent is driving 80% of costs or entering a costly reasoning loop.

`adk-finops` automatically attributes every LLM invocation, thinking token, context caching savings, and tool fee to the specific agent executing the task, with optional per-agent budget limits:

```python
from adk_finops import FinOpsCostPlugin

finops_plugin = FinOpsCostPlugin(
    default_model="gemini-2.5-pro",
    budget_limit_usd=2.00,             # Total session cap: $2.00
    agent_budgets={
        "researcher": 0.50,            # Cap researcher at $0.50
        "coder": 1.00,                 # Cap coder at $1.00
    },
    on_budget_exceeded="halt",         # Halt if any agent breaches its limit
)
```

### Hierarchical Root (Parent) Agent ➔ Sub-Agent Attribution

`FinOpsCostPlugin` automatically inspects the ADK agent tree (`agent.root_agent`, `agent.parent_agent`, and `agent.sub_agents`) during execution and tags every agent entry with:
- `root_agent_name`: The top-level orchestrator / parent agent for the run (e.g., `"supervisor"`).
- `parent_agent_name`: The immediate parent agent (`null` for the root orchestrator, `"supervisor"` for delegated sub-agents).
- `agent_role`: `"root_self"` (the root orchestrator's own direct LLM/tool calls), `"sub_agent"` (a delegated specialist sub-agent), or `"root_rollup"` (the overall parent-level session rollup combining the root agent + all its sub-agents).

Every turn and session summary includes this hierarchy inside `breakdown_by_agent`:

```json
{
  "root_agent_name": "supervisor",
  "breakdown_by_agent": {
    "supervisor": {
      "calls": 1,
      "prompt_tokens": 1000,
      "completion_tokens": 200,
      "thoughts_tokens": 0,
      "cached_tokens": 0,
      "total_tokens": 1200,
      "llm_cost_usd": 0.00225,
      "tool_cost_usd": 0.0,
      "total_cost_usd": 0.00225,
      "savings_usd": 0.0,
      "root_agent_name": "supervisor",
      "parent_agent_name": null,
      "agent_role": "root_self"
    },
    "researcher": {
      "calls": 2,
      "prompt_tokens": 4000,
      "completion_tokens": 500,
      "thoughts_tokens": 0,
      "cached_tokens": 2000,
      "total_tokens": 4500,
      "llm_cost_usd": 0.00189,
      "tool_cost_usd": 0.028,
      "total_cost_usd": 0.02989,
      "savings_usd": 0.00054,
      "savings_pct": 22.2,
      "root_agent_name": "supervisor",
      "parent_agent_name": "supervisor",
      "agent_role": "sub_agent"
    },
    "coder": {
      "calls": 1,
      "prompt_tokens": 8000,
      "completion_tokens": 1500,
      "thoughts_tokens": 300,
      "cached_tokens": 0,
      "total_tokens": 9800,
      "llm_cost_usd": 0.01275,
      "tool_cost_usd": 0.0,
      "total_cost_usd": 0.01275,
      "savings_usd": 0.0,
      "root_agent_name": "supervisor",
      "parent_agent_name": "supervisor",
      "agent_role": "sub_agent"
    }
  }
}
```

### Real-Time Terminal Logs

```text
[FinOps LLM] turn=turn_1 session=sess_1 agent=supervisor model=gemini-2.5-pro tokens=1200 cost=$0.002250
[FinOps Grounding] turn=turn_1 session=sess_1 agent=researcher tool=google_search fee=$0.014000
[FinOps LLM] turn=turn_1 session=sess_1 agent=researcher model=gemini-2.5-flash tokens=4500 cost=$0.001890 | 💰 Saved $0.000540 (22.2%) via Context Caching
[FinOps Agents] supervisor: $0.0023 (1200 tok) | researcher: $0.0299 (4500 tok) | coder: $0.0128 (9800 tok)
```

---

## Automated FinOps Optimization Advisor

Beyond raw telemetry, `adk-finops` includes an **Automated FinOps Optimization Advisor** (`src/adk_finops/advisor.py`) that analyzes completed sessions in `< 1ms` with **zero LLM calls** and **\$0.00 overhead**, computing concrete dollar and percentage savings directly from your active [`RateCardRegistry`](src/adk_finops/rate_card.py):

1. **⚡ Context Caching Opportunity**:
   - Detects agents sending $\ge 2,000$ uncached prompt tokens across $\ge 2$ turns and calculates exact savings from enabling Context Caching (`cached_input_per_1m` vs. `input_per_1m`).
2. **🧠 Thinking Token Alert**:
   - Detects agents where `thoughts_tokens >= 500` account for $\ge 60\%$ of total output token cost, recommending `thinking_budget=0` (or a lower `ThinkingConfig` cap) for routing/classification steps.
3. **🎯 Model Right-Sizing**:
   - Detects agents using flagship models (`gemini-3.5-pro` $\rightarrow$ `gemini-3.5-flash` $\rightarrow$ `gemini-3.5-flash-lite`, `gemini-2.5-pro` $\rightarrow$ `gemini-2.5-flash`, `gpt-4o` $\rightarrow$ `gpt-4o-mini`, `claude-3-5-sonnet` $\rightarrow$ `claude-3-5-haiku`) for short responses (`< 200` average output tokens) with `0` tool calls, calculating exact savings from switching to the lighter tier.

### Enabling or Disabling the Optimization Advisor

The advisor is **enabled by default** and can be toggled on or off in `FinOpsCostPlugin`:

```python
from adk_finops import FinOpsCostPlugin

# Enabled by default (renders in Terminal Box & attaches to get_summary()['optimization_insights'])
finops_plugin = FinOpsCostPlugin(
    enable_optimization_advisor=True,
)

# Disable the advisor if you only want raw telemetry
finops_plugin = FinOpsCostPlugin(
    enable_optimization_advisor=False,
)
```

Or toggle globally via environment variable:
```bash
export ADK_FINOPS_OPTIMIZATION_ADVISOR="false"
```

> [!IMPORTANT]
> **Disclaimer — Directional Hints Only**:
> Optimization insights are generated using **deterministic heuristics and token-level rules** (with zero LLM evaluation of prompt semantics). They should be treated as **directional hints** rather than fully dependable or prescriptive actions. Before changing models, enabling caching, or lowering `thinking_budget` in production, always perform a **deep analysis of your specific use-case, offline/online evaluation datasets (`eval` data), accuracy benchmarks, latency SLAs, and business requirements**.

---

## Rich Terminal Summary Box

`adk-finops` includes an out-of-the-box, color-coded, border-styled terminal summary box. When running in a terminal, it provides instant financial visibility after every turn, displaying turn vs. session costs, context caching ROI, model breakdowns, and sub-agent attributions:

```text
╭───────────────────────── 💸 ADK FinOps Cost Summary ─────────────────────────╮
│                                                                              │
│  Scope           Calls   Tokens   LLM Cost   Tool Fees   Total Cost          │
│  ──────────────────────────────────────────────────────────────────          │
│  Current Turn        2   10,100    $0.0053     $0.0280      $0.0333          │
│  Session Total       2   10,100    $0.0053     $0.0280      $0.0333          │
│                                                                              │
│  💰 Context Caching Savings: $0.0016 saved (4.6% reduction from $0.0349 gross)│
│                                                                              │
│  Model              Calls   Tokens (In/Out)   Cost (USD)           Savings   │
│  ─────────────────────────────────────────────────────────────────────────   │
│  gemini-2.5-pro         1       1,200 / 300      $0.0030                 —   │
│  gemini-2.5-flash       1       8,000 / 600      $0.0023   $0.0016 (41.5%)   │
│                                                                              │
│  Agent             Calls   Tokens   LLM Cost   Tool Fees   Total Cost        │
│  ────────────────────────────────────────────────────────────────────        │
│  🤖 researcher         4   14,400    $0.0078       $0.00      $0.0078        │
│  🤖 router_agent       1    5,900    $0.0130       $0.00      $0.0130        │
│  🤖 formatter          1    1,610    $0.0024       $0.00      $0.0024        │
│                                                                              │
│  🛡️  Budget Guard: $0.0233 / $1.0000 (2.3% utilized)                         │
│                                                                              │
│  💡 Optimization Insights (Est. Savings: $0.0148 | 63.6% cut available)      │
│   ⚡ Context Caching Opportunity: Agent 'researcher' sent >12k uncached      │
│ prompt tokens across 4 turns. Enabling Context Caching would save $0.0026    │
│ (90%).                                                                       │
│   🧠 Thinking Token Alert: Thinking tokens (4,200) were 82% of               │
│ 'router_agent' output cost ($0.0105); consider setting thinking_budget=0.    │
│   🎯 Model Right-Sizing: Agent 'formatter' used gemini-2.5-pro for <150      │
│ output tokens with 0 tool calls; switching to gemini-2.5-flash saves 70%     │
│ ($0.0017).                                                                   │
│   ℹ️  Disclaimer: Insights are deterministic hints; validate against your    │
│ use-case, eval data & business requirements.                                 │
╰───────────────── adk-finops • Universal Token & Cost Engine ─────────────────╯
```

---

### Terminal Box Features

- **Rich 24-Bit Color Styling**: When [`rich`](https://github.com/Textualize/rich) is installed (`pip install "adk-finops[rich]"`), it renders full color highlights, styled headers, and rounded boxes.
- **Pure-Python Unicode Fallback**: If `rich` is not installed, it falls back seamlessly to an aligned pure-Python Unicode box drawing (`╭─╮│╰─╯`) with zero external dependencies.
- **Responsive 80-Column Layout**: Designed to fit standard terminal windows without clipping or wrapping.

### Automatic & On-Demand Integration

`adk-finops` supports both hands-off automated terminal reporting in Google ADK and explicit on-demand rendering for any Python workflow:

#### 1. Automatic Integration (Google ADK)

When using `FinOpsCostPlugin`, the summary box is rendered automatically to stdout at the conclusion of every turn (`after_run_callback`):

```python
from google.adk.agents import Agent
from google.adk.apps import App
from adk_finops import FinOpsCostPlugin

# Automatically renders the Rich box and Optimization Advisor at the end of each turn
finops_plugin = FinOpsCostPlugin(
    default_model="gemini-2.5-pro",
    render_terminal_box=True,          # Enabled by default
    enable_optimization_advisor=True,  # Enabled by default
)

app = App(
    name="support_agent",
    root_agent=Agent(name="support_agent", model="gemini-2.5-pro"),
    plugins=[finops_plugin],
)
```

> [!TIP]
> To disable the box and retain only standard single-line log output (e.g. in headless CI/CD environments), pass `render_terminal_box=False`.

#### 2. On-Demand Integration (Standalone & Custom Workflows)

You can trigger terminal rendering programmatically at any time across your batch scripts, agent loops, or background tasks:

```python
from adk_finops import CostTracker, format_summary_box, print_summary

# 1. Print current turn/session summary box directly to terminal stdout
print_summary()

# Or specify an explicit turn and session ID:
print_summary(run_id="turn_123", session_id="session_abc")

# 2. Or call directly from CostTracker
CostTracker.print_summary()

# 3. Export formatted box string (ANSI or plain text) for custom loggers or webhooks (Slack/Discord)
summary_data = CostTracker.get_summary()
box_text = format_summary_box(summary_data)
logger.info("\n" + box_text)
```

---

## Decoupled Rate Cards (Custom & Enterprise Pricing)

Pricing is completely decoupled from the tracking engine. You can configure rates via:

### 1. Custom JSON Rate Card
Create a `my_rates.json` file:

```json
{
  "models": {
    "custom-fine-tuned-model": {
      "provider": "google",
      "input_per_1m": 0.50,
      "output_per_1m": 2.00,
      "cached_input_per_1m": 0.05
    }
  },
  "tools": {
    "paid_external_api": 0.01
  }
}
```

Pass the file path when initializing the plugin:
```python
finops_plugin = FinOpsCostPlugin(rate_card_path="my_rates.json")
```

### 2. Environment Variable
Set the rate card path globally without touching any code:
```bash
export ADK_FINOPS_RATE_CARD_PATH="/etc/finops/rates.json"
```

### 3. Remote URL / Enterprise Pricing Endpoint
Fetch central rate cards from an internal endpoint or cloud storage bucket (GCS/S3):
```python
from adk_finops import CostTracker

CostTracker.load_rate_card_url("https://internal.corp.com/finops/rates.json")
```

### 4. Enterprise Negotiated Discounts
Apply your organization's contracted cloud discounts:
```python
# Apply a 15% discount across all models and tools
finops_plugin = FinOpsCostPlugin(discount_percent=15.0)

# Or configure dynamically via CostTracker:
CostTracker.set_discount(15.0)                      # Global 15% discount
CostTracker.set_discount(20.0, provider="google")   # 20% discount on Google Cloud models
```

Or set via environment variable:
```bash
export ADK_FINOPS_DISCOUNT_PERCENT="15.0"
```

### 5. Programmatic Model & Tool Registration
If a model is not present in the bundled [`default_rates.json`](src/adk_finops/rates/default_rates.json), `adk-finops` applies the default fallback rate (`$0.30 / $2.50 / $0.03` per 1M tokens), sets `"is_fallback_rate": True` on the recorded usage/summary, and logs a one-time warning prompting you to register the model's exact pricing via `CostTracker.register_rate_card`:

```python
from adk_finops import CostTracker

CostTracker.register_rate_card("claude-opus-5", {
    "provider": "anthropic",
    "input_per_1m": 5.00,
    "output_per_1m": 25.00,
    "cached_input_per_1m": 0.50,
})

CostTracker.register_tool_rate("internal_vector_db", 0.0005)
```

### 6. Regional (`non_global`) & Date-Tiered (`2027`) Pricing
`adk-finops` automatically detects your Google Cloud region and applies `non_global` regional rates (e.g., `us-central1`, `europe-west1`) as well as promotional-to-standard pricing transitions (`standard_pricing_2027` starting Jan 1, 2027):

- **Region Resolution Precedence**:
  1. Explicit `region` passed to `FinOpsCostPlugin(region="us-central1")` or `CostTracker.set_region("us-central1")`
  2. `GOOGLE_CLOUD_LOCATION` environment variable (standard Google ADK `.env` configuration)
  3. `ADK_FINOPS_REGION` environment variable
  4. Defaults to `"global"`

```python
from adk_finops import CostTracker, FinOpsCostPlugin

# Automatically uses GOOGLE_CLOUD_LOCATION from .env if set, or specify explicitly:
finops_plugin = FinOpsCostPlugin(
    region="us-central1",          # Applies non_global rates (+10% regional pricing where applicable)
    effective_date="2027-01-01",   # Optional: simulate or enforce 2027 standard pricing
)
```

### 7. Dynamic Remote Rate Card Syncing & Google Cloud Pricing Extraction Framework
To ensure your agents always calculate costs against the newest model pricing without waiting for a package upgrade, `adk-finops` provides **opt-in Dynamic Remote Rate Card Syncing** (with a 24-hour local disk cache at `~/.cache/adk-finops/remote_rates.json` and graceful offline fallback) as well as a **Google Cloud & Multi-Provider Pricing Extraction Framework** ([`src/adk_finops/pricing_extractor.py`](src/adk_finops/pricing_extractor.py)).

#### Canonical Pricing Sources (`source_urls` & `sources` in `default_rates.json`)
Every rate card generated by `adk-finops` records all upstream sources queried in its top-level `"source_urls"` and `"sources"` metadata:

| Source Name | Canonical URL | Scope & Coverage |
| :--- | :--- | :--- |
| **`gemini_enterprise_pricing`** | `https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing` | Google Cloud Gemini Enterprise Agent Platform (`global`, `non_global` regional +10%, `standard_pricing_2027`, `>200K` context tiers, and Grounding Tools) |
| **`litellm_registry`** | `https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json` | Multi-Provider (`openai`, `anthropic`, `deepseek`, and cross-validation) |
| **`gcp_cloud_billing_catalog_api`** *(optional)* | `https://cloudbilling.googleapis.com/v1/services/C7E2-9256-1C43/skus` | Official Google Cloud Billing Catalog REST API (when `GOOGLE_CLOUD_BILLING_API_KEY` or `--gcp-billing-api-key` is supplied) |

> **Note:** `pricing_extractor.py` seeds Google Cloud model definitions directly from [`default_rates.json`](src/adk_finops/rates/default_rates.json) via `_load_google_baseline_rates()` so curated `non_global`, `standard_pricing_2027`, and `_gt_200k` fields are never overwritten or duplicated.

#### Runtime Remote Rate Card Syncing (`FinOpsCostPlugin` & `CostTracker`)
```python
from adk_finops import CostTracker, FinOpsCostPlugin

# Option A: Enable via FinOpsCostPlugin (or set ADK_FINOPS_SYNC_REMOTE_RATES=1 in .env)
finops_plugin = FinOpsCostPlugin(
    sync_remote_rates=True,             # Syncs latest default_rates.json (cached locally for 24h)
    remote_cache_ttl_seconds=86400,     # Optional custom cache TTL
)

# Option B: Programmatic sync on demand
status = CostTracker.sync_remote_rate_card(force=True)
```

#### Running the Pricing Extraction Framework (4 Ways)

1. **Via the `adk-finops` CLI (`extract-pricing` & `sync-rates`)**:
   ```bash
   # 1. Sync local rate card cache (~/.cache/adk-finops/remote_rates.json) from canonical remote repo
   adk-finops sync-rates --force

   # 2. Dry-run pricing extraction report (prints Rich table comparing live sources vs default_rates.json)
   adk-finops extract-pricing

   # 3. Export merged rate card (including newly discovered models like gpt-4.1, o3, claude-4) to a custom file
   adk-finops extract-pricing --include-new-models --output ./custom_rates.json

   # 4. Update bundled src/adk_finops/rates/default_rates.json in-place
   adk-finops extract-pricing --update-default --include-new-models
   ```

2. **Via the Python Module (`python -m adk_finops.pricing_extractor`)**:
   ```bash
   python -m adk_finops.pricing_extractor --include-new-models --output ./custom_rates.json
   ```

3. **Programmatically in Python (`extract_and_sync_pricing`)**:
   ```python
   from adk_finops import extract_and_sync_pricing

   report = extract_and_sync_pricing(
       update_default_rates=False,
       output_path="./custom_rates.json",
       include_new_models=True,
       quiet=False,
   )
   print("Sources synced:", report.sources_succeeded)
   print("Added models:", list(report.added_models.keys()))
   ```

4. **Automated Weekly GitHub Actions Workflow ([`.github/workflows/sync_pricing.yml`](.github/workflows/sync_pricing.yml))**:
   Runs every Monday at `06:00 UTC` (or manually via `workflow_dispatch` with `include_new_models` enabled by default) to extract live rates and commit any changes to [`src/adk_finops/rates/default_rates.json`](src/adk_finops/rates/default_rates.json).


---

## Smart Tool Classification (MCP vs. Grounding)

`adk-finops` distinguishes between paid cloud grounding services and free tools:

- **MCP Tools (`McpToolset`, `MCPTool`)**: Automatically identified and assigned **\$0.00** fee (e.g. `search_documents`).
- **Database & Custom Tools**: BigQuery toolsets and custom Python functions incur **\$0.00** tool fee.
- **Google Search Grounding & Web Grounding**: Tools named `google_search` or `GoogleSearchTool` incur **\$0.014 / query** (\$14.00 per 1,000 queries; first 5,000 queries/month free). Charged for each individual Grounding Query performed by Gemini. Input tokens returned by search grounding are not charged.
- **Google Maps Grounding**: Incurs **\$0.014 / query** (\$14.00 per 1,000 queries; first 5,000 queries/month free). Input tokens are not charged.
- **Grounding with your data (Vertex AI Search / Datastores)**: Tools matching `vertex_search`, `vertex_ai_search`, or `GroundingTool` incur **\$0.0025 / prompt** (\$2.50 per 1,000 prompts).

---

## Gemini 2.5 & 3.x Thinking Tokens Billing

Gemini 2.5 and Gemini 3.x models separate reasoning/thinking tokens into `thoughtsTokenCount`. 

Per Google Cloud pricing rules:
> **Total Billable Output Tokens** = `candidates_token_count` + `thoughts_token_count`

`adk-finops` automatically incorporates thinking tokens into the output token rate while reporting them as a separate field in `breakdown_by_model` so you can monitor your reasoning overhead.

---

## Standalone Usage (Without ADK)

`adk-finops` is designed to be used in **any Python application** — FastAPI/Flask backends, Celery/Ray background pipelines, LangChain/LlamaIndex workflows, or raw Google GenAI SDK scripts — without requiring Google ADK.


```python
from adk_finops import CostTracker, BigQueryExporter

# Wrap any execution block with automatic lifecycle cleanup
with CostTracker.track_run("request_123"):
    # Record Gemini call with thoughts/reasoning tokens
    CostTracker.record_usage(
        run_id="request_123",
        model_name="gemini-3.7-flash",
        prompt_tokens=1500,
        completion_tokens=250,
        thoughts_tokens=100,
        cached_tokens=0,
    )
    # Record grounding or tool fee ($0.014 / query)
    CostTracker.record_tool_call(
        run_id="request_123",
        tool_name="google_search",
    )

# Print color-coded terminal summary box
CostTracker.print_summary("request_123")

# Get structured metrics dictionary
summary = CostTracker.get_summary("request_123")
print(f"Total Cost: ${summary['total_cost_usd']:.6f}")
print(f"Total Tokens: {summary['total_tokens']}")

# (Optional) Export to BigQuery in 1 line
# exporter = BigQueryExporter("my-project.finops.agent_costs")
# exporter.export_summary(summary)
```

---

## Configuration Reference

### Environment Variables

| Variable | Type | Description |
| :--- | :--- | :--- |
| `GOOGLE_CLOUD_LOCATION` | `str` | Primary ADK environment variable for region detection (e.g., `"global"`, `"us-central1"`). Non-global regions automatically apply `non_global` rates. |
| `ADK_FINOPS_REGION` | `str` | Fallback pricing region if `GOOGLE_CLOUD_LOCATION` is not set (defaults to `"global"`). |
| `ADK_FINOPS_EFFECTIVE_DATE` | `str` | ISO date (`YYYY-MM-DD`) to evaluate date-tiered pricing such as `standard_pricing_2027` (defaults to today's date). |
| `ADK_FINOPS_RATE_CARD_PATH` | `str` | Absolute or relative path to a custom JSON rate card file. |
| `ADK_FINOPS_DISCOUNT_PERCENT` | `float` | Global enterprise discount percentage (e.g., `15.0` for 15%). |
| `ADK_FINOPS_OPTIMIZATION_ADVISOR` | `bool` | Toggle the Automated FinOps Optimization Advisor (`"true"` or `"false"`). |

### Plugin Initialization Parameters

```python
FinOpsCostPlugin(
    name: str = "finops_cost_tracker",
    default_model: str = "gemini-2.5-flash",
    rate_card_path: str | Path | None = None,
    rate_card: dict[str, Any] | None = None,
    discount_percent: float | None = None,
    region: str | None = None,
    effective_date: str | date | None = None,
    budget_limit_usd: float | None = None,
    turn_budget_limit_usd: float | None = None,
    agent_budgets: dict[str, float] | None = None,
    on_budget_exceeded: str = "halt",  # "halt", "warn", or "downgrade"
    fallback_model: str = "gemini-2.5-flash",
    render_terminal_box: bool = True,
    enable_optimization_advisor: bool = True,
)
```

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `default_model` | `str` | `"gemini-2.5-flash"` | Fallback model name if not reported by the LLM response. |
| `rate_card_path` | `str \| Path` | `None` | Path to custom rate card JSON file. |
| `discount_percent` | `float` | `None` | Enterprise volume discount percentage (e.g. `15.0` for 15%). |
| `region` | `str \| None` | `None` | Pricing region override. When `None`, auto-detects from `GOOGLE_CLOUD_LOCATION` $\rightarrow$ `ADK_FINOPS_REGION` $\rightarrow$ `"global"`. |
| `effective_date` | `str \| date` | `None` | Optional ISO date (`"YYYY-MM-DD"`) for date-tiered pricing (e.g. `"2027-01-01"` for 2027 standard rates). |
| `budget_limit_usd` | `float` | `None` | Maximum cumulative spending limit in USD for the entire chat session. |
| `turn_budget_limit_usd` | `float` | `None` | Maximum spending limit in USD for any single user turn. |
| `agent_budgets` | `dict[str, float]` | `None` | Per-agent spending caps in USD (e.g. `{"researcher": 0.50, "coder": 1.00}`). |
| `on_budget_exceeded` | `str` | `"halt"` | Action on budget breach: `"halt"` (raise/block), `"warn"`, or `"downgrade"`. |
| `fallback_model` | `str` | `"gemini-2.5-flash"` | Target model when using `"downgrade"` mode. |
| `render_terminal_box` | `bool` | `True` | Renders a beautiful color-coded summary box to stdout at the end of each turn. |
| `enable_optimization_advisor` | `bool` | `True` | Enables deterministic FinOps Optimization Insights in the summary box and `get_summary()`. |

---

## Built-In Model Rate Cards

The bundled [`default_rates.json`](src/adk_finops/rates/default_rates.json) contains official public pricing (USD per 1M tokens, sourced from [Google Cloud Generative AI Pricing](https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing)):

| Model | Provider | Input / 1M (`<=200k`) | Output / 1M (`<=200k`) | Cached / 1M (`<=200k`) | Context `>200k` (`_gt_200k`) / Regional (`non_global`) / 2027 Notes |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `gemini-3.1-pro-preview` | Google | \$2.00 | \$12.00 | \$0.20 | **`>200k`**: In \$4.00, Out \$18.00, Cached \$0.40 |
| `gemini-3.8-flash-cyber` | Google | \$1.50 | \$7.50 | \$0.15 | **Non-global**: In \$1.65, Out \$8.25, Cached \$0.165 |
| `gemini-3.8-flash` | Google | \$0.75 | \$3.75 | \$0.075 | **Non-global**: \$0.825 / \$4.125 • **2027 Standard**: \$1.50 / \$7.50 (Non-global: \$1.65 / \$8.25) |
| `gemini-3.7-flash` | Google | \$0.75 | \$3.75 | \$0.075 | **Non-global**: \$0.825 / \$4.125 • **2027 Standard**: \$1.50 / \$7.50 (Non-global: \$1.65 / \$8.25) |
| `gemini-3.6-flash` | Google | \$0.75 | \$3.75 | \$0.075 | **Non-global**: \$0.825 / \$4.125 • **2027 Standard**: \$1.50 / \$7.50 (Non-global: \$1.65 / \$8.25) |
| `gemini-3.5-flash` | Google | \$1.50 | \$9.00 | \$0.15 | **Non-global**: In \$1.65, Out \$9.90, Cached \$0.165 |
| `gemini-3.5-flash-lite` | Google | \$0.30 | \$2.50 | \$0.03 | **Non-global**: In \$0.33, Out \$2.75, Cached \$0.033 |
| `gemini-3.1-flash-lite` | Google | \$0.25 | \$1.50 | \$0.025 | **Non-global**: In \$0.275, Out \$1.65, Cached \$0.0275 |
| `gemini-3-flash-preview` | Google | \$0.50 | \$3.00 | \$0.05 | — |
| `gemini-3-pro-image` | Google | \$2.00 | \$12.00 | \$0.20 | **`>200k`**: In \$4.00, Out \$18.00, Cached \$0.40 |
| `gemini-3.1-flash-image` | Google | \$0.50 | \$3.00 | \$0.05 | Nano Banana 2 image generation |
| `gemini-3.1-flash-lite-image` | Google | \$0.25 | \$1.50 | \$0.025 | Nano Banana Lite image generation |
| `gemini-2.5-pro` | Google | \$1.25 | \$10.00 | \$0.125 | **`>200k`**: In \$2.50, Out \$15.00, Cached \$0.25 |
| `gemini-2.5-pro-computer-use-preview` | Google | \$1.25 | \$10.00 | \$0.125 | **`>200k`**: In \$2.50, Out \$15.00, Cached \$0.25 |
| `gemini-2.5-flash` | Google | \$0.30 | \$2.50 | \$0.03 | **`>200k`**: In \$0.30, Out \$2.50, Cached \$0.03 |
| `gemini-2.5-flash-lite` | Google | \$0.10 | \$0.40 | \$0.01 | **`>200k`**: In \$0.10, Out \$0.40, Cached \$0.01 |
| `gemini-2.5-flash-image` | Google | \$0.30 | \$2.50 | \$0.03 | — |
| `gemini-2.5-flash-live` | Google | \$0.50 | \$2.00 | \$0.05 | Live API text rates |
| `gemini-2.0-flash` | Google | \$0.15 | \$0.60 | \$0.0375 | — |
| `codemender` | Google | \$0.75 | \$3.75 | \$0.075 | **2027 Standard**: In \$1.50, Out \$7.50, Cached \$0.15 |
| `gpt-4o` | OpenAI | \$2.50 | \$10.00 | \$1.25 | — |
| `gpt-4o-mini` | OpenAI | \$0.15 | \$0.60 | \$0.075 | — |
| `o1` | OpenAI | \$15.00 | \$60.00 | \$7.50 | — |
| `o3-mini` | OpenAI | \$1.10 | \$4.40 | \$0.55 | — |
| `claude-3-7-sonnet` | Anthropic | \$3.00 | \$15.00 | \$0.30 | — |
| `claude-3-5-sonnet` | Anthropic | \$3.00 | \$15.00 | \$0.30 | — |
| `claude-3-5-haiku` | Anthropic | \$0.80 | \$4.00 | \$0.08 | — |
| `deepseek-v3` | DeepSeek | \$0.14 | \$0.28 | \$0.014 | — |
| `deepseek-r1` | DeepSeek | \$0.55 | \$2.19 | \$0.14 | — |

---

## One-Line BigQuery Exporter

Stream turn, session, model, and sub-agent FinOps telemetry directly into a Google BigQuery dataset with a single configuration parameter:

```python
from adk_finops import FinOpsCostPlugin

finops_plugin = FinOpsCostPlugin(
    default_model="gemini-2.5-flash",
    bigquery_table="my-gcp-project.finops.agent_costs",
    bigquery_export_scope="both",  # "session" (default), "turn", or "both"
    bigquery_tags={"env": "production", "service": "support-agent"},
)
```

> [!TIP]
> **Environment Variable Auto-Discovery**: You can also set `ADK_FINOPS_BIGQUERY_TABLE="my-gcp-project.finops.agent_costs"` in your environment or `.env` file to enable BigQuery streaming without changing a single line of application code!

---

### Zero-Boilerplate Schema Management

When `bigquery_table` is specified, `adk-finops` automatically inspects and provisions the dataset and table with enterprise best practices:

- **Partitioning**: Day-partitioned on `timestamp` to optimize query performance and reduce scan costs.
- **Clustering**: Clustered by `[session_id, agent_name, model_name]` for sub-second filtering in Looker Studio and BI tools.
- **Non-Blocking Background Streaming**: Ingestion runs asynchronously in a background thread pool, adding **zero latency** to agent responses.

#### Table Schema Reference

| Field Name | Type | Description |
| :--- | :--- | :--- |
| `timestamp` | `TIMESTAMP` | Event timestamp in UTC (Partition Key) |
| `session_id` | `STRING` | ADK Session ID (Clustering Key #1) |
| `turn_id` | `STRING` | ADK Turn / Invocation ID |
| `scope` | `STRING` | Record scope: `"turn"` or `"session"` |
| `agent_name` | `STRING` | Attributed Agent name (Clustering Key #2) |
| `model_name` | `STRING` | Model name / version (Clustering Key #3) |
| `prompt_tokens` | `INTEGER` | Input prompt token count |
| `completion_tokens` | `INTEGER` | Output candidate token count |
| `thoughts_tokens` | `INTEGER` | Gemini 2.5 thinking token count |
| `cached_tokens` | `INTEGER` | Context cached token count |
| `total_tokens` | `INTEGER` | Total billable tokens |
| `llm_cost_usd` | `FLOAT` | Net LLM API cost in USD |
| `tool_cost_usd` | `FLOAT` | Search & Grounding fees in USD |
| `total_cost_usd` | `FLOAT` | Total net cost in USD |
| `gross_cost_usd` | `FLOAT` | Gross cost before caching discount |
| `savings_usd` | `FLOAT` | Dollars saved via context caching |
| `savings_pct` | `FLOAT` | Percentage saved via context caching |
| `tool_calls_count` | `INTEGER` | Number of billable grounding/tool calls |
| `budget_limit_usd` | `FLOAT` | Configured budget threshold |
| `budget_utilization_pct` | `FLOAT` | Budget utilization percentage |
| `budget_exceeded` | `BOOLEAN` | Whether budget guard was tripped |
| `breakdown_by_agent` | `JSON` | Multi-agent attribution snapshot |
| `breakdown_by_model` | `JSON` | Model distribution snapshot |
| `tags` | `JSON` | User-provided tags (e.g. `env`, `tenant_id`) |

---

### Understanding Streamed Rows & Dimensions (Rollup vs. Attributed Rows)

When streaming telemetry to BigQuery, `adk-finops` records both **overall aggregates** (for high-level session/turn reporting) and **attributed breakdown rows** (for granular drill-downs by agent or model).

#### Why are `agent_name` or `model_name` NULL in some rows?
In data warehousing and BI rollup patterns, `NULL` is assigned to dimension columns in summary/rollup rows to distinguish between overall totals and specific entity breakdowns:

- **Overall Aggregate Rows (`agent_name IS NULL`)**:
  - Emitted once per turn or session representing the **cumulative total** across all agents and models.
  - `agent_name` is `NULL` (can be displayed as `'OVERALL'` or `'TOTAL'` via `COALESCE(agent_name, 'OVERALL')`).
  - If multiple models were used in the run, `model_name` is `NULL` (the complete distribution is preserved in the `breakdown_by_model` JSON column). If only a single model was used, `model_name` is set to that model.
- **Attributed Agent Rows (`agent_name IS NOT NULL`)**:
  - Emitted for each sub-agent participating in the turn/session (`agent_name = 'research_agent'`, etc.).
  - `model_name` contains the primary model invoked by that agent (e.g. `gemini-2.5-pro`, `gemini-2.5-flash`).

#### Querying Best Practices (Avoiding Double-Counting)
Because both aggregate rows and attributed breakdown rows coexist in the same table, write your SQL queries according to the level of granularity you need:

- **To query overall totals (e.g. total spend per session):**
  ```sql
  SELECT session_id, total_cost_usd, total_tokens
  FROM `my-gcp-project.finops.agent_costs`
  WHERE scope = 'session' AND agent_name IS NULL;
  ```

- **To query per-agent breakdown (without double-counting with the aggregate):**
  ```sql
  SELECT agent_name, SUM(total_cost_usd) AS agent_spend
  FROM `my-gcp-project.finops.agent_costs`
  WHERE scope = 'session' AND agent_name IS NOT NULL
  GROUP BY 1;
  ```

- **To query by model:**
  ```sql
  SELECT model_name, SUM(total_cost_usd) AS model_spend
  FROM `my-gcp-project.finops.agent_costs`
  WHERE scope = 'session' AND model_name IS NOT NULL
  GROUP BY 1;
  ```

---

### Near-Live FinOps Web Dashboard (`adk-finops dashboard`)

![Near-Live FinOps Web Dashboard — Overview](https://raw.githubusercontent.com/dmoonat/adk-finops/main/img/all_dashboard.png)


`adk-finops` includes a built-in, zero-extra-dependency **FastAPI + Chart.js Near-Live Web Dashboard** that auto-refreshes every **2 seconds**, aggregating:
1. **Live In-Memory `CostTracker` State:** Watch tokens and spend accumulate in real time while an agent is mid-execution.
2. **Local Timestamped `.jsonl` & `.csv` Logs:** Automatically scans `logs/<YYYYMMDD_HHMMSS>/*.jsonl` and `*.csv`.
3. **Remote HTTP Push (`POST /api/ingest`):** Receives telemetry pushed over HTTP from remote agent containers (`HTTPExporter` / `dashboard_endpoint`) and persists it to `<log_dir>/ingested_costs.jsonl`.
4. **Optional BigQuery Live Sync:** Toggle the **Include BigQuery** switch in the UI (`15s` cache TTL) to merge cloud warehouse records (`ADK_FINOPS_BIGQUERY_TABLE`).

#### Hierarchical Root (Parent) Agent ➔ Sub-Agent Drilldown & Cascading Filters

![Near-Live FinOps Web Dashboard — Root Agent & Sub-Agent Filtered View](https://raw.githubusercontent.com/dmoonat/adk-finops/main/img/filter_dashboard.png)

- **5 Cascading Hierarchy Filters:**
  1. **`1. 👑 Root (Parent) Agent`:** Filter by a top-level orchestrator/parent agent (e.g., `👑 coordinator_agent`). Selecting a Root Agent automatically computes the **Overall Parent-Level Spend & Tokens (`∑ Parent + All Sub-Agents`)** in the KPI cards while displaying all of its sub-agents in the breakdown charts and hierarchy tree.
  2. **`2. ↳ Sub-Agent Drilldown`:** Dynamically cascades to list only the children belonging to the selected Root Agent (`∑ Overall Parent Total`, `👑 Root Orchestrator Direct Only`, or individual `↳ 🤖 Sub-Agents`).
  3. **`3. Session Filter (Scoped)`:** Automatically scopes the session dropdown so it **only lists sessions belonging to the selected Root Agent (and Sub-Agent)**.
  4. **`4. Model Filter`:** Scoped to the models invoked by the selected Root / Sub-Agent.
  5. **`5. Task Outcome`:** Filter between `✅ Effective Spend Only (Success)` and `🔥 Wasted Spend Only (Failed / Error / Budget)`.
- **`👑 Root (Parent) Agent ➔ Sub-Agents Hierarchy Rollup` Explorer:** Interactive parent-to-child cards showing each Root Agent's overall parent rollup (`∑ Overall Parent Spend` and `Overall Parent Tokens`) alongside each child's spend, token breakdown (`In / Out / Think`), model, and **percentage share of parent spend** with click-to-filter support.
- **5 Executive KPI Cards & 3 Interactive Charts:** Total Spend ($ with active scope badge), Effective Spend ($), Wasted Spend ($ & %), Context Caching Savings ($), Total Tokens (`In / Out / Think`), Spend Efficiency Doughnut, Sub-Agent Stacked Cost Bar (LLM vs Grounding), and Per-Model Token Composition.

---

#### Developer Mode: Embedded Background Server in `FinOpsCostPlugin`

Start the live dashboard automatically in a background daemon thread inside your agent process:

```python
from adk_finops import FinOpsCostPlugin

finops_plugin = FinOpsCostPlugin(
    enable_dashboard=True,                  # or set ADK_FINOPS_ENABLE_DASHBOARD=true
    dashboard_port=8088,                    # default: 8088 (auto-selects next free port if busy)
    jsonl_path="logs/finops_costs.jsonl",   # saves into logs/<YYYYMMDD_HHMMSS>/finops_costs.jsonl
)
```

---

#### Enterprise & Org Admin Mode: Standalone Centralized Dashboard (`3 Patterns`)

An organization admin can run `adk-finops dashboard` as a **single centralized FinOps control plane** (with zero agents running inside the dashboard process) to monitor dozens of independent root agents and sub-agent teams across an organization:

##### Pattern A: Central Cloud Warehouse Mode (BigQuery Only)
All agents across the org stream to a shared BigQuery table (`ADK_FINOPS_BIGQUERY_TABLE="org-project.finops.agent_costs"`). The admin runs the standalone dashboard pointing strictly to BigQuery:
```bash
adk-finops dashboard \
  --bigquery-table org-project.finops.agent_costs \
  --log-dir "" \
  --host 0.0.0.0 \
  --port 8088
```

##### Pattern B: Multi-Project Shared Directory / Volume Mode (`--log-dir`)
Pass comma-separated directories to watch timestamped `.jsonl` / `.csv` files across multiple agent repositories or shared volumes simultaneously:
```bash
adk-finops dashboard \
  --log-dir "/srv/agents/retail_bot/logs,/srv/agents/finance_bot/logs" \
  --port 8088
```

##### Pattern C: Direct HTTP Push Mode (`dashboard_endpoint` $\rightarrow$ `POST /api/ingest`)
When an organization does not use BigQuery and agents run in isolated containers/VMs without a shared disk:

1. **Admin starts the central dashboard server:**
   ```bash
   adk-finops dashboard --host 0.0.0.0 --port 8088 --log-dir central_logs
   ```
2. **Each remote agent sets `dashboard_endpoint` (or `ADK_FINOPS_DASHBOARD_ENDPOINT`):**
   ```python
   from adk_finops import FinOpsCostPlugin

   finops_plugin = FinOpsCostPlugin(
       dashboard_endpoint="http://finops-dash.internal:8088",  # Pushes via HTTPExporter to POST /api/ingest
       export_tags={"team": "payments", "service": "refund_agent"},
   )
   ```
   - **Real-Time Memory + Automatic Disk Persistence (`<log_dir>/ingested_costs.jsonl`):** Every HTTP-pushed batch is immediately served from in-memory cache (`source="http_ingest"`) **and** appended to `central_logs/ingested_costs.jsonl` on the dashboard server. If the dashboard server is restarted later with `--log-dir central_logs`, all previously pushed sessions and agent hierarchies are automatically restored from `ingested_costs.jsonl` without any data loss or double-counting.

---

#### How Hybrid Local + BigQuery Deduplication Works
When **both** local logs (`jsonl_path` / `csv_path`) and `bigquery_table` are enabled, the dashboard deduplicates every record by `(session_id, scope, agent_name)`:
- **Zero Double-Counting:** Active and local sessions load in `0ms` from memory/disk; if the same `session_id` also exists in BigQuery, the duplicate cloud row is ignored.
- **Historical Backfill:** Older runs or teammate sessions that exist **only** in BigQuery are seamlessly merged into the dashboard when **Include BigQuery** is checked.

---

### Cloud-Agnostic & Local Exporters (`JSONL`, `CSV`, `OpenTelemetry`)

You don't need a cloud warehouse to persist FinOps telemetry. `adk-finops` includes built-in local file exporters (`JSONLExporter`, `CSVExporter`) and an `OpenTelemetryExporter` that work completely offline or with any observability backend (DuckDB, Pandas, Jaeger, Datadog, Arize Phoenix, Honeycomb).

#### 1. Configure Local & OTEL Exporters on `FinOpsCostPlugin`

```python
from adk_finops import FinOpsCostPlugin

finops_plugin = FinOpsCostPlugin(
    jsonl_path="logs/finops_costs.jsonl",   # or set ADK_FINOPS_JSONL_PATH
    csv_path="logs/finops_costs.csv",       # or set ADK_FINOPS_CSV_PATH
    enable_otel=True,                       # or set ADK_FINOPS_ENABLE_OTEL=true
    export_scope="session",                 # "session", "turn", or "both"
    export_tags={"env": "local_dev"},
)
```

Each exported row in `.jsonl` and `.csv` automatically includes first-class task outcome columns (`status`, `is_failure`, `error`) alongside token counts, USD costs, context caching savings, and agent/model breakdowns.

#### 2. Query Local `.jsonl` / `.csv` Logs Instantly with DuckDB or Pandas

```sql
-- Query local JSONL file directly using DuckDB CLI
SELECT
  status AS task_outcome,
  is_failure AS is_wasted_spend,
  COUNT(*) AS total_runs,
  ROUND(SUM(total_cost_usd), 4) AS total_spend_usd
FROM read_json_auto('logs/finops_costs.jsonl')
WHERE scope = 'session' AND agent_name IS NULL
GROUP BY 1, 2;
```

#### 3. OpenTelemetry Semantic Conventions (`OpenTelemetryExporter`)

When `enable_otel=True` (or `OpenTelemetryExporter` is used), `adk-finops` enriches the active span and emits `gen_ai.finops.<scope>` spans with standard attributes:
- `gen_ai.usage.input_tokens`, `gen_ai.usage.output_tokens`, `gen_ai.usage.thoughts_tokens`, `gen_ai.usage.cached_tokens`, `gen_ai.usage.total_tokens`
- `gen_ai.usage.cost_usd`, `gen_ai.usage.llm_cost_usd`, `gen_ai.usage.tool_cost_usd`, `gen_ai.usage.savings_usd`
- `gen_ai.finops.task_outcome` (`success`, `failed`, `error`, `budget_exceeded`)
- `gen_ai.finops.is_wasted_spend` (`true` / `false`)

#### 4. Standalone Exporter Usage (`BigQueryExporter`, `JSONLExporter`, `CSVExporter`, `OpenTelemetryExporter`)

You can also use any exporter directly in standalone Python scripts or custom agent frameworks:

```python
from adk_finops import CostTracker
from adk_finops.exporters import (
    BigQueryExporter,
    CSVExporter,
    JSONLExporter,
    OpenTelemetryExporter,
)

exporters = [
    JSONLExporter("finops_costs.jsonl"),
    CSVExporter("finops_costs.csv"),
    OpenTelemetryExporter(),
    # BigQueryExporter(table_id="my-gcp-project.finops.agent_costs"),
]

with CostTracker.track_run("batch_job_42"):
    CostTracker.record_usage(
        run_id="batch_job_42",
        model_name="gemini-2.5-pro",
        prompt_tokens=15000,
        completion_tokens=800,
        cached_tokens=12000,
        agent_name="data_extractor",
    )
    CostTracker.record_task_status(session_id="batch_job_42", status="success")

summary = CostTracker.get_summary("batch_job_42")
for exporter in exporters:
    exporter.export_summary(summary, scope="session", tags={"env": "prod", "pipeline": "etl"})
```

---

### Sample SQL Queries for Looker Studio

Once data streams into BigQuery, power executive dashboards and chargeback reports with standard SQL:

#### 1. Top 5 Most Expensive Agents by LLM & Grounding Spend
```sql
SELECT
  agent_name,
  COUNT(DISTINCT session_id) AS total_sessions,
  SUM(total_tokens) AS total_tokens,
  ROUND(SUM(llm_cost_usd), 4) AS llm_cost,
  ROUND(SUM(tool_cost_usd), 4) AS grounding_fees,
  ROUND(SUM(total_cost_usd), 4) AS total_spend,
  ROUND(SUM(savings_usd), 4) AS caching_dollars_saved
FROM `my-gcp-project.finops.agent_costs`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 30 DAY)
  AND agent_name IS NOT NULL
GROUP BY 1
ORDER BY total_spend DESC
LIMIT 5;
```

#### 2. Context Caching Savings & ROI by Model
```sql
SELECT
  model_name,
  SUM(cached_tokens) AS total_cached_tokens,
  ROUND(SUM(gross_cost_usd), 4) AS gross_spend_without_cache,
  ROUND(SUM(total_cost_usd), 4) AS actual_net_spend,
  ROUND(SUM(savings_usd), 4) AS net_dollars_saved,
  ROUND(SAFE_DIVIDE(SUM(savings_usd), SUM(gross_cost_usd)) * 100, 1) AS overall_savings_pct
FROM `my-gcp-project.finops.agent_costs`
WHERE timestamp >= TIMESTAMP_SUB(CURRENT_TIMESTAMP(), INTERVAL 7 DAY)
  AND scope = 'session'
  AND agent_name IS NULL
GROUP BY 1
ORDER BY net_dollars_saved DESC;
```

#### 3. Successful vs. Wasted Spend & Failure Root-Cause Analysis
```sql
SELECT
  COALESCE(JSON_VALUE(tags, '$.status'), 'success') AS task_outcome,
  COALESCE(JSON_VALUE(tags, '$.is_failure'), 'false') AS is_wasted_spend,
  COUNT(*) AS total_runs,
  ROUND(SUM(total_cost_usd), 4) AS total_spend_usd,
  ROUND(AVG(total_cost_usd), 4) AS avg_cost_per_task,
  ROUND(AVG(total_tokens), 0) AS avg_tokens_per_task
FROM `my-gcp-project.finops.agent_costs`
WHERE scope = 'session'
  AND agent_name IS NULL
GROUP BY 1, 2
ORDER BY total_spend_usd DESC;
```

---

## Limitations & Roadmap (Next Release)

- **Explicit Tool Billing vs. String Matching**: Grounding fees currently rely on tool name matching(e.g., checking if the tool is named google_search, GoogleSearchTool, or vertex_search); upcoming versions would support explicit billing metadata/tags (e.g., `@billable(fee=...)`) and tool config inspection so custom-named tools are never missed.
- **Pre-Flight Budget Guards**: Budget checks currently evaluate reactively after calls finish; future releases would add pre-flight token estimation to block massive requests before the network call occurs.

---

## License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
