# adk-finops

[![PyPI](https://img.shields.io/pypi/v/adk-finops.svg)](https://pypi.org/project/adk-finops/)
[![Downloads](https://img.shields.io/pypi/dm/adk-finops.svg)](https://pypi.org/project/adk-finops/)
[![Python](https://img.shields.io/badge/python-3.9%2B-blue.svg)](pyproject.toml)
[![License](https://img.shields.io/badge/License-Apache%202.0-blue.svg)](LICENSE)
[![Typing: Typed](https://img.shields.io/badge/typing-typed-green.svg)](src/adk_finops/py.typed)

**Universal FinOps cost, token usage, and grounding fee tracking for the Google Agent Development Kit (ADK) and Python AI agents.**

---

## Table of Contents

- [Why adk-finops?](#why-adk-finops)
- [Key Features](#key-features)
- [Installation](#installation)
- [Quickstart with Google ADK](#quickstart-with-google-adk)
- [Dual-Scope Telemetry: Turn vs. Session](#dual-scope-telemetry-turn-vs-session)
- [Budget Guards & Circuit Breakers](#budget-guards--circuit-breakers)
- [Context Caching Savings Analytics (ROI Tracker)](#context-caching-savings-analytics-roi-tracker)
- [Multi-Agent Cost Attribution & Delegation Tracking](#multi-agent-cost-attribution--delegation-tracking)
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
- [Coming Soon](#coming-soon)
  - [One-Line BigQuery Exporter](#-one-line-bigquery-exporter)
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

### With Google ADK
```bash
pip install "adk-finops[adk]"
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

### Agent Breakdown in Session Telemetry

Every turn and session summary includes a granular `breakdown_by_agent` dictionary:

```json
{
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
      "savings_usd": 0.0
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
      "savings_pct": 22.2
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
      "savings_usd": 0.0
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
│  Model              Calls   Tokens (In / Out)   Cost (USD)    Cache Savings  │
│  ──────────────────────────────────────────────────────────────────────────  │
│  gemini-2.5-pro         1         1,200 / 300      $0.0030                —  │
│  gemini-2.5-flash       1         8,000 / 600      $0.0023   $0.0016 (41.5%) │
│                                                                              │
│  Agent           Calls   Tokens   LLM Cost   Tool Fees   Total Cost          │
│  ──────────────────────────────────────────────────────────────────          │
│  🤖 supervisor       1    1,500    $0.0030       $0.00      $0.0030          │
│  🤖 researcher       1    8,600    $0.0023     $0.0280      $0.0303          │
│                                                                              │
│  🛡️  Budget Guard: $0.0333 / $1.0000 (3.3% utilized)                         │
╰───────────────── adk-finops • Universal Token & Cost Engine ─────────────────╯
```

### Features

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

# Automatically renders the Rich box at the end of each turn
finops_plugin = FinOpsCostPlugin(
    default_model="gemini-2.5-pro",
    render_terminal_box=True,  # Enabled by default
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
```python
from adk_finops import CostTracker

CostTracker.register_rate_card("my-internal-model", {
    "provider": "internal",
    "input_per_1m": 0.40,
    "output_per_1m": 1.60,
})

CostTracker.register_tool_rate("internal_vector_db", 0.0005)
```

---

## Smart Tool Classification (MCP vs. Grounding)

`adk-finops` distinguishes between paid cloud grounding services and free tools:

- **MCP Tools (`McpToolset`, `MCPTool`)**: Automatically identified and assigned **\$0.00** fee (e.g. `search_documents`).
- **Database & Custom Tools**: BigQuery toolsets and custom Python functions incur **\$0.00** tool fee.
- **Google Search Grounding & Web Grounding**: Tools named `google_search` or `GoogleSearchTool` incur **\$0.014 / query** (\$14.00 per 1,000 queries; first 5,000 queries/month free). Charged for each individual Grounding Query performed by Gemini. Input tokens returned by search grounding are not charged.
- **Google Maps Grounding**: Incurs **\$0.014 / query** (\$14.00 per 1,000 queries; first 5,000 queries/month free). Input tokens are not charged.
- **Grounding with your data (Vertex AI Search / Datastores)**: Tools matching `vertex_search`, `vertex_ai_search`, or `GroundingTool` incur **\$0.0025 / prompt** (\$2.50 per 1,000 prompts).

---

## Gemini 2.5 Thinking Tokens Billing

Gemini 2.5 Pro and Flash separate reasoning/thinking tokens into `thoughtsTokenCount`. 

Per Google Cloud pricing rules:
> **Total Billable Output Tokens** = `candidates_token_count` + `thoughts_token_count`

`adk-finops` automatically incorporates thinking tokens into the output token rate while reporting them as a separate field in `breakdown_by_model` so you can monitor your reasoning overhead.

---

## Standalone Usage (Without ADK)

You can use `CostTracker` in any Python script, background worker, or web service:

```python
from adk_finops import CostTracker

# Wrap any execution block with automatic lifecycle cleanup
with CostTracker.track_run("request_123"):
    CostTracker.record_usage(
        run_id="request_123",
        model_name="gemini-2.5-flash",
        prompt_tokens=1500,
        completion_tokens=250,
        thoughts_tokens=100,
        cached_tokens=0,
    )

summary = CostTracker.get_summary("request_123")
print(f"Total Cost: ${summary['total_cost_usd']:.6f}")
print(f"Total Tokens: {summary['total_tokens']}")
```

---

## Configuration Reference

### Environment Variables

| Variable | Type | Description |
| :--- | :--- | :--- |
| `ADK_FINOPS_RATE_CARD_PATH` | `str` | Absolute or relative path to a custom JSON rate card file. |
| `ADK_FINOPS_DISCOUNT_PERCENT` | `float` | Global enterprise discount percentage (e.g., `15.0` for 15%). |

### Plugin Initialization Parameters

```python
FinOpsCostPlugin(
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
)
```

| Parameter | Type | Default | Description |
| :--- | :---: | :---: | :--- |
| `default_model` | `str` | `"gemini-2.5-flash"` | Fallback model name if not reported by the LLM response. |
| `rate_card_path` | `str \| Path` | `None` | Path to custom rate card JSON file. |
| `discount_percent` | `float` | `None` | Enterprise volume discount percentage (e.g. `15.0` for 15%). |
| `budget_limit_usd` | `float` | `None` | Maximum cumulative spending limit in USD for the entire chat session. |
| `turn_budget_limit_usd` | `float` | `None` | Maximum spending limit in USD for any single user turn. |
| `agent_budgets` | `dict[str, float]` | `None` | Per-agent spending caps in USD (e.g. `{"researcher": 0.50, "coder": 1.00}`). |
| `on_budget_exceeded` | `str` | `"halt"` | Action on budget breach: `"halt"` (raise/block), `"warn"`, or `"downgrade"`. |
| `fallback_model` | `str` | `"gemini-2.5-flash"` | Target model when using `"downgrade"` mode. |
| `render_terminal_box` | `bool` | `True` | Renders a beautiful color-coded summary box to stdout at the end of each turn. |

---

## Built-In Model Rate Cards

The bundled [`default_rates.json`](src/adk_finops/rates/default_rates.json) contains standard public pricing (USD per 1M tokens):

| Model | Provider | Input / 1M | Output / 1M | Cached / 1M | Context >128k Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `gemini-3.8-flash` | Google | \$0.75 | \$3.75 | \$0.075 | Standard 2027: In \$1.50, Out \$7.50 |
| `gemini-3.7-flash` | Google | \$0.75 | \$3.75 | \$0.075 | Standard 2027: In \$1.50, Out \$7.50 |
| `gemini-3.6-flash` | Google | \$0.75 | \$3.75 | \$0.075 | Standard 2027: In \$1.50, Out \$7.50 |
| `codemender` | Google | \$0.75 | \$3.75 | \$0.075 | Standard 2027: In \$1.50, Out \$7.50 |
| `gemini-3.5-flash` | Google | \$1.50 | \$9.00 | \$0.15 | — |
| `gemini-2.5-flash` | Google | \$0.30 | \$2.50 | \$0.03 | — |
| `gemini-2.5-pro` | Google | \$1.25 | \$5.00 | \$0.3125 | In: \$2.50, Out: \$10.00 |
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

## Coming Soon

We are actively expanding `adk-finops` with enterprise-grade data warehouse integrations and analytics:

### 📊 One-Line BigQuery Exporter

Stream or batch-export turn, session, model, and sub-agent FinOps telemetry directly into a Google BigQuery dataset with a single configuration line:

```python
finops_plugin = FinOpsCostPlugin(
    default_model="gemini-2.5-pro",
    bigquery_table="my-gcp-project.finops.agent_costs",  # Coming soon!
)
```

- **Zero-Boilerplate Schema Management**: Automatically provisions and manages partitioned and clustered BigQuery telemetry tables.
- **Enterprise Reporting & Looker Dashboards**: Power real-time Looker Studio / Looker dashboards for department chargebacks, multi-tenant billing, and cost center attribution.
- **Historical ROI & Trend Analytics**: Track cache hit rates, prompt growth, and grounding fee trends across millions of agent interactions over time.

---

## License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
