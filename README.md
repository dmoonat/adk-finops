# adk-finops

[![PyPI version](https://img.shields.io/pypi/v/adk-finops.svg)](https://pypi.org/project/adk-finops/)
[![Python](https://img.shields.io/pypi/pyversions/adk-finops.svg)](https://pypi.org/project/adk-finops/)
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
- **Decoupled Rate Cards**: Pricing data is stored in clean JSON. Override rates via local file, remote URL, environment variable, or code without modifying the engine.
- **Enterprise Volume Discounts**: Configure global or provider-specific discount multipliers (e.g., 15% Google Cloud negotiated discount).
- **Accurate Thinking Tokens**: Automatically captures and bills `thoughts_token_count` at the output rate while displaying thinking tokens separately in reports.
- **Smart Tool Discrimination**: Automatically excludes `MCPTool`, `McpToolset`, BigQuery, and local function tools (\$0.00 fee) while accurately billing Google Search Grounding (\$0.035) and Vertex AI Search (\$0.0025).
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
- **Google Search Grounding**: Tools named `google_search` or `GoogleSearchTool` incur **\$0.035 / call** (\$35 per 1,000 queries).
- **Vertex AI Search Grounding**: Tools matching `vertex_search`, `vertex_ai_search`, or `GroundingTool` incur **\$0.0025 / call** (\$2.50 per 1,000 queries).

---

## Gemini 2.5 Thinking Tokens Billing

Gemini 2.5 Pro and Flash separate reasoning/thinking tokens into `thoughtsTokenCount`. 

Per Google Cloud pricing rules:
$$\text{Total Billable Output Tokens} = \text{candidates\_token\_count} + \text{thoughts\_token\_count}$$

`adk-finops` automatically incorporates thinking tokens into the output token rate while reporting them as a separate field in `breakdown_by_model` so you can monitor your reasoning overhead.

---

## Standalone Usage (Without ADK)

You can use `CostTracker` in any Python application, LangChain workflow, or FastAPI service:

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
)
```

---

## Built-In Model Rate Cards

The bundled [`default_rates.json`](src/adk_finops/rates/default_rates.json) contains standard public pricing (USD per 1M tokens):

| Model | Provider | Input / 1M | Output / 1M | Cached / 1M | Context >128k Tier |
| :--- | :---: | :---: | :---: | :---: | :---: |
| `gemini-2.5-flash` | Google | \$0.30 | \$2.50 | \$0.03 | — |
| `gemini-2.5-pro` | Google | \$1.25 | \$5.00 | \$0.3125 | In: \$2.50, Out: \$10.00 |
| `gemini-2.0-flash` | Google | \$0.10 | \$0.40 | \$0.025 | — |
| `gemini-1.5-flash` | Google | \$0.075 | \$0.30 | \$0.01875 | In: \$0.15, Out: \$0.60 |
| `gemini-1.5-pro` | Google | \$1.25 | \$5.00 | \$0.3125 | In: \$2.50, Out: \$10.00 |
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

## License

Distributed under the **Apache License 2.0**. See [`LICENSE`](LICENSE) for details.
