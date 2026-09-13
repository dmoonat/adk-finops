# Standalone Usage Examples (Without Google ADK)

This directory contains standalone examples demonstrating how to use `adk-finops` outside the Google ADK framework.

## Files

- **[`basic_usage.py`](basic_usage.py)**: Minimal standalone example using `CostTracker.track_run()`, recording usage, thoughts tokens, caching savings, and printing the Rich summary box.
- **[`advanced_pipeline.py`](advanced_pipeline.py)**: Multi-turn pipeline with per-agent cost attribution, enterprise discounts, budget guard, and BigQuery export.

## Running

```bash
# Basic standalone tracking
python3 basic_usage.py

# Advanced multi-turn pipeline with BigQuery exporter
python3 advanced_pipeline.py
```
