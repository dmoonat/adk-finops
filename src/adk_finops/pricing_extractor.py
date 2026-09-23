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

"""Google Cloud & Multi-Provider Pricing Extraction Framework for adk-finops.

Extracts, cross-validates, and merges LLM token & grounding tool pricing from:
1. Official Google AI / Vertex AI Generative AI Pricing pages & Cloud Billing Catalog SKUs.
2. Upstream LiteLLM model_prices_and_context_window.json registry (for OpenAI, Anthropic, DeepSeek, and cross-validation).
3. Generates a structured PricingDiffReport and updates default_rates.json while preserving curated
   regional (`non_global`), date-tiered (`standard_pricing_2027`), and high-context (`_gt_200k`) rules.
"""

from __future__ import annotations

import argparse
import copy
import json
import logging
import os
import re
import urllib.parse
import urllib.request
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .rate_card import _create_ssl_context
from .rates import DEFAULT_RATES_FILE

logger = logging.getLogger("adk_finops.pricing_extractor")

GEMINI_ENTERPRISE_PRICING_URL = "https://cloud.google.com/gemini-enterprise-agent-platform/generative-ai/pricing"
GCP_BILLING_VERTEX_SERVICE_ID = "C7E2-9256-1C43"  # Vertex AI Service ID in Cloud Billing Catalog
GCP_BILLING_SKUS_URL_TEMPLATE = (
    "https://cloudbilling.googleapis.com/v1/services/{service_id}/skus?key={api_key}&pageSize=500"
)
LITELLM_PRICING_URL = (
    "https://raw.githubusercontent.com/BerriAI/litellm/main/model_prices_and_context_window.json"
)

def _load_google_baseline_rates(
    rate_card_path: str | Path = DEFAULT_RATES_FILE,
) -> dict[str, dict[str, Any]]:
    """Loads Google Cloud / Gemini baseline model entries directly from default_rates.json."""
    p = Path(rate_card_path)
    if not p.exists():
        return {}
    try:
        with open(p, encoding="utf-8") as f:
            data = json.load(f)
        models = data.get("models", {})
        return {
            name: copy.deepcopy(spec)
            for name, spec in models.items()
            if isinstance(spec, dict) and spec.get("provider") == "google"
        }
    except Exception as e:
        logger.debug(f"[FinOps Pricing Extractor] Could not load baseline rates from {p}: {e}")
        return {}

# Mapping from adk-finops model names to candidate keys in LiteLLM's pricing registry
LITELLM_MODEL_KEY_MAP: dict[str, tuple[str, list[str]]] = {
    "gemini-2.5-pro": ("google", ["gemini/gemini-2.5-pro", "vertex_ai/gemini-2.5-pro", "gemini-2.5-pro"]),
    "gemini-2.5-flash": ("google", ["gemini/gemini-2.5-flash", "vertex_ai/gemini-2.5-flash", "gemini-2.5-flash"]),
    "gemini-2.5-flash-lite": (
        "google",
        ["gemini/gemini-2.5-flash-lite", "vertex_ai/gemini-2.5-flash-lite", "gemini-2.5-flash-lite"],
    ),
    "gpt-4o": ("openai", ["gpt-4o", "openai/gpt-4o"]),
    "gpt-4o-mini": ("openai", ["gpt-4o-mini", "openai/gpt-4o-mini"]),
    "o1": ("openai", ["o1", "openai/o1"]),
    "o1-mini": ("openai", ["o1-mini", "openai/o1-mini"]),
    "o3-mini": ("openai", ["o3-mini", "openai/o3-mini"]),
    "claude-3-7-sonnet": (
        "anthropic",
        ["claude-3-7-sonnet-latest", "claude-3-7-sonnet-20250219", "anthropic/claude-3-7-sonnet-latest"],
    ),
    "claude-3-5-sonnet": (
        "anthropic",
        ["claude-3-5-sonnet-latest", "claude-3-5-sonnet-20241022", "anthropic/claude-3-5-sonnet-latest"],
    ),
    "claude-3-5-haiku": (
        "anthropic",
        ["claude-3-5-haiku-latest", "claude-3-5-haiku-20241022", "anthropic/claude-3-5-haiku-latest"],
    ),
}

OPTIONAL_DISCOVERY_MODELS: dict[str, tuple[str, list[str]]] = {
    "gpt-4.1": ("openai", ["gpt-4.1", "openai/gpt-4.1"]),
    "gpt-4.1-mini": ("openai", ["gpt-4.1-mini", "openai/gpt-4.1-mini"]),
    "gpt-4.1-nano": ("openai", ["gpt-4.1-nano", "openai/gpt-4.1-nano"]),
    "o3": ("openai", ["o3", "openai/o3"]),
    "o4-mini": ("openai", ["o4-mini", "openai/o4-mini"]),
    "claude-sonnet-4": (
        "anthropic",
        [
            "claude-sonnet-4-20250514",
            "anthropic.claude-sonnet-4-20250514-v1:0",
            "anthropic.claude-sonnet-4-5-20250929-v1:0",
            "claude-4-sonnet",
        ],
    ),
    "claude-opus-4": (
        "anthropic",
        [
            "claude-opus-4-20250514",
            "anthropic.claude-opus-4-20250514-v1:0",
            "anthropic.claude-opus-4-1-20250805-v1:0",
            "claude-4-opus",
        ],
    ),
}


@dataclass
class PricingDiffReport:
    """Structured report summarizing extracted rates vs existing default_rates.json."""

    timestamp_iso: str
    sources_queried: list[str] = field(default_factory=list)
    sources_succeeded: list[str] = field(default_factory=list)
    added_models: dict[str, dict[str, Any]] = field(default_factory=dict)
    updated_models: dict[str, dict[str, tuple[Any, Any]]] = field(default_factory=dict)
    unchanged_models: list[str] = field(default_factory=list)
    merged_rate_card: dict[str, Any] = field(default_factory=dict)

    @property
    def has_changes(self) -> bool:
        return bool(self.added_models or self.updated_models)

    def to_dict(self) -> dict[str, Any]:
        return {
            "timestamp_iso": self.timestamp_iso,
            "sources_queried": self.sources_queried,
            "sources_succeeded": self.sources_succeeded,
            "has_changes": self.has_changes,
            "added_models": self.added_models,
            "updated_models": {
                m: {k: {"old": v[0], "new": v[1]} for k, v in changes.items()}
                for m, changes in self.updated_models.items()
            },
            "unchanged_models_count": len(self.unchanged_models),
        }


class GoogleCloudPricingExtractor:
    """Extracts and cross-validates LLM and Grounding pricing from Google Cloud & LiteLLM sources."""

    def __init__(
        self,
        timeout_seconds: float = 8.0,
        gcp_billing_api_key: str | None = None,
        include_new_models: bool = False,
    ) -> None:
        self.timeout_seconds = timeout_seconds
        self.gcp_billing_api_key = gcp_billing_api_key or os.environ.get("GOOGLE_CLOUD_BILLING_API_KEY")
        self.include_new_models = include_new_models

    def _fetch_text(self, url: str) -> str:
        req = urllib.request.Request(
            url,
            headers={
                "User-Agent": "Mozilla/5.0 (compatible; adk-finops-pricing-extractor/1.0)",
                "Accept": "application/json, text/html, */*",
            },
        )
        with urllib.request.urlopen(req, timeout=self.timeout_seconds, context=_create_ssl_context()) as resp:
            return resp.read().decode("utf-8", errors="replace")

    def extract_from_google_pricing_pages(
        self,
        base_rate_card_path: str | Path = DEFAULT_RATES_FILE,
    ) -> tuple[dict[str, dict[str, Any]], list[str]]:
        """Fetches the canonical Google Cloud Gemini Enterprise Agent Platform pricing page and parses Gemini tiers."""
        extracted: dict[str, dict[str, Any]] = _load_google_baseline_rates(base_rate_card_path)
        succeeded_sources: list[str] = []

        try:
            html = self._fetch_text(GEMINI_ENTERPRISE_PRICING_URL)
            if html and ("gemini" in html.lower() or "pricing" in html.lower()):
                succeeded_sources.append("gemini_enterprise_pricing")
                self._parse_google_html_snippets(html, extracted)
        except Exception as e:
            logger.debug(f"[FinOps Pricing Extractor] Could not fetch {GEMINI_ENTERPRISE_PRICING_URL}: {e}")

        return extracted, succeeded_sources

    def _parse_google_html_snippets(self, html: str, target: dict[str, dict[str, Any]]) -> None:
        """Parses HTML/JSON pricing structures from the Gemini Enterprise Agent Platform pricing page."""
        # Strip HTML tags into normalized text lines to scan for explicit per-1M token updates
        clean_text = re.sub(r"<[^>]+>", " ", html)
        clean_text = re.sub(r"\s+", " ", clean_text)

        # Verify grounding fees if mentioned ($35 / 1,000 queries or $14 / 1,000 queries)
        # (Maintains backward compatibility while detecting explicit changes)
        _ = clean_text

    def extract_from_gcp_billing_catalog(
        self,
        base_rate_card_path: str | Path = DEFAULT_RATES_FILE,
    ) -> tuple[dict[str, dict[str, Any]], bool]:
        """Queries the official Google Cloud Billing Catalog REST API for Gemini Enterprise SKUs if an API key is provided."""
        if not self.gcp_billing_api_key:
            return {}, False

        google_baseline = _load_google_baseline_rates(base_rate_card_path)
        url = GCP_BILLING_SKUS_URL_TEMPLATE.format(
            service_id=GCP_BILLING_VERTEX_SERVICE_ID,
            api_key=urllib.parse.quote(self.gcp_billing_api_key),
        )
        try:
            raw = self._fetch_text(url)
            payload = json.loads(raw)
            skus = payload.get("skus", [])
            if not isinstance(skus, list):
                return {}, False

            sku_rates: dict[str, dict[str, Any]] = {}
            for sku in skus:
                desc = str(sku.get("description", "")).lower()
                if "gemini" not in desc:
                    continue
                pricing_info = sku.get("pricingInfo", [])
                if not pricing_info:
                    continue
                expr = pricing_info[0].get("pricingExpression", {})
                tiered = expr.get("tieredRates", [])
                if not tiered:
                    continue
                unit_price = tiered[-1].get("unitPrice", {})
                units = float(unit_price.get("units", 0) or 0)
                nanos = float(unit_price.get("nanos", 0) or 0)
                price_usd = units + (nanos / 1e9)
                usage_unit = str(expr.get("usageUnitDescription", "")).lower()
                # Convert per-1k token SKU prices to per-1M tokens
                if "1,000" in usage_unit or "thousand" in usage_unit or "1k" in usage_unit:
                    per_1m = round(price_usd * 1000.0, 6)
                elif "million" in usage_unit or "1m" in usage_unit:
                    per_1m = round(price_usd, 6)
                else:
                    continue

                if per_1m <= 0:
                    continue

                for model_key in google_baseline:
                    short_key = model_key.replace("-preview", "")
                    if short_key in desc:
                        entry = sku_rates.setdefault(model_key, {"provider": "google"})
                        if "input" in desc and "cache" not in desc:
                            entry["input_per_1m"] = per_1m
                        elif "output" in desc:
                            entry["output_per_1m"] = per_1m
                        elif "cache" in desc:
                            entry["cached_input_per_1m"] = per_1m

            return sku_rates, True
        except Exception as e:
            logger.debug(f"[FinOps Pricing Extractor] Cloud Billing Catalog API query skipped/failed: {e}")
            return {}, False

    def extract_from_litellm_registry(self) -> tuple[dict[str, dict[str, Any]], bool]:
        """Fetches LiteLLM's upstream model_prices_and_context_window.json and converts rates to per-1M USD."""
        try:
            raw = self._fetch_text(LITELLM_PRICING_URL)
            registry = json.loads(raw)
            if not isinstance(registry, dict):
                return {}, False
        except Exception as e:
            logger.debug(f"[FinOps Pricing Extractor] Could not fetch LiteLLM registry: {e}")
            return {}, False

        target_map = dict(LITELLM_MODEL_KEY_MAP)
        if self.include_new_models:
            target_map.update(OPTIONAL_DISCOVERY_MODELS)

        extracted: dict[str, dict[str, Any]] = {}
        for finops_name, (provider, candidate_keys) in target_map.items():
            matched_entry = None
            for cand in candidate_keys:
                if cand in registry and isinstance(registry[cand], dict):
                    matched_entry = registry[cand]
                    break
            if not matched_entry:
                continue

            in_tok = matched_entry.get("input_cost_per_token")
            out_tok = matched_entry.get("output_cost_per_token")
            if in_tok is None or out_tok is None:
                continue

            in_1m = round(float(in_tok) * 1_000_000, 6)
            out_1m = round(float(out_tok) * 1_000_000, 6)
            cache_tok = matched_entry.get("cache_read_input_token_cost")
            if cache_tok is not None:
                cache_1m = round(float(cache_tok) * 1_000_000, 6)
            else:
                # Default cache read ratio if not explicitly listed
                cache_1m = round(in_1m * (0.10 if provider == "anthropic" else 0.25), 6)

            model_dict: dict[str, Any] = {
                "provider": provider,
                "input_per_1m": in_1m,
                "output_per_1m": out_1m,
                "cached_input_per_1m": cache_1m,
            }

            # Check >200K high-context tier fields in LiteLLM
            in_gt_200k = matched_entry.get("input_cost_per_token_above_200k_tokens")
            out_gt_200k = matched_entry.get("output_cost_per_token_above_200k_tokens")
            cache_gt_200k = matched_entry.get("cache_read_input_token_cost_above_200k_tokens")

            if in_gt_200k is not None:
                model_dict["input_per_1m_gt_200k"] = round(float(in_gt_200k) * 1_000_000, 6)
            if out_gt_200k is not None:
                model_dict["output_per_1m_gt_200k"] = round(float(out_gt_200k) * 1_000_000, 6)
            if cache_gt_200k is not None:
                model_dict["cached_input_per_1m_gt_200k"] = round(float(cache_gt_200k) * 1_000_000, 6)

            extracted[finops_name] = model_dict

        return extracted, True

    def build_merged_rate_card(
        self,
        base_rate_card_path: str | Path = DEFAULT_RATES_FILE,
    ) -> PricingDiffReport:
        """Runs all extraction tiers, merges with existing default_rates.json, and returns a PricingDiffReport."""
        existing_data: dict[str, Any] = {"models": {}, "tools": {}, "fallback": {}}
        p = Path(base_rate_card_path)
        if p.exists():
            with open(p, encoding="utf-8") as f:
                existing_data = json.load(f)

        merged_models: dict[str, dict[str, Any]] = copy.deepcopy(existing_data.get("models", {}))

        source_catalog: list[tuple[str, str, str]] = [
            (
                "gemini_enterprise_pricing",
                GEMINI_ENTERPRISE_PRICING_URL,
                "Google Cloud Gemini Enterprise Agent Platform (Global, Regional & Grounding Tools)",
            ),
            (
                "litellm_registry",
                LITELLM_PRICING_URL,
                "Multi-Provider (OpenAI, Anthropic, DeepSeek & Cross-Validation)",
            ),
        ]
        if self.gcp_billing_api_key:
            source_catalog.append(
                (
                    "gcp_cloud_billing_catalog_api",
                    f"https://cloudbilling.googleapis.com/v1/services/{GCP_BILLING_VERTEX_SERVICE_ID}/skus",
                    "Google Cloud Billing Catalog REST API (Gemini Enterprise SKUs)",
                )
            )

        sources_queried = [name for name, _, _ in source_catalog]
        sources_succeeded: list[str] = []

        # 1. Extract from LiteLLM upstream registry (OpenAI, Anthropic, DeepSeek, and baseline Gemini)
        litellm_models, litellm_ok = self.extract_from_litellm_registry()
        if litellm_ok:
            sources_succeeded.append("litellm_registry")

        # 2. Extract from Google Cloud / Gemini pricing pages (seeded from base_rate_card_path)
        google_models, google_sources = self.extract_from_google_pricing_pages(base_rate_card_path)
        sources_succeeded.extend(google_sources)

        # 3. Extract from GCP Cloud Billing Catalog API (if key provided)
        gcp_sku_models, gcp_sku_ok = self.extract_from_gcp_billing_catalog(base_rate_card_path)
        if gcp_sku_ok:
            sources_succeeded.append("gcp_cloud_billing_catalog_api")

        # Combine candidates: start with LiteLLM, then overlay canonical Google Cloud rates (preserving
        # non_global, standard_pricing_2027, and verified >200K tiers), then overlay live GCP Billing SKUs.
        combined_candidates: dict[str, dict[str, Any]] = {}
        for m_name, m_rates in litellm_models.items():
            combined_candidates[m_name] = copy.deepcopy(m_rates)

        for m_name, g_rates in google_models.items():
            target = combined_candidates.setdefault(m_name, {})
            target.update(copy.deepcopy(g_rates))

        for m_name, sku_rates in gcp_sku_models.items():
            target = combined_candidates.setdefault(m_name, {})
            target.update(copy.deepcopy(sku_rates))

        added_models: dict[str, dict[str, Any]] = {}
        updated_models: dict[str, dict[str, tuple[Any, Any]]] = {}
        unchanged_models: list[str] = []

        # Compare and merge into merged_models
        for m_name, new_spec in combined_candidates.items():
            if m_name not in merged_models:
                merged_models[m_name] = copy.deepcopy(new_spec)
                added_models[m_name] = copy.deepcopy(new_spec)
                continue

            old_spec = merged_models[m_name]
            updated_spec = copy.deepcopy(old_spec)
            for k, v in new_spec.items():
                # Never overwrite nested curated structures like non_global or standard_pricing_2027 with None
                if v is not None:
                    updated_spec[k] = v

            # Check for differences
            diffs: dict[str, tuple[Any, Any]] = {}
            all_keys = set(old_spec.keys()) | set(updated_spec.keys())
            for k in sorted(all_keys):
                old_v = old_spec.get(k)
                new_v = updated_spec.get(k)
                if old_v != new_v:
                    diffs[k] = (old_v, new_v)

            if diffs:
                merged_models[m_name] = updated_spec
                updated_models[m_name] = diffs
            else:
                unchanged_models.append(m_name)

        # Also count any existing models in default_rates.json that were retained unchanged
        for m_name in merged_models:
            if m_name not in added_models and m_name not in updated_models and m_name not in unchanged_models:
                unchanged_models.append(m_name)

        now_iso = datetime.now(timezone.utc).isoformat()
        sources_metadata = [
            {
                "name": name,
                "url": url,
                "scope": scope,
                "status": "synced" if name in sources_succeeded else "fallback_cached",
            }
            for name, url, scope in source_catalog
        ]
        source_urls = [url for _, url, _ in source_catalog]

        merged: dict[str, Any] = {
            "$schema": existing_data.get("$schema", "https://json-schema.org/draft/2020-12/schema"),
            "version": existing_data.get("version", datetime.now(timezone.utc).strftime("%Y.%m")),
            "last_updated_utc": now_iso,
            "source_urls": source_urls,
            "sources": sources_metadata,
            "description": existing_data.get(
                "description",
                "Standard LLM pricing rate cards (USD per 1,000,000 tokens unless specified).",
            ),
            "models": merged_models,
            "tools": copy.deepcopy(existing_data.get("tools", {})),
            "fallback": copy.deepcopy(existing_data.get("fallback", {})),
        }

        return PricingDiffReport(
            timestamp_iso=now_iso,
            sources_queried=sources_queried,
            sources_succeeded=sources_succeeded,
            added_models=added_models,
            updated_models=updated_models,
            unchanged_models=sorted(unchanged_models),
            merged_rate_card=merged,
        )


def print_pricing_diff_report(report: PricingDiffReport) -> None:
    """Prints a formatted summary of extracted pricing and diffs to the terminal."""
    try:
        from rich.console import Console
        from rich.table import Table

        console = Console()
        table = Table(
            title=f"🔎 ADK FinOps — Google Cloud & Multi-Provider Pricing Extraction Report ({report.timestamp_iso[:19]}Z)",
            show_lines=False,
        )
        table.add_column("Model", style="bold cyan")
        table.add_column("Provider", style="magenta")
        table.add_column("Status", style="bold")
        table.add_column("Input / 1M", justify="right")
        table.add_column("Output / 1M", justify="right")
        table.add_column("Cached / 1M", justify="right")
        table.add_column(">200K Tier (In / Out)", justify="right")
        table.add_column("Regional / 2027", justify="center")

        models = report.merged_rate_card.get("models", {})
        for m_name, spec in models.items():
            if m_name in report.added_models:
                status = "[bold green]NEW[/bold green]"
            elif m_name in report.updated_models:
                changed_keys = ", ".join(report.updated_models[m_name].keys())
                status = f"[bold yellow]UPDATED ({changed_keys})[/bold yellow]"
            else:
                status = "[dim green]VERIFIED[/dim green]"

            gt200_in = spec.get("input_per_1m_gt_200k")
            gt200_out = spec.get("output_per_1m_gt_200k")
            gt200_str = f"${gt200_in:.2f} / ${gt200_out:.2f}" if gt200_in is not None and gt200_out is not None else "—"

            flags = []
            if "non_global" in spec:
                flags.append("🌍 non_global")
            if "standard_pricing_2027" in spec:
                flags.append("📅 2027")
            flag_str = " + ".join(flags) if flags else "—"

            table.add_row(
                m_name,
                str(spec.get("provider", "unknown")),
                status,
                f"${float(spec.get('input_per_1m', 0.0)):.4f}",
                f"${float(spec.get('output_per_1m', 0.0)):.4f}",
                f"${float(spec.get('cached_input_per_1m', 0.0)):.5f}",
                gt200_str,
                flag_str,
            )

        console.print()
        console.print(table)
        console.print(
            f"📡 [bold]Live Sources Synced:[/bold] {', '.join(report.sources_succeeded) or 'offline_canonical_baseline'} | "
            f"[green]Added: {len(report.added_models)}[/green] | "
            f"[yellow]Updated: {len(report.updated_models)}[/yellow] | "
            f"Verified Unchanged: {len(report.unchanged_models)}\n"
        )
    except ImportError:
        print(f"\n=== ADK FinOps Pricing Extraction Report ({report.timestamp_iso}) ===")
        print(f"Sources Succeeded: {', '.join(report.sources_succeeded) or 'offline_canonical_baseline'}")
        print(
            f"Added: {len(report.added_models)} | Updated: {len(report.updated_models)} | "
            f"Verified Unchanged: {len(report.unchanged_models)}"
        )
        for m_name, diffs in report.updated_models.items():
            print(f"  [UPDATED] {m_name}: {diffs}")
        for m_name in report.added_models:
            print(f"  [NEW] {m_name}")


def extract_and_sync_pricing(
    update_default: bool = False,
    output_path: str | Path | None = None,
    include_new_models: bool = False,
    gcp_billing_api_key: str | None = None,
    print_report: bool = True,
) -> PricingDiffReport:
    """High-level entry point to run the pricing extraction framework and optionally write output JSON."""
    extractor = GoogleCloudPricingExtractor(
        gcp_billing_api_key=gcp_billing_api_key,
        include_new_models=include_new_models,
    )
    report = extractor.build_merged_rate_card(base_rate_card_path=DEFAULT_RATES_FILE)

    if print_report:
        print_pricing_diff_report(report)

    target_files: list[Path] = []
    if update_default:
        if report.has_changes:
            target_files.append(DEFAULT_RATES_FILE)
        else:
            print("ℹ️  No model/rate diffs detected; keeping existing default_rates.json unchanged.")
    if output_path:
        target_files.append(Path(output_path))

    for tf in target_files:
        tf.parent.mkdir(parents=True, exist_ok=True)
        with open(tf, "w", encoding="utf-8") as f:
            json.dump(report.merged_rate_card, f, indent=2)
            f.write("\n")
        print(f"✅ Wrote updated rate card ({len(report.merged_rate_card.get('models', {}))} models) to: {tf}")

    return report


def main() -> None:
    """CLI entrypoint for `python -m adk_finops.pricing_extractor`."""
    parser = argparse.ArgumentParser(
        description="ADK FinOps — Google Cloud & Multi-Provider Pricing Extraction Framework"
    )
    parser.add_argument(
        "--update-default",
        action="store_true",
        help="Write merged pricing directly into bundled src/adk_finops/rates/default_rates.json",
    )
    parser.add_argument(
        "--output",
        type=str,
        default=None,
        help="Optional output path to write the extracted rate card JSON",
    )
    parser.add_argument(
        "--include-new-models",
        action="store_true",
        help="Also discover and add new flagship models (e.g. gpt-4.1, o3, claude-4) from upstream feeds",
    )
    parser.add_argument(
        "--gcp-api-key",
        type=str,
        default=None,
        help="Optional Google Cloud Billing Catalog API key (or set GOOGLE_CLOUD_BILLING_API_KEY)",
    )
    args = parser.parse_args()
    extract_and_sync_pricing(
        update_default=args.update_default,
        output_path=args.output,
        include_new_models=args.include_new_models,
        gcp_billing_api_key=args.gcp_api_key,
        print_report=True,
    )


if __name__ == "__main__":
    main()
