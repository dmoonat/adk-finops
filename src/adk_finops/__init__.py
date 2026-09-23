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

"""adk-finops: Universal LLM FinOps and Token Cost Tracking for Google ADK and Python agents."""

from .advisor import ADVISOR_DISCLAIMER, generate_optimization_insights
from .display import (
    format_plain_task_efficiency,
    format_summary_box,
    print_summary,
    print_task_efficiency_summary,
    render_task_efficiency_table,
)
from .billable import (
    BillableToolSpec,
    billable,
    extract_billable_spec,
    is_error_tool_result,
)
from .estimator import estimate_request_tokens, estimate_text_tokens
from .rate_card import ModelRate, RateCardRegistry
from .token_profiles import (
    DEFAULT_PROVIDER_TOKEN_PROFILES,
    PROVIDER_TOKEN_PROFILES,
    TokenProfileRegistry,
    get_token_profile,
    load_token_profiles_file,
    register_token_profile,
    reset_token_profiles,
    update_token_profile,
)
from .tracker import BudgetExceededError, CostTracker

try:
    from .plugin import FinOpsCostPlugin

    __all__ = [
        "CostTracker",
        "BudgetExceededError",
        "FinOpsCostPlugin",
        "RateCardRegistry",
        "ModelRate",
        "TokenProfileRegistry",
        "PROVIDER_TOKEN_PROFILES",
        "DEFAULT_PROVIDER_TOKEN_PROFILES",
        "register_token_profile",
        "update_token_profile",
        "get_token_profile",
        "load_token_profiles_file",
        "reset_token_profiles",
        "BillableToolSpec",
        "billable",
        "extract_billable_spec",
        "estimate_request_tokens",
        "estimate_text_tokens",
        "print_summary",
        "format_summary_box",
        "print_task_efficiency_summary",
        "render_task_efficiency_table",
    ]
except ImportError:
    # Allows CostTracker and RateCardRegistry to be used without google-adk installed
    __all__ = [
        "CostTracker",
        "BudgetExceededError",
        "RateCardRegistry",
        "ModelRate",
        "TokenProfileRegistry",
        "PROVIDER_TOKEN_PROFILES",
        "DEFAULT_PROVIDER_TOKEN_PROFILES",
        "register_token_profile",
        "update_token_profile",
        "get_token_profile",
        "load_token_profiles_file",
        "reset_token_profiles",
        "BillableToolSpec",
        "billable",
        "extract_billable_spec",
        "estimate_request_tokens",
        "estimate_text_tokens",
        "print_summary",
        "format_summary_box",
        "print_task_efficiency_summary",
        "render_task_efficiency_table",
    ]

from .dashboard import create_dashboard_app, start_background_dashboard
from .exporters import (
    BaseExporter,
    CSVExporter,
    HTTPExporter,
    JSONLExporter,
    OpenTelemetryExporter,
)

from .pricing_extractor import (
    GoogleCloudPricingExtractor,
    PricingDiffReport,
    extract_and_sync_pricing,
    print_pricing_diff_report,
)

__all__.extend(
    [
        "BaseExporter",
        "CSVExporter",
        "HTTPExporter",
        "JSONLExporter",
        "OpenTelemetryExporter",
        "create_dashboard_app",
        "start_background_dashboard",
        "generate_optimization_insights",
        "ADVISOR_DISCLAIMER",
        "GoogleCloudPricingExtractor",
        "PricingDiffReport",
        "extract_and_sync_pricing",
        "print_pricing_diff_report",
    ]
)

try:
    from .exporters import BigQueryExporter

    __all__.append("BigQueryExporter")
except ImportError:
    pass

from ._version import __version__
