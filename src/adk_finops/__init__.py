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
from .rate_card import ModelRate, RateCardRegistry
from .tracker import BudgetExceededError, CostTracker

try:
    from .plugin import FinOpsCostPlugin

    __all__ = [
        "CostTracker",
        "BudgetExceededError",
        "FinOpsCostPlugin",
        "RateCardRegistry",
        "ModelRate",
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
    ]
)

try:
    from .exporters import BigQueryExporter

    __all__.append("BigQueryExporter")
except ImportError:
    pass

from ._version import __version__
