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

"""Base exporter interface for adk-finops telemetry."""

from __future__ import annotations

from abc import ABC, abstractmethod
from typing import Any


class BaseExporter(ABC):
    """Abstract base class for telemetry exporters (e.g. BigQuery, PubSub, Cloud Logging)."""

    @abstractmethod
    def export_summary(
        self,
        summary: dict[str, Any],
        scope: str = "session",
        tags: dict[str, Any] | None = None,
        blocking: bool = False,
    ) -> None:
        """Exports a FinOps summary dictionary to the external sink."""
        pass

    def close(self) -> None:
        """Flushes and releases any exporter resources or background thread pools."""
        pass
