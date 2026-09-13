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

import asyncio
import inspect
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import test_rate_card
import test_tracker
import test_plugin
import test_budget
import test_savings
import test_multi_agent
import test_display

modules = [
    test_rate_card,
    test_tracker,
    test_plugin,
    test_budget,
    test_savings,
    test_multi_agent,
    test_display,
]

async def run_all():
    passed = 0
    failed = 0
    print("=" * 60)
    print("Running adk-finops Test Suite")
    print("=" * 60)

    for mod in modules:
        mod_name = mod.__name__
        print(f"\n📂 {mod_name}")
        for attr_name in dir(mod):
            if attr_name.startswith("test_"):
                fn = getattr(mod, attr_name)
                if callable(fn):
                    try:
                        if inspect.iscoroutinefunction(fn):
                            await fn()
                        else:
                            fn()
                        print(f"  ✅ {attr_name}")
                        passed += 1
                    except Exception as e:
                        print(f"  ❌ {attr_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    asyncio.run(run_all())
