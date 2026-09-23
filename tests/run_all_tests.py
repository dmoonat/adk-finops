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

import asyncio
import inspect
import sys
from pathlib import Path

# Add src to sys.path
sys.path.insert(0, str(Path(__file__).parent.parent / "src"))

import test_advisor
import test_bigquery_exporter
import test_budget
import test_display
import test_exporters
import test_multi_agent
import test_plugin
import test_preflight_guard
import test_rate_card
import test_savings
import test_security
import test_task_efficiency
import test_tracker
import unittest

modules = [
    test_rate_card,
    test_tracker,
    test_plugin,
    test_budget,
    test_savings,
    test_multi_agent,
    test_display,
    test_bigquery_exporter,
    test_task_efficiency,
    test_advisor,
    test_exporters,
    test_preflight_guard,
    test_security,
]

import contextlib
import io
import os
import tempfile


class _MiniMonkeyPatch:
    def __init__(self):
        self._env_backups = {}
        self._attr_backups = []

    def setenv(self, name, value):
        if name not in self._env_backups:
            self._env_backups[name] = os.environ.get(name)
        os.environ[name] = str(value)

    def delenv(self, name, raising=True):
        if name not in self._env_backups:
            self._env_backups[name] = os.environ.get(name)
        os.environ.pop(name, None)

    def setattr(self, target, name, value):
        orig = getattr(target, name)
        self._attr_backups.append((target, name, orig))
        setattr(target, name, value)

    def undo(self):
        for k, v in self._env_backups.items():
            if v is None:
                os.environ.pop(k, None)
            else:
                os.environ[k] = v
        for target, name, orig in reversed(self._attr_backups):
            setattr(target, name, orig)


class _MiniCapSys:
    def __init__(self, out_buf, err_buf):
        self._out = out_buf
        self._err = err_buf

    def readouterr(self):
        from collections import namedtuple
        Res = namedtuple("CaptureResult", ["out", "err"])
        return Res(self._out.getvalue(), self._err.getvalue())


def run_all():
    passed = 0
    failed = 0
    print("=" * 60)
    print("Running adk-finops Complete Test Suite (All 13 Modules)")
    print("=" * 60)

    for mod in modules:
        mod_name = mod.__name__
        print(f"\n📂 {mod_name}")
        for attr_name in dir(mod):
            if attr_name.startswith("test_"):
                fn = getattr(mod, attr_name)
                if callable(fn):
                    mp = _MiniMonkeyPatch()
                    out_buf, err_buf = io.StringIO(), io.StringIO()
                    try:
                        sig = inspect.signature(fn)
                        with tempfile.TemporaryDirectory() as tmpdir:
                            kwargs = {}
                            if "tmp_path" in sig.parameters:
                                kwargs["tmp_path"] = Path(tmpdir)
                            if "monkeypatch" in sig.parameters:
                                kwargs["monkeypatch"] = mp
                            if "capsys" in sig.parameters:
                                kwargs["capsys"] = _MiniCapSys(out_buf, err_buf)
                                with contextlib.redirect_stdout(out_buf), contextlib.redirect_stderr(err_buf):
                                    if inspect.iscoroutinefunction(fn):
                                        asyncio.run(fn(**kwargs))
                                    else:
                                        fn(**kwargs)
                            else:
                                if inspect.iscoroutinefunction(fn):
                                    asyncio.run(fn(**kwargs))
                                else:
                                    fn(**kwargs)
                        print(f"  ✅ {attr_name}")
                        passed += 1
                    except Exception as e:
                        print(f"  ❌ {attr_name}: {e}")
                        import traceback
                        traceback.print_exc()
                        failed += 1
                    finally:
                        mp.undo()
            elif isinstance(getattr(mod, attr_name), type) and issubclass(getattr(mod, attr_name), unittest.TestCase):
                cls_obj = getattr(mod, attr_name)
                suite = unittest.defaultTestLoader.loadTestsFromTestCase(cls_obj)
                for test_case in suite:
                    test_name = test_case._testMethodName
                    try:
                        res = unittest.TestResult()
                        test_case.run(res)
                        if res.errors or res.failures:
                            err_msg = (res.errors + res.failures)[0][1]
                            print(f"  ❌ {cls_obj.__name__}.{test_name}:\n{err_msg}")
                            failed += 1
                        else:
                            print(f"  ✅ {cls_obj.__name__}.{test_name}")
                            passed += 1
                    except Exception as e:
                        print(f"  ❌ {cls_obj.__name__}.{test_name}: {e}")
                        failed += 1

    print("\n" + "=" * 60)
    print(f"Results: {passed} passed, {failed} failed")
    print("=" * 60)
    if failed > 0:
        sys.exit(1)

if __name__ == "__main__":
    run_all()
