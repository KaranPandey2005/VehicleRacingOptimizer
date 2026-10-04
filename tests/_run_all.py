"""
Minimal standalone test runner for environments without pytest installed.
Discovers test_*.py files in this directory and runs every test_* function,
reporting pass/fail. Real development should use `pytest` directly.
"""
import sys, os, traceback, importlib

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), "..")))

try:
    import pytest  # noqa: F401
except ImportError:
    # Minimal shim so test files that use `pytest.raises` can still run in
    # this sandbox. Real development should install pytest via requirements.txt.
    import types
    from contextlib import contextmanager

    @contextmanager
    def _raises(exc_type):
        try:
            yield
        except exc_type:
            return
        else:
            raise AssertionError(f"Expected {exc_type.__name__} to be raised")

    fake_pytest = types.ModuleType("pytest")
    fake_pytest.raises = _raises
    sys.modules["pytest"] = fake_pytest

TEST_DIR = os.path.dirname(__file__)
passed, failed = 0, 0
failures = []

for fname in sorted(os.listdir(TEST_DIR)):
    if not (fname.startswith("test_") and fname.endswith(".py")):
        continue
    modname = f"tests.{fname[:-3]}"
    mod = importlib.import_module(modname)
    for attr in sorted(dir(mod)):
        if attr.startswith("test_"):
            fn = getattr(mod, attr)
            if callable(fn):
                try:
                    fn()
                    passed += 1
                    print(f"PASS  {fname}::{attr}")
                except Exception as e:
                    failed += 1
                    failures.append((fname, attr, e))
                    print(f"FAIL  {fname}::{attr}  -> {e}")

print(f"\n{passed} passed, {failed} failed")
if failures:
    print("\n--- Failure details ---")
    for fname, attr, e in failures:
        print(f"\n{fname}::{attr}")
        traceback.print_exception(type(e), e, e.__traceback__)
    sys.exit(1)
