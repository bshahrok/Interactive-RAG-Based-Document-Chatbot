#!/usr/bin/env python3
"""Lightweight test runner to execute test functions in tests/ without pytest.

Finds modules under `tests/` named `test_*.py`, imports them, and runs any
callable named `test_*`. Reports failures and a summary exit code.
"""
import importlib.util
import os
import sys
import traceback

# Ensure project root is on sys.path so tests can import `src` package
ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)


def discover_tests(tests_dir="tests", paths=None):
    if paths:
        for p in paths:
            yield p
        return
    for fname in os.listdir(tests_dir):
        if fname.startswith("test_") and fname.endswith(".py"):
            yield os.path.join(tests_dir, fname)


def load_module_from_path(path):
    name = os.path.splitext(os.path.basename(path))[0]
    spec = importlib.util.spec_from_file_location(name, path)
    mod = importlib.util.module_from_spec(spec)
    sys.modules[name] = mod
    spec.loader.exec_module(mod)
    return mod


def run(paths=None):
    failures = []
    successes = 0
    for path in discover_tests(paths=paths):
        print(f"Running tests in {path}")
        mod = load_module_from_path(path)
        for attr in dir(mod):
            if attr.startswith("test_") and callable(getattr(mod, attr)):
                fn = getattr(mod, attr)
                try:
                    fn()
                    print(f"  PASS: {attr}")
                    successes += 1
                except AssertionError:
                    print(f"  FAIL: {attr} (AssertionError)")
                    traceback.print_exc()
                    failures.append((path, attr, sys.exc_info()))
                except Exception:
                    print(f"  ERROR: {attr} (Exception)")
                    traceback.print_exc()
                    failures.append((path, attr, sys.exc_info()))

    print("\nSummary:")
    print(f"  Passed: {successes}")
    print(f"  Failed: {len(failures)}")
    if failures:
        return 1
    return 0


if __name__ == "__main__":
    args = sys.argv[1:]
    if args:
        sys.exit(run(paths=args))
    sys.exit(run())
