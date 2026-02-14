"""Unified local checks runner.

Profiles:
- simple: fast minimal checks
- smoke: broader deterministic checks (default)
"""

import argparse
import os
import subprocess
import sys


def run(cmd: list[str], env: dict[str, str]) -> bool:
    """Run command and return whether it passed."""
    print(f"$ {' '.join(cmd)}")
    result = subprocess.run(cmd, env=env)
    ok = result.returncode == 0
    print(f"{'✓ PASS' if ok else '✗ FAIL'}\n")
    return ok


def build_checks(profile: str) -> dict[str, list[str]]:
    python = sys.executable
    base = {
        "ruff": [python, "-m", "ruff", "check", "src", "evaluation", "tests"],
        "output filter": [python, "-m", "pytest", "tests/test_output_filter.py", "-q"],
    }

    if profile == "simple":
        return base

    return {
        **base,
        "evaluation metrics": [python, "-m", "pytest", "tests/test_evaluation_metrics.py", "-q"],
    }


def main() -> int:
    parser = argparse.ArgumentParser(description="Run local validation checks")
    parser.add_argument(
        "--profile",
        choices=["simple", "smoke"],
        default="smoke",
        help="Check profile to run",
    )
    args = parser.parse_args()

    env = os.environ.copy()
    env.setdefault("PYTHONPATH", ".")
    env.setdefault("OPENAI_API_KEY", "sk-test")

    checks = build_checks(args.profile)
    results = {name: run(cmd, env) for name, cmd in checks.items()}

    passed = sum(results.values())
    total = len(results)
    print(f"Total: {passed}/{total} checks passed")
    return 0 if passed == total else 1


if __name__ == "__main__":
    raise SystemExit(main())
