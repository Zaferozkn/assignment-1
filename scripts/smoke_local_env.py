"""Small smoke check for the opt-in host-local execution backend.

This is intentionally a script rather than a test under tests/: the assignment
rules prohibit modifying the provided test suite.
"""
from __future__ import annotations

import argparse
from pathlib import Path

from assignment.local_env import LocalEnvironment


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--cwd", type=Path, default=Path.cwd())
    args = parser.parse_args()

    with LocalEnvironment(cwd=args.cwd) as env:
        result = env.execute("python -c \"print('local backend ready')\"")
    if result.get("returncode") != 0:
        print(result)
        return 1
    print(result.get("stdout", "").strip())
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
