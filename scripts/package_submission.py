"""Validate and create the required Assignment 1 submission archive.

The script deliberately refuses to package an incomplete or fabricated set of
experiment artifacts. Run it after all required model and evaluation runs have
produced their outputs.
"""
from __future__ import annotations

import argparse
from pathlib import Path
import zipfile

REQUIRED_ARTIFACTS = (
    "fix.patch",
    "part1-trajectory.json",
    "django__django-15368.patch",
    "django__django-15368-trajectory.json",
    "token-usage-analysis.md",
    "part3-trajectory.json",
    "game-result.json",
    "part3-no-legal-moves-deepseek.json",
    "part3-no-legal-moves-deepseek-result.json",
    "part3-legal-moves-deepseek.json",
    "part3-legal-moves-deepseek-result.json",
    "part3-no-legal-moves-gpt-oss.json",
    "part3-no-legal-moves-gpt-oss-result.json",
    "part3-legal-moves-gpt-oss.json",
    "part3-legal-moves-gpt-oss-result.json",
    "observation-experiment.md",
    "part3-python-skill-trajectory.json",
)


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("assignment-1-submission.zip"))
    args = parser.parse_args()

    root = args.root.resolve()
    missing = [f"artifacts/{name}" for name in REQUIRED_ARTIFACTS
               if not (root / "artifacts" / name).is_file()]
    for required in ("AI_USAGE.md", "src/assignment/agent"):
        if not (root / required).exists():
            missing.append(required)
    if missing:
        print("Submission not packaged. Missing required files:")
        for item in missing:
            print(f"  - {item}")
        return 1

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for folder in ("src", "artifacts"):
            for path in sorted((root / folder).rglob("*")):
                if path.is_file():
                    archive.write(path, path.relative_to(root))
        archive.write(root / "AI_USAGE.md", "AI_USAGE.md")
    print(f"Created {output}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
