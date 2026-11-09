"""Validate and package only the files allowed by ASSIGNMENT.md.

Required experiment evidence must exist and be parseable. This checker does
not create, repair, or synthesize any experiment evidence.
"""
from __future__ import annotations

import argparse
import json
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


def validate(root: Path) -> tuple[list[str], list[Path]]:
    problems: list[str] = []
    included: list[Path] = []

    for name in REQUIRED_ARTIFACTS:
        relative = Path("artifacts") / name
        file = root / relative
        if not file.is_file() or file.stat().st_size == 0:
            problems.append(f"Missing or empty: {relative}")
            continue
        if file.is_symlink():
            problems.append(f"Symlink not allowed: {relative}")
            continue
        if file.suffix == ".json":
            try:
                data = json.loads(file.read_text(encoding="utf-8"))
                if not isinstance(data, dict):
                    raise ValueError("Expected a JSON object")
            except (UnicodeError, json.JSONDecodeError, ValueError) as exc:
                problems.append(f"Invalid JSON in {relative}: {exc}")
                continue
        included.append(relative)

    source = root / "src" / "assignment" / "agent"
    if not source.is_dir():
        problems.append("Missing: src/assignment/agent")
    else:
        for file in sorted(source.rglob("*.py")):
            if file.is_symlink():
                problems.append(f"Symlink not allowed: {file.relative_to(root)}")
            elif file.is_file():
                included.append(file.relative_to(root))

    prompts = Path("src/assignment/prompts.py")
    if (root / prompts).is_file():
        # Only include the optional prompts file if the student edited it.
        # Unchanged provided files are not part of the submission.
        # This can be checked externally with git diff; omitted by default.
        pass

    usage = Path("AI_USAGE.md")
    if not (root / usage).is_file() or not (root / usage).read_text(encoding="utf-8").strip():
        problems.append("Missing or empty: AI_USAGE.md")
    else:
        included.append(usage)

    # Ensure at least the expected implementation modules are present.
    for name in ("base.py", "code_agent.py", "chess_agent.py", "chess_tools.py", "tools.py"):
        if not (source / name).is_file():
            problems.append(f"Missing: src/assignment/agent/{name}")

    return problems, included


def main() -> int:
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--root", type=Path, default=Path.cwd())
    parser.add_argument("--output", type=Path, default=Path("assignment-1-submission.zip"))
    args = parser.parse_args()
    root = args.root.resolve()

    problems, included = validate(root)
    if problems:
        print("Submission not packaged. Resolve these issues first:")
        for item in problems:
            print(f"  - {item}")
        return 1

    output = args.output if args.output.is_absolute() else root / args.output
    output.parent.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(output, "w", compression=zipfile.ZIP_DEFLATED) as archive:
        for relative in sorted(set(included)):
            archive.write(root / relative, relative.as_posix())
    print(f"Created {output} ({len(set(included))} files)")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
