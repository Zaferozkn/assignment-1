from pathlib import Path

from assignment.local_env import LocalEnvironment


def test_local_environment_runs_commands_in_configured_directory(tmp_path: Path):
    (tmp_path / "hello.txt").write_text("hello")
    with LocalEnvironment(cwd=tmp_path) as env:
        result = env.execute("python -c \"from pathlib import Path; print(Path('hello.txt').read_text())\"")
    assert result["returncode"] == 0
    assert result["stdout"].strip() == "hello"
    assert result["exception_info"] == ""


def test_local_environment_reports_nonzero_exit_without_raising(tmp_path: Path):
    result = LocalEnvironment(cwd=tmp_path).execute("python -c \"raise SystemExit(7)\"")
    assert result["returncode"] == 7
    assert result["exception_info"] == ""


def test_local_environment_reports_timeout(tmp_path: Path):
    result = LocalEnvironment(cwd=tmp_path).execute(
        "python -c \"import time; time.sleep(2)\"", timeout=0.05
    )
    assert result["returncode"] == -1
    assert "timed out" in result["exception_info"].lower()
