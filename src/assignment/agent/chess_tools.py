"""Chess tool implementations, decoupled from the agent that registers them."""
from __future__ import annotations
import base64
import json
import shlex
from typing import Any
import httpx

CHESS_PORT = 8000


def _request_state(client: httpx.Client, method: str, endpoint: str, **kwargs: Any) -> dict[str, Any]:
    response = client.request(method, endpoint, **kwargs)
    try:
        payload = response.json()
    except ValueError as exc:
        raise RuntimeError(f"Chess server returned non-JSON ({response.status_code}).") from exc
    if response.status_code >= 400:
        detail = payload.get("detail", payload) if isinstance(payload, dict) else payload
        raise ValueError(str(detail))
    if not isinstance(payload, dict):
        raise RuntimeError("Chess server response must be a JSON object.")
    return payload


def _error(exc: Exception) -> str:
    return f"<chess_error>{type(exc).__name__}: {exc}</chess_error>"


def _parse_arguments(arguments: str) -> dict[str, Any]:
    if not isinstance(arguments, str):
        raise ValueError("Tool arguments must be a JSON string.")
    try:
        parsed = json.loads(arguments)
    except json.JSONDecodeError as exc:
        raise ValueError(f"Malformed JSON arguments: {exc.msg}") from exc
    if not isinstance(parsed, dict):
        raise ValueError("Tool arguments must be a JSON object.")
    return parsed


def _simulate_move(client: httpx.Client, arguments: str) -> str:
    try:
        args = _parse_arguments(arguments)
        fen = args.get("fen")
        move = args.get("move")
        if not isinstance(fen, str) or not fen.strip():
            raise ValueError("fen is required and must be a non-empty string.")
        if move is not None and not isinstance(move, str):
            raise ValueError("move must be a string when provided.")
        payload = {"fen": fen}
        if move is not None:
            payload["move"] = move
        return json.dumps(_request_state(client, "POST", "/api/simulate", json=payload))
    except Exception as exc:
        return _error(exc)


def _play_move(client: httpx.Client, arguments: str) -> str:
    try:
        args = _parse_arguments(arguments)
        move = args.get("move")
        if not isinstance(move, str) or not move.strip():
            raise ValueError("move is required and must be a non-empty string.")
        state = _request_state(client, "POST", "/api/move", json={"move": move})
        if not isinstance(state.get("fen"), str) or not state["fen"].strip():
            raise ValueError("Chess server response is missing a string fen.")
        return json.dumps(state)
    except Exception as exc:
        return _error(exc)


def _run_python(env: Any, port: int, arguments: str) -> str:
    try:
        args = _parse_arguments(arguments)
        code = args.get("code")
        if not isinstance(code, str):
            raise ValueError("code is required and must be a string.")
        encoded = base64.b64encode(code.encode("utf-8")).decode("ascii")
        command = f"python /opt/assignment/sandbox_python.py {int(port)} {shlex.quote(encoded)}"
        result = env.execute(command, timeout=120)
        if result.get("returncode", -1) != 0:
            detail = result.get("exception_info") or result.get("stderr") or result.get("output") or "Sandbox command failed."
            return f"<chess_error>{detail}</chess_error>"
        output = result.get("output", "")
        # The runner writes its JSON payload to stdout; retain the exact JSON.
        return output.strip()
    except Exception as exc:
        return _error(exc)


def _invoke_skill(skills: dict[str, dict[str, str]], arguments: str) -> str:
    try:
        args = _parse_arguments(arguments)
        name = args.get("name")
        if not isinstance(name, str) or name not in skills:
            raise ValueError(f"Unknown skill: {name!r}")
        return skills[name]["content"]
    except Exception as exc:
        return _error(exc)


def _game_state(client: httpx.Client, reset: bool = False) -> dict:
    method, endpoint = ("POST", "/api/reset") if reset else ("GET", "/api/state")
    return _request_state(client, method, endpoint)
