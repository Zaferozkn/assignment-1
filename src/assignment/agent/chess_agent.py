"""The Part 3 chess agent: reuse the shared loop with chess tools."""

from __future__ import annotations
import json
from typing import Any
import httpx

from assignment.agent.base import Agent
from assignment.agent.chess_tools import (
    _game_state, _invoke_skill, _play_move, CHESS_PORT,
    _run_python, _simulate_move,
)
from assignment.agent.tools import (
    INVOKE_SKILL_TOOL, PLAY_MOVE_TOOL, RUN_PYTHON_TOOL, SIMULATE_MOVE_TOOL,
)
from assignment.prompts import (
    CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE,
    CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE,
    PROGRAMMATIC_CHESS_PROMPT,
)
from typing import TYPE_CHECKING

if TYPE_CHECKING:
    from assignment.env import Environment


def format_chess_state(state: dict[str, Any], *, include_legal_moves: bool = True) -> str:
    squares = state.get("squares", {})
    board_lines = ["    a b c d e f g h"]
    for rank in range(8, 0, -1):
        pieces = [squares.get(f"{file}{rank}", ".") for file in "abcdefgh"]
        board_lines.append(f"{rank} | {' '.join(pieces)} | {rank}")
    board_lines.append("    a b c d e f g h")
    recent_history = state.get("history", [])[-8:]
    history = " ".join(f"{item.get('ply', '?')}:{item.get('san', '?')}" for item in recent_history) or "(none)"
    legal_moves = " ".join(state.get("legal_moves", [])) or "(none)"
    observation = (
        "<chess_state>\n"
        f"status: {state.get('status', 'unknown')}\n"
        f"turn: {state.get('turn', 'unknown')}\n"
        f"in_check: {state.get('in_check', False)}\n"
        f"game_over: {state.get('game_over', False)}\n"
        f"fen: {state.get('fen', '')}\n"
        f"human_move: {state.get('human_move') or '(none)'}\n"
        f"engine_move: {state.get('engine_move') or '(none)'}\n"
        "board:\n" + "\n".join(board_lines) + "\n"
        f"recent_history: {history}\n"
    )
    if include_legal_moves:
        observation += f"legal_moves: {legal_moves}\n"
    return observation + "</chess_state>"


class ChessAgent(Agent):
    """An agent that plays White against the server's deterministic Black bot."""

    def __init__(
        self, environment: Environment, model: str | None = None,
        logs_save_path: str | None = None, step_limit: int = 200,
        skills_path: str | None = None, auto_stop_environment: bool = True,
        http_client: Any | None = None, reset_game: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = None,
        compaction_max_tokens: int = None, programmatic_tools: bool = False,
        python_sandbox_port: int = CHESS_PORT, include_legal_moves: bool = True,
    ):
        super().__init__(
            environment=environment, model=model, logs_save_path=logs_save_path,
            step_limit=step_limit, skills_path=skills_path,
            auto_stop_environment=auto_stop_environment,
            compact_threshold_tokens=compact_threshold_tokens,
            compaction_keep_recent_steps=compaction_keep_recent_steps,
            compaction_max_tokens=compaction_max_tokens,
        )
        self.tools.append(PLAY_MOVE_TOOL)
        if programmatic_tools:
            self.tools.extend([SIMULATE_MOVE_TOOL, RUN_PYTHON_TOOL])
        if self.skills:
            # The base class already registered invoke_skill for a loaded catalog.
            pass
        self.python_sandbox_port = python_sandbox_port
        self.include_legal_moves = include_legal_moves
        if http_client is None:
            server_url = getattr(environment, "server_url", "")
            if not server_url:
                raise ValueError("ChessAgent needs an environment with server_url or an http_client.")
            http_client = httpx.Client(base_url=server_url, timeout=20)
        self.chess_client = http_client
        initial_state = _game_state(self.chess_client, reset=reset_game)
        self.last_state = initial_state
        self.finished = bool(initial_state.get("game_over"))
        template = CHESS_AGENT_SYSTEM_PROMPT_TEMPLATE if include_legal_moves else CHESS_AGENT_NO_LEGAL_MOVES_PROMPT_TEMPLATE
        self.system_prompt = template.render()
        if programmatic_tools:
            self.system_prompt += "\n\n" + PROGRAMMATIC_CHESS_PROMPT
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            self.system_prompt += (
                "\n\nReusable skills are available. Call invoke_skill with a skill name "
                "to load its complete instructions.\n<skills>\n" + catalog + "\n</skills>"
            )
        opening = "Choose one move from legal_moves and call play_move." if include_legal_moves else "Infer a legal move from the board and FEN, then call play_move."
        self.task_prompt = f"Play this game as White. {opening}\n\n{self.format_state(initial_state)}"

    def format_state(self, state: dict[str, Any]) -> str:
        return format_chess_state(state, include_legal_moves=self.include_legal_moves)

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        observations: list[dict[str, str]] = []
        live_move_used = False
        for call in tool_calls:
            call_id = str(call.get("id", ""))
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments", "{}")
            if not isinstance(raw, str):
                raw = json.dumps(raw)
            try:
                if name == "play_move":
                    if live_move_used:
                        raise ValueError("Only one live move may be played per assistant turn; inspect the updated position first.")
                    live_move_used = True
                    result = _play_move(self.chess_client, raw)
                    if result.startswith("<chess_error>"):
                        content = result
                    else:
                        state = json.loads(result)
                        self.last_state = state
                        self.finished = bool(state.get("game_over"))
                        content = self.format_state(state)
                elif name == "simulate_move" and SIMULATE_MOVE_TOOL in self.tools:
                    content = _simulate_move(self.chess_client, raw)
                elif name == "run_python" and RUN_PYTHON_TOOL in self.tools:
                    if live_move_used:
                        raise ValueError("Only one live-board action may run per assistant turn; inspect the updated position first.")
                    # A Python snippet may call play_move, so it counts as the
                    # single state-changing action allowed in this batch.
                    live_move_used = True
                    result = _run_python(self.env, self.python_sandbox_port, raw)
                    # A snippet may have committed a live move before returning
                    # an error. Always refresh the authoritative board after the
                    # sandbox invocation so the next model turn cannot act on a
                    # stale position.
                    state = _game_state(self.chess_client, reset=False)
                    self.last_state = state
                    self.finished = bool(state.get("game_over"))
                    content = result + "\n" + self.format_state(state)
                elif name == "invoke_skill" and self.skills:
                    content = _invoke_skill(self.skills, raw)
                else:
                    raise ValueError(f"Unknown or unavailable tool: {name!r}")
            except Exception as exc:
                content = f"<chess_error>{type(exc).__name__}: {exc}</chess_error>"
            observations.append({"role": "tool", "tool_call_id": call_id, "content": content})
        return observations
