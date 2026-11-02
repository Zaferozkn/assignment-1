"""The Part 1 coding agent: fix a software issue and submit a git patch."""

from __future__ import annotations
import json
from typing import Any

from assignment.agent.base import DEFAULT_COMPACTION_KEEP_RECENT_STEPS, DEFAULT_COMPACTION_MAX_TOKENS, Agent, format_tool_output
from assignment.agent.tools import EXECUTE_TOOL, SEND_MESSAGE_TOOL
from assignment.env import Environment


class CodeAgent(Agent):
    """An agent that fixes a software issue and submits a git patch."""

    def __init__(self, task: str, environment: Environment, model: str | None = None,
                 logs_save_path: str | None = None, step_limit: int = 100,
                 skills_path: str | None = None, auto_stop_environment: bool = True,
                 compact_threshold_tokens: int | None = None,
                 compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
                 compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS):
        super().__init__(environment, model, logs_save_path, step_limit, skills_path,
                         auto_stop_environment, compact_threshold_tokens,
                         compaction_keep_recent_steps, compaction_max_tokens)
        self.task = task
        self.submitted_patch = ""
        self.tools.extend([EXECUTE_TOOL, SEND_MESSAGE_TOOL])
        system = (
            "You are a careful software engineering agent working in a remote terminal. "
            "Inspect the repository, reproduce the issue, make the smallest correct change, "
            "run relevant tests, and report verified results. Use execute for terminal commands. "
            "Do not claim tests passed unless you ran them.\n"
            "<system_information>\n"
            + json.dumps({
                "machine": environment.machine, "release": environment.release,
                "system": environment.system, "version": environment.version,
            }, indent=2)
            + "\n</system_information>"
        )
        if self.skills:
            catalog = "\n".join(skill["metadata"] for skill in self.skills.values())
            system += (
                "\n\nAvailable skills are listed below. Call invoke_skill with a skill name "
                "to load its full instructions before following that workflow.\n"
                "<skills>\n" + catalog + "\n</skills>"
            )
        self.system_prompt = system
        self.task_prompt = task

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        observations: list[dict[str, str]] = []
        for call in tool_calls:
            call_id = str(call.get("id", ""))
            function = call.get("function") or {}
            name = function.get("name")
            raw = function.get("arguments", "{}")
            try:
                args = json.loads(raw) if isinstance(raw, str) else raw
                if not isinstance(args, dict):
                    raise ValueError("Tool arguments must be a JSON object.")
                if name == "execute":
                    if "command" not in args:
                        raise ValueError("Missing required argument: command")
                    allowed = {k: args[k] for k in ("timeout", "cwd", "env", "shell") if k in args}
                    result = self.env.execute(args["command"], **allowed)
                    content = format_tool_output(result)
                elif name == "send_message":
                    summary = args.get("summary")
                    if not isinstance(summary, str):
                        raise ValueError("send_message requires a string summary.")
                    content = summary
                    self.finished = True
                elif name == "invoke_skill":
                    skill_name = args.get("name")
                    if not isinstance(skill_name, str) or skill_name not in self.skills:
                        raise ValueError(f"Unknown skill: {skill_name!r}")
                    content = self.skills[skill_name]["content"]
                else:
                    raise ValueError(f"Unknown tool: {name!r}")
            except Exception as exc:
                content = f"<tool_error>{type(exc).__name__}: {exc}</tool_error>"
            observations.append({"role": "tool", "tool_call_id": call_id, "content": content})
        return observations
