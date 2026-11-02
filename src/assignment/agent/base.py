"""The domain-independent ReAct loop shared by both agents."""

from __future__ import annotations

from copy import deepcopy
import json
import logging
import math
import os
from pathlib import Path
from typing import Any

import yaml
from dotenv import load_dotenv
from openai import OpenAI

from assignment.env import Environment
from assignment.agent.tools import INVOKE_SKILL_TOOL

load_dotenv()
logger = logging.getLogger(__name__)

DEFAULT_COMPACTION_KEEP_RECENT_STEPS = 1
DEFAULT_COMPACTION_MAX_TOKENS = 1_200
MAX_OBSERVATION_CHARS = 10_000

COMPACTION_SYSTEM_PROMPT = """You maintain concise, factual working memory for a software engineering agent.
Summarize the earlier interaction without inventing facts. Preserve the objective,
constraints, relevant files, commands run, edits made, concrete outputs and test
results, failed approaches, blockers, and the most useful next action. Distinguish
verified facts from hypotheses. Do not reproduce long terminal output or reasoning.
Return only the working-memory summary."""


class StepLimitError(Exception):
    """Raised when an agent exhausts its model-call budget."""


def format_tool_output(output: dict[str, Any]) -> str:
    """Format a terminal result as a compact, tagged model observation."""
    elements: list[str] = []
    for key in sorted(output):
        value = output[key]
        if isinstance(value, str) and len(value) > MAX_OBSERVATION_CHARS:
            retained = 4_900
            omitted = len(value) - 2 * retained
            value = f"{value[:retained]}\n[{omitted} characters elided; read a narrower range]\n{value[-retained:]}"
        elements.append(f"<{key}>{value}</{key}>")
    return "\n".join(elements)


def rough_message_tokens(messages: list[dict[str, Any]]) -> int:
    serialized = json.dumps(messages, ensure_ascii=False, separators=(",", ":"))
    return max(1, math.ceil(len(serialized) / 4))


class Agent:
    """Base class for a ReAct agent with pluggable tools."""

    def __init__(
        self, environment: Environment, model: str | None = None,
        logs_save_path: str | None = None, step_limit: int = 100,
        skills_path: str | None = None, auto_stop_environment: bool = True,
        compact_threshold_tokens: int | None = None,
        compaction_keep_recent_steps: int = DEFAULT_COMPACTION_KEEP_RECENT_STEPS,
        compaction_max_tokens: int = DEFAULT_COMPACTION_MAX_TOKENS,
    ):
        self.env = environment
        self.model = model or os.environ.get("OPENAI_MODEL")
        if not self.model:
            raise RuntimeError("OPENAI_MODEL is not set.")
        api_key = os.environ.get("OPENAI_API_KEY")
        if not api_key:
            raise RuntimeError("OPENAI_API_KEY is not set.")
        base_url = os.environ.get("OPENAI_BASE_URL")
        if not base_url:
            raise RuntimeError("OPENAI_BASE_URL is not set.")
        try:
            max_retries = int(os.environ.get("OPENAI_MAX_RETRIES", "5"))
        except ValueError as exc:
            raise RuntimeError("OPENAI_MAX_RETRIES must be an integer.") from exc
        if max_retries < 0:
            raise RuntimeError("OPENAI_MAX_RETRIES must be non-negative.")
        self.client = OpenAI(api_key=api_key, base_url=base_url, max_retries=max_retries)
        self.logs_save_path = logs_save_path
        self.step_limit = step_limit
        self.auto_stop_environment = auto_stop_environment
        if compact_threshold_tokens is not None and compact_threshold_tokens <= 0:
            raise ValueError("compact_threshold_tokens must be positive or None")
        if compaction_keep_recent_steps is not None and compaction_keep_recent_steps < 1:
            raise ValueError("compaction_keep_recent_steps must be at least 1")
        if compaction_max_tokens is not None and compaction_max_tokens < 1:
            raise ValueError("compaction_max_tokens must be positive")
        self.compact_threshold_tokens = compact_threshold_tokens
        self.compaction_keep_recent_steps = (
            DEFAULT_COMPACTION_KEEP_RECENT_STEPS if compaction_keep_recent_steps is None
            else compaction_keep_recent_steps
        )
        self.compaction_max_tokens = (
            DEFAULT_COMPACTION_MAX_TOKENS if compaction_max_tokens is None
            else compaction_max_tokens
        )
        self.system_prompt: str = ""
        self.task_prompt: str = ""
        self.api_prompts: list[list[dict[str, Any]]] = []
        self.api_responses: list[dict[str, Any]] = []
        self.compaction_events: list[dict[str, Any]] = []
        self.tools: list[dict[str, Any]] = []
        self.finished = False
        self.steps_taken = 0
        self.messages: list[dict[str, Any]] = []
        self.working_memory: str | None = None
        self.skills_path = Path(skills_path) if skills_path is not None else None
        self.skills = self.load_skills(self.skills_path) if self.skills_path is not None else {}
        if self.skills:
            self.tools.append(INVOKE_SKILL_TOOL)

    def load_skills(self, skills_path: Path) -> dict[str, dict[str, str]]:
        """Load and validate one Agent Skill file from each child directory."""
        if not skills_path.exists() or not skills_path.is_dir():
            raise ValueError(f"Skills path is not a directory: {skills_path}")
        skills: dict[str, dict[str, str]] = {}
        for directory in sorted(skills_path.iterdir()):
            if not directory.is_dir():
                continue
            skill_file = directory / "SKILL.md"
            if not skill_file.is_file():
                raise ValueError(f"Missing SKILL.md in skill directory: {directory}")
            content = skill_file.read_text(encoding="utf-8")
            lines = content.splitlines()
            if not lines or lines[0].strip() != "---":
                raise ValueError(f"Malformed YAML frontmatter in {skill_file}")
            try:
                end = lines.index("---", 1)
            except ValueError as exc:
                raise ValueError(f"Missing closing frontmatter delimiter in {skill_file}") from exc
            try:
                metadata_obj = yaml.safe_load("\n".join(lines[1:end]))
            except yaml.YAMLError as exc:
                raise ValueError(f"Invalid YAML frontmatter in {skill_file}: {exc}") from exc
            if not isinstance(metadata_obj, dict):
                raise ValueError(f"Frontmatter must be a mapping in {skill_file}")
            name, description = metadata_obj.get("name"), metadata_obj.get("description")
            if not isinstance(name, str) or not name.strip():
                raise ValueError(f"Missing or invalid skill name in {skill_file}")
            if not isinstance(description, str) or not description.strip():
                raise ValueError(f"Missing or invalid skill description in {skill_file}")
            if name in skills:
                raise ValueError(f"Duplicate skill name {name!r} in {skill_file}")
            metadata = f"name: {name}\ndescription: {description}"
            skills[name] = {"metadata": metadata, "content": content}
        return skills

    def query_language_model(self) -> dict[str, Any]:
        messages = self.build_prompt()
        self.api_prompts.append(deepcopy(messages))
        step_number = self.steps_taken + 1
        print(f"[agent] step {step_number}/{self.step_limit}: requesting action", flush=True)
        try:
            response = self.client.chat.completions.create(
                model=self.model, messages=messages, tools=self.tools,
                reasoning_effort="medium", max_completion_tokens=4096,
            )
        except Exception as exc:
            print(f"[agent] step {step_number}: model request failed after retries ({type(exc).__name__}: {exc})", flush=True)
            raise
        self.api_responses.append(response.model_dump(mode="json"))
        self.steps_taken += 1
        message = self.process_response(response)
        names = [c.get("function", {}).get("name", "unknown") for c in message.get("tool_calls", []) if isinstance(c, dict)]
        if names:
            print(f"[agent] step {step_number}: tool call(s): {', '.join(names)}", flush=True)
        else:
            print(f"[agent] step {step_number}: response contained no parsed tool call", flush=True)
        return message

    def process_response(self, response: Any) -> dict[str, Any]:
        return response.choices[0].message.model_dump(exclude_none=True)

    def build_prompt(self) -> list[dict[str, Any]]:
        prompt = [{"role": "system", "content": self.system_prompt},
                  {"role": "user", "content": self.task_prompt}]
        if self.working_memory:
            prompt.append({"role": "user", "content": "<working_memory>\n" + self.working_memory + "\n</working_memory>"})
        prompt.extend(deepcopy(self.messages))
        return prompt

    def estimate_active_prompt_tokens(self) -> int:
        current = self.build_prompt()
        rough = rough_message_tokens(current)
        if not self.api_prompts or not self.api_responses:
            return rough
        usage = self.api_responses[-1].get("usage") or {}
        actual = usage.get("prompt_tokens")
        if not isinstance(actual, int):
            return rough
        previous = rough_message_tokens(self.api_prompts[-1])
        return actual + max(0, rough - previous)

    @property
    def compaction_enabled(self) -> bool:
        return self.compact_threshold_tokens is not None

    def compact_context(self):
        prompt_before = self.build_prompt()
        # Retain the most recent complete assistant action and its tool results.
        assistant_positions = [i for i, msg in enumerate(self.messages) if msg.get("role") == "assistant"]
        keep_from = len(self.messages)
        if assistant_positions:
            target = max(0, len(assistant_positions) - self.compaction_keep_recent_steps)
            keep_from = assistant_positions[target]
            # A tool response is linked to the preceding assistant action; keep
            # the whole suffix from that action onward.
        old_messages = self.messages[:keep_from]
        recent_messages = self.messages[keep_from:]
        if not old_messages:
            compaction_prompt = [
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {"role": "user", "content": json.dumps(prompt_before, ensure_ascii=False)},
            ]
            # No eligible old transcript: return an empty summary rather than
            # discarding any live context.
            self.working_memory = self.working_memory or ""
        else:
            prior = self.working_memory or ""
            compaction_prompt = [
                {"role": "system", "content": COMPACTION_SYSTEM_PROMPT},
                {
                    "role": "user",
                    "content": (
                        "Original agent system instructions (preserve these constraints):\n"
                        + self.system_prompt
                        + "\n\nOriginal task (the objective to retain):\n"
                        + self.task_prompt
                        + "\n\nExisting working memory:\n"
                        + prior
                        + "\n\nEarlier transcript to summarize:\n"
                        + json.dumps(old_messages, ensure_ascii=False)
                    ),
                },
            ]
        ### Do not modify this section ###
        compaction_response = self.client.chat.completions.create(
            model=self.model,
            messages=compaction_prompt,
            reasoning_effort="medium",
            max_completion_tokens=self.compaction_max_tokens,
        )
        ##################################
        summary = compaction_response.choices[0].message.content or ""
        if old_messages and summary.strip():
            self.working_memory = summary.strip()
            self.messages = recent_messages
        ### Do not modify this section ###
        return compaction_prompt, compaction_response.model_dump(mode="json")
        ##################################

    def maybe_compact_context(self) -> bool:
        if not self.compaction_enabled or self.estimate_active_prompt_tokens() < self.compact_threshold_tokens:
            return False
        before = deepcopy(self.build_prompt())
        if len([m for m in before if m.get("role") == "assistant"]) <= self.compaction_keep_recent_steps:
            return False
        compaction_prompt, response = self.compact_context()
        after = deepcopy(self.build_prompt())
        self.compaction_events.append({
            "step": self.steps_taken,
            "estimated_tokens_before": rough_message_tokens(before),
            "estimated_tokens_after": rough_message_tokens(after),
            "active_prompt_before": before,
            "compaction_prompt": compaction_prompt,
            "compaction_response": response,
        })
        return True

    def run(self) -> None:
        try:
            while not self.finished:
                # Do not spend a compaction request when the step budget means
                # there cannot be another action request.
                if self.steps_taken >= self.step_limit:
                    raise StepLimitError(f"Agent exceeded step limit ({self.step_limit}).")
                self.maybe_compact_context()
                action = self.query_language_model()
                self.messages.append(action)
                calls = action.get("tool_calls") or []
                if not calls:
                    self.finished = True
                    break
                observations = self.execute_tool_calls(calls)
                self.messages.extend(observations)
        finally:
            if self.logs_save_path:
                path = Path(self.logs_save_path)
                path.parent.mkdir(parents=True, exist_ok=True)
                path.write_text(json.dumps({
                    "prompts": self.api_prompts, "responses": self.api_responses,
                    "compactions": self.compaction_events,
                }, indent=2))
            if self.auto_stop_environment:
                stop = getattr(self.env, "stop", None)
                if callable(stop):
                    stop()

    def execute_tool_calls(self, tool_calls: list[dict[str, Any]]) -> list[dict[str, str]]:
        raise NotImplementedError
