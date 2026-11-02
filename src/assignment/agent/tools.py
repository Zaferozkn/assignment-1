"""Tool definitions exposed to the model in OpenAI function-calling format."""

EXECUTE_TOOL = {
    "type": "function",
    "function": {
        "name": "execute",
        "description": "Run a bash command in the sandbox and return stdout, stderr, and exit code.",
        "strict": False,
        "parameters": {
            "type": "object",
            "properties": {
                "command": {"anyOf": [{"type": "string"}, {"type": "array", "items": {"type": "string"}}]},
                "shell": {"type": ["boolean", "null"]},
                "cwd": {"type": ["string", "null"]},
                "timeout": {"type": ["number", "null"]},
                "env": {"type": ["object", "null"], "additionalProperties": {"type": "string"}},
            },
            "required": ["command"], "additionalProperties": False,
        },
    },
}
SEND_MESSAGE_TOOL = {
    "type": "function",
    "function": {"name": "send_message", "description": "Send a final message to the user.",
                 "strict": True, "parameters": {"type": "object",
                 "properties": {"summary": {"type": "string"}},
                 "required": ["summary"], "additionalProperties": False}},
}
INVOKE_SKILL_TOOL = {
    "type": "function",
    "function": {"name": "invoke_skill", "description": "Load a skill's complete instructions by name.",
                 "strict": True, "parameters": {"type": "object",
                 "properties": {"name": {"type": "string"}},
                 "required": ["name"], "additionalProperties": False}},
}
PLAY_MOVE_TOOL = {
    "type": "function",
    "function": {"name": "play_move",
                 "description": "Play one legal chess move in UCI notation, such as e2e4 or e7e8q.",
                 "strict": True, "parameters": {"type": "object",
                 "properties": {"move": {"type": "string", "description": "A chess move in UCI notation, e.g. e2e4 or e7e8q."}},
                 "required": ["move"], "additionalProperties": False}},
}
SIMULATE_MOVE_TOOL = {
    "type": "function",
    "function": {"name": "simulate_move", "description": "Simulate a position from a six-field FEN, optionally applying one UCI move without changing the live game.",
                 "strict": True, "parameters": {"type": "object",
                 "properties": {"fen": {"type": "string"}, "move": {"type": ["string", "null"]}},
                 "required": ["fen", "move"], "additionalProperties": False}},
}
RUN_PYTHON_TOOL = {
    "type": "function",
    "function": {"name": "run_python", "description": "Run Python code in the sandbox with simulate_move and play_move helper functions.",
                 "strict": True, "parameters": {"type": "object",
                 "properties": {"code": {"type": "string"}},
                 "required": ["code"], "additionalProperties": False}},
}
