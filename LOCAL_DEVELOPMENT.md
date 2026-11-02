# Local development without Modal

The assignment's reference evaluation uses Modal testbed images. For development
on a machine without Modal credentials, the Part 1 coding agent also supports a
host-local execution backend:

```bash
uv sync --all-groups
# Configure an OpenAI-compatible endpoint (Ollama is one option):
# OPENAI_BASE_URL=http://localhost:11434/v1
# OPENAI_API_KEY=local
# OPENAI_MODEL=<a model installed in Ollama>
uv run assignment-code-agent --backend local \
  --trajectory artifacts/local-part1-trajectory.json \
  --patch-output artifacts/local-fix.patch
```

The local backend executes shell commands directly on the host, with the task
source checkout as its working directory. It is **not an isolation boundary**:
only use it with a trusted model and a disposable/clean checkout. It does not
build the pinned Modal image and local results are not a substitute for the
official Modal/SWE-bench evaluation. The task's `chess_app` source checkout
must be initialized and at the task's pinned base commit.

For the reference environment and billable evaluation runs, keep using the
default `--backend modal` path and the documented Modal setup.
