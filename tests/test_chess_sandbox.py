"""Billable integration test for the chess server hosted in a Modal sandbox."""

import modal
import pytest

from assignment.chess_sandbox import ChessSandbox, IllegalMove


pytestmark = pytest.mark.modal

def running_sandboxes(app: modal.App) -> list[modal.Sandbox]:
    return [sandbox for sandbox in modal.Sandbox.list(app_id=app.app_id) if sandbox.poll() is None]


def test_chess_app_is_playable_through_modal_tunnel():
    # This public test verifies the tunnel and API infrastructure without
    # depending on the intentionally broken engine search. The private grader
    # applies a candidate patch before exercising successful moves.
    with ChessSandbox() as sandbox:
        own_sandbox = sandbox.deployment._sandbox
        assert own_sandbox is not None and own_sandbox.poll() is None

        current = sandbox.state()
        assert current["turn"] == "white"
        assert current["history"] == []

        with pytest.raises(IllegalMove):
            sandbox.play("e2e5")

        assert sandbox.reset()["history"] == []

    assert own_sandbox.poll() is not None, "Chess sandbox was not terminated after the test."
