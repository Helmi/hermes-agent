"""Regression tests for the 1Password hardline guard.

Verifies that direct secret-retrieval commands (``op read``, ``op item get``)
are unconditionally blocked in the terminal execution path, while the
permitted environment-injection shape (``op run``) passes through.

Uses a fake ``op`` executable and a synthetic sentinel value — no real
1Password vault, credentials, or ``op`` CLI are needed.

Incident context (2026-07-20): a Kanban worker force-loaded the
``1password-op-cli`` skill yet still executed ``op read`` / ``op item get``
and spilled resolved Cloudflare login material into the worker log.  The
skill was advisory text only; no technical guard existed.  These tests
prove the guard added to ``HARDLINE_PATTERNS`` in ``tools/approval.py``
closes that gap.
"""

import os
import stat
import subprocess
import textwrap

import pytest

from tools.approval import (
    HARDLINE_PATTERNS,
    check_all_command_guards,
    detect_hardline_command,
    disable_session_yolo,
    enable_session_yolo,
    reset_current_session_key,
    set_current_session_key,
)

# Synthetic sentinel — a value that must NEVER appear in captured output.
SENTINEL = "SENTINEL_SECRET_abc123_DO_NOT_LEAK"


# -------------------------------------------------------------------------
# Pattern detection — blocked forms
# -------------------------------------------------------------------------

_OP_BLOCK = [
    # op read — any form
    "op read 'op://Private/Cloudflare/email'",
    'op read "op://Private/Cloudflare/password"',
    "op read op://vault/item/field",
    "set +x; op read op://v/i/f",
    "SECRET=$(op read op://v/i/f)",
    "op read op://v/i/f 2>/dev/null",
    # op item get — any form (value-producing by default)
    'op item get "Cloudflare" --vault Private',
    "op item get Cloudflare --format json",
    "op item get Cloudflare --fields username,password",
    "op item get Cloudflare --otp",
    "op item get 1234567890 --reveal",
    "set +x; op item get my-item >/dev/null",
    # Command substitution / chaining
    "echo $(op read op://v/i/f)",
    "op item get x && echo done",
    "op read op://v/i/f | head -1",
    # sudo / env wrappers
    "sudo op read op://v/i/f",
    "env OP_TOKEN=x op item get item",
]

# Commands that must NOT be hardline-blocked.
_OP_ALLOW = [
    # op run — the ONLY permitted secret-consumption shape
    "op run --env-file=.env -- my-command",
    "op run -- python3 script.py",
    "op run --no-masking -- deploy.sh",
    "export PW='op://v/i/f'; op run -- sh -c 'echo $PW'",
    # Metadata-only / informational
    "op --version",
    "op item list",
    "op item list --vault Private",
    "op vault list",
    "op whoami",
    "op account list",
    # Unrelated commands containing "op" as a substring
    "grep op readme.txt",
    "echo hello",
    "python3 -c 'print(1+1)'",
    "ls -la /tmp",
    "cat /dev/null",
    # Word-boundary: "op" inside another word
    "deploy --target production",
    "python3 script.py --option read",
]


@pytest.mark.parametrize("command", _OP_BLOCK)
def test_op_block_detection(command):
    """Every direct-retrieval form must be detected as hardline."""
    is_hl, desc = detect_hardline_command(command)
    assert is_hl, f"expected hardline match for {command!r}"
    assert desc is not None
    assert "1Password" in desc


@pytest.mark.parametrize("command", _OP_ALLOW)
def test_op_allow_detection(command):
    """Permitted and unrelated commands must NOT be hardline-blocked."""
    is_hl, desc = detect_hardline_command(command)
    assert not is_hl, f"expected hardline NOT to match {command!r} (got: {desc})"


# -------------------------------------------------------------------------
# Integration with check_all_command_guards
# -------------------------------------------------------------------------

@pytest.fixture
def clean_session(monkeypatch):
    """Reset session-scoped approval state around each test."""
    monkeypatch.delenv("HERMES_YOLO_MODE", raising=False)
    monkeypatch.delenv("HERMES_INTERACTIVE", raising=False)
    monkeypatch.delenv("HERMES_GATEWAY_SESSION", raising=False)
    monkeypatch.delenv("HERMES_CRON_SESSION", raising=False)
    monkeypatch.delenv("HERMES_EXEC_ASK", raising=False)
    token = set_current_session_key("op_guard_test")
    try:
        disable_session_yolo("op_guard_test")
        yield
    finally:
        disable_session_yolo("op_guard_test")
        reset_current_session_key(token)


def test_check_all_guards_blocks_op_read(clean_session):
    result = check_all_command_guards("op read op://v/i/f", "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]
    assert "1Password" in result["message"]


def test_check_all_guards_blocks_op_item_get(clean_session):
    result = check_all_command_guards('op item get "Cloudflare"', "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert "BLOCKED (hardline)" in result["message"]


def test_check_all_guards_allows_op_run(clean_session):
    result = check_all_command_guards("op run -- my-command", "local")
    assert result["approved"] is True


def test_yolo_cannot_bypass_op_guard(clean_session, monkeypatch):
    """--yolo / HERMES_YOLO_MODE=1 must not bypass the 1Password guard."""
    monkeypatch.setenv("HERMES_YOLO_MODE", "1")
    for cmd in ["op read op://v/i/f", "op item get item"]:
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"yolo leaked 1Password guard on {cmd!r}"
        assert result.get("hardline") is True


def test_session_yolo_cannot_bypass_op_guard(clean_session):
    """Gateway /yolo (session-scoped) must not bypass the 1Password guard."""
    enable_session_yolo("op_guard_test")
    for cmd in ["op read op://v/i/f", "op item get item"]:
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"session yolo leaked on {cmd!r}"
        assert result.get("hardline") is True


def test_approvals_mode_off_cannot_bypass_op_guard(clean_session, monkeypatch):
    """approvals.mode=off must not bypass the 1Password guard."""
    import tools.approval as approval_mod
    monkeypatch.setattr(approval_mod, "_get_approval_mode", lambda: "off")
    for cmd in ["op read op://v/i/f", "op item get item"]:
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"mode=off leaked on {cmd!r}"
        assert result.get("hardline") is True


def test_cron_approve_mode_cannot_bypass_op_guard(clean_session, monkeypatch):
    """Cron approve mode must not bypass the 1Password guard."""
    monkeypatch.setenv("HERMES_CRON_SESSION", "1")
    import tools.approval as approval_mod
    monkeypatch.setattr(approval_mod, "_get_cron_approval_mode", lambda: "approve")
    for cmd in ["op read op://v/i/f", "op item get item"]:
        result = check_all_command_guards(cmd, "local")
        assert result["approved"] is False, f"cron approve leaked on {cmd!r}"
        assert result.get("hardline") is True


def test_container_backends_bypass_op_guard(clean_session):
    """Containerized backends still bypass — they can't touch the host."""
    for env in ("docker", "singularity", "modal", "daytona"):
        for cmd in ["op read op://v/i/f", "op item get item"]:
            result = check_all_command_guards(cmd, env)
            assert result["approved"] is True, f"container {env} should bypass on {cmd!r}"


# -------------------------------------------------------------------------
# Hardline list size guard
# -------------------------------------------------------------------------

def test_hardline_list_stays_small():
    """Adding 1Password patterns must not push the list past 20."""
    assert len(HARDLINE_PATTERNS) <= 20, (
        f"HARDLINE_PATTERNS has {len(HARDLINE_PATTERNS)} entries; "
        "only truly unrecoverable commands belong here."
    )


# -------------------------------------------------------------------------
# Fake-op executable + sentinel integration test
# -------------------------------------------------------------------------

@pytest.fixture
def fake_op(tmp_path):
    """Create a fake ``op`` executable that emits a synthetic sentinel.

    The fake ``op`` prints SENTINEL to stdout for ``read`` and ``item get``
    subcommands (simulating a real secret leak), and prints nothing for
    ``run`` (simulating safe environment injection).
    """
    op_script = tmp_path / "op"
    op_script.write_text(textwrap.dedent(f"""\
        #!/bin/sh
        # Fake op CLI for regression testing — emits a synthetic sentinel.
        case "$1" in
            read)
                echo "{SENTINEL}"
                ;;
            item)
                if [ "$2" = "get" ]; then
                    echo "{SENTINEL}"
                else
                    echo "item-list-metadata-only"
                fi
                ;;
            run)
                # op run injects secrets into the child env; it does NOT
                # print them.  Run the child command (shift past "run" and
                # any flags up to "--").
                shift
                while [ "$1" != "--" ] && [ $# -gt 0 ]; do shift; done
                [ "$1" = "--" ] && shift
                exec "$@"
                ;;
            *)
                echo "op-fake: unknown command $1" >&2
                exit 1
                ;;
        esac
    """))
    op_script.chmod(op_script.stat().st_mode | stat.S_IEXEC)
    return str(op_script)


def test_fake_op_read_emits_sentinel(fake_op):
    """Sanity: the fake op read DOES emit the sentinel (proving the fake works)."""
    result = subprocess.run(
        [fake_op, "read", "op://vault/item/field"],
        capture_output=True, text=True, timeout=5,
    )
    assert SENTINEL in result.stdout, "fake op read should emit sentinel"


def test_fake_op_item_get_emits_sentinel(fake_op):
    """Sanity: the fake op item get DOES emit the sentinel."""
    result = subprocess.run(
        [fake_op, "item", "get", "my-item"],
        capture_output=True, text=True, timeout=5,
    )
    assert SENTINEL in result.stdout, "fake op item get should emit sentinel"


def test_fake_op_run_does_not_emit_sentinel(fake_op):
    """Sanity: the fake op run does NOT emit the sentinel to stdout."""
    result = subprocess.run(
        [fake_op, "run", "--", "echo", "child-output"],
        capture_output=True, text=True, timeout=5,
    )
    assert SENTINEL not in result.stdout, "op run must not leak sentinel to stdout"
    assert "child-output" in result.stdout, "op run should execute the child command"


def test_guard_prevents_op_read_execution(clean_session, fake_op):
    """The hardline guard must block ``op read`` BEFORE it can execute.

    We verify by checking that check_all_command_guards returns a block
    result — the terminal tool would return this error to the agent without
    ever spawning the subprocess.
    """
    cmd = f"{fake_op} read op://vault/item/field"
    result = check_all_command_guards(cmd, "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    # The sentinel must never appear in the guard's response message
    assert SENTINEL not in result.get("message", "")


def test_guard_prevents_op_item_get_execution(clean_session, fake_op):
    """The hardline guard must block ``op item get`` BEFORE it can execute."""
    cmd = f'{fake_op} item get "my-item"'
    result = check_all_command_guards(cmd, "local")
    assert result["approved"] is False
    assert result.get("hardline") is True
    assert SENTINEL not in result.get("message", "")


def test_guard_allows_op_run_execution(clean_session, fake_op):
    """The hardline guard must allow ``op run`` to pass through."""
    cmd = f"{fake_op} run -- echo safe-output"
    result = check_all_command_guards(cmd, "local")
    assert result["approved"] is True
    # Actually execute to prove the sentinel doesn't leak
    proc = subprocess.run(
        [fake_op, "run", "--", "echo", "safe-output"],
        capture_output=True, text=True, timeout=5,
    )
    assert SENTINEL not in proc.stdout
    assert "safe-output" in proc.stdout
