"""Focused contract tests for scripts/hermes_update_supervisor.py."""
from __future__ import annotations

import importlib.util
import json
from argparse import Namespace
from pathlib import Path
from types import SimpleNamespace

import pytest


@pytest.fixture
def supervisor():
    script = Path(__file__).resolve().parents[2] / "scripts" / "hermes_update_supervisor.py"
    spec = importlib.util.spec_from_file_location("hermes_update_supervisor", script)
    module = importlib.util.module_from_spec(spec)
    assert spec.loader
    spec.loader.exec_module(module)
    return module


@pytest.fixture
def layout(tmp_path):
    source, home, state = tmp_path / "source", tmp_path / "home", tmp_path / "state"
    launcher = source / ".hermes" / "bin" / "hermes"
    launcher.parent.mkdir(parents=True)
    launcher.write_text("#!/bin/sh\n", encoding="utf-8")
    home.mkdir()
    return source, home, state


def receipt(module, state, source, home, *, status="queued", delivery="pending"):
    path = state / ("a" * 32)
    module.atomic_write(path, {
        "schema": 1, "run_id": "a" * 32, "status": status, "source": str(source), "home": str(home),
        "profiles": ["hermie", "mia", "smoerrebroed"], "update_timeout_s": 9, "restart_timeout_s": 7,
        "origin": {**module.DEFAULT_ORIGIN, "session_id": "real-session-id"}, "delivery_state": delivery,
    })
    return path


def test_start_hands_off_immediately_without_running_update(supervisor, layout, monkeypatch, capsys):
    source, home, state = layout
    monkeypatch.setattr(supervisor, "session_exists", lambda *_: True)
    started = []
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: started.append((a, kw)) or SimpleNamespace(pid=4321))
    args = Namespace(source=str(source), home=str(home), state_dir=str(state), origin_session_id="real-session-id", activate=True,
                     profile=["hermie", "mia", "smoerrebroed"], update_timeout_s=9, restart_timeout_s=7)
    assert supervisor.start(args) == 0
    result = json.loads(capsys.readouterr().out)
    assert result["worker_pid"] == 4321
    assert started[0][1]["start_new_session"] is True
    assert supervisor.read_state(Path(result["receipt"]))["status"] == "queued"


def test_start_refuses_missing_origin_session(supervisor, layout, monkeypatch):
    source, home, state = layout
    monkeypatch.setattr(supervisor, "session_exists", lambda *_: False)
    args = Namespace(source=str(source), home=str(home), state_dir=str(state), origin_session_id="not-a-real-session", activate=True,
                     profile=["hermie"], update_timeout_s=9, restart_timeout_s=7)
    with pytest.raises(ValueError, match="origin session"):
        supervisor.start(args)


def test_start_discovers_only_one_native_origin_session(supervisor, layout, monkeypatch):
    source, home, state = layout
    monkeypatch.setattr(supervisor, "discover_origin_sessions", lambda _: ["discovered-session"])
    monkeypatch.setattr(supervisor, "session_exists", lambda _source, _home, sid: sid == "discovered-session")
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: SimpleNamespace(pid=4321))
    args = Namespace(source=str(source), home=str(home), state_dir=str(state), origin_session_id=None, activate=True,
                     profile=["hermie"], update_timeout_s=9, restart_timeout_s=7)
    assert supervisor.start(args) == 0
    receipt = next(state.glob("*.json"))
    assert supervisor.read_state(receipt)["origin"]["session_id"] == "discovered-session"


def test_start_refuses_ambiguous_origin_session_lookup(supervisor, layout, monkeypatch):
    source, home, state = layout
    monkeypatch.setattr(supervisor, "discover_origin_sessions", lambda _: ["old", "current"])
    args = Namespace(source=str(source), home=str(home), state_dir=str(state), origin_session_id=None, activate=True,
                     profile=["hermie"], update_timeout_s=9, restart_timeout_s=7)
    with pytest.raises(ValueError, match="not unique"):
        supervisor.start(args)


def test_start_requires_explicit_activation(supervisor, layout, monkeypatch):
    source, home, state = layout
    monkeypatch.setattr(supervisor, "session_exists", lambda *_: True)
    args = Namespace(source=str(source), home=str(home), state_dir=str(state), origin_session_id="real-session-id", activate=False,
                     profile=["hermie"], update_timeout_s=9, restart_timeout_s=7)
    with pytest.raises(ValueError, match="--activate"):
        supervisor.start(args)


def test_worker_uses_native_update_then_all_profile_restarts(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home)
    calls = []
    monkeypatch.setattr(supervisor, "session_exists", lambda *_: True)
    monkeypatch.setattr(supervisor, "run_command", lambda argv, **kw: calls.append(argv) or (0, ""))
    assert supervisor.worker(path) == 0
    assert calls[0][1:] == ["update", "--channel", "main", "--no-backup", "--yes", "--no-gateway-restart"]
    assert [call[-2:] for call in calls[1:4]] == [["gateway", "restart"]] * 3
    assert supervisor.read_state(path)["status"] == "succeeded"


def test_failed_restart_is_reported_and_routed_to_exact_origin(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home)
    calls = []
    monkeypatch.setattr(supervisor, "session_exists", lambda *_: True)
    def run(argv, **kw):
        calls.append(argv)
        return (5 if argv[-2:] == ["gateway", "restart"] else 0, "")
    monkeypatch.setattr(supervisor, "run_command", run)
    assert supervisor.worker(path) == 1
    data = supervisor.read_state(path)
    assert data["status"] == "failed"
    assert data["delivery_state"] == "sent"
    send = calls[-1]
    assert send[send.index("--to") + 1] == "discord:1565697700836540429:1565697700836540429"


def test_missing_origin_at_completion_refuses_without_send(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home, status="failed")
    monkeypatch.setattr(supervisor, "session_exists", lambda *_: False)
    monkeypatch.setattr(supervisor, "run_command", lambda *a, **kw: pytest.fail("must not send"))
    supervisor.deliver_completion(path, source, home, path.with_suffix(".log"))
    assert supervisor.read_state(path)["delivery_state"] == "refused"


def test_attempted_delivery_never_retries_after_crash_window(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home, status="succeeded", delivery="attempted")
    monkeypatch.setattr(supervisor, "run_command", lambda *a, **kw: pytest.fail("duplicate send"))
    supervisor.deliver_completion(path, source, home, path.with_suffix(".log"))
    assert supervisor.read_state(path)["delivery_state"] == "attempted"


def test_recover_refuses_while_worker_is_live(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home)
    supervisor.update_state(path, worker_pid=123)
    monkeypatch.setattr(supervisor, "is_live_pid", lambda _: True)
    with pytest.raises(RuntimeError, match="still live"):
        supervisor.recover(Namespace(receipt=str(path)))


def test_recover_orphan_marks_failure_without_replaying_update(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home)
    monkeypatch.setattr(supervisor, "is_live_pid", lambda _: False)
    monkeypatch.setattr(supervisor, "session_exists", lambda *_: False)
    assert supervisor.recover(Namespace(receipt=str(path))) == 1
    assert supervisor.read_state(path)["status"] == "failed"
    assert supervisor.read_state(path)["delivery_state"] == "refused"
