"""Focused contract tests for scripts/hermes_update_supervisor.py."""
from __future__ import annotations

import importlib.util
import json
import sqlite3
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


def origin():
    return {"platform": "discord", "chat_id": "chat", "thread_id": "thread", "parent_chat_id": "parent",
            "profile": "hermie", "session_id": "session"}


def args(source, home, state, **overrides):
    values = dict(source=str(source), home=str(home), state_dir=str(state), activate=True, expected_sha=None,
                  profile=["hermie"], update_profile="hermie", update_timeout_s=9, restart_timeout_s=7,
                  origin_platform="discord", origin_chat_id="chat", origin_thread_id="thread",
                  origin_parent_chat_id="parent", origin_profile="hermie", origin_session_id="session")
    values.update(overrides)
    return Namespace(**values)


def receipt(module, state, source, home, *, status="queued", delivery="pending"):
    path = state / ("a" * 32)
    module.atomic_write(path, {"schema": 2, "run_id": "a" * 32, "status": status, "source": str(source),
        "home": str(home), "expected_sha": "b" * 40, "profiles": ["hermie"], "update_profile": "hermie",
        "update_timeout_s": 9, "restart_timeout_s": 7, "origin": origin(), "delivery_state": delivery})
    return path


def install_native_session(home):
    db = sqlite3.connect(home / "state.db")
    db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, chat_id TEXT, thread_id TEXT, profile_name TEXT, origin_json TEXT)")
    db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
               ("session", "discord", "chat", "thread", "hermie", json.dumps({"parent_chat_id": "parent"})))
    db.commit()
    db.close()


def test_exact_origin_requires_id_and_every_native_route_field(supervisor, layout):
    _, home, _ = layout
    install_native_session(home)
    assert supervisor.exact_origin_exists(home, origin())
    for field in origin():
        altered = origin()
        altered[field] = "wrong"
        assert not supervisor.exact_origin_exists(home, altered), field


def test_start_passes_inherited_lock_and_explicit_scopes(supervisor, layout, monkeypatch, capsys):
    source, home, state = layout
    started = []
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: True)
    monkeypatch.setattr(supervisor, "freeze_target", lambda *_: "b" * 40)
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: started.append((a, kw)) or SimpleNamespace(pid=4321))
    assert supervisor.start(args(source, home, state)) == 0
    result = json.loads(capsys.readouterr().out)
    argv, kwargs = started[0]
    assert "--lock-fd" in argv[0]
    assert kwargs["pass_fds"] and kwargs["start_new_session"] is True
    assert supervisor.read_state(Path(result["receipt"]))["expected_sha"] == "b" * 40


def test_start_rejects_another_active_transaction(supervisor, layout, monkeypatch):
    source, home, state = layout
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: True)
    monkeypatch.setattr(supervisor, "freeze_target", lambda *_: "b" * 40)
    state.mkdir()
    lock = (state / ".lock").open("a+")
    supervisor.fcntl.flock(lock.fileno(), supervisor.fcntl.LOCK_EX | supervisor.fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="active"):
            supervisor.start(args(source, home, state))
    finally:
        lock.close()


def test_freeze_target_rejects_dirty_or_nonmatching_source(supervisor, layout, monkeypatch):
    source, _, _ = layout
    monkeypatch.setattr(supervisor, "git", lambda *_: "dirty" if _[-1] == "--porcelain" else "a" * 40)
    with pytest.raises(ValueError, match="dirty"):
        supervisor.freeze_target(source, None)
    def clean_git(_source, *git_args):
        if git_args[-1] == "--porcelain":
            return ""
        if git_args[-1].endswith("^{commit}"):
            return "b" * 40
        return "a" * 40
    monkeypatch.setattr(supervisor, "git", clean_git)
    with pytest.raises(ValueError, match="HEAD"):
        supervisor.freeze_target(source, "b" * 40)


def test_profile_home_scopes_each_native_invocation(supervisor, layout):
    _, home, _ = layout
    assert supervisor.profile_home(home, "hermie", "hermie") == home
    assert supervisor.profile_home(home, "hermie", "mia") == home.parent / "mia"


def test_worker_scopes_update_restart_and_requires_runtime_sha(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home)
    calls = []
    monkeypatch.setattr(supervisor, "git", lambda *_: "" if _[-1] == "--porcelain" else "b" * 40)
    monkeypatch.setattr(supervisor, "run_command", lambda argv, **kw: calls.append((argv, kw)) or (0, ""))
    monkeypatch.setattr(supervisor, "health_record", lambda h, sha: {"code_sha": sha, "home": str(h)})
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: False)
    assert supervisor.worker(path) == 0
    assert calls[0][0][1:4] == ["-p", "hermie", "update"]
    assert calls[1][0][1:5] == ["-p", "hermie", "gateway", "restart"]
    assert calls[0][1]["home"] == home
    assert supervisor.read_state(path)["gateway_health"]["hermie"]["code_sha"] == "b" * 40


def test_health_record_refuses_stale_wrong_sha_or_wrong_home(supervisor, layout):
    _, home, _ = layout
    (home / "gateway_state.json").write_text(json.dumps({"gateway_state": "running", "hermes_home": str(home),
                                                            "code_sha": "a" * 40}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="does not equal"):
        supervisor.health_record(home, "b" * 40)
    (home / "gateway_state.json").write_text(json.dumps({"gateway_state": "draining", "hermes_home": str(home),
                                                            "code_sha": "b" * 40}), encoding="utf-8")
    with pytest.raises(RuntimeError, match="not ready"):
        supervisor.health_record(home, "b" * 40)


def test_delivery_rechecks_exact_route_and_parses_native_exit(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home, status="failed")
    calls = []
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: True)
    monkeypatch.setattr(supervisor, "run_command", lambda argv, **kw: calls.append(argv) or (0, '{"success": true}'))
    supervisor.deliver_completion(path, path.with_suffix(".log"))
    assert supervisor.read_state(path)["delivery_state"] == "sent"
    assert supervisor.read_state(path)["delivery_wake"] == "unsupported"
    assert calls[0][calls[0].index("--to") + 1] == "discord:chat:thread"


def test_delivery_rejects_exit_zero_without_native_json_success(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home, status="failed")
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: True)
    monkeypatch.setattr(supervisor, "run_command", lambda *a, **kw: (0, '{"success": false}'))
    supervisor.deliver_completion(path, path.with_suffix(".log"))
    assert supervisor.read_state(path)["delivery_state"] == "failed"


def test_delivery_refuses_when_route_changes(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home, status="failed")
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: False)
    monkeypatch.setattr(supervisor, "run_command", lambda *a, **kw: pytest.fail("must not send"))
    supervisor.deliver_completion(path, path.with_suffix(".log"))
    assert supervisor.read_state(path)["delivery_state"] == "refused"


def test_recovery_marks_orphan_and_never_replays_update(supervisor, layout, monkeypatch):
    source, home, state = layout
    path = receipt(supervisor, state, source, home)
    monkeypatch.setattr(supervisor, "is_live_pid", lambda _: False)
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: False)
    assert supervisor.recover(Namespace(receipt=str(path))) == 1
    data = supervisor.read_state(path)
    assert data["status"] == "orphaned"
    assert "never replayed" in data["error"]
