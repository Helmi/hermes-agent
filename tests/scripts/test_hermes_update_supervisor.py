"""Contract tests for the detached Hermes update supervisor."""
from __future__ import annotations

import fcntl
import importlib.util
import json
import os
import sqlite3
import subprocess
import sys
import time
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
    values = dict(source=str(source), home=str(home), state_dir=str(state), activate=True, expected_sha="b" * 40,
                  profile=["hermie"], update_profile="hermie", update_timeout_s=2, restart_timeout_s=2,
                  origin_platform="discord", origin_chat_id="chat", origin_thread_id="thread",
                  origin_parent_chat_id="parent", origin_profile="hermie", origin_session_id="session")
    values.update(overrides)
    return Namespace(**values)


def receipt(module, state, source, home, *, status="queued", delivery="pending", profiles=None):
    state.mkdir(exist_ok=True)
    path = state / ("a" * 32 + ".json")
    module.atomic_write(path, {"schema": 3, "run_id": "a" * 32, "status": status, "source": str(source),
        "home": str(home), "pre_sha": "a" * 40, "expected_sha": "b" * 40,
        "profiles": profiles or ["hermie"], "update_profile": "hermie", "update_timeout_s": 2,
        "restart_timeout_s": 2, "origin": origin(), "delivery_state": delivery})
    return path


def install_native_session(home):
    db = sqlite3.connect(home / "state.db")
    db.execute("CREATE TABLE sessions (id TEXT PRIMARY KEY, source TEXT, chat_id TEXT, thread_id TEXT, profile_name TEXT, origin_json TEXT)")
    db.execute("INSERT INTO sessions VALUES (?, ?, ?, ?, ?, ?)",
               ("session", "discord", "chat", "thread", "hermie", json.dumps({"parent_chat_id": "parent"})))
    db.commit(); db.close()


def test_exact_origin_requires_id_and_every_native_route_field(supervisor, layout):
    _, home, _ = layout; install_native_session(home)
    assert supervisor.exact_origin_exists(home, origin())
    for field in origin():
        altered = origin(); altered[field] = "wrong"
        assert not supervisor.exact_origin_exists(home, altered), field


def test_freeze_target_permits_old_to_new_only_when_local_and_remote_match(supervisor, layout, monkeypatch):
    source, _, _ = layout
    def fake_git(_source, *parts):
        if parts[-1] == "--porcelain": return ""
        if parts[:3] == ("remote", "get-url", "fork"): return "fork-url"
        if parts[-1].endswith("^{commit}"): return "b" * 40
        return "a" * 40
    monkeypatch.setattr(supervisor, "git", fake_git)
    monkeypatch.setattr(supervisor, "fork_main_sha", lambda _: "b" * 40)
    assert supervisor.freeze_target(source, "b" * 40) == {"pre_sha": "a" * 40, "expected_sha": "b" * 40}
    monkeypatch.setattr(supervisor, "fork_main_sha", lambda _: "c" * 40)
    with pytest.raises(ValueError, match="fork main"):
        supervisor.freeze_target(source, "b" * 40)


def test_start_has_no_parent_receipt_mutation_after_real_spawn(supervisor, layout, monkeypatch, capsys):
    source, home, state = layout; started = []
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: True)
    monkeypatch.setattr(supervisor, "freeze_target", lambda *_: {"pre_sha": "a" * 40, "expected_sha": "b" * 40})
    monkeypatch.setattr(supervisor.subprocess, "Popen", lambda *a, **kw: started.append((a, kw)) or SimpleNamespace(pid=4321))
    assert supervisor.start(args(source, home, state)) == 0
    result = json.loads(capsys.readouterr().out); data = supervisor.read_state(Path(result["receipt"]))
    assert "worker_pid" not in data and data["status"] == "queued"
    assert "--lock-fd" in started[0][0][0] and started[0][1]["pass_fds"]


def test_inherited_flock_remains_exclusive_after_parent_exit(tmp_path):
    lock_path = tmp_path / ".lock"
    code = "import fcntl,os,sys,time; f=open(sys.argv[1],'a+'); fcntl.flock(f,fcntl.LOCK_EX); print('locked',flush=True); time.sleep(.5)"
    child = subprocess.Popen([sys.executable, "-c", code, str(lock_path)], stdout=subprocess.PIPE, text=True)
    assert child.stdout and child.stdout.readline().strip() == "locked"
    contender = lock_path.open("a+")
    try:
        with pytest.raises(BlockingIOError):
            fcntl.flock(contender.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    finally:
        contender.close(); child.wait(timeout=2)


def test_worker_collects_all_profile_failures_and_never_stops_on_first(supervisor, layout, monkeypatch):
    source, home, state = layout; path = receipt(supervisor, state, source, home, profiles=["hermie", "mia"]); calls = []; restarted = []
    monkeypatch.setattr(supervisor, "git", lambda *_: "" if _[-1] == "--porcelain" else "b" * 40)
    monkeypatch.setattr(supervisor, "fork_main_sha", lambda _: "b" * 40)
    def fake_run(argv, **kwargs):
        calls.append(argv)
        return (1, "") if argv[-1] == "restart" and argv[2] == "hermie" else (0, "")
    monkeypatch.setattr(supervisor, "run_command", fake_run)
    def fake_restart(_launcher, profile, *_args, **_kwargs):
        restarted.append(profile)
        if profile == "hermie":
            raise RuntimeError("restart failed")
        return {"code_sha": "b" * 40}
    monkeypatch.setattr(supervisor, "restart_and_wait_for_health", fake_restart)
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: False)
    assert supervisor.worker(path) == 1
    data = supervisor.read_state(path)
    assert data["status"] == "failed" and "error" in data["gateway_results"]["hermie"]
    assert restarted == ["hermie", "mia"]


def test_health_requires_live_pid_start_heartbeat_and_identity(supervisor, layout, monkeypatch):
    _, home, _ = layout
    pid = os.getpid(); started = 123
    monkeypatch.setattr(supervisor, "process_start_time", lambda _: started)
    payload = {"gateway_state": "running", "code_sha": "b" * 40, "pid": pid,
               "start_time": started, "updated_at": "2030-01-01T00:00:00+00:00"}
    (home / "gateway_state.json").write_text(json.dumps(payload), encoding="utf-8")
    (home / "gateway.pid").write_text(json.dumps({"pid": pid, "start_time": started,
                                                    "hermes_home": str(home)}), encoding="utf-8")
    assert supervisor.health_record(home, "b" * 40)["pid"] == pid
    payload["start_time"] = 999; (home / "gateway_state.json").write_text(json.dumps(payload), encoding="utf-8")
    with pytest.raises(RuntimeError, match="identity"):
        supervisor.health_record(home, "b" * 40)


def test_restart_retries_only_the_explicit_busy_refusal(supervisor, layout, monkeypatch):
    source, home, _ = layout; attempts = []; sleeps = []
    # Verbatim native refusal captured from a live restart (identifiers anonymized).
    busy = ("✗ Refusing to restart the gateway: 2 live session(s) would be interrupted.\n"
            "  • cron job 000000000000 (example) in-process pid 1234, running 16m25s\n"
            "  • managed run pid 1234\n"
            "  Let them finish, or interrupt them deliberately:\n"
            "    hermes gateway restart --yes   # interrupts 2 live session(s)\n")
    outcomes = [(1, busy), (0, "restarted")]
    monkeypatch.setattr(supervisor, "run_command", lambda argv, **kwargs: attempts.append(argv) or outcomes.pop(0))
    monkeypatch.setattr(supervisor, "wait_for_health", lambda *_: {"ready": True})
    monkeypatch.setattr(supervisor.time, "sleep", sleeps.append)
    assert supervisor.restart_and_wait_for_health(source / ".hermes/bin/hermes", "hermie", home, "b" * 40,
                                                  timeout=10, log=home / "supervisor.log") == {"ready": True}
    assert len(attempts) == 2 and sleeps == [1.0]


def test_restart_refusal_does_not_retry_other_errors(supervisor, layout, monkeypatch):
    source, home, _ = layout; attempts = []
    monkeypatch.setattr(supervisor, "run_command", lambda argv, **kwargs: attempts.append(argv) or (1, "bad configuration"))
    with pytest.raises(RuntimeError, match="exited 1"):
        supervisor.restart_and_wait_for_health(source / ".hermes/bin/hermes", "hermie", home, "b" * 40,
                                                timeout=10, log=home / "supervisor.log")
    assert len(attempts) == 1


def test_restart_readiness_waits_through_the_drain_until_restart_deadline(supervisor, layout, monkeypatch):
    # A drain-first restart returns before the old gateway exits; readiness must span the drain.
    source, home, _ = layout; deadlines = []
    monkeypatch.setattr(supervisor.time, "monotonic", lambda: 100.0)
    monkeypatch.setattr(supervisor, "run_command", lambda *_args, **_kwargs: (0, "✓ Service restart requested"))
    monkeypatch.setattr(supervisor, "wait_for_health", lambda _home, _sha, deadline: deadlines.append(deadline) or {"ready": True})
    supervisor.restart_and_wait_for_health(source / ".hermes/bin/hermes", "hermie", home, "b" * 40,
                                           timeout=1800, log=home / "supervisor.log")
    assert deadlines == [100.0 + 1800]


def test_timeout_is_uncertain_and_blocks_new_transaction(supervisor, layout, monkeypatch):
    source, home, state = layout; path = receipt(supervisor, state, source, home)
    monkeypatch.setattr(supervisor, "git", lambda *_: "" if _[-1] == "--porcelain" else "b" * 40)
    monkeypatch.setattr(supervisor, "fork_main_sha", lambda _: "b" * 40)
    monkeypatch.setattr(supervisor, "run_command", lambda *a, **kw: (_ for _ in ()).throw(supervisor.CommandTimeout("alive child")))
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: False)
    assert supervisor.worker(path) == 1 and supervisor.read_state(path)["status"] == "uncertain"
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: True)
    monkeypatch.setattr(supervisor, "freeze_target", lambda *_: {"pre_sha": "a" * 40, "expected_sha": "b" * 40})
    with pytest.raises(RuntimeError, match="uncertain"):
        supervisor.start(args(source, home, state))


def test_delivery_is_marked_attempting_before_send_and_never_replayed(supervisor, layout, monkeypatch):
    source, home, state = layout; path = receipt(supervisor, state, source, home, status="failed"); seen = []
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: True)
    def fail_send(*a, **kw):
        seen.append(supervisor.read_state(path)["delivery_state"])
        raise RuntimeError("transport dropped")
    monkeypatch.setattr(supervisor, "run_command", fail_send)
    supervisor.deliver_completion(path, path.with_suffix(".log"))
    assert seen == ["attempting"] and supervisor.read_state(path)["delivery_state"] == "ambiguous"
    supervisor.deliver_completion(path, path.with_suffix(".log")); assert seen == ["attempting"]


def test_recovery_refuses_held_lock_and_orphans_only_after_lock_absent(supervisor, layout, monkeypatch):
    source, home, state = layout; path = receipt(supervisor, state, source, home)
    lock = (state / ".lock").open("a+"); fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    try:
        with pytest.raises(RuntimeError, match="lock"):
            supervisor.recover(Namespace(receipt=str(path)))
    finally:
        lock.close()
    monkeypatch.setattr(supervisor, "exact_origin_exists", lambda *_: False)
    assert supervisor.recover(Namespace(receipt=str(path))) == 1
    assert supervisor.read_state(path)["status"] == "orphaned"


def test_monitor_wakes_only_for_terminal_receipt(supervisor, layout, capsys):
    source, home, state = layout; path = receipt(supervisor, state, source, home)
    assert supervisor.monitor(Namespace(receipt=str(path))) == 0
    assert json.loads(capsys.readouterr().out)["wakeAgent"] is False
    supervisor.finish(path, status="failed", error="x")
    supervisor.monitor(Namespace(receipt=str(path)))
    assert json.loads(capsys.readouterr().out)["wakeAgent"] is True
