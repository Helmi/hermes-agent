#!/usr/bin/env python3
"""Detached, at-most-once Hermes source-update supervisor.

Run ``start`` from a gateway turn.  It records intent and forks a new session,
then returns immediately so the invoking turn can drain.  The detached worker
uses Hermes' supported updater, gateway restart, session store, and ``send``
CLI; it does not import platform credentials or signal/kill gateway processes.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

DEFAULT_PROFILES = ("hermie", "mia", "smoerrebroed")
DEFAULT_ORIGIN = {
    "platform": "discord",
    "chat_id": "1565697700836540429",
    "thread_id": "1565697700836540429",
    "parent_chat_id": "1487881681615458354",
    "profile": "hermie",
}


def now() -> int:
    return time.time_ns()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 1:
        raise ValueError("invalid supervisor receipt")
    return data


def state_path(state_dir: Path, run_id: str) -> Path:
    if len(run_id) != 32 or any(c not in "0123456789abcdef" for c in run_id):
        raise ValueError("run id must be 32 lowercase hex characters")
    return state_dir / f"{run_id}.json"


def is_live_pid(pid: Any) -> bool:
    if not isinstance(pid, int) or pid <= 0:
        return False
    try:
        os.kill(pid, 0)
    except ProcessLookupError:
        return False
    except PermissionError:
        return True
    return True


def canonical_launcher(source: Path) -> Path:
    launcher = source / ".hermes" / "bin" / "hermes"
    if not launcher.is_file():
        raise FileNotFoundError(f"canonical Hermes launcher is missing: {launcher}")
    return launcher


def session_exists(source: Path, home: Path, session_id: str) -> bool:
    """Read only the named profile's native session row; never infer an id.

    This intentionally uses SQLite's stdlib interface rather than importing the
    checkout: the supervisor has to run before and after a Python-runtime swap.
    """
    del source
    if not session_id or not (home / "state.db").is_file():
        return False
    import sqlite3
    try:
        connection = sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)
        try:
            return connection.execute("SELECT 1 FROM sessions WHERE id=? LIMIT 1", (session_id,)).fetchone() is not None
        finally:
            connection.close()
    except sqlite3.Error:
        return False


def discover_origin_sessions(home: Path) -> list[str]:
    """Look up, rather than derive, Discord sessions bound to the fixed origin route."""
    import sqlite3
    if not (home / "state.db").is_file():
        return []
    try:
        connection = sqlite3.connect(f"file:{home / 'state.db'}?mode=ro", uri=True)
        try:
            rows = connection.execute(
                "SELECT id FROM sessions WHERE source=? AND chat_id=? AND thread_id=? AND profile_name=? "
                "ORDER BY last_activity_at DESC, started_at DESC",
                (DEFAULT_ORIGIN["platform"], DEFAULT_ORIGIN["chat_id"], DEFAULT_ORIGIN["thread_id"],
                 DEFAULT_ORIGIN["profile"]),
            ).fetchall()
            return [str(row[0]) for row in rows]
        finally:
            connection.close()
    except sqlite3.Error:
        return []


def update_state(path: Path, **changes: Any) -> dict[str, Any]:
    receipt = read_state(path)
    receipt.update(changes)
    receipt["updated_at_ns"] = now()
    atomic_write(path, receipt)
    return receipt


def run_command(argv: list[str], *, timeout: float, log: Path) -> tuple[int, str]:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(argv) + "\n")
        stream.flush()
        result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                text=True, timeout=timeout, check=False)
        stream.write(f"[exit={result.returncode}]\n")
        stream.flush()
        return result.returncode, ""


def completion_text(receipt: dict[str, Any]) -> str:
    if receipt["status"] == "succeeded":
        return f"Hermes-Update {receipt['run_id'][:12]} abgeschlossen; Gateways neu gestartet."
    return (f"Hermes-Update {receipt['run_id'][:12]} fehlgeschlagen: "
            f"{receipt.get('error', 'unbekannter Fehler')}")


def deliver_completion(receipt_path: Path, source: Path, home: Path, log: Path) -> None:
    receipt = read_state(receipt_path)
    if receipt.get("delivery_state") != "pending":
        return
    origin = receipt["origin"]
    if not session_exists(source, home, origin["session_id"]):
        update_state(receipt_path, delivery_state="refused", delivery_error="origin session no longer exists")
        return
    # Mark before the native send. A crash may lose a notice but cannot duplicate one.
    update_state(receipt_path, delivery_state="attempted")
    message_path = receipt_path.with_suffix(".message.txt")
    message_path.write_text(completion_text(read_state(receipt_path)), encoding="utf-8")
    launcher = canonical_launcher(source)
    target = f"{origin['platform']}:{origin['chat_id']}:{origin['thread_id']}"
    try:
        code, _ = run_command([str(launcher), "-p", origin["profile"], "send", "--to", target,
                               "--file", str(message_path), "--json"], timeout=120, log=log)
    except Exception as exc:
        update_state(receipt_path, delivery_state="failed", delivery_error=str(exc))
        return
    if code == 0:
        update_state(receipt_path, delivery_state="sent")
    else:
        update_state(receipt_path, delivery_state="failed", delivery_error=f"native send exited {code}")


def worker(receipt_path: Path) -> int:
    receipt = read_state(receipt_path)
    source, home = Path(receipt["source"]), Path(receipt["home"])
    log = receipt_path.with_suffix(".log")
    if receipt["status"] != "queued":
        return 0
    update_state(receipt_path, status="running", worker_pid=os.getpid(), started_at_ns=now())
    launcher = canonical_launcher(source)
    try:
        code, _ = run_command([str(launcher), "update", "--channel", "main", "--no-backup", "--yes",
                               "--no-gateway-restart"], timeout=receipt["update_timeout_s"], log=log)
        if code != 0:
            raise RuntimeError(f"native update exited {code}")
        for profile in receipt["profiles"]:
            code, _ = run_command([str(launcher), "-p", profile, "gateway", "restart"],
                                  timeout=receipt["restart_timeout_s"], log=log)
            if code != 0:
                raise RuntimeError(f"gateway restart for {profile} exited {code}")
    except Exception as exc:
        update_state(receipt_path, status="failed", error=str(exc), finished_at_ns=now())
    else:
        update_state(receipt_path, status="succeeded", finished_at_ns=now())
    deliver_completion(receipt_path, source, home, log)
    return 0 if read_state(receipt_path)["status"] == "succeeded" else 1


def start(args: argparse.Namespace) -> int:
    source, home, state_dir = Path(args.source).resolve(), Path(args.home).resolve(), Path(args.state_dir).resolve()
    if not args.activate:
        raise ValueError("refusing to start without --activate")
    origin_session_id = args.origin_session_id
    if not origin_session_id:
        matches = discover_origin_sessions(home)
        if len(matches) != 1:
            raise ValueError("origin session lookup was not unique; pass --origin-session-id explicitly")
        origin_session_id = matches[0]
    if not session_exists(source, home, origin_session_id):
        raise ValueError("origin session was not found in the hermie SessionDB")
    canonical_launcher(source)
    lock_path = state_dir / ".lock"
    state_dir.mkdir(parents=True, exist_ok=True)
    with lock_path.open("a+") as lock:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        run_id = uuid.uuid4().hex
        receipt_path = state_path(state_dir, run_id)
        receipt = {
            "schema": 1, "run_id": run_id, "status": "queued", "created_at_ns": now(),
            "source": str(source), "home": str(home), "profiles": list(args.profile),
            "update_timeout_s": args.update_timeout_s, "restart_timeout_s": args.restart_timeout_s,
            "origin": {**DEFAULT_ORIGIN, "session_id": origin_session_id},
            "delivery_state": "pending",
        }
        atomic_write(receipt_path, receipt)
        log = receipt_path.with_suffix(".log")
        with log.open("a", encoding="utf-8") as stream:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--receipt", str(receipt_path)],
                                    stdin=subprocess.DEVNULL, stdout=stream, stderr=subprocess.STDOUT,
                                    start_new_session=True, close_fds=True)
        update_state(receipt_path, worker_pid=proc.pid, handoff_at_ns=now())
    print(json.dumps({"run_id": run_id, "receipt": str(receipt_path), "worker_pid": proc.pid}))
    return 0


def validate(args: argparse.Namespace) -> int:
    source, home = Path(args.source).resolve(), Path(args.home).resolve()
    matches = discover_origin_sessions(home)
    canonical_launcher(source)
    print(json.dumps({"origin": DEFAULT_ORIGIN, "session_ids": matches, "unique": len(matches) == 1}))
    return 0 if len(matches) == 1 else 1


def recover(args: argparse.Namespace) -> int:
    receipt_path = Path(args.receipt).resolve()
    receipt = read_state(receipt_path)
    if receipt["status"] in {"succeeded", "failed"}:
        print(json.dumps(receipt, sort_keys=True))
        return 0
    if is_live_pid(receipt.get("worker_pid")):
        raise RuntimeError("supervisor worker is still live; refusing a concurrent recovery")
    update_state(receipt_path, status="failed", error="orphaned worker recovered without replay", finished_at_ns=now())
    receipt = read_state(receipt_path)
    deliver_completion(receipt_path, Path(receipt["source"]), Path(receipt["home"]), receipt_path.with_suffix(".log"))
    print(json.dumps(read_state(receipt_path), sort_keys=True))
    return 1


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    start_p = sub.add_parser("start")
    start_p.add_argument("--source", required=True)
    start_p.add_argument("--home", required=True)
    start_p.add_argument("--state-dir", required=True)
    start_p.add_argument("--origin-session-id")
    start_p.add_argument("--activate", action="store_true")
    start_p.add_argument("--profile", action="append", default=None)
    start_p.add_argument("--update-timeout-s", type=float, default=3600.0)
    start_p.add_argument("--restart-timeout-s", type=float, default=1800.0)
    worker_p = sub.add_parser("worker")
    worker_p.add_argument("--receipt", required=True)
    recover_p = sub.add_parser("recover")
    recover_p.add_argument("--receipt", required=True)
    discover_p = sub.add_parser("discover-origin")
    discover_p.add_argument("--home", required=True)
    validate_p = sub.add_parser("validate")
    validate_p.add_argument("--source", required=True)
    validate_p.add_argument("--home", required=True)
    args = parser.parse_args()
    if args.command == "start":
        if args.profile is None:
            args.profile = list(DEFAULT_PROFILES)
        return start(args)
    if args.command == "worker":
        return worker(Path(args.receipt).resolve())
    if args.command == "discover-origin":
        print(json.dumps({"origin": DEFAULT_ORIGIN, "session_ids": discover_origin_sessions(Path(args.home).resolve())}))
        return 0
    if args.command == "validate":
        return validate(args)
    return recover(args)


if __name__ == "__main__":
    raise SystemExit(main())
