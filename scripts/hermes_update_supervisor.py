#!/usr/bin/env python3
"""Detached, serialized Hermes source-update supervisor.

The launcher records an explicit target and origin route, then hands an inherited
advisory lock to the detached worker.  The worker uses only supported Hermes CLI
entry points; it never signals gateway processes or bypasses updater safeguards.
"""
from __future__ import annotations

import argparse
import fcntl
import json
import os
import re
import sqlite3
import subprocess
import sys
import time
import uuid
from pathlib import Path
from typing import Any

_SHA = re.compile(r"^[0-9a-f]{40}$")
_TERMINAL = {"succeeded", "failed", "orphaned"}


def now() -> int:
    return time.time_ns()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.with_suffix(path.suffix + ".tmp")
    tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
    os.replace(tmp, path)


def read_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 2:
        raise ValueError("invalid supervisor receipt")
    return data


def update_state(path: Path, **changes: Any) -> dict[str, Any]:
    receipt = read_state(path)
    receipt.update(changes)
    receipt["updated_at_ns"] = now()
    atomic_write(path, receipt)
    return receipt


def state_path(state_dir: Path, run_id: str) -> Path:
    if not re.fullmatch(r"[0-9a-f]{32}", run_id):
        raise ValueError("run id must be 32 lowercase hex characters")
    return state_dir / f"{run_id}.json"


def canonical_launcher(source: Path) -> Path:
    launcher = source / ".hermes" / "bin" / "hermes"
    if not launcher.is_file():
        raise FileNotFoundError(f"canonical Hermes launcher is missing: {launcher}")
    return launcher


def git(source: Path, *args: str) -> str:
    result = subprocess.run(["git", "-C", str(source), *args], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, check=False)
    if result.returncode:
        raise RuntimeError(f"git {' '.join(args)} failed: {result.stderr.strip()}")
    return result.stdout.strip()


def freeze_target(source: Path, expected_sha: str | None) -> str:
    if git(source, "status", "--porcelain"):
        raise ValueError("refusing dirty source checkout")
    head = git(source, "rev-parse", "HEAD")
    target = expected_sha or head
    if not _SHA.fullmatch(target):
        raise ValueError("expected SHA must be a full lowercase 40-hex commit")
    if git(source, "rev-parse", f"{target}^{{commit}}") != target:
        raise ValueError("expected SHA is not a commit in the source checkout")
    if head != target:
        raise ValueError("source HEAD must equal the frozen expected SHA before handoff")
    return target


def route_from_args(args: argparse.Namespace) -> dict[str, str]:
    values = {"platform": args.origin_platform, "chat_id": args.origin_chat_id,
              "thread_id": args.origin_thread_id, "parent_chat_id": args.origin_parent_chat_id,
              "profile": args.origin_profile, "session_id": args.origin_session_id}
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("all explicit --origin-* values, including --origin-session-id, are required")
    return {key: value.strip() for key, value in values.items()}


def exact_origin_exists(home: Path, origin: dict[str, str]) -> bool:
    """Require the requested session id AND all native route fields to match."""
    db = home / "state.db"
    if not db.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
            row = connection.execute(
                "SELECT source, chat_id, thread_id, profile_name, origin_json FROM sessions WHERE id=?",
                (origin["session_id"],),
            ).fetchone()
    except sqlite3.Error:
        return False
    if row is None or tuple(str(x or "") for x in row[:4]) != (
        origin["platform"], origin["chat_id"], origin["thread_id"], origin["profile"]):
        return False
    try:
        metadata = json.loads(row[4] or "{}")
    except json.JSONDecodeError:
        return False
    return isinstance(metadata, dict) and str(metadata.get("parent_chat_id") or "") == origin["parent_chat_id"]


def profile_home(origin_home: Path, origin_profile: str, profile: str) -> Path:
    """Resolve each CLI call's profile home instead of retaining gateway-call internals."""
    return origin_home if profile == origin_profile else origin_home.parent / profile


def command_env(home: Path) -> dict[str, str]:
    """Do not inherit an invoking gateway's session/control internals."""
    env = {key: value for key, value in os.environ.items()
           if not key.startswith("HERMES_") or key in {"HERMES_LOG_LEVEL"}}
    env["HERMES_HOME"] = str(home)
    return env


def run_command(argv: list[str], *, timeout: float, log: Path, home: Path) -> tuple[int, str]:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(argv) + "\n")
        stream.flush()
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    env=command_env(home), timeout=timeout, check=False, text=True)
        except subprocess.TimeoutExpired as exc:
            # subprocess.run kills only its direct CLI child; receipt records uncertainty instead of replaying.
            stream.write(f"[timeout after {timeout}s: {exc}]\n")
            stream.flush()
            raise RuntimeError(f"command timed out after {timeout}s; manual inspection required") from exc
        output = result.stdout or ""
        stream.write(output)
        stream.write(f"[exit={result.returncode}]\n")
        stream.flush()
        return result.returncode, output


def health_record(home: Path, expected_sha: str) -> dict[str, Any]:
    """Native gateway state proves a fresh process, heartbeat, home and code identity."""
    path = home / "gateway_state.json"
    try:
        state = json.loads(path.read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("gateway runtime status is unavailable") from None
    if not isinstance(state, dict):
        raise RuntimeError("gateway runtime status is invalid")
    if state.get("gateway_state") != "running":
        raise RuntimeError(f"gateway is not ready (state={state.get('gateway_state')!r})")
    if Path(str(state.get("hermes_home", ""))).resolve() != home.resolve():
        raise RuntimeError("gateway runtime status belongs to another HERMES_HOME")
    if state.get("code_sha") != expected_sha:
        raise RuntimeError(f"gateway code SHA {state.get('code_sha')!r} does not equal expected {expected_sha}")
    return {key: state.get(key) for key in ("pid", "updated_at", "code_sha", "code_version")}


def completion_text(receipt: dict[str, Any]) -> str:
    if receipt["status"] == "succeeded":
        return f"Hermes update {receipt['run_id'][:12]} completed and verified."
    return f"Hermes update {receipt['run_id'][:12]} requires manual recovery: {receipt.get('error', 'unknown error')}"


def parse_native_send_result(output: str) -> dict[str, Any]:
    """Accept only a structured native send success, never an exit code alone."""
    decoder = json.JSONDecoder()
    for offset, char in enumerate(output):
        if char != "{":
            continue
        try:
            value, _ = decoder.raw_decode(output[offset:])
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            return value
    raise RuntimeError("native send did not return a JSON result")


def deliver_completion(receipt_path: Path, log: Path) -> None:
    receipt = read_state(receipt_path)
    if receipt.get("delivery_state") != "pending":
        return
    source, home, origin = Path(receipt["source"]), Path(receipt["home"]), receipt["origin"]
    if not exact_origin_exists(home, origin):
        update_state(receipt_path, delivery_state="refused", delivery_error="exact origin route no longer matches")
        return
    message = receipt_path.with_suffix(".message.txt")
    message.write_text(completion_text(receipt), encoding="utf-8")
    launcher = canonical_launcher(source)
    target = f"{origin['platform']}:{origin['chat_id']}:{origin['thread_id']}"
    # No supported CLI/control-socket continuation API exists here. This is verified channel delivery only.
    try:
        code, output = run_command([str(launcher), "-p", origin["profile"], "send", "--to", target,
                                    "--file", str(message), "--json"], timeout=120, log=log, home=home)
        if code:
            raise RuntimeError(f"native send exited {code}")
        result = parse_native_send_result(output)
        if result.get("success") is not True:
            raise RuntimeError(f"native send did not confirm success: {result!r}")
    except Exception as exc:
        update_state(receipt_path, delivery_state="failed", delivery_error=str(exc))
        return
    update_state(receipt_path, delivery_state="sent", delivery_wake="unsupported")


def finish(receipt_path: Path, *, status: str, error: str | None = None) -> None:
    changes: dict[str, Any] = {"status": status, "finished_at_ns": now()}
    if error:
        changes["error"] = error
    update_state(receipt_path, **changes)


def worker(receipt_path: Path, inherited_lock_fd: int | None = None) -> int:
    receipt = read_state(receipt_path)
    if receipt["status"] != "queued":
        return 0
    source, home = Path(receipt["source"]), Path(receipt["home"])
    log = receipt_path.with_suffix(".log")
    lock = None
    try:
        if inherited_lock_fd is None:
            lock = (receipt_path.parent / ".lock").open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            # The launcher passed a locked open-file description. Keep it open until terminal state.
            os.fstat(inherited_lock_fd)
        update_state(receipt_path, status="running", worker_pid=os.getpid(), started_at_ns=now())
        launcher = canonical_launcher(source)
        if git(source, "status", "--porcelain"):
            raise RuntimeError("source became dirty after handoff")
        update_home = profile_home(home, receipt["origin"]["profile"], receipt["update_profile"])
        code, _ = run_command([str(launcher), "-p", receipt["update_profile"], "update", "--channel", "main",
                               "--no-backup", "--yes", "--no-gateway-restart"], timeout=receipt["update_timeout_s"],
                              log=log, home=update_home)
        if code:
            raise RuntimeError(f"native update exited {code}")
        if git(source, "rev-parse", "HEAD") != receipt["expected_sha"]:
            raise RuntimeError("installed source SHA does not equal frozen expected SHA")
        health: dict[str, Any] = {}
        for profile in receipt["profiles"]:
            target_home = profile_home(home, receipt["origin"]["profile"], profile)
            code, _ = run_command([str(launcher), "-p", profile, "gateway", "restart"],
                                  timeout=receipt["restart_timeout_s"], log=log, home=target_home)
            if code:
                raise RuntimeError(f"gateway restart for {profile} exited {code}")
            health[profile] = health_record(target_home, receipt["expected_sha"])
        finish(receipt_path, status="succeeded")
        update_state(receipt_path, gateway_health=health)
    except Exception as exc:
        finish(receipt_path, status="failed", error=str(exc))
    finally:
        if lock is not None:
            lock.close()
        if inherited_lock_fd is not None:
            os.close(inherited_lock_fd)
    deliver_completion(receipt_path, log)
    return 0 if read_state(receipt_path)["status"] == "succeeded" else 1


def start(args: argparse.Namespace) -> int:
    if not args.activate:
        raise ValueError("refusing to start without --activate")
    source, home, state_dir = Path(args.source).resolve(), Path(args.home).resolve(), Path(args.state_dir).resolve()
    origin = route_from_args(args)
    if not exact_origin_exists(home, origin):
        raise ValueError("origin session does not match the exact requested native route")
    launcher = canonical_launcher(source)
    del launcher
    expected_sha = freeze_target(source, args.expected_sha)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / ".lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("another supervisor transaction is active") from exc
    try:
        run_id, receipt_path = uuid.uuid4().hex, None
        receipt_path = state_path(state_dir, run_id)
        receipt = {"schema": 2, "run_id": run_id, "status": "queued", "created_at_ns": now(),
                   "source": str(source), "home": str(home), "expected_sha": expected_sha,
                   "profiles": list(args.profile), "update_profile": args.update_profile,
                   "update_timeout_s": args.update_timeout_s, "restart_timeout_s": args.restart_timeout_s,
                   "origin": origin, "delivery_state": "pending"}
        atomic_write(receipt_path, receipt)
        log = receipt_path.with_suffix(".log")
        os.set_inheritable(lock.fileno(), True)
        with log.open("a", encoding="utf-8") as stream:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--receipt",
                                     str(receipt_path), "--lock-fd", str(lock.fileno())], stdin=subprocess.DEVNULL,
                                    stdout=stream, stderr=subprocess.STDOUT, start_new_session=True,
                                    close_fds=True, pass_fds=(lock.fileno(),))
        update_state(receipt_path, worker_pid=proc.pid, handoff_at_ns=now())
    finally:
        lock.close()  # inherited descriptor preserves the flock in the worker.
    print(json.dumps({"run_id": run_id, "receipt": str(receipt_path), "worker_pid": proc.pid,
                      "expected_sha": expected_sha}))
    return 0


def validate(args: argparse.Namespace) -> int:
    source, home = Path(args.source).resolve(), Path(args.home).resolve()
    origin = route_from_args(args)
    canonical_launcher(source)
    expected_sha = freeze_target(source, args.expected_sha)
    ok = exact_origin_exists(home, origin)
    print(json.dumps({"origin": origin, "expected_sha": expected_sha, "exact_route_match": ok}))
    return 0 if ok else 1


def is_live_pid(pid: Any) -> bool:
    try:
        os.kill(int(pid), 0)
    except (TypeError, ValueError, ProcessLookupError):
        return False
    except PermissionError:
        return True
    return True


def recover(args: argparse.Namespace) -> int:
    receipt_path = Path(args.receipt).resolve()
    receipt = read_state(receipt_path)
    if receipt["status"] not in _TERMINAL:
        if is_live_pid(receipt.get("worker_pid")):
            raise RuntimeError("supervisor worker is still live; refusing concurrent recovery")
        finish(receipt_path, status="orphaned", error="orphaned worker; transaction state uncertain; never replayed")
    deliver_completion(receipt_path, receipt_path.with_suffix(".log"))
    print(json.dumps(read_state(receipt_path), sort_keys=True))
    return 0 if read_state(receipt_path)["status"] == "succeeded" else 1


def add_origin_flags(parser: argparse.ArgumentParser) -> None:
    parser.add_argument("--origin-platform")
    parser.add_argument("--origin-chat-id")
    parser.add_argument("--origin-thread-id")
    parser.add_argument("--origin-parent-chat-id")
    parser.add_argument("--origin-profile")
    parser.add_argument("--origin-session-id")


def main() -> int:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    start_p = sub.add_parser("start")
    for name in ("source", "home", "state_dir"):
        start_p.add_argument(f"--{name.replace('_', '-')}", required=True)
    add_origin_flags(start_p)
    start_p.add_argument("--expected-sha")
    start_p.add_argument("--activate", action="store_true")
    start_p.add_argument("--profile", action="append", required=True)
    start_p.add_argument("--update-profile", required=True)
    start_p.add_argument("--update-timeout-s", type=float, default=3600.0)
    start_p.add_argument("--restart-timeout-s", type=float, default=1800.0)
    worker_p = sub.add_parser("worker")
    worker_p.add_argument("--receipt", required=True)
    worker_p.add_argument("--lock-fd", type=int)
    recover_p = sub.add_parser("recover")
    recover_p.add_argument("--receipt", required=True)
    validate_p = sub.add_parser("validate")
    validate_p.add_argument("--source", required=True)
    validate_p.add_argument("--home", required=True)
    validate_p.add_argument("--expected-sha")
    add_origin_flags(validate_p)
    args = parser.parse_args()
    if args.command == "start":
        return start(args)
    if args.command == "worker":
        return worker(Path(args.receipt).resolve(), args.lock_fd)
    if args.command == "validate":
        return validate(args)
    return recover(args)


if __name__ == "__main__":
    raise SystemExit(main())
