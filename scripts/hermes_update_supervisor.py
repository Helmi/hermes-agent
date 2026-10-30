#!/usr/bin/env python3
"""Detached, serialized Hermes source-update supervisor.

The launcher freezes a clean installed SHA and an explicit, remote-verified target,
then hands its advisory lock to a detached worker. The worker uses supported Hermes
CLI entry points only; it neither signals gateways nor bypasses their drain guards.
"""
from __future__ import annotations

import argparse
import datetime as dt
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
_TERMINAL = {"succeeded", "failed", "orphaned", "uncertain"}


def now() -> int:
    return time.time_ns()


def atomic_write(path: Path, value: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    tmp = path.parent / f".{path.name}.{uuid.uuid4().hex}.tmp"
    try:
        tmp.write_text(json.dumps(value, sort_keys=True, indent=2) + "\n", encoding="utf-8")
        os.replace(tmp, path)
    finally:
        with contextlib.suppress(FileNotFoundError):
            tmp.unlink()


# Kept local to avoid importing the live gateway runtime into this detached helper.
import contextlib


def read_state(path: Path) -> dict[str, Any]:
    data = json.loads(path.read_text(encoding="utf-8"))
    if not isinstance(data, dict) or data.get("schema") != 3:
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


def fork_main_sha(source: Path) -> str:
    """Read the configured fork remote directly; do not trust a stale local ref."""
    remote = git(source, "remote", "get-url", "fork")
    result = subprocess.run(["git", "ls-remote", remote, "refs/heads/main"], stdin=subprocess.DEVNULL,
                            capture_output=True, text=True, timeout=20, check=False)
    if result.returncode or not result.stdout.strip():
        raise RuntimeError("could not resolve fork main with bounded git ls-remote")
    sha = result.stdout.split()[0]
    if not _SHA.fullmatch(sha):
        raise RuntimeError("fork main did not resolve to a full commit SHA")
    return sha


def freeze_target(source: Path, expected_sha: str | None) -> dict[str, str]:
    if git(source, "status", "--porcelain"):
        raise ValueError("refusing dirty source checkout")
    pre_sha = git(source, "rev-parse", "HEAD")
    target = expected_sha or pre_sha
    if not _SHA.fullmatch(target):
        raise ValueError("expected SHA must be a full lowercase 40-hex commit")
    if git(source, "rev-parse", f"{target}^{{commit}}") != target:
        raise ValueError("expected SHA is not a commit in the source checkout")
    if fork_main_sha(source) != target:
        raise ValueError("expected SHA must exactly match fork main at handoff")
    return {"pre_sha": pre_sha, "expected_sha": target}


def route_from_args(args: argparse.Namespace) -> dict[str, str]:
    values = {"platform": args.origin_platform, "chat_id": args.origin_chat_id,
              "thread_id": args.origin_thread_id, "parent_chat_id": args.origin_parent_chat_id,
              "profile": args.origin_profile, "session_id": args.origin_session_id}
    if any(not isinstance(value, str) or not value.strip() for value in values.values()):
        raise ValueError("all explicit --origin-* values, including --origin-session-id, are required")
    return {key: value.strip() for key, value in values.items()}


def exact_origin_exists(home: Path, origin: dict[str, str]) -> bool:
    db = home / "state.db"
    if not db.is_file():
        return False
    try:
        with sqlite3.connect(f"file:{db}?mode=ro", uri=True) as connection:
            row = connection.execute("SELECT source, chat_id, thread_id, profile_name, origin_json FROM sessions WHERE id=?",
                                     (origin["session_id"],)).fetchone()
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
    return origin_home if profile == origin_profile else origin_home.parent / profile


def command_env(home: Path) -> dict[str, str]:
    """Keep native lifecycle context intact; only scope the requested profile home."""
    env = os.environ.copy()
    env["HERMES_HOME"] = str(home)
    return env


class CommandTimeout(RuntimeError):
    pass


def run_command(argv: list[str], *, timeout: float, log: Path, home: Path) -> tuple[int, str]:
    with log.open("a", encoding="utf-8") as stream:
        stream.write("$ " + subprocess.list2cmdline(argv) + "\n")
        stream.flush()
        try:
            result = subprocess.run(argv, stdin=subprocess.DEVNULL, stdout=subprocess.PIPE, stderr=subprocess.STDOUT,
                                    env=command_env(home), timeout=timeout, check=False, text=True)
        except subprocess.TimeoutExpired as exc:
            stream.write(f"[timeout after {timeout}s: {exc}]\n")
            stream.flush()
            raise CommandTimeout(f"command timed out after {timeout}s; descendants may still be active") from exc
        output = result.stdout or ""
        stream.write(output + f"[exit={result.returncode}]\n")
        stream.flush()
        return result.returncode, output


_START_TIME_DRIFT_TOLERANCE = 200
# Native drain-first restarts return before the old process exits; readiness may take as long as the drain.
_BUSY_RESTART_REFUSAL = re.compile(r"Refusing to restart the gateway: \d+ live session\(s\) would be interrupted")


def process_start_time(pid: int) -> int | None:
    """Return the native, clock-step-stable process-birth fingerprint."""
    try:
        import psutil  # type: ignore
        from hermes_cli.process_identity import process_birth
        return round(process_birth(psutil.Process(pid)) * 100)
    except Exception:
        return None


def start_times_match(recorded: Any, current: Any) -> bool:
    try:
        return abs(int(current) - int(recorded)) <= _START_TIME_DRIFT_TOLERANCE
    except (TypeError, ValueError):
        return False


def is_busy_restart_refusal(code: int, output: str) -> bool:
    """Only retry the native drain guard; every other restart failure is terminal."""
    return code != 0 and _BUSY_RESTART_REFUSAL.search(output) is not None


def parse_updated_at(value: Any) -> float | None:
    if not isinstance(value, str):
        return None
    try:
        return dt.datetime.fromisoformat(value).timestamp()
    except ValueError:
        return None


def health_record(home: Path, expected_sha: str, *, max_age_s: float = 120) -> dict[str, Any]:
    try:
        state = json.loads((home / "gateway_state.json").read_text(encoding="utf-8"))
        pid_record = json.loads((home / "gateway.pid").read_text(encoding="utf-8"))
    except (OSError, json.JSONDecodeError):
        raise RuntimeError("gateway runtime status is unavailable") from None
    if not isinstance(state, dict) or state.get("gateway_state") != "running":
        raise RuntimeError(f"gateway is not ready (state={state.get('gateway_state') if isinstance(state, dict) else None!r})")
    if not isinstance(pid_record, dict) or Path(str(pid_record.get("hermes_home", ""))).resolve() != home.resolve():
        raise RuntimeError("gateway PID record belongs to another HERMES_HOME")
    if state.get("code_sha") != expected_sha:
        raise RuntimeError(f"gateway code SHA {state.get('code_sha')!r} does not equal expected {expected_sha}")
    pid, state_start, pid_start = state.get("pid"), state.get("start_time"), pid_record.get("start_time")
    if (not isinstance(pid, int) or pid_record.get("pid") != pid
            or not start_times_match(state_start, pid_start)
            or not start_times_match(pid_start, process_start_time(pid))):
        raise RuntimeError("gateway PID/start-time identity is not live")
    updated_at = parse_updated_at(state.get("updated_at"))
    if updated_at is None or time.time() - updated_at > max_age_s:
        raise RuntimeError("gateway heartbeat is stale or invalid")
    return {key: state.get(key) for key in ("pid", "start_time", "updated_at", "code_sha", "code_version")} | {"hermes_home": pid_record["hermes_home"]}


def wait_for_health(home: Path, expected_sha: str, deadline: float) -> dict[str, Any]:
    last_error = "gateway readiness deadline elapsed"
    while time.monotonic() < deadline:
        try:
            return health_record(home, expected_sha)
        except RuntimeError as exc:
            last_error = str(exc)
            time.sleep(min(1.0, max(0.0, deadline - time.monotonic())))
    raise RuntimeError(last_error)


def restart_and_wait_for_health(launcher: Path, profile: str, home: Path, expected_sha: str, *, timeout: float, log: Path) -> dict[str, Any]:
    """Respect the native drain guard, retrying only its explicit busy refusal."""
    deadline = time.monotonic() + timeout
    backoff = 1.0
    while True:
        remaining = deadline - time.monotonic()
        if remaining <= 0:
            raise RuntimeError("gateway restart remained busy until its bounded deadline")
        code, output = run_command([str(launcher), "-p", profile, "gateway", "restart"],
                                   timeout=min(remaining, 120.0), log=log, home=home)
        if code == 0:
            return wait_for_health(home, expected_sha, deadline)
        if not is_busy_restart_refusal(code, output):
            raise RuntimeError(f"gateway restart exited {code}")
        sleep_for = min(backoff, max(0.0, deadline - time.monotonic()))
        if sleep_for <= 0:
            raise RuntimeError("gateway restart remained busy until its bounded deadline")
        time.sleep(sleep_for)
        backoff = min(backoff * 2, 15.0)


def completion_text(receipt: dict[str, Any]) -> str:
    if receipt["status"] == "succeeded":
        return f"Hermes update {receipt['run_id'][:12]} completed and verified."
    return f"Hermes update {receipt['run_id'][:12]} requires manual recovery: {receipt.get('error', 'unknown error')}"


def parse_native_send_result(output: str) -> dict[str, Any]:
    decoder = json.JSONDecoder()
    for offset, char in enumerate(output):
        if char == "{":
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
    update_state(receipt_path, delivery_state="attempting", delivery_attempted_at_ns=now())
    try:
        code, output = run_command([str(canonical_launcher(source)), "-p", origin["profile"], "send", "--to",
                                    f"{origin['platform']}:{origin['chat_id']}:{origin['thread_id']}", "--file", str(message), "--json"],
                                   timeout=120, log=log, home=home)
        if code:
            raise RuntimeError(f"native send exited {code}")
        result = parse_native_send_result(output)
        if result.get("success") is not True:
            raise RuntimeError(f"native send did not confirm success: {result!r}")
    except Exception as exc:
        # 'attempting' is intentionally retained: external delivery is ambiguous and never retried automatically.
        update_state(receipt_path, delivery_state="ambiguous", delivery_error=str(exc), delivery_wake="unsupported")
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
    source, home, log = Path(receipt["source"]), Path(receipt["home"]), receipt_path.with_suffix(".log")
    lock = None
    try:
        if inherited_lock_fd is None:
            lock = (receipt_path.parent / ".lock").open("a+")
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        else:
            os.fstat(inherited_lock_fd)
        update_state(receipt_path, status="running", worker_pid=os.getpid(), worker_started_at_ns=now())
        if git(source, "status", "--porcelain"):
            raise RuntimeError("source became dirty after handoff")
        if fork_main_sha(source) != receipt["expected_sha"]:
            raise RuntimeError("fork main changed after handoff")
        update_home = profile_home(home, receipt["origin"]["profile"], receipt["update_profile"])
        code, _ = run_command([str(canonical_launcher(source)), "-p", receipt["update_profile"], "update", "--channel", "main",
                               "--no-backup", "--yes", "--no-gateway-restart"], timeout=receipt["update_timeout_s"], log=log, home=update_home)
        if code:
            raise RuntimeError(f"native update exited {code}")
        if git(source, "rev-parse", "HEAD") != receipt["expected_sha"]:
            raise RuntimeError("installed source SHA does not equal frozen expected SHA")
        results: dict[str, Any] = {}
        for profile in receipt["profiles"]:
            target_home = profile_home(home, receipt["origin"]["profile"], profile)
            try:
                health = restart_and_wait_for_health(canonical_launcher(source), profile, target_home,
                                                     receipt["expected_sha"], timeout=receipt["restart_timeout_s"],
                                                     log=log)
                results[profile] = {"health": health}
            except Exception as exc:
                results[profile] = {"error": str(exc)}
        update_state(receipt_path, gateway_results=results)
        failures = {p: value["error"] for p, value in results.items() if "error" in value}
        if failures:
            raise RuntimeError(f"gateway activation failures: {failures}")
        finish(receipt_path, status="succeeded")
    except CommandTimeout as exc:
        finish(receipt_path, status="uncertain", error=str(exc))
    except Exception as exc:
        finish(receipt_path, status="failed", error=str(exc))
    finally:
        if lock is not None:
            lock.close()
        if inherited_lock_fd is not None:
            os.close(inherited_lock_fd)
    deliver_completion(receipt_path, log)
    return 0 if read_state(receipt_path)["status"] == "succeeded" else 1


def unresolved_uncertain(state_dir: Path) -> bool:
    for receipt_path in state_dir.glob("*.json"):
        try:
            if read_state(receipt_path).get("status") == "uncertain":
                return True
        except (OSError, ValueError, json.JSONDecodeError):
            continue
    return False


def start(args: argparse.Namespace) -> int:
    if not args.activate:
        raise ValueError("refusing to start without --activate")
    source, home, state_dir = Path(args.source).resolve(), Path(args.home).resolve(), Path(args.state_dir).resolve()
    origin = route_from_args(args)
    if not exact_origin_exists(home, origin):
        raise ValueError("origin session does not match the exact requested native route")
    canonical_launcher(source)
    frozen = freeze_target(source, args.expected_sha)
    state_dir.mkdir(parents=True, exist_ok=True)
    lock = (state_dir / ".lock").open("a+")
    try:
        try:
            fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        except BlockingIOError as exc:
            raise RuntimeError("another supervisor transaction is active") from exc
        if unresolved_uncertain(state_dir):
            raise RuntimeError("an uncertain supervisor transaction requires explicit reconciliation")
        run_id, receipt_path = uuid.uuid4().hex, None
        receipt_path = state_path(state_dir, run_id)
        receipt = {"schema": 3, "run_id": run_id, "status": "queued", "created_at_ns": now(), "source": str(source),
                   "home": str(home), **frozen, "profiles": list(args.profile), "update_profile": args.update_profile,
                   "update_timeout_s": args.update_timeout_s, "restart_timeout_s": args.restart_timeout_s,
                   "origin": origin, "delivery_state": "pending"}
        atomic_write(receipt_path, receipt)
        os.set_inheritable(lock.fileno(), True)
        with receipt_path.with_suffix(".log").open("a", encoding="utf-8") as stream:
            proc = subprocess.Popen([sys.executable, str(Path(__file__).resolve()), "worker", "--receipt", str(receipt_path),
                                     "--lock-fd", str(lock.fileno())], stdin=subprocess.DEVNULL, stdout=stream,
                                    stderr=subprocess.STDOUT, start_new_session=True, close_fds=True, pass_fds=(lock.fileno(),))
    finally:
        lock.close()
    print(json.dumps({"run_id": run_id, "receipt": str(receipt_path), "worker_pid": proc.pid, **frozen}))
    return 0


def validate(args: argparse.Namespace) -> int:
    source, home = Path(args.source).resolve(), Path(args.home).resolve()
    origin = route_from_args(args)
    canonical_launcher(source)
    frozen = freeze_target(source, args.expected_sha)
    ok = exact_origin_exists(home, origin)
    print(json.dumps({"origin": origin, **frozen, "exact_route_match": ok}))
    return 0 if ok else 1


def recover(args: argparse.Namespace) -> int:
    receipt_path = Path(args.receipt).resolve()
    lock = (receipt_path.parent / ".lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
    except BlockingIOError as exc:
        lock.close()
        raise RuntimeError("supervisor transaction lock is held; refusing concurrent recovery") from exc
    try:
        receipt = read_state(receipt_path)
        if receipt["status"] not in _TERMINAL:
            finish(receipt_path, status="orphaned", error="worker lock absent; transaction state uncertain; never replayed")
        deliver_completion(receipt_path, receipt_path.with_suffix(".log"))
        print(json.dumps(read_state(receipt_path), sort_keys=True))
        return 0 if read_state(receipt_path)["status"] == "succeeded" else 1
    finally:
        lock.close()


def reconcile(args: argparse.Namespace) -> int:
    receipt_path = Path(args.receipt).resolve()
    if args.status not in {"failed", "orphaned"}:
        raise ValueError("reconciliation status must be failed or orphaned")
    lock = (receipt_path.parent / ".lock").open("a+")
    try:
        fcntl.flock(lock.fileno(), fcntl.LOCK_EX | fcntl.LOCK_NB)
        receipt = read_state(receipt_path)
        if receipt["status"] != "uncertain":
            raise RuntimeError("only an uncertain transaction can be reconciled")
        finish(receipt_path, status=args.status, error="explicit manual reconciliation: " + args.note)
    finally:
        lock.close()
    print(json.dumps(read_state(receipt_path), sort_keys=True))
    return 0


def monitor(args: argparse.Namespace) -> int:
    receipt = read_state(Path(args.receipt).resolve())
    terminal = receipt.get("status") in _TERMINAL
    print(json.dumps({"run_id": receipt["run_id"], "status": receipt["status"], "delivery_state": receipt.get("delivery_state"),
                      "wakeAgent": terminal}))
    return 0


def add_origin_flags(parser: argparse.ArgumentParser) -> None:
    for name in ("platform", "chat-id", "thread-id", "parent-chat-id", "profile", "session-id"):
        parser.add_argument(f"--origin-{name}")


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
    worker_p = sub.add_parser("worker"); worker_p.add_argument("--receipt", required=True); worker_p.add_argument("--lock-fd", type=int)
    recover_p = sub.add_parser("recover"); recover_p.add_argument("--receipt", required=True)
    reconcile_p = sub.add_parser("reconcile"); reconcile_p.add_argument("--receipt", required=True); reconcile_p.add_argument("--status", required=True); reconcile_p.add_argument("--note", required=True)
    monitor_p = sub.add_parser("monitor"); monitor_p.add_argument("--receipt", required=True)
    validate_p = sub.add_parser("validate"); validate_p.add_argument("--source", required=True); validate_p.add_argument("--home", required=True); validate_p.add_argument("--expected-sha"); add_origin_flags(validate_p)
    args = parser.parse_args()
    return {"start": start, "worker": lambda a: worker(Path(a.receipt).resolve(), a.lock_fd), "validate": validate,
            "recover": recover, "reconcile": reconcile, "monitor": monitor}[args.command](args)


if __name__ == "__main__":
    raise SystemExit(main())
