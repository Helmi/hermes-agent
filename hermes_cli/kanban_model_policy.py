"""Profile-local model routing guard for Kanban workers.

A policy is opt-in under ``kanban.worker_model_policy``.  It is resolved by
the assigned profile owner immediately before the worker's first model call,
so a dispatcher cannot accidentally inherit its own profile's default route.
"""
from __future__ import annotations

from dataclasses import dataclass
from typing import Any, Mapping


class KanbanModelPolicyError(RuntimeError):
    """A task's requested route violates its assigned profile's worker policy."""


_VISUAL_TERMS = ("image", "photo", "picture", "screenshot", "vision", "visual", "ocr")
_CLAUDE_TERMS = ("claude", "opus", "sonnet")


@dataclass(frozen=True)
class WorkerRoute:
    model: str
    provider: str


def _route(value: Any, name: str) -> WorkerRoute:
    if not isinstance(value, Mapping):
        raise KanbanModelPolicyError(f"kanban.worker_model_policy.{name} must name provider and model")
    model, provider = str(value.get("model") or "").strip(), str(value.get("provider") or "").strip()
    if not model or not provider:
        raise KanbanModelPolicyError(f"kanban.worker_model_policy.{name} must name provider and model")
    if _is_claude(model, provider):
        raise KanbanModelPolicyError(f"kanban.worker_model_policy.{name} must not route background work to Claude")
    return WorkerRoute(model, provider)


def _is_claude(model: str, provider: str) -> bool:
    route = f"{provider}/{model}".lower()
    return provider.lower() == "anthropic" or any(term in route for term in _CLAUDE_TERMS)


def _approved_claude(task: Any, model: str, provider: str) -> bool:
    """An approval is an exact, task-local body record, never a comment inference.

    The deliberate marker is intended to be pasted only after Frank's approval:
    ``FRANK-APPROVED-CLAUDE: provider/model``.  It binds one route and one card.
    """
    marker = f"FRANK-APPROVED-CLAUDE: {provider}/{model}".lower()
    return any(line.strip().lower() == marker for line in str(getattr(task, "body", "") or "").splitlines())


def resolve_worker_route(task: Any, config: Mapping[str, Any]) -> WorkerRoute | None:
    """Return a policy route or raise before a Kanban worker can enter a model.

    No policy means upstream behaviour.  Explicit non-Claude task pins remain
    intact.  Claude pins require the exact task-local approval marker; unpinned
    work receives the profile's text default, with obvious visual tasks routed
    to its configured visual default.
    """
    kanban = config.get("kanban") if isinstance(config, Mapping) else None
    policy = kanban.get("worker_model_policy") if isinstance(kanban, Mapping) else None
    if not policy:
        return None
    if not isinstance(policy, Mapping):
        raise KanbanModelPolicyError("kanban.worker_model_policy must be a mapping")

    model = str(getattr(task, "model_override", "") or "").strip()
    provider = str(getattr(task, "provider_override", "") or "").strip()
    if model:
        if not provider:
            raise KanbanModelPolicyError(
                f"Kanban task {task.id} pins model {model!r} without --provider; "
                "this profile's background policy requires a provider-qualified safe pin."
            )
        if _is_claude(model, provider) and not _approved_claude(task, model, provider):
            raise KanbanModelPolicyError(
                f"Kanban task {task.id} requests Claude ({provider}/{model}) without a task-local Frank approval. "
                f"Add an exact line `FRANK-APPROVED-CLAUDE: {provider}/{model}` to this task after approval, "
                "or use a non-Claude --model/--provider pin."
            )
        return WorkerRoute(model, provider)

    title_body = f"{getattr(task, 'title', '')}\n{getattr(task, 'body', '')}".lower()
    route_name = "visual" if any(term in title_body for term in _VISUAL_TERMS) and policy.get("visual") else "default"
    return _route(policy.get(route_name), route_name)
