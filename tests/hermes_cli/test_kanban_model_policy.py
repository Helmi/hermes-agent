"""Contract tests for the opt-in Kanban background model guard."""
from types import SimpleNamespace

import pytest

from hermes_cli.kanban_model_policy import KanbanModelPolicyError, resolve_worker_route


POLICY = {
    "kanban": {
        "worker_model_policy": {
            "default": {"provider": "zai", "model": "glm-5.3"},
            "visual": {"provider": "zai", "model": "glm-5.3-flash"},
        }
    }
}


def task(**changes):
    values = {"id": "t_policy", "title": "normal background task", "body": "", "model_override": None, "provider_override": None}
    values.update(changes)
    return SimpleNamespace(**values)


def test_unpinned_mia_worker_gets_safe_text_route():
    assert resolve_worker_route(task(), POLICY).model == "glm-5.3"
    assert resolve_worker_route(task(), POLICY).provider == "zai"


def test_explicit_safe_pin_is_preserved():
    route = resolve_worker_route(task(model_override="MiniMax-M3", provider_override="minimax"), POLICY)
    assert (route.provider, route.model) == ("minimax", "MiniMax-M3")


def test_unqualified_pin_fails_closed_instead_of_inheriting_interactive_provider():
    with pytest.raises(KanbanModelPolicyError, match="without --provider"):
        resolve_worker_route(task(model_override="glm-5.3"), POLICY)


def test_claude_pin_requires_exact_task_local_frank_approval():
    blocked = task(model_override="claude-opus-5-5", provider_override="anthropic")
    with pytest.raises(KanbanModelPolicyError, match="task-local Frank approval"):
        resolve_worker_route(blocked, POLICY)

    approved = task(
        model_override="claude-opus-5-5", provider_override="anthropic",
        body="FRANK-APPROVED-CLAUDE: anthropic/claude-opus-5-5",
    )
    route = resolve_worker_route(approved, POLICY)
    assert (route.provider, route.model) == ("anthropic", "claude-opus-5-5")


def test_visual_task_gets_configured_safe_vision_route():
    route = resolve_worker_route(task(title="Inspect screenshot for broken layout"), POLICY)
    assert (route.provider, route.model) == ("zai", "glm-5.3-flash")


def test_profiles_without_policy_are_unchanged():
    assert resolve_worker_route(task(), {"kanban": {}}) is None
