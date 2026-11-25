"""Fork regression: guarded routes must enter build_policy, not dataclass.replace."""
import json
from contextlib import closing
from types import SimpleNamespace

import pytest


@pytest.mark.parametrize('title,model,provider,body,expected', [
    ('Text task', None, None, '', ('glm-5.3', 'zai')),
    ('Inspect screenshot', None, None, '', ('MiniMax-M3', 'minimax')),
    ('Pinned task', 'glm-5.3', 'zai', '', ('glm-5.3', 'zai')),
    ('Unsafe pin', 'claude-opus-4-6', 'anthropic', '', None),
    ('Unqualified pin', 'glm-5.3', None, '', None),
    ('Approved pin', 'claude-opus-4-6', 'anthropic',
     'FRANK-APPROVED-CLAUDE: anthropic/claude-opus-4-6', ('claude-opus-4-6', 'anthropic')),
])
def test_background_route_enters_supported_policy(tmp_path, monkeypatch, title, model, provider, body, expected):
    from hermes_cli import kanban_db as kb
    from hermes_cli.kanban_db_connect import connect
    from gateway.session_contract import SessionActor
    from gateway.session_kanban import build_kanban_policy
    from gateway.session_policy import restore_policy
    from dataclasses import asdict
    from persistence.core.errors import AuthorityRefusal

    for key in ('HERMES_HOME', 'HERMES_KANBAN_HOME'):
        monkeypatch.setenv(key, str(tmp_path))
    monkeypatch.setenv('HOME', str(tmp_path / 'home'))
    # The home-I/O guard treats sqlite file: URIs lexically relative to cwd.
    monkeypatch.chdir(tmp_path)
    monkeypatch.delenv('HERMES_KANBAN_DB', raising=False)
    monkeypatch.delenv('HERMES_DELEGATED_CHILD', raising=False)
    monkeypatch.setattr('hermes_cli.profiles.resolve_profile_env', lambda name: str(tmp_path))
    kb.create_board('owned')
    with closing(connect(board='owned')) as conn:
        tid = kb.create_task(conn, title=title, body=body, assignee='default',
                             workspace_kind='dir', workspace_path=str(tmp_path),
                             model_override=model, provider_override=provider)
        task = kb.claim_task(conn, tid)
    actor = SessionActor('owner', str(tmp_path), frozenset({'session:create'}), 'transport')
    connection = SimpleNamespace(authority=SimpleNamespace(profile_id=str(tmp_path)), actor=actor, native_owner=True)
    params = dict(board='owned', task_id=tid, run_id=task.current_run_id, claim_lock=task.claim_lock)
    config = {'model': {'default': 'claude-opus-4-6', 'provider': 'anthropic',
                        'base_url': 'https://old-route.example.test'},
              'platform_toolsets': {'cli': ['terminal', 'file']},
              'kanban': {'worker_model_policy': {
                  'default': {'model': 'glm-5.3', 'provider': 'zai'},
                  'visual': {'model': 'MiniMax-M3', 'provider': 'minimax'}}}}
    if expected is None:
        with pytest.raises(AuthorityRefusal):
            build_kanban_policy(connection, params, config)
        return
    policy, _ = build_kanban_policy(connection, params, config)
    for frozen in (policy, restore_policy(asdict(policy))):
        assert (frozen.model, frozen.provider) == expected
        request = json.loads(frozen.request_json)
        assert (request['model'], request['provider']) == expected
        assert frozen.source == 'kanban' and 'kanban' in frozen.toolsets
        assert json.loads(frozen.kanban_json)['task_id'] == tid
        if expected[1] != 'anthropic':
            assert frozen.base_url is None  # never carry the interactive provider's endpoint
    assert config['model']['provider'] == 'anthropic'  # construction never mutates profile config
