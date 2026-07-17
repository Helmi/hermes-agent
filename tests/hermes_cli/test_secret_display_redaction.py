"""Regression tests for presence-only secret display in CLI diagnostics.

When ``security.redact_secrets`` is enabled (the default), ``hermes config``
and ``hermes status`` must emit NO credential characters or fragments in
their API Keys sections — only presence state (``(set)`` / ``(not set)``).
With redaction explicitly disabled, masked prefix/suffix fragments remain
(the pre-existing debuggability behavior).

All credentials in this file are DUMMY values that can never authenticate.
"""

from types import SimpleNamespace

from hermes_cli.config import (
    _secret_redaction_enabled,
    get_hermes_home,
    secret_display_label,
)

# DUMMY credentials — long enough that mask_secret would normally preserve a
# prefix/suffix, and carrying a unique sentinel so we can assert its absence.
_SENTINEL = "LEAKSENTINEL"
_DUMMY_OPENROUTER = f"sk-or-{_SENTINEL}aaaaBBBB4321"  # ~30 chars
_DUMMY_FIRECRAWL = f"fc-{_SENTINEL}ccccDDDD8765"      # ~24 chars
_DUMMY_SHORT = "short1234"


def _set_redact(redact: bool) -> None:
    """Write a config.yaml with only the security.redact_secrets toggle.

    load_config() deep-merges this over DEFAULT_CONFIG, so every other
    setting keeps its default. The file lands in the per-test HERMES_HOME
    that the autouse ``_isolate_hermes_home`` fixture redirects to.
    """
    cfg_path = get_hermes_home() / "config.yaml"
    cfg_path.write_text(
        f"security:\n  redact_secrets: {str(redact).lower()}\n",
        encoding="utf-8",
    )


# ---------------------------------------------------------------------------
# Unit tests — the helper and the flag reader
# ---------------------------------------------------------------------------


class TestSecretDisplayLabel:
    def test_presence_only_when_redact_true(self):
        assert secret_display_label(_DUMMY_OPENROUTER, redact=True) == "(set)"

    def test_presence_only_for_short_value_when_redact_true(self):
        # Even a short value reports presence only — no "***" placeholder leak.
        assert secret_display_label(_DUMMY_SHORT, redact=True) == "(set)"

    def test_not_set_placeholder_when_empty_redact_true(self):
        label = secret_display_label("", redact=True)
        assert "not set" in label.lower()

    def test_not_set_placeholder_when_none(self):
        label = secret_display_label(None, redact=True)
        assert "not set" in label.lower()

    def test_masked_fragment_when_redact_false(self):
        # Compat contract: masked prefix/suffix preserved when disabled.
        label = secret_display_label(_DUMMY_OPENROUTER, redact=False)
        assert label.startswith("sk-o")
        assert label.endswith("4321")
        assert "..." in label
        # The full value and the hidden middle never appear.
        assert _DUMMY_OPENROUTER not in label
        assert _SENTINEL not in label

    def test_masked_short_value_when_redact_false(self):
        # Below the mask floor → fully masked placeholder.
        assert secret_display_label(_DUMMY_SHORT, redact=False) == "***"


class TestSecretRedactionFlag:
    def test_defaults_true_when_unset(self):
        # No config.yaml in the temp HERMES_HOME → DEFAULT_CONFIG → True.
        assert _secret_redaction_enabled() is True

    def test_true_when_explicitly_enabled(self):
        _set_redact(True)
        assert _secret_redaction_enabled() is True

    def test_false_when_explicitly_disabled(self):
        _set_redact(False)
        assert _secret_redaction_enabled() is False


# ---------------------------------------------------------------------------
# E2E — the real show_config rendering path against a temp HERMES_HOME
# ---------------------------------------------------------------------------


class TestShowConfigRedaction:
    def test_no_credential_fragments_when_redaction_enabled(self, monkeypatch, capsys):
        _set_redact(True)
        monkeypatch.setenv("OPENROUTER_API_KEY", _DUMMY_OPENROUTER)
        monkeypatch.setenv("FIRECRAWL_API_KEY", _DUMMY_FIRECRAWL)

        from hermes_cli.config import show_config

        show_config()
        out = capsys.readouterr().out

        # Presence is reported for configured keys; absence for others.
        assert "(set)" in out
        assert "not set" in out.lower()
        # No dummy credential value leaks.
        assert _DUMMY_OPENROUTER not in out
        assert _DUMMY_FIRECRAWL not in out
        # No recognizable fragment / sentinel leaks.
        assert _SENTINEL not in out
        assert "sk-or-" not in out
        assert "fc-" not in out
        # Distinctive tail characters must not appear.
        assert "4321" not in out
        assert "8765" not in out

    def test_masked_fragments_when_redaction_disabled(self, monkeypatch, capsys):
        _set_redact(False)
        monkeypatch.setenv("OPENROUTER_API_KEY", _DUMMY_OPENROUTER)

        from hermes_cli.config import show_config

        show_config()
        out = capsys.readouterr().out

        # Compat contract: masked prefix/suffix present, full value absent.
        assert "sk-o" in out
        assert "4321" in out
        assert "..." in out
        assert _DUMMY_OPENROUTER not in out
        assert _SENTINEL not in out
        # Presence-only label must NOT appear when redaction is off.
        assert "(set)" not in out


# ---------------------------------------------------------------------------
# E2E — the real show_status rendering path (API Keys section)
# ---------------------------------------------------------------------------


class TestShowStatusRedaction:
    def test_api_keys_section_no_fragments_when_redaction_enabled(self, monkeypatch, capsys):
        _set_redact(True)
        monkeypatch.setenv("OPENROUTER_API_KEY", _DUMMY_OPENROUTER)

        from hermes_cli.status import show_status

        show_status(SimpleNamespace(all=False, deep=False))
        out = capsys.readouterr().out

        assert "(set)" in out
        assert _DUMMY_OPENROUTER not in out
        assert _SENTINEL not in out
        assert "sk-or-" not in out

    def test_status_all_still_redacts(self, monkeypatch, capsys):
        # `hermes status --all` used to be a full-value escape hatch; upstream
        # closed it in c080a530a ("redact status API keys with --all"). The
        # presence-only label must hold on that path too.
        _set_redact(True)
        monkeypatch.setenv("OPENROUTER_API_KEY", _DUMMY_OPENROUTER)

        from hermes_cli.status import show_status

        show_status(SimpleNamespace(all=True, deep=False))
        out = capsys.readouterr().out

        assert "(set)" in out
        assert _DUMMY_OPENROUTER not in out
        assert _SENTINEL not in out
