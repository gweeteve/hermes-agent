import json
from pathlib import Path
from types import SimpleNamespace

from gateway import data_isolation
from gateway.session_context import clear_session_vars, set_session_vars


def _reset_isolation(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    monkeypatch.setenv("LANG", "C.UTF-8")
    data_isolation.reload_config(force=True, path=tmp_path / "data_isolation.json")


def test_default_config_is_guest_and_restrictive(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    cfg = data_isolation.reload_config()

    assert cfg["version"] == 2
    assert cfg["guest_allowed_tools"] == ["vision_analyze", "web_extract", "web_search"]
    assert "send_message_list" in cfg["trusted_allowed_tools"]
    assert cfg["trusted_read_paths"] == ["/workspace/homes/{identity_key}", "/workspace/shared"]
    assert cfg["trusted_write_paths"] == ["/workspace/homes/{identity_key}"]
    assert data_isolation.level_for_identity("telegram:user:alice") == "guest"
    assert data_isolation.check_tool_access(
        "terminal",
        {},
        identity_key="telegram:user:alice",
        level="guest",
    ).allowed is False


def test_v1_config_migrates_to_v2_without_losing_state(monkeypatch, tmp_path):
    monkeypatch.setenv("HERMES_HOME", str(tmp_path))
    config_file = tmp_path / "data_isolation.json"
    config_file.write_text(
        json.dumps(
            {
                "version": 1,
                "enabled": True,
                "default_level": "guest",
                "guest_allowed_tools": [],
                "trusted_allowed_tools": [],
                "contacts": {"telegram:user:alice": {"level": "trusted", "display_name": "Alice"}},
                "project_grants": [{"identity_key": "telegram:user:alice", "tools": ["terminal"]}],
                "denial_counts": {"telegram:user:bob": {"total": 1, "tools": {"terminal": 1}}},
            }
        )
    )

    cfg = data_isolation.reload_config(force=True, path=config_file)

    assert cfg["version"] == 2
    assert cfg["contacts"]["telegram:user:alice"]["level"] == "trusted"
    assert cfg["project_grants"][0]["tools"] == ["terminal"]
    assert cfg["denial_counts"]["telegram:user:bob"]["total"] == 1
    assert cfg["guest_allowed_tools"] == ["vision_analyze", "web_extract", "web_search"]
    assert "send_message_send" in cfg["trusted_allowed_tools"]
    assert json.loads(config_file.read_text())["version"] == 2


def test_trusted_paths_are_canonicalized(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    identity = "telegram:user:alice"

    allowed = data_isolation.check_tool_access(
        "write_file",
        {"path": f"/workspace/homes/{identity}/notes.txt"},
        identity_key=identity,
        level="trusted",
    )
    denied = data_isolation.check_tool_access(
        "write_file",
        {"path": f"/workspace/homes/{identity}/../other/notes.txt"},
        identity_key=identity,
        level="trusted",
    )
    shared_read = data_isolation.check_tool_access(
        "read_file",
        {"path": "/workspace/shared/readme.md"},
        identity_key=identity,
        level="trusted",
    )
    shared_write = data_isolation.check_tool_access(
        "write_file",
        {"path": "/workspace/shared/readme.md"},
        identity_key=identity,
        level="trusted",
    )

    assert allowed.allowed is True
    assert denied.allowed is False
    assert shared_read.allowed is True
    assert shared_write.allowed is False


def test_trusted_paths_are_configurable(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    cfg = data_isolation.empty_config()
    cfg["trusted_read_paths"] = ["/workspace/private/{identity_key}"]
    cfg["trusted_write_paths"] = ["/workspace/private/{identity_key}/drafts"]
    data_isolation.save_config(cfg, tmp_path / "data_isolation.json")

    identity = "telegram:user:alice"
    assert data_isolation.check_tool_access(
        "read_file",
        {"path": "/workspace/private/telegram:user:alice/notes.txt"},
        identity_key=identity,
        level="trusted",
    ).allowed is True
    assert data_isolation.check_tool_access(
        "write_file",
        {"path": "/workspace/private/telegram:user:alice/notes.txt"},
        identity_key=identity,
        level="trusted",
    ).allowed is False
    assert data_isolation.check_tool_access(
        "write_file",
        {"path": "/workspace/private/telegram:user:alice/drafts/notes.txt"},
        identity_key=identity,
        level="trusted",
    ).allowed is True


def test_search_files_requires_explicit_trusted_path(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)

    decision = data_isolation.check_tool_access(
        "search_files",
        {},
        identity_key="telegram:user:alice",
        level="trusted",
    )

    assert decision.allowed is False
    assert decision.reason == "missing path"


def test_guest_defaults_allow_only_conversation_tools(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)

    assert data_isolation.check_tool_access("web_search", {}, level="guest").allowed is True
    assert data_isolation.check_tool_access("web_extract", {}, level="guest").allowed is True
    assert data_isolation.check_tool_access("vision_analyze", {}, level="guest").allowed is True
    assert data_isolation.check_tool_access("send_message", {"action": "list"}, level="guest").allowed is False
    assert data_isolation.check_tool_access("send_message", {"action": "send"}, level="guest").allowed is False


def test_trusted_send_message_actions_and_known_contacts(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    cfg = data_isolation.empty_config()
    cfg["contacts"] = {
        "telegram:user:888933588": {"level": "admin", "display_name": "Admin"},
        "telegram:user:428830317": {"level": "trusted", "display_name": "Elva"},
    }
    data_isolation.save_config(cfg, tmp_path / "data_isolation.json")

    assert data_isolation.can_advertise_tool(
        "send_message",
        identity_key="telegram:user:341294910",
        level="trusted",
    )
    assert data_isolation.check_tool_access(
        "send_message",
        {"action": "list"},
        identity_key="telegram:user:341294910",
        level="trusted",
    ).allowed is True
    assert data_isolation.check_tool_access(
        "send_message",
        {"action": "send", "target": "telegram:user:428830317", "message": "Salut"},
        identity_key="telegram:user:341294910",
        level="trusted",
    ).allowed is True
    assert data_isolation.check_tool_access(
        "send_message",
        {"action": "send", "target": "telegram:888933588", "message": "Salut"},
        identity_key="telegram:user:341294910",
        level="trusted",
    ).allowed is True
    assert data_isolation.check_tool_access(
        "send_message",
        {"action": "send", "target": "telegram:user:999999999", "message": "Salut"},
        identity_key="telegram:user:341294910",
        level="trusted",
    ).allowed is False


def test_trusted_send_message_allows_current_channel_only(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    tokens = set_session_vars(
        platform="telegram",
        chat_id="341294910",
        session_key="telegram:dm:341294910",
        identity_key="telegram:user:341294910",
        data_isolation_level="trusted",
    )
    try:
        assert data_isolation.check_tool_access(
            "send_message",
            {"target": "telegram", "message": "Salut"},
            identity_key="telegram:user:341294910",
            level="trusted",
        ).allowed is True
        assert data_isolation.check_tool_access(
            "send_message",
            {"target": "discord", "message": "Salut"},
            identity_key="telegram:user:341294910",
            level="trusted",
        ).allowed is False
    finally:
        clear_session_vars(tokens)


def test_admin_bypasses_unknown_tools(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)

    assert data_isolation.can_advertise_tool("unknown_tool", level="admin") is True
    assert data_isolation.check_tool_access("unknown_tool", {}, level="admin").allowed is True


def test_configured_admin_overrides_stale_context_level(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    cfg = data_isolation.empty_config()
    cfg["contacts"] = {
        "telegram:user:admin": {"level": "admin", "display_name": "Admin"},
    }
    data_isolation.save_config(cfg, tmp_path / "data_isolation.json")

    assert data_isolation.can_advertise_tool(
        "read_file",
        identity_key="telegram:user:admin",
        level="guest",
    ) is True
    assert data_isolation.check_tool_access(
        "read_file",
        {"path": "/workspace/projects/persona/inner_state.json"},
        identity_key="telegram:user:admin",
        level="guest",
    ).allowed is True


def test_calendar_home_admin_can_advertise_memory_but_guest_synthetic_cannot(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    cfg = data_isolation.empty_config()
    cfg["contacts"] = {
        "telegram:user:888933588": {"level": "admin", "display_name": "Admin"},
    }
    data_isolation.save_config(cfg, tmp_path / "data_isolation.json")

    assert data_isolation.can_advertise_tool(
        "hindsight_retain",
        identity_key="telegram:user:888933588",
        level="guest",
    ) is True
    assert data_isolation.can_advertise_tool(
        "hindsight_retain",
        identity_key="telegram:user:system:calendar",
        level="guest",
    ) is False


def test_active_project_grant_overrides_role_tools(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    cfg = data_isolation.empty_config()
    cfg["project_grants"] = [{"identity_key": "telegram:user:alice", "tools": ["terminal"]}]
    data_isolation.save_config(cfg, tmp_path / "data_isolation.json")

    decision = data_isolation.check_tool_access(
        "terminal",
        {},
        identity_key="telegram:user:alice",
        level="guest",
    )

    assert decision.allowed is True


def test_expired_project_grant_is_denied(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    cfg = data_isolation.empty_config()
    cfg["project_grants"] = [
        {
            "identity_key": "telegram:user:alice",
            "tools": ["terminal"],
            "expires_at": "2000-01-01T00:00:00Z",
        }
    ]
    data_isolation.save_config(cfg, tmp_path / "data_isolation.json")

    decision = data_isolation.check_tool_access(
        "terminal",
        {},
        identity_key="telegram:user:alice",
        level="guest",
    )

    assert decision.allowed is False


def test_schema_filter_uses_explicit_context_not_env(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    monkeypatch.setenv("HERMES_SESSION_KEY", "stale-env-session")
    monkeypatch.setenv("HERMES_IDENTITY_KEY", "telegram:user:alice")
    monkeypatch.setenv("HERMES_DATA_ISOLATION_LEVEL", "guest")
    import model_tools

    tools = [
        {"type": "function", "function": {"name": "terminal"}},
        {"type": "function", "function": {"name": "read_file"}},
    ]

    assert model_tools._filter_data_isolation_tool_definitions(tools) == tools


def test_empty_session_key_does_not_activate_data_isolation(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    import model_tools

    tokens = set_session_vars(session_key="")
    try:
        tools = [
            {"type": "function", "function": {"name": "terminal"}},
            {"type": "function", "function": {"name": "read_file"}},
        ]
        assert model_tools._filter_data_isolation_tool_definitions(tools) == tools
        assert model_tools._data_isolation_guard("terminal", {}, "default") is None
    finally:
        clear_session_vars(tokens)


def test_schema_filter_and_runtime_guard_for_gateway_context(monkeypatch, tmp_path):
    _reset_isolation(monkeypatch, tmp_path)
    import model_tools

    tokens = set_session_vars(
        session_key="telegram:dm:1",
        identity_key="telegram:user:alice",
        data_isolation_level="guest",
    )
    try:
        tools = [
            {"type": "function", "function": {"name": "terminal"}},
            {"type": "function", "function": {"name": "read_file"}},
        ]
        assert model_tools._filter_data_isolation_tool_definitions(tools) == []

        result = json.loads(model_tools._data_isolation_guard("terminal", {}, "default"))
        assert result == {"error": "Je ne peux pas faire ça."}
    finally:
        clear_session_vars(tokens)


def test_contacts_cli_set_level_and_grant(monkeypatch, tmp_path, capsys):
    _reset_isolation(monkeypatch, tmp_path)
    from hermes_cli.contacts import contacts_command

    contacts_command(
        SimpleNamespace(
            contacts_action="set-level",
            identity_key="telegram:user:alice",
            level="trusted",
            display_name="Alice",
        )
    )
    contacts_command(
        SimpleNamespace(
            contacts_action="grant",
            identity_key="telegram:user:alice",
            tools="read_file,write_file",
            path=str(tmp_path / "project"),
            expires_at="2099-01-01T00:00:00Z",
        )
    )

    out = capsys.readouterr().out
    assert "trusted" in out
    assert "read_file, write_file" in out
    assert data_isolation.level_for_identity("telegram:user:alice") == "trusted"
    assert data_isolation.reload_config()["project_grants"][0]["tools"] == ["read_file", "write_file"]
