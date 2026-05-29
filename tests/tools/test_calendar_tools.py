import json

from hermes_cli import calendar_db
from tools import calendar_tools


def test_calendar_add_handler(monkeypatch):
    called = {}

    def fake_add_event(**kwargs):
        called.update(kwargs)
        return {"id": 1, "title": kwargs["title"]}

    monkeypatch.setattr(calendar_db, "add_event", fake_add_event)

    result = json.loads(calendar_tools._handle_add({
        "title": "Wake",
        "scheduled_at": "2026-05-27T10:00:00Z",
        "tags": ["memoire"],
    }))

    assert result["ok"] is True
    assert result["result"]["id"] == 1
    assert called["title"] == "Wake"
    assert called["tags"] == ["memoire"]


def test_calendar_done_handler_missing_event(monkeypatch):
    monkeypatch.setattr(calendar_db, "mark_done", lambda *a, **kw: None)

    result = json.loads(calendar_tools._handle_done({"id": 404}))

    assert "not found" in result["error"]
