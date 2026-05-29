from unittest.mock import MagicMock

import pytest
from aiohttp import web
from aiohttp.test_utils import TestClient, TestServer

from gateway.config import PlatformConfig
from gateway.platforms.api_server import APIServerAdapter, cors_middleware


def _make_adapter(api_key: str = "") -> APIServerAdapter:
    extra = {}
    if api_key:
        extra["key"] = api_key
    return APIServerAdapter(PlatformConfig(enabled=True, extra=extra))


def _create_app(adapter: APIServerAdapter) -> web.Application:
    app = web.Application(middlewares=[cors_middleware])
    app["api_server_adapter"] = adapter
    app.router.add_get("/api/calendar", adapter._handle_calendar_list)
    app.router.add_post("/api/calendar", adapter._handle_calendar_create)
    app.router.add_get("/api/calendar/upcoming", adapter._handle_calendar_upcoming)
    app.router.add_get("/api/calendar/due", adapter._handle_calendar_due)
    app.router.add_patch("/api/calendar/{event_id:\\d+}", adapter._handle_calendar_update)
    app.router.add_delete("/api/calendar/{event_id:\\d+}", adapter._handle_calendar_delete)
    return app


@pytest.mark.asyncio
async def test_calendar_create(adapter=None):
    adapter = _make_adapter()
    fake_db = MagicMock()
    fake_db.add_event.return_value = {"id": 1, "title": "Wake"}
    adapter._calendar_db = lambda: fake_db
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.post("/api/calendar", json={
            "title": "Wake",
            "scheduled_at": "2026-05-27T10:00:00Z",
            "tags": ["memoire"],
        })
        data = await resp.json()

    assert resp.status == 201
    assert data["event"]["id"] == 1
    fake_db.add_event.assert_called_once()


@pytest.mark.asyncio
async def test_calendar_due_claims_events():
    adapter = _make_adapter()
    fake_db = MagicMock()
    fake_db.claim_due_events.return_value = [{"id": 2, "status": "firing"}]
    adapter._calendar_db = lambda: fake_db
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/calendar/due?limit=3")
        data = await resp.json()

    assert resp.status == 200
    assert data["events"][0]["status"] == "firing"
    fake_db.claim_due_events.assert_called_once_with(limit=3)


@pytest.mark.asyncio
async def test_calendar_delete_soft_cancels():
    adapter = _make_adapter()
    fake_db = MagicMock()
    fake_db.cancel_event.return_value = {"id": 3, "status": "cancelled"}
    adapter._calendar_db = lambda: fake_db
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.delete("/api/calendar/3")
        data = await resp.json()

    assert resp.status == 200
    assert data["event"]["status"] == "cancelled"
    fake_db.cancel_event.assert_called_once_with(3)


@pytest.mark.asyncio
async def test_calendar_auth_enforced():
    adapter = _make_adapter(api_key="sk-secret")
    adapter._calendar_db = lambda: MagicMock()
    app = _create_app(adapter)

    async with TestClient(TestServer(app)) as cli:
        resp = await cli.get("/api/calendar")

    assert resp.status == 401
