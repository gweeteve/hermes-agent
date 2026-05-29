"""Calendar tools for Judy's agentic agenda."""

from __future__ import annotations

import json
from typing import Any

from tools.registry import registry, tool_error


def _ok(**fields: Any) -> str:
    return json.dumps({"ok": True, **fields}, ensure_ascii=False)


def _call(fn, *args, **kwargs) -> str:
    try:
        return _ok(result=fn(*args, **kwargs))
    except Exception as exc:
        return tool_error(str(exc))


def _handle_add(args: dict, **kw) -> str:
    from hermes_cli import calendar_db

    return _call(
        calendar_db.add_event,
        title=args.get("title"),
        scheduled_at=args.get("scheduled_at"),
        description=args.get("description"),
        recurrence=args.get("recurrence"),
        tags=args.get("tags"),
        context=args.get("context"),
    )


def _handle_list(args: dict, **kw) -> str:
    from hermes_cli import calendar_db

    return _call(
        calendar_db.list_events,
        status=args.get("status"),
        from_=args.get("from"),
        to=args.get("to"),
        tags=args.get("tags"),
        limit=args.get("limit", 50),
    )


def _handle_upcoming(args: dict, **kw) -> str:
    from hermes_cli import calendar_db

    return _call(calendar_db.upcoming_events, limit=args.get("limit", 5))


def _handle_cancel(args: dict, **kw) -> str:
    from hermes_cli import calendar_db

    event = calendar_db.cancel_event(int(args.get("id")))
    if not event:
        return tool_error("calendar event not found")
    return _ok(result=event)


def _handle_update(args: dict, **kw) -> str:
    from hermes_cli import calendar_db

    event_id = int(args.get("id"))
    updates = {k: args.get(k) for k in ("title", "scheduled_at", "description", "tags", "context", "recurrence")}
    event = calendar_db.update_event(event_id, **updates)
    if not event:
        return tool_error("calendar event not found")
    return _ok(result=event)


def _handle_done(args: dict, **kw) -> str:
    from hermes_cli import calendar_db

    event = calendar_db.mark_done(int(args.get("id")), notes=args.get("notes"))
    if not event:
        return tool_error("calendar event not found")
    return _ok(result=event)


_EVENT_ID = {"type": "integer", "description": "Calendar event id."}
_TAGS = {"type": "array", "items": {"type": "string"}, "description": "Optional tags."}
_CONTEXT = {"type": "object", "description": "Free-form JSON context for the wakeup."}
_RECURRENCE = {
    "type": "string",
    "enum": ["daily", "weekly", "monthly"],
    "description": "Simple v1 recurrence. Omit for one-shot events.",
}


registry.register(
    name="calendar_add",
    toolset="calendar",
    schema={
        "name": "calendar_add",
        "description": "Create a Judy calendar event. scheduled_at must be an ISO-8601 timestamp with timezone.",
        "parameters": {
            "type": "object",
            "properties": {
                "title": {"type": "string"},
                "scheduled_at": {"type": "string"},
                "description": {"type": "string"},
                "recurrence": _RECURRENCE,
                "tags": _TAGS,
                "context": _CONTEXT,
            },
            "required": ["title", "scheduled_at"],
        },
    },
    handler=_handle_add,
    emoji="📅",
)

registry.register(
    name="calendar_list",
    toolset="calendar",
    schema={
        "name": "calendar_list",
        "description": "List Judy calendar events with optional status/date/tag filters.",
        "parameters": {
            "type": "object",
            "properties": {
                "status": {"type": "string", "enum": ["pending", "firing", "done", "cancelled"]},
                "from": {"type": "string"},
                "to": {"type": "string"},
                "tags": _TAGS,
                "limit": {"type": "integer", "description": "Default 50, max 200."},
            },
            "required": [],
        },
    },
    handler=_handle_list,
    emoji="📅",
)

registry.register(
    name="calendar_upcoming",
    toolset="calendar",
    schema={
        "name": "calendar_upcoming",
        "description": "List upcoming pending Judy calendar events.",
        "parameters": {
            "type": "object",
            "properties": {"limit": {"type": "integer", "description": "Default 5, max 200."}},
            "required": [],
        },
    },
    handler=_handle_upcoming,
    emoji="📅",
)

registry.register(
    name="calendar_cancel",
    toolset="calendar",
    schema={
        "name": "calendar_cancel",
        "description": "Cancel a Judy calendar event without deleting its row.",
        "parameters": {"type": "object", "properties": {"id": _EVENT_ID}, "required": ["id"]},
    },
    handler=_handle_cancel,
    emoji="🗑",
)

registry.register(
    name="calendar_update",
    toolset="calendar",
    schema={
        "name": "calendar_update",
        "description": "Update mutable fields on a Judy calendar event.",
        "parameters": {
            "type": "object",
            "properties": {
                "id": _EVENT_ID,
                "title": {"type": "string"},
                "scheduled_at": {"type": "string"},
                "description": {"type": "string"},
                "recurrence": _RECURRENCE,
                "tags": _TAGS,
                "context": _CONTEXT,
            },
            "required": ["id"],
        },
    },
    handler=_handle_update,
    emoji="✏",
)

registry.register(
    name="calendar_done",
    toolset="calendar",
    schema={
        "name": "calendar_done",
        "description": "Mark a calendar event done. Recurring events are advanced and returned to pending.",
        "parameters": {
            "type": "object",
            "properties": {"id": _EVENT_ID, "notes": {"type": "string"}},
            "required": ["id"],
        },
    },
    handler=_handle_done,
    emoji="✔",
)
