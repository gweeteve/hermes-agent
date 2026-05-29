---
name: judy-calendar
description: Use when Judy needs to manage her agentic calendar, create voluntary wakeups, list upcoming calendar events, cancel or update events, or mark a scheduled wakeup complete with calendar_done.
---

# Judy Calendar

This is Judy's agentic calendar: voluntary reminders and wakeups stored in Hermes, not Google Calendar.

Use these tools directly:

- `calendar_add(title, scheduled_at, description?, recurrence?, tags?, context?)`
- `calendar_list(status?, from?, to?, tags?, limit?)`
- `calendar_upcoming(limit?)`
- `calendar_cancel(id)`
- `calendar_update(id, title?, scheduled_at?, description?, recurrence?, tags?, context?)`
- `calendar_done(id, notes?)`

## Wakeup Flow

When a planned wakeup arrives, the prompt includes the calendar event id. Do the planned work, then call:

```text
calendar_done(id=<event_id>, notes="<short factual result>")
```

Call `calendar_done` only after the wakeup has actually been handled. If the event should not be handled anymore, use `calendar_cancel` instead.

Recurring events (`daily`, `weekly`, `monthly`) are advanced automatically by `calendar_done`; do not create a duplicate event for the next occurrence.

## Timestamp Rules

Use ISO-8601 timestamps with an explicit timezone, for example:

```text
2026-05-28T10:00:00Z
2026-05-28T10:00:00+02:00
```

Do not use naive timestamps without timezone.
