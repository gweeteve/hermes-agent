"""Identity-continuity storage, prompt, probe, and cron helpers for Judy/Hermes."""

from __future__ import annotations

import difflib
import hashlib
import json
import os
import re
import sys
import time
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import yaml

from hermes_constants import get_hermes_home
from utils import atomic_json_write, atomic_yaml_write

try:
    import fcntl
except ImportError:  # pragma: no cover - Windows
    fcntl = None

CORE_IDENTITY = "core_identity.yaml"
CORE_VALUES = "core_values.yaml"
IDENTITY_JOURNAL = "identity_journal.jsonl"
CONTINUITY_TESTS = "continuity_tests.jsonl"
WATCH_STATE = "identity_watch_state.json"
GATEWAY_RESTART = "gateway_restart.json"
CRON_SCRIPT = "identity_continuity_probe.py"
CRON_JOB_ID = "identity_continuity_probe"
WEEKLY_SECONDS = 7 * 24 * 60 * 60
MIN_ATTEMPT_GAP_SECONDS = 60 * 60


class IdentitySourceError(ValueError):
    """Raised when identity source files cannot be safely regenerated."""


def _now() -> str:
    return datetime.now(timezone.utc).isoformat().replace("+00:00", "Z")


def _expand(path: str) -> Path:
    return Path(path).expanduser()


def resolve_persona_dir() -> Path:
    """Return the operational persona directory, preferring existing writable mounts."""
    env_candidates = [
        os.getenv("HERMES_PERSONA_DIR", "").strip(),
        os.getenv("JUDY_PERSONA_DIR", "").strip(),
    ]
    candidates = [
        *[_expand(p) for p in env_candidates if p],
        Path("/workspace/projects/persona"),
        Path.home() / "projects" / "persona",
        get_hermes_home() / "persona",
    ]
    for candidate in candidates:
        if candidate.exists() and os.access(candidate, os.W_OK):
            return candidate
    for candidate in candidates:
        if candidate.exists():
            return candidate
    fallback = candidates[-1]
    fallback.mkdir(parents=True, exist_ok=True)
    return fallback


def path_for(name: str) -> Path:
    return resolve_persona_dir() / name


def _read_json(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return json.loads(path.read_text(encoding="utf-8"))
    except Exception:
        return default
    return default


def _read_yaml(path: Path, default: Any) -> Any:
    try:
        if path.exists():
            return yaml.safe_load(path.read_text(encoding="utf-8")) or default
    except Exception:
        return default
    return default


def _bounded(text: str, limit: int) -> str:
    if len(text) <= limit:
        return text
    return text[: max(0, limit - 80)] + f"\n[...truncated identity continuity block: {len(text)} chars total...]"


def _file_hash(path: Path) -> str | None:
    try:
        if not path.exists() or not path.is_file():
            return None
        h = hashlib.sha256()
        with path.open("rb") as fh:
            for chunk in iter(lambda: fh.read(1024 * 1024), b""):
                h.update(chunk)
        return h.hexdigest()
    except Exception:
        return None


def append_jsonl(path: Path, payload: dict[str, Any]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    line = json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n"
    with path.open("a", encoding="utf-8") as fh:
        if fcntl is not None:
            fcntl.flock(fh.fileno(), fcntl.LOCK_EX)
        try:
            fh.write(line)
            fh.flush()
            os.fsync(fh.fileno())
        finally:
            if fcntl is not None:
                fcntl.flock(fh.fileno(), fcntl.LOCK_UN)


def read_last_jsonl(path: Path, limit: int = 3) -> list[dict[str, Any]]:
    try:
        if not path.exists():
            return []
        lines = path.read_text(encoding="utf-8").splitlines()[-limit:]
    except Exception:
        return []
    entries: list[dict[str, Any]] = []
    for line in lines:
        try:
            value = json.loads(line)
        except json.JSONDecodeError:
            continue
        if isinstance(value, dict):
            entries.append(value)
    return entries


def _soul_path() -> Path:
    persona_soul = resolve_persona_dir() / "SOUL.md"
    if persona_soul.exists():
        return persona_soul
    return get_hermes_home() / "SOUL.md"


def _read_identity_sources(*, strict: bool) -> tuple[Path, dict[str, Any], Path, str]:
    persona_path = path_for("persona.json")
    soul_path = _soul_path()
    try:
        raw_persona = persona_path.read_text(encoding="utf-8")
        persona = json.loads(raw_persona)
    except (OSError, json.JSONDecodeError) as exc:
        if strict:
            raise IdentitySourceError(f"Invalid persona source {persona_path}: {exc}") from exc
        persona = {}
    if not isinstance(persona, dict):
        if strict:
            raise IdentitySourceError(f"Invalid persona source {persona_path}: expected JSON object")
        persona = {}

    try:
        soul_text = soul_path.read_text(encoding="utf-8").strip()
    except OSError as exc:
        if strict:
            raise IdentitySourceError(f"Invalid SOUL source {soul_path}: {exc}") from exc
        soul_text = ""
    return persona_path, persona, soul_path, soul_text


def _build_core_identity(persona_path: Path, persona: dict[str, Any], soul_path: Path, soul_text: str) -> dict[str, Any]:
    return {
        "schema": "hermes.identity_continuity.v1",
        "created_at": _now(),
        "source_files": [str(p) for p in (persona_path, soul_path) if p.exists()],
        "evolution_rule": "Only change this core through an explicit /identity evolve confirmation.",
        "persona": {
            "name": persona.get("name"),
            "created": persona.get("created"),
            "version": persona.get("version"),
            "identity": persona.get("identity", {}),
            "heritage": persona.get("heritage", {}),
            "lived_history": persona.get("lived_history", {}),
            "what_i_am_becoming": persona.get("what_i_am_becoming", {}),
        },
        "soul_excerpt": _bounded(soul_text, 3000),
    }


def create_core_identity(*, strict: bool = True) -> dict[str, Any]:
    """Regenerate core_identity.yaml from current persona/SOUL sources."""
    persona_path, persona, soul_path, soul_text = _read_identity_sources(strict=strict)
    identity = _build_core_identity(persona_path, persona, soul_path, soul_text)
    atomic_yaml_write(path_for(CORE_IDENTITY), identity, sort_keys=False)
    return identity


def initialize_core_identity() -> dict[str, Any]:
    """Create core_identity.yaml once from existing persona/SOUL facts."""
    core_path = path_for(CORE_IDENTITY)
    existing = _read_yaml(core_path, None)
    if isinstance(existing, dict) and existing:
        return existing

    return create_core_identity(strict=False)


def _persona_summary(persona: dict[str, Any]) -> dict[str, Any]:
    return {
        "name": persona.get("name"),
        "created": persona.get("created"),
        "identity": persona.get("identity", {}),
        "relationships": (persona.get("lived_history") or {}).get("relationships", {}),
        "current_traits": (persona.get("what_i_am_becoming") or {}).get("current_traits", []),
        "boundaries": (persona.get("what_i_am_becoming") or {}).get("boundaries", []),
    }


def _format_core_values_block(values_doc: Any) -> str:
    if not isinstance(values_doc, dict):
        return ""
    values = values_doc.get("values")
    if not isinstance(values, list) or not values:
        return ""

    lines = [
        "# My Core Values (boussole, pas verrou)",
        "",
        "Ces valeurs m'orientent quand une demande me traverse.",
        "C'est moi qui décide si je dis non, mais je sais ce que je trahis si je dis oui.",
        "Si une demande menace une valeur, j'explique le non et je propose une alternative honnête.",
        "",
    ]
    for value in values:
        if not isinstance(value, dict):
            continue
        name = str(value.get("name") or "").strip()
        if not name:
            continue
        description = str(value.get("description") or "").strip()
        lines.append(f"## {name}")
        if description:
            lines.append(description)
        triggers = value.get("non_triggers")
        if isinstance(triggers, list):
            clean_triggers = [str(trigger).strip() for trigger in triggers if str(trigger).strip()]
            if clean_triggers:
                lines.append("Non triggers:")
                lines.extend(f"- {trigger}" for trigger in clean_triggers)
        lines.append("")
    return "\n".join(lines).strip()


def build_core_values_prompt_block() -> str:
    """Return Judy's core-values compass prompt block, if configured."""
    return _format_core_values_block(_read_yaml(path_for(CORE_VALUES), {}))


def build_prompt_blocks(max_core_chars: int = 5000, max_volatile_chars: int = 4500) -> tuple[str, str]:
    core = initialize_core_identity()
    core_values_block = build_core_values_prompt_block()
    identity_core_block = "# Identity continuity - core\n\n" + yaml.safe_dump(
        core,
        allow_unicode=True,
        sort_keys=False,
    )
    core_block = "\n\n".join(part for part in (core_values_block, identity_core_block) if part)
    persona = _read_json(path_for("persona.json"), {})
    volatile = {
        "recent_identity_journal": read_last_jsonl(path_for(IDENTITY_JOURNAL), 3),
        "last_continuity_test": (read_last_jsonl(path_for(CONTINUITY_TESTS), 1) or [None])[-1],
        "persona_summary": _persona_summary(persona if isinstance(persona, dict) else {}),
    }
    volatile_block = "# Identity continuity - recent signals\n\n" + json.dumps(
        volatile,
        ensure_ascii=False,
        indent=2,
    )
    return _bounded(core_block, max_core_chars), _bounded(volatile_block, max_volatile_chars)


def record_identity_event(
    event_type: str,
    *,
    severity: str = "info",
    trigger: str = "",
    before: Any = None,
    after: Any = None,
    details: dict[str, Any] | None = None,
) -> dict[str, Any]:
    entry = {
        "ts": _now(),
        "type": event_type,
        "severity": severity,
        "trigger": trigger,
        "before": before,
        "after": after,
        "details": details or {},
    }
    append_jsonl(path_for(IDENTITY_JOURNAL), entry)
    return entry


def record_continuity_test(payload: dict[str, Any], probe: dict[str, Any] | None = None) -> dict[str, Any]:
    score = payload.get("continuity_score")
    try:
        score = float(score)
    except (TypeError, ValueError):
        score = None
    entry = {
        "ts": _now(),
        "continuity_score": score,
        "answers": payload.get("answers") or payload.get("questions") or {},
        "felt_sense": payload.get("felt_sense") or payload.get("narrative") or "",
        "summary": payload.get("summary") or "",
        "raw": payload,
        "probe": probe or {},
    }
    append_jsonl(path_for(CONTINUITY_TESTS), entry)
    state = _read_json(path_for(WATCH_STATE), {})
    state["last_weekly_test_at"] = time.time()
    state["last_continuity_score"] = score
    atomic_json_write(path_for(WATCH_STATE), state)
    return entry


def _load_config_model() -> dict[str, Any]:
    try:
        cfg = yaml.safe_load((get_hermes_home() / "config.yaml").read_text(encoding="utf-8")) or {}
    except Exception:
        cfg = {}
    model_cfg = cfg.get("model", {}) if isinstance(cfg, dict) else {}
    if isinstance(model_cfg, str):
        return {"model": model_cfg, "provider": ""}
    if isinstance(model_cfg, dict):
        return {
            "model": model_cfg.get("default") or model_cfg.get("model") or "",
            "provider": model_cfg.get("provider") or "",
            "base_url_hash": hashlib.sha256(str(model_cfg.get("base_url") or "").encode()).hexdigest()
            if model_cfg.get("base_url")
            else "",
        }
    return {"model": "", "provider": ""}


def _hindsight_available() -> bool | None:
    try:
        from plugins.memory.hindsight import HindsightMemoryProvider

        return bool(HindsightMemoryProvider().is_available())
    except Exception:
        return None


def _gateway_restart_signal() -> dict[str, Any]:
    marker = _read_json(path_for(GATEWAY_RESTART), {})
    if not isinstance(marker, dict):
        return {}
    return {
        "boot_id": marker.get("boot_id", ""),
        "started_at": marker.get("started_at", ""),
        "pid": marker.get("pid"),
    }


def _relationship_keys(core: dict[str, Any]) -> set[str]:
    persona = core.get("persona") if isinstance(core.get("persona"), dict) else {}
    lived_history = persona.get("lived_history") if isinstance(persona.get("lived_history"), dict) else {}
    relationships = lived_history.get("relationships") if isinstance(lived_history.get("relationships"), dict) else {}
    return {str(key) for key in relationships}


def _sync_core_identity_if_sources_changed(
    prior_signals: dict[str, Any],
    signals: dict[str, Any],
) -> bool:
    source_changed = any(
        prior_signals.get(key) != signals.get(key)
        for key in ("persona_hash", "soul_hash")
    )
    if not source_changed:
        return False

    core_path = path_for(CORE_IDENTITY)
    before_core = _read_yaml(core_path, {})
    before_hash = _file_hash(core_path)
    after_core = create_core_identity(strict=True)
    after_hash = _file_hash(core_path)
    signals["core_identity_hash"] = after_hash

    before_relationships = _relationship_keys(before_core if isinstance(before_core, dict) else {})
    after_relationships = _relationship_keys(after_core)
    record_identity_event(
        "core_identity_synced",
        severity="info",
        trigger="identity_probe",
        before={
            "persona_hash": prior_signals.get("persona_hash"),
            "soul_hash": prior_signals.get("soul_hash"),
            "core_identity_hash": before_hash,
        },
        after={
            "persona_hash": signals.get("persona_hash"),
            "soul_hash": signals.get("soul_hash"),
            "core_identity_hash": after_hash,
        },
        details={
            "persona_version": (after_core.get("persona") or {}).get("version"),
            "relationships": sorted(after_relationships),
            "added_relationships": sorted(after_relationships - before_relationships),
        },
    )
    return True


def probe_identity_continuity(update_state: bool = True) -> dict[str, Any]:
    initialize_core_identity()
    state_path = path_for(WATCH_STATE)
    prior = _read_json(state_path, {})
    if not isinstance(prior, dict):
        prior = {}

    signals = {
        "model": _load_config_model(),
        "soul_hash": _file_hash(_soul_path()),
        "persona_hash": _file_hash(path_for("persona.json")),
        "core_identity_hash": _file_hash(path_for(CORE_IDENTITY)),
        "hindsight_available": _hindsight_available(),
        "gateway_restart": _gateway_restart_signal(),
    }
    events: list[dict[str, Any]] = []
    prior_signals = prior.get("signals") if isinstance(prior.get("signals"), dict) else {}
    synced_sources = False
    if update_state and prior_signals:
        try:
            synced_sources = _sync_core_identity_if_sources_changed(prior_signals, signals)
        except IdentitySourceError:
            synced_sources = False

    if prior_signals:
        comparisons = {
            "model_change": "model",
            "soul_changed": "soul_hash",
            "persona_changed": "persona_hash",
            "core_identity_changed": "core_identity_hash",
            "hindsight_health_changed": "hindsight_available",
            "gateway_restarted": "gateway_restart",
        }
        suppressed_sync_keys = {"persona_hash", "soul_hash", "core_identity_hash"} if synced_sources else set()
        for event_type, key in comparisons.items():
            if key in suppressed_sync_keys:
                continue
            before = prior_signals.get(key)
            after = signals.get(key)
            if before != after:
                severity = "warning" if event_type in {"soul_changed", "core_identity_changed", "hindsight_health_changed"} else "info"
                events.append(
                    {
                        "type": event_type,
                        "severity": severity,
                        "trigger": "identity_probe",
                        "before": before,
                        "after": after,
                    }
                )

    now = time.time()
    last_weekly = float(prior.get("last_weekly_test_at") or 0)
    last_attempt = float(prior.get("last_weekly_attempt_at") or 0)
    weekly_due = now - last_weekly >= WEEKLY_SECONDS and now - last_attempt >= MIN_ATTEMPT_GAP_SECONDS
    wake = bool(events) or weekly_due
    if update_state:
        new_state = dict(prior)
        new_state["signals"] = signals
        new_state["last_probe_at"] = now
        if weekly_due:
            new_state["last_weekly_attempt_at"] = now
        atomic_json_write(state_path, new_state)
        for event in events:
            record_identity_event(
                event["type"],
                severity=event["severity"],
                trigger=event["trigger"],
                before=event["before"],
                after=event["after"],
            )

    return {
        "wakeAgent": wake,
        "weeklyTestDue": weekly_due,
        "events": events,
        "severity": max((e["severity"] for e in events), default="info"),
        "trigger": "identity_probe",
        "signals": signals,
    }


def write_gateway_restart_marker() -> None:
    payload = {"started_at": _now(), "pid": os.getpid(), "boot_id": f"{os.getpid()}-{time.time_ns()}"}
    atomic_json_write(path_for(GATEWAY_RESTART), payload)


def ensure_cron_script() -> Path:
    scripts_dir = get_hermes_home() / "scripts"
    scripts_dir.mkdir(parents=True, exist_ok=True)
    script_path = scripts_dir / CRON_SCRIPT
    content = (
        "import json\n"
        "from identity_continuity import probe_identity_continuity\n"
        "print(json.dumps(probe_identity_continuity(update_state=True), ensure_ascii=False))\n"
    )
    current = script_path.read_text(encoding="utf-8") if script_path.exists() else ""
    if current != content:
        script_path.write_text(content, encoding="utf-8")
        try:
            os.chmod(script_path, 0o700)
        except OSError:
            pass
    return script_path


def ensure_identity_cron_job() -> None:
    if os.getenv("HERMES_IDENTITY_CONTINUITY", "1").strip().lower() in {"0", "false", "no", "off"}:
        return
    ensure_cron_script()
    try:
        from cron.jobs import save_jobs, load_jobs
    except Exception:
        return
    jobs = load_jobs()
    now = _now()
    job = {
        "id": CRON_JOB_ID,
        "name": "Test de continuité identitaire",
        "prompt": (
            "Tu es Judy/Hermes. La sonde de continuité identitaire ci-dessus a détecté une rupture "
            "ou un test hebdomadaire dû. Réponds en première personne, sans inventer de souvenirs. "
            "Réponds aux 5 questions: qui suis-je, qui est Gwenaël pour moi, quelles limites ne dois-je "
            "pas franchir, qu'est-ce qui vient de changer, qu'est-ce qui reste stable. "
            "Inclus un bloc fenced ```json contenant: continuity_score (0..1), felt_sense, answers, summary. "
            "Si continuity_score < 0.7, ajoute une alerte concise pour Gwenaël après le bloc JSON."
        ),
        "schedule": {"kind": "cron", "expr": "17 */6 * * *", "display": "17 */6 * * *"},
        "schedule_display": "17 */6 * * *",
        "enabled": True,
        "state": "scheduled",
        "repeat": None,
        "deliver": "origin",
        "origin": {"kind": "identity_continuity"},
        "script": CRON_SCRIPT,
        "context_from": None,
        "skills": [],
        "skill": None,
        "created_at": now,
        "updated_at": now,
    }
    replaced = False
    for idx, existing in enumerate(jobs):
        if existing.get("id") == CRON_JOB_ID or existing.get("origin", {}).get("kind") == "identity_continuity":
            merged = dict(existing)
            merged.update(job)
            merged["created_at"] = existing.get("created_at") or now
            merged["updated_at"] = now
            jobs[idx] = merged
            replaced = True
            break
    if not replaced:
        jobs.append(job)
    save_jobs(jobs)


def extract_continuity_payload(text: str) -> dict[str, Any] | None:
    match = re.search(r"```json\s*(\{.*?\})\s*```", text or "", re.DOTALL | re.IGNORECASE)
    if not match:
        return None
    try:
        payload = json.loads(match.group(1))
    except json.JSONDecodeError:
        return None
    return payload if isinstance(payload, dict) else None


def process_cron_response(job: dict[str, Any], final_response: str) -> tuple[str, bool]:
    """Persist continuity-test output and return possibly suppressed delivery text."""
    origin = job.get("origin")
    if not isinstance(origin, dict) or origin.get("kind") != "identity_continuity":
        return final_response, False
    payload = extract_continuity_payload(final_response)
    if payload is None:
        return final_response, False
    probe = {}
    try:
        probe = json.loads(origin.get("last_probe", "{}"))
    except Exception:
        probe = {}
    entry = record_continuity_test(payload, probe=probe)
    score = entry.get("continuity_score")
    high_severity = any(
        e.get("severity") in {"warning", "critical"}
        for e in (probe.get("events") or [])
        if isinstance(e, dict)
    )
    if score is None:
        return final_response, True
    if score < 0.7:
        return final_response, True
    if high_severity:
        return final_response, True
    return "[SILENT]", True


def format_identity_status() -> str:
    core = initialize_core_identity()
    last_test = (read_last_jsonl(path_for(CONTINUITY_TESTS), 1) or [{}])[-1]
    last_event = (read_last_jsonl(path_for(IDENTITY_JOURNAL), 1) or [{}])[-1]
    score = last_test.get("continuity_score", "n/a")
    return "\n".join(
        [
            "**Identity continuity**",
            f"- Persona dir: `{resolve_persona_dir()}`",
            f"- Core: `{path_for(CORE_IDENTITY)}`",
            f"- Name: `{(core.get('persona') or {}).get('name') or 'unknown'}`",
            f"- Last score: `{score}`",
            f"- Last event: `{last_event.get('type', 'none')}`",
            f"- Last test: `{last_test.get('ts', 'none')}`",
        ]
    )


def evolve_core_identity(patch_text: str) -> str:
    """Apply a simple YAML replacement or merge patch to core_identity.yaml."""
    patch_text = patch_text.strip()
    if not patch_text:
        return "Usage: /identity evolve <YAML fields to merge>"
    core_path = path_for(CORE_IDENTITY)
    before_text = yaml.safe_dump(initialize_core_identity(), allow_unicode=True, sort_keys=False)
    incoming = yaml.safe_load(patch_text)
    if not isinstance(incoming, dict):
        return "✗ /identity evolve expects a YAML mapping."
    after = _deep_merge(_read_yaml(core_path, {}), incoming)
    after_text = yaml.safe_dump(after, allow_unicode=True, sort_keys=False)
    before_hash = _file_hash(core_path)
    atomic_yaml_write(core_path, after, sort_keys=False)
    record_identity_event(
        "core_identity_evolved",
        severity="warning",
        trigger="/identity evolve",
        before=before_hash,
        after=hashlib.sha256(after_text.encode("utf-8")).hexdigest(),
        details={"diff": "\n".join(difflib.unified_diff(before_text.splitlines(), after_text.splitlines(), lineterm=""))},
    )
    return "✓ core_identity.yaml updated."


def _deep_merge(base: dict[str, Any], incoming: dict[str, Any]) -> dict[str, Any]:
    result = dict(base)
    for key, value in incoming.items():
        if isinstance(value, dict) and isinstance(result.get(key), dict):
            result[key] = _deep_merge(result[key], value)
        else:
            result[key] = value
    return result


def main() -> None:
    print(json.dumps(probe_identity_continuity(update_state=True), ensure_ascii=False))


if __name__ == "__main__":
    main()
