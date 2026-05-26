"""Gateway conversation-turn policy helpers.

This module keeps small social-turn suppression and optional proactive
rebounds outside the main gateway runner. It is deliberately conservative:
auxiliary model failures, invalid JSON, stale persona state, or ambiguity all
fall back to normal replies.
"""

from __future__ import annotations

import json
import logging
import time
import unicodedata
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from zoneinfo import ZoneInfo

from hermes_cli.config import get_hermes_home

logger = logging.getLogger(__name__)

PARIS_TZ = ZoneInfo("Europe/Paris")
SUPPORTED_STATES = {
    "active_request",
    "terminal_ack",
    "social_closure",
    "ambiguous",
    "other",
}
SUPPRESS_CONFIDENCE_THRESHOLD = 0.95
INNER_STATE_MAX_AGE_SECONDS = 6 * 60 * 60
REBOUND_THRESHOLD = 0.65
REBOUND_EXPANDED_THRESHOLD = 0.85
REBOUND_WINDOW_SECONDS = 6 * 60 * 60
REBOUND_WINDOW_LIMIT = 3
GWENAEL_RELATIONAL_BASELINE = 0.82
REBOUND_TRAIT_WEIGHTS = {
    "momentum": ("rebond_momentum", 0.6),
    "profondeur": ("rebond_profondeur", 0.5),
    "nouveaute": ("rebond_nouveaute", 0.4),
    "resonance": ("rebond_resonance", 0.5),
    "elan": ("rebond_elan", 0.7),
}


@dataclass(frozen=True)
class TurnPolicyDecision:
    conversation_state: str = "other"
    should_reply: bool = True
    confidence: float = 0.0
    reason: str = ""
    source: str = "fallback"


@dataclass(frozen=True)
class SuppressionResult:
    suppress: bool
    decision: TurnPolicyDecision
    reason: str


@dataclass(frozen=True)
class ReboundResult:
    response: str
    added: bool
    reason: str
    score: Optional[float] = None
    dimensions: Optional[Dict[str, float]] = None
    finish_reason: Optional[str] = None
    weights: Optional[Dict[str, float]] = None
    weight_fallbacks: Optional[List[str]] = None


@dataclass(frozen=True)
class ReboundDecision:
    generated: bool
    score: float
    dimensions: Dict[str, float]
    reason: str
    weights: Optional[Dict[str, float]] = None
    weight_fallbacks: Optional[List[str]] = None


def _coerce_epoch(value: Any) -> Optional[float]:
    if value is None:
        return None
    if isinstance(value, (int, float)):
        try:
            return float(value)
        except (TypeError, ValueError):
            return None
    if isinstance(value, datetime):
        dt = value if value.tzinfo else value.replace(tzinfo=timezone.utc)
        return dt.timestamp()
    if isinstance(value, str):
        raw = value.strip()
        if not raw:
            return None
        try:
            return float(raw)
        except ValueError:
            pass
        try:
            return datetime.fromisoformat(raw.replace("Z", "+00:00")).timestamp()
        except ValueError:
            return None
    return None


def _format_utc_and_paris(epoch: float) -> str:
    utc_dt = datetime.fromtimestamp(epoch, tz=timezone.utc)
    paris_dt = utc_dt.astimezone(PARIS_TZ)
    return (
        f"{utc_dt.isoformat().replace('+00:00', 'Z')} / "
        f"{paris_dt.strftime('%H:%M')} Europe/Paris"
    )


def _format_age(seconds: float) -> str:
    seconds = max(0, int(seconds))
    if seconds < 90:
        return f"{seconds}s ago"
    minutes = seconds // 60
    if minutes < 90:
        return f"{minutes}min ago"
    hours = minutes // 60
    if hours < 48:
        return f"{hours}h ago"
    days = hours // 24
    return f"{days}d ago"


def _last_role_timestamp(history: Iterable[Dict[str, Any]], role: str) -> Optional[float]:
    for msg in reversed(list(history or [])):
        if msg.get("role") != role:
            continue
        ts = _coerce_epoch(msg.get("timestamp"))
        if ts is not None:
            return ts
    return None


def build_message_time_context(
    *,
    current_timestamp: Any,
    history: Iterable[Dict[str, Any]],
    now: Any = None,
) -> str:
    """Build a compact internal time-context block for the current turn."""
    current_epoch = _coerce_epoch(current_timestamp) or time.time()
    now_epoch = _coerce_epoch(now) or current_epoch
    previous_user = _last_role_timestamp(history, "user")
    previous_assistant = _last_role_timestamp(history, "assistant")

    lines = [
        "[Conversation time context]",
        f"Current message: {_format_utc_and_paris(current_epoch)}",
    ]
    if previous_user is not None:
        lines.append(f"Previous user message: {_format_age(now_epoch - previous_user)}")
    if previous_assistant is not None:
        lines.append(
            f"Previous assistant message: {_format_age(now_epoch - previous_assistant)}"
        )
    lines.append(
        "Use these timestamps only for conversational continuity; do not display them unless useful or asked."
    )
    return "\n".join(lines)


def _response_content(response: Any) -> str:
    try:
        return str(response.choices[0].message.content or "").strip()
    except Exception:
        return ""


def _finish_reason(response: Any) -> Optional[str]:
    try:
        value = getattr(response.choices[0], "finish_reason", None)
    except Exception:
        return None
    return str(value) if value is not None else None


def _parse_decision(raw: str) -> TurnPolicyDecision:
    data = json.loads(raw)
    if not isinstance(data, dict):
        raise ValueError("turn policy response is not an object")
    state = str(data.get("conversation_state") or "other").strip()
    if state not in SUPPORTED_STATES:
        state = "other"
    confidence = float(data.get("confidence") or 0.0)
    confidence = max(0.0, min(1.0, confidence))
    return TurnPolicyDecision(
        conversation_state=state,
        should_reply=bool(data.get("should_reply", True)),
        confidence=confidence,
        reason=str(data.get("reason") or "")[:240],
        source="auxiliary",
    )


def _has_structural_request_signal(message_text: str) -> bool:
    text = (message_text or "").strip()
    return bool(
        "?" in text
        or "\n" in text
        or len(text) > 80
        or text.startswith(("/", "!", "@"))
        or any(ch.isdigit() for ch in text)
    )


def _normalized_ascii_text(message_text: str) -> str:
    normalized = unicodedata.normalize("NFKD", message_text or "")
    return normalized.encode("ascii", "ignore").decode("ascii").lower()


def _assistant_awaits_short_reply(message_text: Optional[str]) -> bool:
    text = (message_text or "").strip()
    if not text:
        return False
    if "?" in text:
        return True

    ascii_text = _normalized_ascii_text(text)
    binary_reply_prompts = (
        "tu veux",
        "veux-tu",
        "veux tu",
        "souhaites-tu",
        "souhaites tu",
        "est-ce que tu veux",
        "est ce que tu veux",
        "prefere-tu",
        "preferes-tu",
        "prefere tu",
        "preferes tu",
        "dois-je",
        "dois je",
        "do you want",
        "would you like",
        "shall i",
        "should i",
    )
    return any(prompt in ascii_text for prompt in binary_reply_prompts)


def _ascii_content_tokens(message_text: str) -> set[str]:
    return _content_tokens(_normalized_ascii_text(message_text))


def _has_meaningful_social_signal(message_text: str) -> bool:
    tokens = _ascii_content_tokens(message_text)
    affective = {
        "bravo",
        "content",
        "contente",
        "encourage",
        "felicitation",
        "felicitations",
        "fier",
        "fiere",
        "fiers",
        "fieres",
        "fierte",
        "heureux",
        "heureuse",
        "impressionne",
        "impressionnee",
        "merci",
        "reconnaissant",
        "reconnaissante",
        "touche",
        "touchee",
    }
    relational = {
        "autonomie",
        "autonome",
        "confiance",
        "existe",
        "grandi",
        "grandis",
        "judy",
        "presence",
        "progres",
        "progresse",
        "relation",
        "toi",
        "vivante",
    }
    return bool((tokens & affective) and (tokens & relational))


def _has_coordination_update_signal(message_text: str) -> bool:
    """Return whether a short ack is also a useful work-status update."""
    tokens = _ascii_content_tokens(message_text)
    completion_tokens = {
        "cree",
        "creee",
        "done",
        "execute",
        "executee",
        "fait",
        "fini",
        "termine",
        "terminee",
    }
    work_context_tokens = {
        "build",
        "codex",
        "commit",
        "compose",
        "docker",
        "gateway",
        "hermes",
        "link",
        "rebuild",
        "restart",
        "symlink",
    }
    return bool((tokens & completion_tokens) and (tokens & work_context_tokens))


def _eligible_gateway(source: Any) -> bool:
    # This policy is only for Judy's Telegram social-turn handling.
    platform = getattr(getattr(source, "platform", None), "value", None) or str(
        getattr(source, "platform", "") or ""
    )
    return platform == "telegram"


def _is_telegram_dm(source: Any) -> bool:
    return _eligible_gateway(source) and getattr(source, "chat_type", None) == "dm"


def classify_conversation_turn(
    *,
    message_text: str,
    history: List[Dict[str, Any]],
    source: Any = None,
    has_media: bool = False,
    is_command: bool = False,
    call_llm_fn: Any = None,
) -> TurnPolicyDecision:
    """Classify whether a short turn needs a reply.

    The prefilter is structural only. It intentionally does not match closure
    words or phrases.
    """
    text = (message_text or "").strip()
    if not _eligible_gateway(source):
        return TurnPolicyDecision(reason="gateway_not_eligible")
    if getattr(source, "chat_type", None) not in {None, "dm"} and not getattr(source, "thread_id", None):
        return TurnPolicyDecision(reason="ambiguous_group")
    if has_media or is_command:
        return TurnPolicyDecision(reason="media_or_command")
    if not text or len(text) > 80 or "\n" in text:
        return TurnPolicyDecision(reason="not_short_text")
    last_assistant_ts = _last_role_timestamp(history, "assistant")
    if last_assistant_ts is None or time.time() - last_assistant_ts > 20 * 60:
        return TurnPolicyDecision(reason="no_recent_assistant")
    if _has_structural_request_signal(text):
        return TurnPolicyDecision(reason="structural_request_signal")

    try:
        if call_llm_fn is None:
            from agent.auxiliary_client import call_llm as call_llm_fn

        response = call_llm_fn(
            task="conversation_turn_policy",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "Classify the user's latest chat turn. Return strict JSON only with "
                        "conversation_state, should_reply, confidence, reason. "
                        "States: active_request, terminal_ack, social_closure, ambiguous, other. "
                        "Be conservative: if there may be a request, should_reply must be true. "
                        "A short yes/no answer to a recent assistant question or binary proposal "
                        "is an active_request and should_reply must be true. "
                        "If the turn contains emotionally meaningful praise, concern, appreciation, "
                        "or a relational update addressed to the assistant, should_reply must be true."
                    ),
                },
                {
                    "role": "user",
                    "content": json.dumps(
                        {
                            "latest_user_message": text,
                            "recent_history": [
                                {
                                    "role": m.get("role"),
                                    "content": str(m.get("content") or "")[:500],
                                }
                                for m in (history or [])[-4:]
                                if m.get("role") in {"user", "assistant"}
                            ],
                        },
                        ensure_ascii=False,
                    ),
                },
            ],
            temperature=0,
            max_tokens=512,
            extra_body={"response_format": {"type": "json_object"}},
        )
        return _parse_decision(_response_content(response))
    except Exception as exc:
        logger.debug("conversation turn classifier unavailable: %s", exc)
        return TurnPolicyDecision(reason="classifier_failed")


def should_suppress_conversation_turn(
    *,
    decision: TurnPolicyDecision,
    message_text: str,
    source: Any = None,
    recent_assistant_message: Optional[str] = None,
) -> SuppressionResult:
    if not _eligible_gateway(source):
        return SuppressionResult(False, decision, "gateway_not_eligible")
    if not _is_telegram_dm(source):
        return SuppressionResult(False, decision, "reply_by_default")
    if (
        not decision.should_reply
        and decision.source == "auxiliary"
        and decision.conversation_state in {"terminal_ack", "social_closure"}
        and decision.confidence >= SUPPRESS_CONFIDENCE_THRESHOLD
        and not _has_structural_request_signal(message_text)
        and not _has_meaningful_social_signal(message_text)
        and not _has_coordination_update_signal(message_text)
    ):
        if _assistant_awaits_short_reply(recent_assistant_message):
            return SuppressionResult(False, decision, "assistant_question_pending")
        return SuppressionResult(True, decision, "telegram_dm_closure_silence")
    return SuppressionResult(False, decision, "dm_reply_by_default")


def _runtime_log_path() -> Path:
    return get_hermes_home() / "runtime" / "conversation_turn_policy.jsonl"


def write_turn_policy_log(event: Dict[str, Any], *, path: Optional[Path] = None) -> None:
    payload = {
        "timestamp": datetime.now(timezone.utc).isoformat().replace("+00:00", "Z"),
        **event,
    }
    log_path = path or _runtime_log_path()
    try:
        log_path.parent.mkdir(parents=True, exist_ok=True)
        with log_path.open("a", encoding="utf-8") as fh:
            fh.write(json.dumps(payload, ensure_ascii=False, sort_keys=True) + "\n")
    except Exception as exc:
        logger.debug("failed to write conversation turn policy log: %s", exc)


def _read_json(path: Path) -> Any:
    return json.loads(path.read_text(encoding="utf-8"))


def _load_inner_state(path: Path, *, now: Optional[float] = None) -> tuple[Optional[dict], Optional[float], str]:
    try:
        data = _read_json(path)
    except Exception:
        return None, None, "missing_or_invalid_inner_state"
    if not isinstance(data, dict):
        return None, None, "invalid_inner_state"
    ts = _coerce_epoch(data.get("timestamp"))
    if ts is None:
        return data, None, "missing_inner_state_timestamp"
    age = (now or time.time()) - ts
    if age > INNER_STATE_MAX_AGE_SECONDS:
        return data, age, "stale_inner_state"
    return data, age, "fresh"


def _desire_weight(path: Path, name: str) -> float:
    traits_path = path.with_name("desire_traits.json")
    try:
        data = _read_json(traits_path if traits_path.exists() else path)
    except Exception:
        return 0.0
    if isinstance(data, dict):
        data = data.get("traits", [])
    if not isinstance(data, list):
        return 0.0
    for item in data:
        if isinstance(item, dict) and item.get("name") == name:
            try:
                return max(0.0, min(1.0, float(item.get("weight") or 0.0)))
            except (TypeError, ValueError):
                return 0.0
    return 0.0


def _load_trait_items(path: Path) -> List[dict]:
    traits_path = path.with_name("desire_traits.json")
    try:
        data = _read_json(traits_path if traits_path.exists() else path)
    except Exception:
        return []
    if isinstance(data, dict):
        data = data.get("traits", [])
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict)]


def _rebound_trait_weights(path: Path) -> tuple[Dict[str, float], List[str]]:
    traits = {str(item.get("name") or ""): item for item in _load_trait_items(path)}
    weights: Dict[str, float] = {}
    fallbacks: List[str] = []
    for dimension, (trait_name, default_weight) in REBOUND_TRAIT_WEIGHTS.items():
        item = traits.get(trait_name)
        if item is None:
            fallbacks.append(trait_name)
            weights[dimension] = default_weight
            continue
        try:
            weights[dimension] = max(0.0, min(1.0, float(item.get("weight"))))
        except (TypeError, ValueError):
            fallbacks.append(trait_name)
            weights[dimension] = default_weight
    if fallbacks:
        logger.warning(
            "missing or invalid rebound desire traits, using default weights: %s",
            ", ".join(fallbacks),
        )
    return weights, fallbacks


def _weighted_rebound_score(dimensions: Dict[str, float], weights: Dict[str, float]) -> float:
    total_weight = sum(max(0.0, weight) for weight in weights.values())
    if total_weight <= 0:
        logger.warning("all rebound desire trait weights are zero, using unweighted score")
        return round(sum(dimensions.values()) / len(dimensions), 4)
    weighted = sum(_clamp_score(dimensions.get(name)) * max(0.0, weights.get(name, 0.0)) for name in dimensions)
    return round(weighted / total_weight, 4)


def _clamp_score(value: Any) -> float:
    try:
        return max(0.0, min(1.0, float(value)))
    except (TypeError, ValueError):
        return 0.0


def _content_tokens(*parts: str) -> set[str]:
    text = " ".join(part or "" for part in parts).lower()
    current = []
    tokens: set[str] = set()
    for ch in text:
        if ch.isalnum() or ch in {"_", "-"}:
            current.append(ch)
        elif current:
            token = "".join(current).strip("_-")
            if len(token) >= 4:
                tokens.add(token)
            current = []
    if current:
        token = "".join(current).strip("_-")
        if len(token) >= 4:
            tokens.add(token)
    return tokens


def _shares_tokens(tokens: set[str], value: Any) -> bool:
    if not tokens:
        return False
    if isinstance(value, dict):
        return any(_shares_tokens(tokens, item) for item in value.values())
    if isinstance(value, list):
        return any(_shares_tokens(tokens, item) for item in value)
    return bool(tokens & _content_tokens(str(value or "")))


def _last_assistant_message(history: List[Dict[str, Any]]) -> Optional[Dict[str, Any]]:
    for msg in reversed(history or []):
        if msg.get("role") == "assistant":
            return msg
    return None


def _has_recent_rebound(history: List[Dict[str, Any]], *, now: float) -> bool:
    count = 0
    for msg in history or []:
        if msg.get("role") != "assistant" or not msg.get("conversation_rebound"):
            continue
        ts = _coerce_epoch(msg.get("timestamp"))
        if ts is not None and 0 <= now - ts <= REBOUND_WINDOW_SECONDS:
            count += 1
    return count >= REBOUND_WINDOW_LIMIT


def _critical_alert_signal(message_text: str) -> bool:
    tokens = _content_tokens(message_text)
    return bool(tokens & {"alerte", "incident", "securite", "corruption", "critique"})


def _momentum_emotionnel(inner_state: dict) -> float:
    return round(
        _clamp_score(inner_state.get("social_temperature")) * 0.4
        + _clamp_score(inner_state.get("satisfaction")) * 0.35
        + _clamp_score(inner_state.get("energy")) * 0.25,
        4,
    )


def _profondeur_echange(
    *,
    state: str,
    message_text: str,
    response: str,
) -> tuple[float, str]:
    if state in {"terminal_ack", "social_closure"}:
        return 0.0, state
    combined = f"{message_text}\n{response}"
    tokens = _content_tokens(combined)
    if tokens & {"intime", "vulnerable", "amour", "peur"}:
        return 0.9, "intimate"
    if tokens & {"sens", "identite", "conscience", "mort", "philosophie"}:
        return 0.85, "philosophical"
    if (
        "```" in combined
        or tokens
        & {
            "debug",
            "debugger",
            "docker",
            "traceback",
            "erreur",
            "pytest",
        }
    ):
        return 0.65, "debugging"
    if tokens & {
        "gateway",
        "runtime",
        "implementation",
        "fonction",
        "code",
        "module",
        "test",
        "tests",
    }:
        return 0.6, "technical"
    if state == "active_request":
        return 0.3, "active_request_simple"
    if state == "other" and ("?" in message_text or len(message_text) > 80):
        return 0.3, "active_request_simple"
    if tokens & {"salut", "hello", "bonjour", "coucou"}:
        return 0.12, "greeting"
    return 0.15, "small_talk"


def _read_jsonl_recent(path: Path, *, now: float, max_age_seconds: float) -> List[dict]:
    try:
        lines = path.read_text(encoding="utf-8").splitlines()
    except Exception:
        return []
    rows: List[dict] = []
    for line in reversed(lines[-200:]):
        try:
            item = json.loads(line)
        except Exception:
            continue
        if not isinstance(item, dict):
            continue
        ts = _coerce_epoch(item.get("timestamp"))
        if ts is None or now - ts > max_age_seconds:
            if rows:
                break
            continue
        rows.append(item)
    return rows


def _nouveaute_recente(path: Path, *, now: float) -> float:
    recent_4h = _read_jsonl_recent(path, now=now, max_age_seconds=4 * 60 * 60)
    if recent_4h:
        for item in recent_4h:
            if item.get("delivered") is True:
                continue
            decision = str(item.get("decision") or "").lower()
            novelty = item.get("novelty_score")
            if decision in {"retain", "stored", "learned"} or _clamp_score(novelty) >= 0.55:
                return 0.8
        return 0.2
    recent_8h = _read_jsonl_recent(path, now=now, max_age_seconds=8 * 60 * 60)
    return 0.2 if recent_8h else 0.0


def _active_open_loops(path: Path) -> List[dict]:
    try:
        data = _read_json(path)
    except Exception:
        return []
    if not isinstance(data, list):
        return []
    return [item for item in data if isinstance(item, dict) and item.get("status") != "resolved"]


def _resonance_memoire(
    *,
    message_text: str,
    response: str,
    history: List[Dict[str, Any]],
    inner_state: dict,
    open_loops_path: Path,
) -> float:
    tokens = _content_tokens(message_text, response)
    score = 0.0
    open_loops = _active_open_loops(open_loops_path)
    if open_loops and any(_shares_tokens(tokens, loop) for loop in open_loops):
        score += 0.3
    elif _shares_tokens(tokens, inner_state.get("open_loops")):
        score += 0.3

    previous_text = " ".join(
        str(msg.get("content") or "")
        for msg in (history or [])[-12:-1]
        if msg.get("role") in {"user", "assistant"}
    )
    if tokens & _content_tokens(previous_text):
        score += 0.4

    project_memory = [
        inner_state.get("attention_targets"),
        inner_state.get("current_obsessions"),
        inner_state.get("dominant_thought"),
    ]
    if any(_shares_tokens(tokens, item) for item in project_memory):
        score += 0.2
    return round(min(1.0, score), 4)


def _load_relationship_metrics(path: Path, source: Any = None) -> Optional[dict]:
    try:
        data = _read_json(path)
    except Exception:
        return None
    if not isinstance(data, dict):
        return None
    candidates = [data]
    user_id = str(getattr(source, "user_id", "") or "")
    for key in ("gwenael", "gwenaël", "default", user_id):
        item = data.get(key)
        if isinstance(item, dict):
            candidates.insert(0, item)
    for item in candidates:
        rel = item.get("relationship") if isinstance(item.get("relationship"), dict) else item
        if isinstance(rel, dict) and any(k in rel for k in ("trust", "intimacy", "reciprocity", "alignment")):
            return rel
    return None


def _elan_relationnel(path: Path, source: Any = None) -> float:
    metrics = _load_relationship_metrics(path, source=source)
    if metrics is None:
        return GWENAEL_RELATIONAL_BASELINE
    return round(
        _clamp_score(metrics.get("trust")) * 0.3
        + _clamp_score(metrics.get("intimacy")) * 0.3
        + _clamp_score(metrics.get("reciprocity")) * 0.2
        + _clamp_score(metrics.get("alignment")) * 0.2,
        4,
    )


def _rebound_decision(
    *,
    message_text: str,
    response: str,
    history: List[Dict[str, Any]],
    source: Any,
    suppression: Optional[SuppressionResult],
    inner_state: dict,
    desires_path: Path,
    curiosity_log_path: Path,
    open_loops_path: Path,
    relationship_path: Path,
    now: float,
) -> ReboundDecision:
    state = (suppression.decision.conversation_state if suppression else "other") or "other"
    if state in {"terminal_ack", "social_closure"} and not _critical_alert_signal(message_text):
        return ReboundDecision(False, 0.0, {}, "terminal_or_social_closure")

    ne_pas_deranger = _desire_weight(desires_path, "ne_pas_deranger")
    if ne_pas_deranger > 0.85:
        return ReboundDecision(False, 0.0, {}, "do_not_disturb")

    profondeur, _depth_reason = _profondeur_echange(
        state=state,
        message_text=message_text,
        response=response,
    )
    dimensions = {
        "momentum": _momentum_emotionnel(inner_state),
        "profondeur": profondeur,
        "nouveaute": _nouveaute_recente(curiosity_log_path, now=now),
        "resonance": _resonance_memoire(
            message_text=message_text,
            response=response,
            history=history,
            inner_state=inner_state,
            open_loops_path=open_loops_path,
        ),
        "elan": _elan_relationnel(relationship_path, source=source),
    }
    weights, weight_fallbacks = _rebound_trait_weights(desires_path)
    score = _weighted_rebound_score(dimensions, weights)
    if score < REBOUND_THRESHOLD:
        return ReboundDecision(False, score, dimensions, "score_below_threshold", weights, weight_fallbacks)
    last_assistant = _last_assistant_message(history)
    if last_assistant and last_assistant.get("conversation_rebound"):
        return ReboundDecision(False, score, dimensions, "consecutive_cooldown", weights, weight_fallbacks)
    if _has_recent_rebound(history, now=now):
        return ReboundDecision(False, score, dimensions, "window_cooldown", weights, weight_fallbacks)
    return ReboundDecision(True, score, dimensions, "score_threshold", weights, weight_fallbacks)

def maybe_add_conversation_rebound(
    *,
    response: str,
    message_text: str,
    history: List[Dict[str, Any]],
    source: Any = None,
    session_key: str = "",
    suppression: Optional[SuppressionResult] = None,
    inner_state_path: Optional[Path] = None,
    desires_path: Optional[Path] = None,
    curiosity_log_path: Optional[Path] = None,
    open_loops_path: Optional[Path] = None,
    relationship_path: Optional[Path] = None,
    call_llm_fn: Any = None,
    now: Optional[float] = None,
) -> ReboundResult:
    if not _eligible_gateway(source):
        return ReboundResult(response=response, added=False, reason="gateway_not_eligible")
    state = (suppression.decision.conversation_state if suppression else "other") or "other"
    if state in {"terminal_ack", "social_closure"} and not _critical_alert_signal(message_text):
        return ReboundResult(
            response=response,
            added=False,
            reason="terminal_or_social_closure",
            score=0.0,
            dimensions={},
        )
    clean_response = (response or "").strip()
    if not clean_response or clean_response.startswith("⚠️"):
        return ReboundResult(response=response, added=False, reason="empty_or_error_response")
    current_epoch = now or time.time()
    last_user_ts = _last_role_timestamp(history, "user")
    if last_user_ts is None or current_epoch - last_user_ts > 30 * 60:
        return ReboundResult(response=response, added=False, reason="conversation_not_active")

    inner_path = inner_state_path or Path("/workspace/projects/persona/inner_state.json")
    inner_state, age, freshness = _load_inner_state(inner_path, now=current_epoch)
    if freshness != "fresh" or inner_state is None:
        return ReboundResult(response=response, added=False, reason=freshness)

    persona_root = inner_path.parent
    desire_path = desires_path or persona_root / "desires.json"
    curiosity_path = curiosity_log_path or persona_root / "curiosity_log.jsonl"
    loops_path = open_loops_path or persona_root / "open_loops.json"
    rel_path = relationship_path or persona_root / "relationships.json"

    decision = _rebound_decision(
        message_text=message_text,
        response=clean_response,
        history=history,
        source=source,
        suppression=suppression,
        inner_state=inner_state,
        desires_path=desire_path,
        curiosity_log_path=curiosity_path,
        open_loops_path=loops_path,
        relationship_path=rel_path,
        now=current_epoch,
    )
    if not decision.generated:
        return ReboundResult(
            response=response,
            added=False,
            reason=decision.reason,
            score=decision.score,
            dimensions=decision.dimensions,
            weights=decision.weights,
            weight_fallbacks=decision.weight_fallbacks,
        )

    sentence_limit = "2-3 sentences" if decision.score >= REBOUND_EXPANDED_THRESHOLD else "1-2 sentences"
    try:
        if call_llm_fn is None:
            from agent.auxiliary_client import call_llm as call_llm_fn
        rebound_messages = [
            {
                "role": "system",
                "content": (
                    "Write a short proactive follow-up in French, "
                    f"{sentence_limit} max. It must be concrete, useful, and not needy. "
                    "Return only the text."
                ),
            },
            {
                "role": "user",
                "content": json.dumps(
                    {
                        "user_message": (message_text or "")[:500],
                        "assistant_response": clean_response[:1200],
                        "rebound_score": decision.score,
                        "rebound_dimensions": decision.dimensions,
                        "inner_state": {
                            "mood": inner_state.get("mood"),
                            "dominant_thought": inner_state.get("dominant_thought"),
                            "energy": inner_state.get("energy"),
                            "satisfaction": inner_state.get("satisfaction"),
                            "social_temperature": inner_state.get("social_temperature"),
                            "age_seconds": age,
                        },
                    },
                    ensure_ascii=False,
                ),
            },
        ]
        aux_response = call_llm_fn(
            task="conversation_rebound",
            messages=rebound_messages,
            temperature=0.35,
            max_tokens=512,
            timeout=5,
        )
        rebound = _response_content(aux_response)
        finish_reason = _finish_reason(aux_response)
        if not rebound and finish_reason == "length":
            aux_response = call_llm_fn(
                task="conversation_rebound",
                messages=rebound_messages,
                temperature=0.35,
                max_tokens=2048,
                timeout=5,
            )
            rebound = _response_content(aux_response)
            finish_reason = _finish_reason(aux_response)
    except Exception as exc:
        logger.debug("conversation rebound unavailable: %s", exc)
        return ReboundResult(
            response=response,
            added=False,
            reason="rebound_model_failed",
            score=decision.score,
            dimensions=decision.dimensions,
            weights=decision.weights,
            weight_fallbacks=decision.weight_fallbacks,
        )

    if not rebound:
        return ReboundResult(
            response=response,
            added=False,
            reason="empty_rebound",
            score=decision.score,
            dimensions=decision.dimensions,
            weights=decision.weights,
            weight_fallbacks=decision.weight_fallbacks,
            finish_reason=finish_reason,
        )
    rebound = " ".join(rebound.split())[:500]
    return ReboundResult(
        response=f"{clean_response}\n\n{rebound}",
        added=True,
        reason="added",
        score=decision.score,
        dimensions=decision.dimensions,
        finish_reason=finish_reason,
        weights=decision.weights,
        weight_fallbacks=decision.weight_fallbacks,
    )
