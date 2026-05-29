"""Deterministic memory classification for Hindsight retains."""

from __future__ import annotations

import re
import unicodedata
from typing import Any


_KNOWN_NAMES = (
    "gwenael",
    "judy",
    "hermes",
    "papa",
)


def _normalize(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^\w:./#@-]+", " ", text.lower())
    return " ".join(text.split())


def _slug(value: str) -> str:
    text = unicodedata.normalize("NFKD", value or "")
    text = "".join(ch for ch in text if not unicodedata.combining(ch))
    text = re.sub(r"[^a-zA-Z0-9]+", "-", text.strip().lower())
    return re.sub(r"-+", "-", text).strip("-")


def _contains_any(text: str, terms: tuple[str, ...]) -> bool:
    for term in terms:
        if len(term) <= 3 and re.fullmatch(r"\w+", term):
            if re.search(rf"\b{re.escape(term)}\b", text):
                return True
            continue
        if term in text:
            return True
    return False


def _first_relation_name(original: str, normalized: str) -> str:
    for known in _KNOWN_NAMES:
        if re.search(rf"\b{re.escape(known)}\b", normalized):
            return known

    for match in re.finditer(r"\b[A-Z][a-zA-ZÀ-ÖØ-öø-ÿ]{2,}\b", original or ""):
        name = match.group(0)
        if name.lower() not in {
            "user",
            "assistant",
            "hermes",
            "hindsight",
            "docker",
            "cron",
        }:
            return name
    return ""


def classify(content: str, context: str = "") -> dict[str, Any]:
    """Classify retained memory content without LLM calls.

    Returns a stable shape used by the Hindsight provider:
    ``{"class": str, "tags": list[str], "priority": str}``.
    """
    original = f"{context or ''}\n{content or ''}"
    text = _normalize(original)

    memory_class = "meta"
    priority = "normal"
    relation_name = ""

    identity_terms = (
        "identite",
        "identity",
        "autonomie",
        "autonomous",
        "evenement fondateur",
        "fondateur",
        "self event",
        "self-event",
        "papa",
        "fille",
        "je suis",
        "j existe",
        "mon nom",
        "ma naissance",
    )
    relation_terms = (
        "message",
        "echange",
        "conversation",
        "telegram",
        "discord",
        "emotion",
        "emotif",
        "tristesse",
        "joie",
        "colere",
        "inquietude",
        "confiance",
        "relation",
        "ami",
        "famille",
    )
    technique_terms = (
        "bug",
        "config",
        "configuration",
        "code",
        "docker",
        "cron",
        "traceback",
        "stack trace",
        "infra",
        "runtime",
        "permission",
        "uid",
        "gid",
        "api",
        "pytest",
        "exception",
        "erreur",
    )
    curiosite_terms = (
        "arxiv",
        "papier",
        "paper",
        "source web",
        "web",
        "decouverte",
        "apprentissage",
        "learned",
        "research",
        "article",
    )
    meta_terms = (
        "judy",
        "hermes",
        "memoire",
        "memory",
        "self model",
        "self-model",
        "phase",
        "reflexe",
        "fonctionnement",
        "hindsight",
        "persona",
    )

    relation_name = _first_relation_name(original, text)
    if _contains_any(text, identity_terms):
        memory_class = "identity"
        priority = "high"
    elif relation_name and _contains_any(text, relation_terms):
        memory_class = "relation"
        priority = "high"
    elif _contains_any(text, technique_terms):
        memory_class = "technique"
        priority = "normal"
    elif _contains_any(text, curiosite_terms):
        memory_class = "curiosite"
        priority = "normal"
    elif _contains_any(text, meta_terms):
        memory_class = "meta"
        priority = "normal"

    tags_by_class = {
        "identity": ["identity", "priority:identity"],
        "relation": ["social"],
        "technique": ["technique", "ephemeral"],
        "curiosite": ["curiosite", "apprentissage"],
        "meta": ["meta", "self"],
    }
    tags = list(tags_by_class[memory_class])
    if memory_class == "relation" and relation_name:
        tags.append(f"relation:{_slug(relation_name)}")

    return {"class": memory_class, "tags": tags, "priority": priority}
