import json
from types import SimpleNamespace

from agent.conversation_turn_policy import (
    SuppressionResult,
    TurnPolicyDecision,
    maybe_add_conversation_rebound,
)


class _Response:
    def __init__(self, content: str, finish_reason: str = "stop"):
        self.choices = [
            SimpleNamespace(
                message=SimpleNamespace(content=content),
                finish_reason=finish_reason,
            )
        ]


def _source():
    return SimpleNamespace(platform=SimpleNamespace(value="telegram"), chat_type="dm")


def _write_persona(
    tmp_path,
    *,
    timestamp="2026-05-22T20:40:00Z",
    ne_pas_deranger=0.5,
    relationship=None,
):
    inner = tmp_path / "inner_state.json"
    desires = tmp_path / "desires.json"
    curiosity = tmp_path / "curiosity_log.jsonl"
    loops = tmp_path / "open_loops.json"
    relationships = tmp_path / "relationships.json"
    inner.write_text(
        json.dumps(
            {
                "timestamp": timestamp,
                "social_temperature": 0.9,
                "satisfaction": 0.9,
                "energy": 0.9,
                "attention_targets": ["gateway", "runtime"],
                "current_obsessions": ["rebond deterministe"],
                "open_loops": ["stabiliser le rebond gateway"],
            }
        )
    )
    desires.write_text(
        json.dumps([{"name": "ne_pas_deranger", "weight": ne_pas_deranger}])
    )
    curiosity.write_text(
        json.dumps(
            {
                "timestamp": "2026-05-22T20:30:00Z",
                "decision": "retain",
                "novelty_score": 0.7,
                "delivered": False,
            }
        )
        + "\n"
    )
    loops.write_text(
        json.dumps(
            [
                {
                    "id": "gateway-rebond",
                    "status": "active",
                    "summary": "stabiliser le rebond gateway deterministe",
                }
            ]
        )
    )
    relationships.write_text(json.dumps(relationship or {}))
    return inner, desires, curiosity, loops, relationships


def _history(*, rebound=False, old_rebounds=0):
    rows = [
        {
            "role": "user",
            "content": "On stabilise le rebond gateway deterministe.",
            "timestamp": 1779482100.0,
        },
        {
            "role": "assistant",
            "content": "Je regarde le runtime gateway et le rebond deterministe.",
            "timestamp": 1779482200.0,
        },
        {
            "role": "user",
            "content": "Peux-tu debugger le rebond gateway ?",
            "timestamp": 1779482400.0,
        },
    ]
    for idx in range(old_rebounds):
        rows.insert(
            0,
            {
                "role": "assistant",
                "content": f"Ancien rebond {idx}",
                "timestamp": 1779482475.0 - ((idx + 1) * 1000),
                "conversation_rebound": True,
            },
        )
    if rebound:
        rows.append(
            {
                "role": "assistant",
                "content": "Rebond precedent.",
                "timestamp": 1779482450.0,
                "conversation_rebound": True,
            }
        )
    return rows


def _paths(paths):
    inner, desires, curiosity, loops, relationships = paths
    return {
        "inner_state_path": inner,
        "desires_path": desires,
        "curiosity_log_path": curiosity,
        "open_loops_path": loops,
        "relationship_path": relationships,
    }


def _write_traits(tmp_path, traits):
    (tmp_path / "desire_traits.json").write_text(
        json.dumps({"schema_version": 1, "traits": traits})
    )


def test_rebound_blocked_when_inner_state_stale(tmp_path):
    paths = _write_persona(tmp_path, timestamp="2026-05-21T00:00:00Z")

    result = maybe_add_conversation_rebound(
        response="Reponse normale.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )

    assert result.added is False
    assert result.reason == "stale_inner_state"


def test_rebound_gate_blocks_terminal_ack_without_scoring(tmp_path):
    paths = _write_persona(tmp_path)
    suppression = SuppressionResult(
        True,
        TurnPolicyDecision(conversation_state="terminal_ack", should_reply=False, confidence=0.95),
        "telegram_dm_closure_silence",
    )

    result = maybe_add_conversation_rebound(
        response="Reponse normale.",
        message_text="ok merci",
        history=_history(),
        source=_source(),
        suppression=suppression,
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )

    assert result.added is False
    assert result.reason == "terminal_or_social_closure"
    assert result.score == 0.0
    assert result.dimensions == {}


def test_rebound_skips_when_score_is_below_threshold(tmp_path):
    paths = _write_persona(tmp_path)
    inner = paths[0]
    data = json.loads(inner.read_text())
    data.update({"social_temperature": 0.2, "satisfaction": 0.2, "energy": 0.2})
    inner.write_text(json.dumps(data))
    paths[2].write_text("")

    result = maybe_add_conversation_rebound(
        response="Reponse normale.",
        message_text="petit retour",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )

    assert result.added is False
    assert result.reason == "score_below_threshold"
    assert result.score < 0.65
    assert set(result.dimensions) == {"momentum", "profondeur", "nouveaute", "resonance", "elan"}


def test_rebound_added_when_deterministic_score_allows(tmp_path):
    paths = _write_persona(tmp_path)

    result = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: _Response("Je peux aussi surveiller le prochain tour pour confirmer que le log est propre."),
        **_paths(paths),
    )

    assert result.added is True
    assert result.score >= 0.65
    assert result.response.endswith("confirmer que le log est propre.")


def test_rebound_score_uses_configured_rebond_trait_weights(tmp_path):
    paths = _write_persona(tmp_path)
    _write_traits(
        tmp_path,
        [
            {"name": "ne_pas_deranger", "weight": 0.5},
            {"name": "rebond_momentum", "weight": 0.0},
            {"name": "rebond_profondeur", "weight": 1.0},
            {"name": "rebond_nouveaute", "weight": 0.0},
            {"name": "rebond_resonance", "weight": 0.0},
            {"name": "rebond_elan", "weight": 0.0},
        ],
    )

    result = maybe_add_conversation_rebound(
        response="Le module runtime reste stable.",
        message_text="Architecture gateway runtime",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )

    assert result.added is False
    assert result.reason == "score_below_threshold"
    assert result.score == result.dimensions["profondeur"] == 0.6
    assert result.weights["profondeur"] == 1.0


def test_rebound_missing_trait_uses_default_weight_with_warning(tmp_path, caplog):
    paths = _write_persona(tmp_path)
    _write_traits(
        tmp_path,
        [
            {"name": "ne_pas_deranger", "weight": 0.5},
            {"name": "rebond_momentum", "weight": 0.0},
            {"name": "rebond_profondeur", "weight": 0.0},
            {"name": "rebond_nouveaute", "weight": 0.0},
            {"name": "rebond_resonance", "weight": 0.0},
        ],
    )
    caplog.set_level("WARNING", logger="agent.conversation_turn_policy")

    result = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: _Response("Je peux aussi surveiller le prochain tour."),
        **_paths(paths),
    )

    assert result.added is True
    assert result.weight_fallbacks == ["rebond_elan"]
    assert result.weights["elan"] == 0.7
    assert result.score == result.dimensions["elan"] == 0.82
    assert "rebond_elan" in caplog.text


def test_rebound_retries_when_model_exhausts_output_budget(tmp_path):
    paths = _write_persona(tmp_path)
    calls = []

    def call_llm_fn(**kwargs):
        calls.append((kwargs["max_tokens"], kwargs.get("timeout")))
        if len(calls) == 1:
            return _Response("", finish_reason="length")
        return _Response("Je peux continuer le test avec toi.")

    result = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=call_llm_fn,
        **_paths(paths),
    )

    assert calls == [(512, 5), (2048, 5)]
    assert result.added is True
    assert result.finish_reason == "stop"


def test_rebound_passes_timeout_to_model_call(tmp_path):
    paths = _write_persona(tmp_path)
    calls = []

    def call_llm_fn(**kwargs):
        calls.append(kwargs)
        return _Response("Je peux aussi surveiller le prochain tour.")

    result = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=call_llm_fn,
        **_paths(paths),
    )

    assert result.added is True
    assert calls[0]["task"] == "conversation_rebound"
    assert calls[0]["timeout"] == 5


def test_rebound_uses_expanded_sentence_budget_above_ceiling(tmp_path):
    paths = _write_persona(
        tmp_path,
        relationship={"gwenael": {"trust": 1, "intimacy": 1, "reciprocity": 1, "alignment": 1}},
    )
    calls = []

    def call_llm_fn(**kwargs):
        calls.append(kwargs)
        return _Response("Je garde ce fil ouvert et je te propose la suite technique.")

    result = maybe_add_conversation_rebound(
        response="Le rebond deterministe touche le gateway, le runtime et l'identite intime de Judy.",
        message_text="Cette histoire de conscience et d'identite dans le gateway me travaille.",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=call_llm_fn,
        **_paths(paths),
    )

    assert result.added is True
    assert result.score >= 0.85
    assert "2-3 sentences" in calls[0]["messages"][0]["content"]


def test_rebound_blocked_by_do_not_disturb(tmp_path):
    paths = _write_persona(tmp_path, ne_pas_deranger=0.9)

    result = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )

    assert result.added is False
    assert result.reason == "do_not_disturb"


def test_rebound_do_not_disturb_uses_migrated_desire_traits(tmp_path):
    paths = _write_persona(tmp_path, ne_pas_deranger=0.1)
    (tmp_path / "desire_traits.json").write_text(
        json.dumps({"schema_version": 1, "traits": [{"name": "ne_pas_deranger", "weight": 0.9}]})
    )

    result = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )

    assert result.added is False
    assert result.reason == "do_not_disturb"


def test_rebound_cooldowns_block_consecutive_and_window(tmp_path):
    paths = _write_persona(tmp_path)

    consecutive = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(rebound=True),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )
    window = maybe_add_conversation_rebound(
        response="J'ai corrige le runtime gateway pour le rebond deterministe.",
        message_text="Peux-tu debugger le rebond gateway ?",
        history=_history(old_rebounds=3),
        source=_source(),
        now=1779482475.0,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
        **_paths(paths),
    )

    assert consecutive.reason == "consecutive_cooldown"
    assert consecutive.score >= 0.65
    assert set(consecutive.dimensions) == {"momentum", "profondeur", "nouveaute", "resonance", "elan"}
    assert window.reason == "window_cooldown"
    assert window.score >= 0.65
