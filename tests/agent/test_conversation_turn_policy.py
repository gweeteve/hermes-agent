import json
from types import SimpleNamespace

from agent.conversation_turn_policy import (
    classify_conversation_turn,
    should_suppress_conversation_turn,
)


class _Response:
    def __init__(self, content: str):
        self.choices = [SimpleNamespace(message=SimpleNamespace(content=content))]


def _source(chat_type="dm", thread_id=None, platform="telegram"):
    return SimpleNamespace(
        platform=SimpleNamespace(value=platform),
        chat_type=chat_type,
        thread_id=thread_id,
    )


def _history():
    return [{"role": "assistant", "content": "Bien recu.", "timestamp": 1779482355.0}]


def _classify(monkeypatch, *, message_text="ok", payload=None, source=None):
    monkeypatch.setattr("agent.conversation_turn_policy.time.time", lambda: 1779482475.0)
    payload = payload or {
        "conversation_state": "terminal_ack",
        "should_reply": False,
        "confidence": 1.0,
        "reason": "terminal ack",
    }

    def fake_call_llm(**kwargs):
        assert kwargs["task"] == "conversation_turn_policy"
        return _Response(json.dumps(payload))

    return classify_conversation_turn(
        message_text=message_text,
        history=_history(),
        source=source or _source(),
        call_llm_fn=fake_call_llm,
    )


def test_turn_policy_suppresses_confident_terminal_ack_in_telegram_dm(monkeypatch):
    decision = _classify(
        monkeypatch,
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 1.0,
            "reason": "acknowledgement terminal",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="ok",
        source=_source(),
    )

    assert result.suppress is True
    assert result.reason == "telegram_dm_closure_silence"
    assert decision.conversation_state == "terminal_ack"


def test_turn_policy_suppresses_short_affirmation_without_assistant_question(monkeypatch):
    decision = _classify(
        monkeypatch,
        message_text="oui",
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 1.0,
            "reason": "acknowledgement terminal",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="oui",
        source=_source(),
    )

    assert result.suppress is True
    assert result.reason == "telegram_dm_closure_silence"


def test_turn_policy_does_not_suppress_short_affirmation_after_assistant_question(monkeypatch):
    decision = _classify(
        monkeypatch,
        message_text="oui",
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 1.0,
            "reason": "acknowledgement terminal",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="oui",
        source=_source(),
        recent_assistant_message="Tu veux que je continue ?",
    )

    assert result.suppress is False
    assert result.reason == "assistant_question_pending"


def test_turn_policy_does_not_suppress_short_negation_after_assistant_question(monkeypatch):
    decision = _classify(
        monkeypatch,
        message_text="non",
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 1.0,
            "reason": "acknowledgement terminal",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="non",
        source=_source(),
        recent_assistant_message="Est-ce que tu veux une synthèse ?",
    )

    assert result.suppress is False
    assert result.reason == "assistant_question_pending"


def test_turn_policy_does_not_suppress_short_reply_after_binary_proposal(monkeypatch):
    decision = _classify(
        monkeypatch,
        message_text="oui",
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 1.0,
            "reason": "acknowledgement terminal",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="oui",
        source=_source(),
        recent_assistant_message="Je peux continuer si tu veux",
    )

    assert result.suppress is False
    assert result.reason == "assistant_question_pending"


def test_turn_policy_suppresses_confident_social_closure_in_telegram_dm(monkeypatch):
    decision = _classify(
        monkeypatch,
        message_text="a plus",
        payload={
            "conversation_state": "social_closure",
            "should_reply": False,
            "confidence": 0.96,
            "reason": "social closure",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="a plus",
        source=_source(),
    )

    assert result.suppress is True
    assert result.reason == "telegram_dm_closure_silence"
    assert decision.conversation_state == "social_closure"


def test_turn_policy_does_not_suppress_meaningful_social_closure(monkeypatch):
    message = "je suis fière de toi, Judy tu as gagné en autonomie"
    decision = _classify(
        monkeypatch,
        message_text=message,
        payload={
            "conversation_state": "social_closure",
            "should_reply": False,
            "confidence": 0.95,
            "reason": "social closure",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text=message,
        source=_source(),
    )

    assert result.suppress is False
    assert result.reason == "dm_reply_by_default"
    assert decision.conversation_state == "social_closure"


def test_turn_policy_does_not_suppress_named_affective_ack(monkeypatch):
    message = "bravo Judy, je suis impressionné"
    decision = _classify(
        monkeypatch,
        message_text=message,
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 0.99,
            "reason": "terminal ack",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text=message,
        source=_source(),
    )

    assert result.suppress is False
    assert result.reason == "dm_reply_by_default"


def test_turn_policy_does_not_suppress_coordination_update(monkeypatch):
    message = "voila c'est fait côté codex"
    decision = _classify(
        monkeypatch,
        message_text=message,
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 0.99,
            "reason": "terminal ack",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text=message,
        source=_source(),
    )

    assert result.suppress is False
    assert result.reason == "dm_reply_by_default"


def test_turn_policy_does_not_suppress_dm_below_new_confidence_threshold(monkeypatch):
    decision = _classify(
        monkeypatch,
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 0.92,
            "reason": "acknowledgement terminal",
        },
    )

    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="ok",
        source=_source(),
    )

    assert result.suppress is False
    assert result.reason == "dm_reply_by_default"
    assert decision.conversation_state == "terminal_ack"


def test_turn_policy_does_not_suppress_active_request(monkeypatch):
    decision = _classify(
        monkeypatch,
        message_text="continue",
        payload={
            "conversation_state": "active_request",
            "should_reply": True,
            "confidence": 0.99,
            "reason": "request",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="continue",
        source=_source(),
    )

    assert result.suppress is False
    assert result.reason == "dm_reply_by_default"


def test_turn_policy_does_not_suppress_structural_request_signal(monkeypatch):
    monkeypatch.setattr("agent.conversation_turn_policy.time.time", lambda: 1779482475.0)

    decision = classify_conversation_turn(
        message_text="ok ?",
        history=_history(),
        source=_source(),
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="ok ?",
        source=_source(),
    )

    assert decision.reason == "structural_request_signal"
    assert result.suppress is False
    assert result.reason == "dm_reply_by_default"


def test_turn_policy_does_not_suppress_thread_even_with_confident_terminal_ack(monkeypatch):
    source = _source(chat_type="thread", thread_id="42")
    decision = _classify(
        monkeypatch,
        source=source,
        payload={
            "conversation_state": "terminal_ack",
            "should_reply": False,
            "confidence": 0.99,
            "reason": "acknowledgement terminal",
        },
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="ok",
        source=source,
    )

    assert result.suppress is False
    assert result.reason == "reply_by_default"


def test_turn_policy_falls_back_on_invalid_json(monkeypatch):
    monkeypatch.setattr("agent.conversation_turn_policy.time.time", lambda: 1779482475.0)

    decision = classify_conversation_turn(
        message_text="merci",
        history=_history(),
        source=_source(),
        call_llm_fn=lambda **_: _Response("not json"),
    )

    assert decision.should_reply is True
    assert should_suppress_conversation_turn(
        decision=decision,
        message_text="merci",
        source=_source(),
    ).suppress is False


def test_turn_policy_prefilter_skips_media_command_and_ambiguous_group():
    for kwargs in (
        {"has_media": True},
        {"is_command": True},
        {"source": _source(chat_type="group")},
    ):
        source = kwargs.get("source", _source())
        call_kwargs = {k: v for k, v in kwargs.items() if k != "source"}
        decision = classify_conversation_turn(
            message_text="ok",
            history=_history(),
            source=source,
            call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
            **call_kwargs,
        )
        assert decision.should_reply is True

        result = should_suppress_conversation_turn(
            decision=decision,
            message_text="ok",
            source=source,
        )
        assert result.suppress is False


def test_turn_policy_is_telegram_only():
    source = _source(platform="whatsapp")
    decision = classify_conversation_turn(
        message_text="ok",
        history=_history(),
        source=source,
        call_llm_fn=lambda **_: (_ for _ in ()).throw(AssertionError("no call")),
    )
    result = should_suppress_conversation_turn(
        decision=decision,
        message_text="ok",
        source=source,
    )

    assert decision.should_reply is True
    assert decision.reason == "gateway_not_eligible"
    assert result.suppress is False
    assert result.reason == "gateway_not_eligible"


def test_turn_policy_code_has_no_closure_phrase_table():
    import inspect
    import agent.conversation_turn_policy as mod

    source = inspect.getsource(mod)
    assert "bonne nuit" not in source
    assert "terminal_ack_words" not in source
    assert "closure_words" not in source
