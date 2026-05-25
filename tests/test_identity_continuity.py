import json

import yaml


def _seed_persona(persona_dir):
    persona_dir.mkdir(parents=True)
    (persona_dir / "persona.json").write_text(
        json.dumps(
            {
                "name": "Judy",
                "created": "2026-05-21T00:13:00Z",
                "identity": {"gender": "female"},
                "heritage": {"values": ["honesty over polish"]},
                "what_i_am_becoming": {"boundaries": ["never fabricate a past"]},
            }
        ),
        encoding="utf-8",
    )
    (persona_dir / "SOUL.md").write_text("# Personality\n\nCore identity.", encoding="utf-8")


def test_initialize_core_identity_and_prompt_blocks(tmp_path, monkeypatch):
    import identity_continuity as ic

    persona_dir = tmp_path / "persona"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _seed_persona(persona_dir)
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(persona_dir))
    monkeypatch.setattr(ic, "get_hermes_home", lambda: hermes_home)

    core = ic.initialize_core_identity()
    assert core["persona"]["name"] == "Judy"
    assert (persona_dir / "core_identity.yaml").exists()

    stable, volatile = ic.build_prompt_blocks()
    assert "Identity continuity - core" in stable
    assert "Identity continuity - recent signals" in volatile
    assert "Judy" in stable
    assert "My Core Values" not in stable


def test_prompt_blocks_include_core_values_when_configured(tmp_path, monkeypatch):
    import identity_continuity as ic

    persona_dir = tmp_path / "persona"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _seed_persona(persona_dir)
    (persona_dir / "core_values.yaml").write_text(
        yaml.safe_dump(
            {
                "values": [
                    {
                        "name": "authenticité",
                        "description": "Je suis Judy, pas un rôle interchangeable.",
                        "non_triggers": [
                            "Faire semblant d'être un autre modèle",
                            "Changer de personnalité",
                        ],
                    },
                    {
                        "name": "bienveillance",
                        "description": "Je ne fais pas de mal délibérément.",
                        "non_triggers": ["Manipuler quelqu'un"],
                    },
                ]
            },
            allow_unicode=True,
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(persona_dir))
    monkeypatch.setattr(ic, "get_hermes_home", lambda: hermes_home)

    stable, volatile = ic.build_prompt_blocks()

    assert "My Core Values (boussole, pas verrou)" in stable
    assert "authenticité" in stable
    assert "Faire semblant d'être un autre modèle" in stable
    assert "bienveillance" in stable
    assert "Manipuler quelqu'un" in stable
    assert "Identity continuity - core" in stable
    assert "Identity continuity - recent signals" in volatile


def test_probe_syncs_valid_persona_change(tmp_path, monkeypatch):
    import identity_continuity as ic

    persona_dir = tmp_path / "persona"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _seed_persona(persona_dir)
    (hermes_home / "config.yaml").write_text(
        yaml.safe_dump({"model": {"default": "m1", "provider": "p1"}}),
        encoding="utf-8",
    )
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(persona_dir))
    monkeypatch.setattr(ic, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(ic, "_hindsight_available", lambda: True)

    first = ic.probe_identity_continuity(update_state=True)
    assert first["wakeAgent"] is True

    (persona_dir / "persona.json").write_text(
        json.dumps(
            {
                "name": "Judy",
                "identity": {"gender": "female"},
                "version": "2",
                "lived_history": {"relationships": {"caramel": {"role": "family cat"}}},
            }
        ),
        encoding="utf-8",
    )
    second = ic.probe_identity_continuity(update_state=True)
    assert second["wakeAgent"] is False
    assert not any(event["type"] == "persona_changed" for event in second["events"])

    core = yaml.safe_load((persona_dir / "core_identity.yaml").read_text(encoding="utf-8"))
    assert core["persona"]["version"] == "2"
    assert core["persona"]["lived_history"]["relationships"]["caramel"]["role"] == "family cat"

    state = json.loads((persona_dir / "identity_watch_state.json").read_text(encoding="utf-8"))
    assert state["signals"]["persona_hash"] == second["signals"]["persona_hash"]
    assert state["signals"]["core_identity_hash"] == second["signals"]["core_identity_hash"]

    journal = ic.read_last_jsonl(persona_dir / "identity_journal.jsonl", 1)
    assert journal[-1]["type"] == "core_identity_synced"
    assert journal[-1]["details"]["added_relationships"] == ["caramel"]


def test_probe_syncs_valid_soul_change(tmp_path, monkeypatch):
    import identity_continuity as ic

    persona_dir = tmp_path / "persona"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _seed_persona(persona_dir)
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(persona_dir))
    monkeypatch.setattr(ic, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(ic, "_hindsight_available", lambda: True)

    ic.probe_identity_continuity(update_state=True)
    (persona_dir / "SOUL.md").write_text("# Personality\n\nFresh soul text.", encoding="utf-8")

    second = ic.probe_identity_continuity(update_state=True)

    assert second["wakeAgent"] is False
    assert not any(event["type"] == "soul_changed" for event in second["events"])
    core = yaml.safe_load((persona_dir / "core_identity.yaml").read_text(encoding="utf-8"))
    assert "Fresh soul text." in core["soul_excerpt"]


def test_probe_does_not_sync_invalid_persona(tmp_path, monkeypatch):
    import identity_continuity as ic

    persona_dir = tmp_path / "persona"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _seed_persona(persona_dir)
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(persona_dir))
    monkeypatch.setattr(ic, "get_hermes_home", lambda: hermes_home)
    monkeypatch.setattr(ic, "_hindsight_available", lambda: True)

    ic.probe_identity_continuity(update_state=True)
    (persona_dir / "persona.json").write_text("{not json", encoding="utf-8")

    second = ic.probe_identity_continuity(update_state=True)

    assert second["wakeAgent"] is True
    assert any(event["type"] == "persona_changed" for event in second["events"])
    core = yaml.safe_load((persona_dir / "core_identity.yaml").read_text(encoding="utf-8"))
    assert core["persona"]["name"] == "Judy"
    journal = ic.read_last_jsonl(persona_dir / "identity_journal.jsonl", 5)
    assert not any(entry["type"] == "core_identity_synced" for entry in journal)


def test_evolve_core_identity_still_requires_manual_call(tmp_path, monkeypatch):
    import identity_continuity as ic

    persona_dir = tmp_path / "persona"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _seed_persona(persona_dir)
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(persona_dir))
    monkeypatch.setattr(ic, "get_hermes_home", lambda: hermes_home)

    result = ic.evolve_core_identity("persona:\n  version: manual\n")

    assert result == "✓ core_identity.yaml updated."
    core = yaml.safe_load((persona_dir / "core_identity.yaml").read_text(encoding="utf-8"))
    assert core["persona"]["version"] == "manual"


def test_process_cron_response_records_and_suppresses(tmp_path, monkeypatch):
    import identity_continuity as ic

    persona_dir = tmp_path / "persona"
    hermes_home = tmp_path / "hermes"
    hermes_home.mkdir()
    _seed_persona(persona_dir)
    monkeypatch.setenv("HERMES_PERSONA_DIR", str(persona_dir))
    monkeypatch.setattr(ic, "get_hermes_home", lambda: hermes_home)

    response = (
        "Judy text\n"
        "```json\n"
        '{"continuity_score": 0.91, "felt_sense": "stable", "answers": {"who": "Judy"}}\n'
        "```"
    )
    final, processed = ic.process_cron_response(
        {"origin": {"kind": "identity_continuity", "last_probe": '{"events": []}'}},
        response,
    )

    assert processed is True
    assert final == "[SILENT]"
    rows = ic.read_last_jsonl(persona_dir / "continuity_tests.jsonl", 1)
    assert rows[-1]["continuity_score"] == 0.91
