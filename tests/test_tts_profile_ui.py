from pathlib import Path

import pytest
import yaml

from dubbing.ui import gradio_app


def _profile(**overrides):
    profile = {
        "tts_system": "gemini",
        "model": "gemini-2.5-pro-preview-tts",
        "voice_name": "Kore",
        "style_prompt": "calm",
        "reference_mode": None,
        "reference_audio": None,
        "reference_text": None,
        "params": {},
    }
    profile.update(overrides)
    return profile


def test_profile_state_loads_new_and_legacy_shapes_with_new_style_precedence():
    state = gradio_app.voice_profiles_to_state(
        {
            "voices": {"SPEAKER_00": _profile(tts_system="gemini")},
            "tts_system_mapping": {
                "SPEAKER_00": "openai",
                "SPEAKER_01": "omnivoice",
            },
            "voice_prompt": {"SPEAKER_01": "warm"},
        }
    )

    assert state["SPEAKER_00"]["tts_system"] == "gemini"
    assert state["SPEAKER_01"]["tts_system"] == "omnivoice"
    assert state["SPEAKER_01"]["style_prompt"] == "warm"
    assert gradio_app.voice_profile_table_rows(state) == [
        ["SPEAKER_00", "gemini", "gemini-2.5-pro-preview-tts", "Kore", ""],
        ["SPEAKER_01", "omnivoice", "", "", ""],
    ]


def test_profile_state_promotes_old_top_level_defaults_to_star():
    state = gradio_app.voice_profiles_to_state(
        {
            "tts_system": "openai",
            "tts_model": "tts-1-hd",
            "voice_name": "nova",
            "reference_audio": "D:/voice.wav",
            "reference_text": "sample",
        }
    )

    assert state["*"] == {
        "tts_system": "openai",
        "model": "tts-1-hd",
        "voice_name": "nova",
        "style_prompt": None,
        "reference_audio": "D:/voice.wav",
        "reference_text": "sample",
        "reference_mode": "configured",
        "params": {},
    }


def test_profile_add_update_delete_and_validation(tmp_path):
    state = {"*": _profile()}

    status, state, rows, selected = gradio_app.save_voice_profile(
        state,
        None,
        "SPEAKER_00",
        "",
        "",
        "Aoede",
        "bright",
        "",
        "",
        "",
        "temperature: 0.7",
    )
    assert status == "Saved profile SPEAKER_00."
    assert selected == "SPEAKER_00"
    assert state["SPEAKER_00"]["voice_name"] == "Aoede"
    assert state["SPEAKER_00"]["params"] == {"temperature": 0.7}
    assert rows[-1][0] == "SPEAKER_00"

    unchanged = state
    status, new_state, _rows, _selected = gradio_app.save_voice_profile(
        state, None, "SPEAKER_00", "gemini", "flash", "", "", "", "", "", ""
    )
    assert "already exists" in status
    assert new_state == unchanged

    status, new_state, _rows, _selected = gradio_app.save_voice_profile(
        state, None, "speaker-2", "gemini", "flash", "", "", "", "", "", ""
    )
    assert "must be '*' or match SPEAKER_" in status
    assert new_state == unchanged

    status, new_state, _rows, _selected = gradio_app.save_voice_profile(
        state, None, "SPEAKER_02", "gemini", "flash", "", "", "", "", "", "- item"
    )
    assert "params must decode to a mapping" in status
    assert new_state == unchanged

    status, state, rows, selected = gradio_app.delete_voice_profile(state, "SPEAKER_00")
    assert status == "Deleted profile SPEAKER_00."
    assert "SPEAKER_00" not in state
    assert rows == [["*", "gemini", "gemini-2.5-pro-preview-tts", "Kore", ""]]
    assert selected is None


def test_updating_or_deleting_star_cannot_invalidate_named_profiles():
    state = {
        "*": _profile(tts_system="gemini", model="gemini-model"),
        "SPEAKER_00": _profile(
            tts_system="openai",
            model=None,
            voice_name="nova",
            style_prompt=None,
        ),
    }

    status, updated, _rows, _selected = gradio_app.save_voice_profile(
        state,
        "*",
        "*",
        "higgs",
        "",
        "",
        "",
        "segment",
        "",
        "",
        "",
    )
    assert "A model is required for openai" in status
    assert updated == state

    status, updated, _rows, selected = gradio_app.delete_voice_profile(state, "*")
    assert "A model is required for openai" in status
    assert updated == state
    assert selected == "*"


def test_library_assignment_updates_only_selected_profile(tmp_path):
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"audio")
    state = {
        "SPEAKER_00": _profile(tts_system="higgs", model=None),
        "SPEAKER_01": _profile(tts_system="higgs", model=None, voice_name="Other"),
    }

    status, updated, rows = gradio_app.assign_library_reference_to_profile(
        ["Library label", str(audio), "reference words"],
        state,
        "SPEAKER_00",
    )

    assert status == "Assigned library entry 'Library label' to SPEAKER_00."
    assert updated["SPEAKER_00"]["reference_audio"] == str(audio)
    assert updated["SPEAKER_00"]["reference_text"] == "reference words"
    assert updated["SPEAKER_00"]["reference_mode"] == "configured"
    assert updated["SPEAKER_01"] == state["SPEAKER_01"]
    assert rows[0][-1] == "configured"


def test_library_assignment_refreshes_selected_editor_fields(tmp_path):
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"audio")
    result = gradio_app._assign_library_reference_in_ui(
        ["Narrator", str(audio), "sample words"],
        {"SPEAKER_00": _profile(tts_system="higgs", model=None)},
        "SPEAKER_00",
    )

    assert result[3] == str(audio)
    assert result[4] == "sample words"
    assert result[5]["value"] == "configured"


def test_library_assignment_rejects_provider_without_reference_support(tmp_path):
    audio = tmp_path / "reference.wav"
    audio.write_bytes(b"audio")
    state = {"SPEAKER_00": _profile(tts_system="gemini")}

    status, updated, _rows = gradio_app.assign_library_reference_to_profile(
        ["Narrator", str(audio), "sample words"], state, "SPEAKER_00"
    )

    assert "does not support reference_mode" in status
    assert updated == state


def test_save_settings_writes_only_voices_and_removes_obsolete_tts_keys(tmp_path):
    config_path = tmp_path / "settings.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "tts_system": "openai",
                "tts_model": "tts-1",
                "tts_fallback_model": "old-fallback",
                "voice_name": {"SPEAKER_00": "nova"},
                "voice_prompt": {"SPEAKER_00": "warm"},
                "tts_system_mapping": {"SPEAKER_00": "openai"},
                "reference_audio": "old.wav",
                "reference_text": "old",
                "reference_audio_mapping": {"SPEAKER_00": "old.wav"},
                "reference_text_mapping": {"SPEAKER_00": "old"},
                "voices": {"SPEAKER_00": {"fallback_model": "also-old"}},
                "target_language": "be",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )
    state = {"*": _profile(tts_system="openai", model="tts-1", voice_name="nova")}

    status = gradio_app.save_settings({"voices": state}, config_path=str(config_path))
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))

    assert status == f"Settings saved to {config_path}"
    assert saved["voices"] == state
    assert saved["target_language"] == "be"
    for obsolete in {
        "tts_system",
        "tts_model",
        "tts_fallback_model",
        "voice_name",
        "voice_prompt",
        "tts_system_mapping",
        "reference_audio",
        "reference_text",
        "reference_audio_mapping",
        "reference_text_mapping",
    }:
        assert obsolete not in saved
    assert "fallback_model" not in yaml.safe_dump(saved)


def test_save_settings_rejects_invalid_profile_id_without_changing_file(tmp_path):
    config_path = tmp_path / "settings.yml"
    original = "target_language: be\n"
    config_path.write_text(original, encoding="utf-8")

    with pytest.raises(ValueError, match="SPEAKER_XX"):
        gradio_app.save_settings(
            {"voices": {"speaker-1": _profile()}},
            config_path=str(config_path),
        )

    assert config_path.read_text(encoding="utf-8") == original


def test_tts_profile_choices_follow_provider_and_keep_custom_values():
    models, voices, modes = gradio_app.get_tts_profile_choices(
        "gemini", "custom-model", "CustomVoice"
    )
    assert "gemini-2.5-pro-preview-tts" in models
    assert "custom-model" in models
    assert "Kore" in voices
    assert "CustomVoice" in voices
    assert modes == []

    models, voices, modes = gradio_app.get_tts_profile_choices("higgs", "", "")
    assert models == []
    assert voices == []
    assert modes == ["configured", "segment", "speaker"]


def test_build_app_exposes_structured_profiles_and_removes_legacy_controls():
    app = gradio_app.build_app()
    labels = {
        component.get("props", {}).get("label")
        for component in app.config["components"]
    }

    assert "Voice profiles" in labels
    assert "Profile speaker ID" in labels
    assert "Profile TTS system" in labels
    assert "Profile params (YAML)" in labels
    button_values = {
        component.get("props", {}).get("value")
        for component in app.config["components"]
        if component.get("type") == "button"
    }
    assert "Use in selected profile" in button_values
    for removed in {
        "Voices (YAML) — per-speaker TTS profiles",
        "Speaker reference mappings",
        "Fallback TTS model",
        "TTS system mapping JSON",
        "Voice prompt JSON",
        "Reference audio path",
        "Reference text",
    }:
        assert removed not in labels
