"""Tests for dubbing.core.voice_profiles.normalize_voices."""

import warnings

import pytest

import dubbing.core.config as config_module
from dubbing.core.runner import build_config_from_overrides
from dubbing.core.voice_profiles import (
    FALLBACK_SPEAKER,
    VoiceProfile,
    normalize_voices,
    resolve_profile,
)


def _patch_projects_root(monkeypatch, tmp_path):
    projects_root = tmp_path / "prj"
    monkeypatch.setattr(config_module, "DEFAULT_PROJECTS_ROOT", projects_root, raising=False)
    return projects_root


def test_normalize_empty_config_returns_empty_dict():
    assert normalize_voices({}) == {}


def test_normalize_new_style_voices_block_parses_params_bag():
    config = {
        "voices": {
            "SPEAKER_00": {
                "tts_system": "gemini",
                "model": "gemini-2.5-flash-preview-tts",
                "voice_name": "Kore",
                "style_prompt": "calm narrator",
                "params": {"temperature": 0.9},
            },
            "SPEAKER_01": {
                "tts_system": "omnivoice",
                "reference_audio": "D:/voice.wav",
                "reference_text": "sample",
                "instruct": "",
                "num_steps": 32,
            },
        }
    }
    voices = normalize_voices(config)

    assert voices["SPEAKER_00"].tts_system == "gemini"
    assert voices["SPEAKER_00"].model == "gemini-2.5-flash-preview-tts"
    assert voices["SPEAKER_00"].voice_name == "Kore"
    assert voices["SPEAKER_00"].style_prompt == "calm narrator"
    assert voices["SPEAKER_00"].params == {"temperature": 0.9}

    # Unknown keys fall into params (per-provider bag).
    assert voices["SPEAKER_01"].tts_system == "omnivoice"
    assert voices["SPEAKER_01"].reference_audio == "D:/voice.wav"
    assert voices["SPEAKER_01"].reference_text == "sample"
    assert voices["SPEAKER_01"].params == {"instruct": "", "num_steps": 32}


def test_normalize_legacy_mappings_emits_deprecation_and_folds_into_voices():
    config = {
        "tts_system_mapping": {"SPEAKER_00": "gemini", "SPEAKER_01": "omnivoice"},
        "voice_name": {"SPEAKER_00": "Kore"},
        "voice_prompt": {"SPEAKER_00": "calm narrator"},
        "reference_audio_mapping": {"SPEAKER_01": "D:/voice.wav"},
        "reference_text_mapping": {"SPEAKER_01": "sample"},
    }

    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        voices = normalize_voices(config)

    assert any(issubclass(w.category, DeprecationWarning) for w in caught), (
        "Legacy fields must trigger a DeprecationWarning"
    )

    assert voices["SPEAKER_00"].tts_system == "gemini"
    assert voices["SPEAKER_00"].voice_name == "Kore"
    assert voices["SPEAKER_00"].style_prompt == "calm narrator"
    assert voices["SPEAKER_00"].reference_audio is None

    assert voices["SPEAKER_01"].tts_system == "omnivoice"
    assert voices["SPEAKER_01"].voice_name is None
    assert voices["SPEAKER_01"].reference_audio == "D:/voice.wav"
    assert voices["SPEAKER_01"].reference_text == "sample"


def test_normalize_string_voice_name_becomes_fallback_voice():
    config = {"voice_name": "Kore"}
    voices = normalize_voices(config)
    assert FALLBACK_SPEAKER in voices
    assert voices[FALLBACK_SPEAKER].voice_name == "Kore"


def test_new_style_voices_takes_precedence_over_legacy_for_same_speaker():
    config = {
        "voices": {
            "SPEAKER_00": {"tts_system": "gemini", "voice_name": "Kore"},
        },
        "tts_system_mapping": {
            "SPEAKER_00": "omnivoice",  # legacy — should be ignored for SPEAKER_00
            "SPEAKER_01": "openai",     # legacy — still picked up (partial migration)
        },
    }
    voices = normalize_voices(config)

    assert voices["SPEAKER_00"].tts_system == "gemini"
    assert voices["SPEAKER_00"].voice_name == "Kore"
    assert voices["SPEAKER_01"].tts_system == "openai"


def test_resolve_profile_uses_star_fallback_and_default_system():
    profiles = {
        FALLBACK_SPEAKER: VoiceProfile(tts_system="omnivoice", voice_name="default"),
        "SPEAKER_00": VoiceProfile(voice_name="Kore"),
    }

    resolved_named = resolve_profile(profiles, "SPEAKER_00", tts_system_default="coqui")
    assert resolved_named.tts_system == "omnivoice"  # from "*" fallback
    assert resolved_named.voice_name == "Kore"       # from own profile

    resolved_unknown = resolve_profile(profiles, "SPEAKER_99", tts_system_default="coqui")
    assert resolved_unknown.tts_system == "omnivoice"
    assert resolved_unknown.voice_name == "default"

    resolved_empty = resolve_profile({}, "SPEAKER_00", tts_system_default="coqui")
    assert resolved_empty.tts_system == "coqui"


def test_pool_key_groups_identical_profiles_and_separates_by_model():
    a = VoiceProfile(tts_system="gemini", model="flash", params={"temperature": 0.5})
    b = VoiceProfile(tts_system="gemini", model="flash", params={"temperature": 0.5})
    c = VoiceProfile(tts_system="gemini", model="pro", params={"temperature": 0.5})
    d = VoiceProfile(tts_system="gemini", model="flash", params={"temperature": 0.9})

    assert a.pool_key() == b.pool_key()
    assert a.pool_key() != c.pool_key()
    assert a.pool_key() != d.pool_key()


def test_build_config_from_overrides_parses_voices_json_string(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    voices_json = (
        '{"SPEAKER_00": {"tts_system": "gemini", "voice_name": "Kore"}, '
        '"*": {"tts_system": "omnivoice"}}'
    )

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "voices": voices_json,
        }
    )

    voices = config.get("voices")
    assert isinstance(voices, dict)
    assert voices["SPEAKER_00"].tts_system == "gemini"
    assert voices["SPEAKER_00"].voice_name == "Kore"
    assert voices[FALLBACK_SPEAKER].tts_system == "omnivoice"


def test_build_config_from_overrides_folds_legacy_fields_into_voices(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = build_config_from_overrides(
            {
                "config": "",  # don't inherit repo dubbing_config.yml
                "input": str(video_path),
                "source_language": "en",
                "target_language": "be",
                "tts_system_mapping": '{"SPEAKER_00": "gemini"}',
                "voice_prompt": '{"SPEAKER_00": "calm"}',
                "reference_audio_mapping": '{"SPEAKER_00": "D:/voice.wav"}',
                "reference_text_mapping": '{"SPEAKER_00": "sample"}',
            }
        )

    voices = config.get("voices")
    assert isinstance(voices, dict)
    assert voices["SPEAKER_00"].tts_system == "gemini"
    assert voices["SPEAKER_00"].style_prompt == "calm"
    assert voices["SPEAKER_00"].reference_audio == "D:/voice.wav"
    assert voices["SPEAKER_00"].reference_text == "sample"
