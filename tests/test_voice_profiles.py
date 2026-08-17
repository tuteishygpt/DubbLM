"""Tests for dubbing.core.voice_profiles.normalize_voices."""

import warnings

import pytest

import dubbing.core.config as config_module
from dubbing.core.runner import build_config_from_overrides
from dubbing.core.voice_profiles import (
    FALLBACK_SPEAKER,
    LegacyVoiceConfigError,
    VoiceProfile,
    normalize_voices,
    reject_legacy_voice_config,
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


@pytest.mark.parametrize(
    ("config", "key"),
    [
        ({"tts_system_mapping": {}}, "tts_system_mapping"),
        ({"voice_prompt": None}, "voice_prompt"),
        ({"reference_audio_mapping": {}}, "reference_audio_mapping"),
        ({"reference_text_mapping": {}}, "reference_text_mapping"),
        ({"voice_name": {"SPEAKER_00": "Kore"}}, "voice_name"),
        ({"voice_name": "SPEAKER_00:Kore,SPEAKER_01:Aoede"}, "voice_name"),
    ],
)
def test_reject_legacy_voice_config_has_exact_migration_error(config, key):
    with pytest.raises(
        LegacyVoiceConfigError,
        match=(
            rf"^Legacy per-speaker TTS setting '{key}' is no longer supported in direct test; "
            r"migrate speaker configuration to 'voices:'\.$"
        ),
    ):
        reject_legacy_voice_config(config, source="direct test")


def test_normalize_voices_rejects_legacy_config_at_direct_library_boundary():
    with pytest.raises(
        LegacyVoiceConfigError,
        match=(
            r"^Legacy per-speaker TTS setting 'voice_prompt' is no longer supported in "
            r"normalize_voices; migrate speaker configuration to 'voices:'\.$"
        ),
    ):
        normalize_voices({"voice_prompt": {"SPEAKER_00": "calm narrator"}})


def test_normalize_string_voice_name_becomes_fallback_voice():
    config = {"voice_name": "Kore"}
    voices = normalize_voices(config)
    assert FALLBACK_SPEAKER in voices
    assert voices[FALLBACK_SPEAKER].voice_name == "Kore"


def test_normalize_top_level_tts_defaults_become_star_profile():
    voices = normalize_voices(
        {
            "tts_system": "openai",
            "tts_model": "tts-1-hd",
            "voice_name": "nova",
            "reference_audio": "D:/voice.wav",
            "reference_text": "sample",
        }
    )

    fallback = voices[FALLBACK_SPEAKER]
    assert fallback.tts_system == "openai"
    assert fallback.model == "tts-1-hd"
    assert fallback.voice_name == "nova"
    assert fallback.reference_audio == "D:/voice.wav"
    assert fallback.reference_text == "sample"
    assert fallback.reference_mode == "configured"


def test_removed_fallback_models_are_ignored_and_warned():
    with warnings.catch_warnings(record=True) as caught:
        warnings.simplefilter("always")
        voices = normalize_voices(
            {
                "tts_fallback_model": "top-level-old",
                "voices": {
                    "SPEAKER_00": {
                        "tts_system": "gemini",
                        "model": "selected-model",
                        "fallback_model": "profile-old",
                    }
                },
            }
        )

    assert voices["SPEAKER_00"].model == "selected-model"
    assert "fallback_model" not in voices["SPEAKER_00"].params
    assert sum("fallback_model" in str(item.message) for item in caught) == 1


def test_scalar_global_and_nested_voice_names_remain_allowed():
    voices = normalize_voices(
        {
            "voice_name": "nova",
            "voices": {
                "SPEAKER_00": {"tts_system": "gemini", "voice_name": "Kore"},
            },
        }
    )

    assert voices[FALLBACK_SPEAKER].voice_name == "nova"
    assert voices["SPEAKER_00"].voice_name == "Kore"


def test_reference_mode_is_parsed_inherited_and_excluded_from_pool_identity():
    voices = normalize_voices(
        {
            "voices": {
                "*": {"tts_system": "higgs", "reference_mode": "segment"},
                "SPEAKER_00": {"reference_mode": "configured"},
                "SPEAKER_01": {"reference_mode": "speaker"},
                "SPEAKER_02": {"reference_mode": "none"},
            }
        }
    )

    assert resolve_profile(voices, "SPEAKER_00").reference_mode == "configured"
    assert resolve_profile(voices, "SPEAKER_01").reference_mode == "speaker"
    assert resolve_profile(voices, "SPEAKER_02").reference_mode == "none"
    assert resolve_profile(voices, "SPEAKER_99").reference_mode == "segment"
    assert VoiceProfile(tts_system="higgs", reference_mode="segment").pool_key() == (
        VoiceProfile(tts_system="higgs", reference_mode="configured").pool_key()
    )


def test_explicit_provider_switch_does_not_inherit_wildcard_reference_mode():
    voices = normalize_voices(
        {
            "voices": {
                "*": {"tts_system": "higgs", "reference_mode": "segment"},
                "SPEAKER_00": {"tts_system": "gemini", "voice_name": "Kore"},
            }
        }
    )

    resolved = resolve_profile(voices, "SPEAKER_00")
    assert resolved.tts_system == "gemini"
    assert resolved.reference_mode is None


def test_provider_switch_clears_mode_when_wildcard_provider_comes_from_global_default():
    voices = normalize_voices(
        {
            "voices": {
                "*": {"reference_mode": "segment"},
                "SPEAKER_00": {"tts_system": "gemini"},
            }
        }
    )

    resolved = resolve_profile(
        voices,
        "SPEAKER_00",
        tts_system_default="higgs",
    )
    assert resolved.tts_system == "gemini"
    assert resolved.reference_mode is None


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


def test_build_config_from_overrides_rejects_raw_legacy_override(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    with pytest.raises(
        LegacyVoiceConfigError,
        match=(
            r"^Legacy per-speaker TTS setting 'tts_system_mapping' is no longer supported in "
            r"runner overrides; migrate speaker configuration to 'voices:'\.$"
        ),
    ):
        build_config_from_overrides(
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


def test_load_from_yaml_rejects_raw_legacy_mapping_before_merge(tmp_path):
    config_path = tmp_path / "legacy.yml"
    config_path.write_text("voice_prompt: null\ntarget_language: be\n", encoding="utf-8")
    config = config_module.DubbingConfig()
    original = config.to_dict()

    with pytest.raises(
        LegacyVoiceConfigError,
        match=(
            r"^Legacy per-speaker TTS setting 'voice_prompt' is no longer supported in "
            r"YAML config; migrate speaker configuration to 'voices:'\.$"
        ),
    ):
        config.load_from_yaml(str(config_path))

    assert config.to_dict() == original


def test_process_special_parameters_rejects_legacy_config_safety_net():
    config = config_module.DubbingConfig()
    config.set("reference_text_mapping", {"SPEAKER_00": "sample"})

    with pytest.raises(
        LegacyVoiceConfigError,
        match=(
            r"^Legacy per-speaker TTS setting 'reference_text_mapping' is no longer supported in "
            r"merged config; migrate speaker configuration to 'voices:'\.$"
        ),
    ):
        config.process_special_parameters()
