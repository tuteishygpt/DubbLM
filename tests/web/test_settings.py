"""Framework-independent settings and voice-profile behavior."""

from hashlib import sha256
from concurrent.futures import ThreadPoolExecutor
from threading import Barrier

import pytest
import yaml

from dubbing.web import schema
from dubbing.web.settings import (
    SettingsConflictError,
    SettingsService,
    SettingsValidationError,
    SettingsWriteError,
)


def test_schema_preserves_exact_legacy_field_inventory():
    assert schema.WORKFLOW_FIELDS == [
        "input", "source_language", "target_language", "output", "config",
        "run_step", "generate_speaker_report", "save_original_subtitles",
        "save_translated_subtitles", "keep_background", "include_original_audio",
        "remove_pauses", "isolated_tracks_files", "isolated_tracks_labels",
        "inner_transcription_system",
    ]
    assert schema.SETTINGS_FIELDS == [
        "transcription_model", "transcription_system", "start_time", "duration",
        "no_cache", "translator_type", "llm_provider", "llm_model_name",
        "llm_temperature", "translation_prompt_prefix", "glossary",
        "refinement_llm_provider", "refinement_model_name",
        "refinement_temperature", "refinement_max_tokens", "refinement_persona",
        "voice_auto_selection", "voices", "tts_prompt_prefix",
        "enable_emotion_analysis", "emotion_provider", "emotion_model",
        "segment_reference_min_duration", "watermark_path", "watermark_text",
        "keep_original_audio_ranges", "min_pause_duration", "keyframe_buffer",
        "use_two_pass_encoding", "dubbed_volume", "background_volume",
        "timing_short_segment_threshold", "timing_short_segment_max_speed",
        "timing_max_speed", "timing_max_stretch", "timing_max_overflow",
        "semantic_split_enabled", "tts_preferred_segment_duration",
        "tts_hard_segment_duration", "semantic_split_search_window", "debug_info",
        "debug_tts", "debug_diarize_only",
    ]


def test_schema_exposes_legacy_run_and_provider_options():
    assert schema.RUN_MODES == [
        "full_pipeline", "from_scratch", "transcribe_only", "translate_only",
        "analyze_emotions_only", "combine_video", "tts_to_end",
    ]
    assert schema.INNER_TRANSCRIPTION_SYSTEM_CHOICES == ["deepgram", "assemblyai", "gemini"]
    assert schema.TRANSCRIPTION_SYSTEM_CHOICES == [
        "whisper", "openai", "pyannote_openai", "whisperx", "assemblyai", "gemini", "deepgram",
    ]
    assert schema.LLM_PROVIDER_CHOICES == ["gemini", "openrouter"]
    assert schema.EMOTION_PROVIDER_CHOICES == ["gemini", "speechbrain"]
    assert schema.TTS_PROVIDER_CHOICES == [
        "coqui", "xtts", "f5", "openai", "gemini", "bextts", "omnivoice", "higgs",
    ]
    assert schema.REFINEMENT_PERSONA_CHOICES == [
        "normal", "casual_manager", "child", "housewife", "science_popularizer", "it_buddy", "ai_buddy",
    ]


def test_schema_exposes_legacy_model_voice_and_reference_choices():
    assert schema.TRANSCRIPTION_MODEL_CHOICES["assemblyai"] == ["best", "nano"]
    assert schema.TRANSCRIPTION_MODEL_DEFAULTS["deepgram"] == "nova-3"
    assert schema.EMOTION_MODEL_CHOICES[0] == "gemini-3.1-flash-lite"
    assert schema.TTS_MODEL_CHOICES["openai"] == ["tts-1", "tts-1-hd"]
    assert schema.TTS_REFERENCE_CAPABILITIES == {
        "coqui": "required", "xtts": "required", "f5": "required",
        "f5_tts": "required", "omnivoice": "required", "higgs": "required",
        "bextts": "optional", "gemini": "unsupported", "openai": "unsupported",
    }
    assert schema.get_tts_profile_choices("gemini")[2] == []
    assert schema.get_tts_profile_choices("higgs")[2] == ["configured", "segment", "speaker"]
    assert schema.get_tts_profile_choices("bextts")[2] == ["none", "configured", "segment", "speaker"]


def _write_config(path, values):
    path.write_text(yaml.safe_dump(values, sort_keys=False, allow_unicode=True), encoding="utf-8")


def test_settings_loads_yaml_with_content_hash_revision(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"transcription_system": "assemblyai", "custom_key": {"keep": True}})

    loaded = SettingsService(config_path).load()

    assert loaded.values == {"transcription_system": "assemblyai", "custom_key": {"keep": True}}
    assert loaded.revision == sha256(config_path.read_bytes()).hexdigest()


def test_settings_save_preserves_unrelated_yaml_keys_and_parses_structured_values(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"custom_key": {"keep": True}, "duration": 30})
    service = SettingsService(config_path)
    loaded = service.load()

    saved = service.save(
        {
            "glossary": '{"hello": "привет"}',
            "keep_original_audio_ranges": "00:10-00:15\n01:02-01:08\n",
        },
        revision=loaded.revision,
    )

    persisted = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert persisted == {
        "custom_key": {"keep": True},
        "duration": 30,
        "glossary": {"hello": "привет"},
        "keep_original_audio_ranges": ["00:10-00:15", "01:02-01:08"],
    }
    assert saved.values == persisted
    assert saved.revision == sha256(config_path.read_bytes()).hexdigest()


def test_settings_save_removes_zero_duration(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"duration": 30, "source_language": "en"})
    service = SettingsService(config_path)

    service.save({"duration": 0}, revision=service.load().revision)

    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {"source_language": "en"}


def test_settings_save_rejects_stale_content_revision(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"llm_temperature": 0.5})
    service = SettingsService(config_path)
    stale = service.load()
    _write_config(config_path, {"llm_temperature": 0.7})

    with pytest.raises(SettingsConflictError):
        service.save({"llm_temperature": 0.9}, revision=stale.revision)

    assert yaml.safe_load(config_path.read_text(encoding="utf-8")) == {"llm_temperature": 0.7}


def test_settings_save_serializes_same_revision_writers(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"llm_temperature": 0.5})
    revision = SettingsService(config_path).load().revision
    barrier = Barrier(2)

    def save(value):
        barrier.wait()
        try:
            SettingsService(config_path).save({"llm_temperature": value}, revision=revision)
            return "saved"
        except SettingsConflictError:
            return "conflict"

    with ThreadPoolExecutor(max_workers=2) as executor:
        outcomes = list(executor.map(save, [0.7, 0.9]))

    assert sorted(outcomes) == ["conflict", "saved"]


def test_settings_atomic_save_preserves_prior_file_when_replace_fails(tmp_path, monkeypatch):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"llm_temperature": 0.5})
    service = SettingsService(config_path)
    revision = service.load().revision
    original = config_path.read_bytes()

    monkeypatch.setattr("dubbing.web.settings.os.replace", lambda *_args: (_ for _ in ()).throw(OSError("disk error")))

    with pytest.raises(SettingsWriteError):
        service.save({"llm_temperature": 0.9}, revision=revision)

    assert config_path.read_bytes() == original
    assert list(tmp_path.glob(".dubbing_config.yml.*.tmp")) == []


def test_profile_validation_rejects_invalid_speaker_and_incomplete_provider_configuration(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"voices": {}})
    service = SettingsService(config_path)
    revision = service.load().revision

    with pytest.raises(SettingsValidationError, match="Speaker ID"):
        service.put_profile("narrator", {"tts_system": "gemini", "model": "model"}, revision=revision)
    with pytest.raises(SettingsValidationError, match="model"):
        service.put_profile("SPEAKER_00", {"tts_system": "gemini"}, revision=revision)
    with pytest.raises(SettingsValidationError, match="reference_mode"):
        service.put_profile("SPEAKER_00", {"tts_system": "higgs"}, revision=revision)


def test_profiles_list_and_put_upsert_create_or_replace_only_matching_profile(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(
        config_path,
        {
            "voices": {
                "SPEAKER_00": {"tts_system": "gemini", "model": "old", "voice_name": "old voice"},
                "SPEAKER_01": {"tts_system": "gemini", "model": "keep", "voice_name": "keep voice"},
            }
        },
    )
    service = SettingsService(config_path)
    listed = service.list_profiles()
    assert listed.profiles["SPEAKER_01"]["voice_name"] == "keep voice"

    created = service.put_profile(
        "SPEAKER_02",
        {"tts_system": "gemini", "model": "new", "voice_name": "new voice"},
        revision=listed.revision,
    )
    replaced = service.put_profile(
        "SPEAKER_00",
        {"tts_system": "gemini", "model": "replacement"},
        revision=created.revision,
    )

    assert replaced.profiles == {
        "SPEAKER_00": {"tts_system": "gemini", "model": "replacement"},
        "SPEAKER_01": {"tts_system": "gemini", "model": "keep", "voice_name": "keep voice"},
        "SPEAKER_02": {"tts_system": "gemini", "model": "new", "voice_name": "new voice"},
    }


def test_profile_delete_and_reference_assignment_share_settings_revision(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(
        config_path,
        {"voices": {
            "SPEAKER_00": {"tts_system": "higgs", "reference_mode": "segment"},
            "SPEAKER_01": {"tts_system": "gemini", "model": "keep"},
        }},
    )
    reference_audio = tmp_path / "SPEAKER_00.wav"
    reference_audio.write_bytes(b"reference")
    service = SettingsService(config_path)
    assigned = service.assign_reference(
        "SPEAKER_00",
        reference_audio=str(reference_audio),
        reference_text="Reference transcript",
        revision=service.list_profiles().revision,
    )
    deleted = service.delete_profile("SPEAKER_01", revision=assigned.revision)

    assert deleted.profiles == {
        "SPEAKER_00": {
            "tts_system": "higgs",
            "reference_mode": "configured",
            "reference_audio": str(reference_audio),
            "reference_text": "Reference transcript",
        }
    }
    with pytest.raises(SettingsConflictError):
        service.delete_profile("SPEAKER_00", revision=assigned.revision)


def test_partial_named_profile_inherits_star_fallback_for_validation(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(
        config_path,
        {"voices": {"*": {"tts_system": "gemini", "model": "fallback-model"}}},
    )
    service = SettingsService(config_path)

    saved = service.put_profile(
        "SPEAKER_00",
        {"voice_name": "Kore"},
        revision=service.list_profiles().revision,
    )

    assert saved.profiles["SPEAKER_00"] == {"voice_name": "Kore"}


def test_star_update_or_delete_cannot_invalidate_partial_named_profiles(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(
        config_path,
        {"voices": {
            "*": {"tts_system": "gemini", "model": "fallback-model"},
            "SPEAKER_00": {"voice_name": "Kore"},
        }},
    )
    service = SettingsService(config_path)
    revision = service.list_profiles().revision

    with pytest.raises(SettingsValidationError, match="tts_system"):
        service.put_profile("*", {"voice_name": "other"}, revision=revision)
    with pytest.raises(SettingsValidationError, match="tts_system"):
        service.delete_profile("*", revision=revision)


def test_configured_reference_profile_requires_existing_audio_path(tmp_path):
    config_path = tmp_path / "dubbing_config.yml"
    _write_config(config_path, {"voices": {}})
    service = SettingsService(config_path)

    with pytest.raises(SettingsValidationError, match="Reference file does not exist"):
        service.put_profile(
            "SPEAKER_00",
            {
                "tts_system": "higgs",
                "reference_mode": "configured",
                "reference_audio": str(tmp_path / "missing.wav"),
            },
            revision=service.list_profiles().revision,
        )
