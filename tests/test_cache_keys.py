import importlib
from pathlib import Path

import pytest

import dubbing.core.config as config_module
import dubbing.core.smart_dubbing as smart_dubbing_module
from dubbing.core.runner import build_config_from_overrides
from dubbing.core.smart_dubbing import SmartDubbing
from dubbing.core.voice_profiles import VoiceProfile
from translation.llm_translator import LLMTranslator


class _CacheManager:
    def generate_cache_key(self, *dimensions):
        return "audio-" + "|".join(str(value) for value in dimensions)


def _dubber(**overrides):
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = _CacheManager()
    dubber.config = {
        "source_language": "en",
        "target_language": "be",
        "whisper_model": "large-v3",
        "start_time": 0,
        "duration": 60,
        "translator_type": "llm",
        "llm_provider": "gemini",
        "llm_model_name": "model-a",
        "llm_temperature": 0.5,
        "llm_max_tokens": 16384,
        "refinement_llm_provider": "gemini",
        "refinement_model_name": "model-b",
        "refinement_temperature": 1.0,
        "refinement_max_tokens": 16384,
        "refinement_persona": "normal",
        "glossary": {"API": "інтэрфейс"},
        "translation_prompt_prefix": "Audience: engineers",
        "emotion_provider": "gemini",
        "emotion_model": "emotion-a",
    }
    dubber.config.update(overrides)
    return dubber


def _cache_keys_module():
    return importlib.import_module("dubbing.core.pipeline.cache_keys")


@pytest.mark.parametrize(
    ("facade_name", "module_name", "args", "kwargs"),
    [
        ("_cache_fingerprint", "cache_fingerprint", ({"value": 1},), {}),
        ("_effective_tts_cache_fingerprint", "effective_tts_cache_fingerprint", (["SPEAKER_00"],), {}),
        ("_file_content_identity", "file_content_identity", (None,), {}),
        ("_shared_audio_transcription_identity", "shared_audio_transcription_identity", ("source.wav",), {}),
        ("_effective_translation_cache_dimensions", "effective_translation_cache_dimensions", (), {}),
        ("_build_dubbing_text_snapshot_key", "build_dubbing_text_snapshot_key", ("source.wav",), {}),
        ("_build_translation_cache_key", "build_translation_cache_key", ("source.wav",), {}),
        ("_build_emotions_cache_key", "build_emotions_cache_key", ("source.wav", []), {}),
        ("_isolated_tracks_cache_key", "isolated_tracks_cache_key", ("source.wav", {}), {}),
        ("_isolated_tracks_raw_cache_key", "isolated_tracks_raw_cache_key", ({},), {}),
        ("_semantic_classifier_identity", "semantic_classifier_identity", (), {}),
        ("_tts_selection_cache_fingerprint", "tts_selection_cache_fingerprint", ([],), {}),
        ("_segment_cache_metadata_path", "segment_cache_metadata_path", (Path("segment.wav"),), {}),
        (
            "_raw_tts_segment_cache_key",
            "raw_tts_segment_cache_key",
            (),
            {
                "base_cache_prefix": "base",
                "tts_system": "fake",
                "segment": {},
                "speaker": "SPEAKER_00",
                "translation": "hello",
                "style_prompt": "",
                "reference_audio_path": None,
            },
        ),
    ],
)
def test_cache_key_facade_methods_delegate_to_module_at_call_time(
    monkeypatch, facade_name, module_name, args, kwargs
):
    module = _cache_keys_module()
    sentinel = object()
    calls = []

    def implementation(*implementation_args, **implementation_kwargs):
        calls.append((implementation_args, implementation_kwargs))
        return sentinel

    monkeypatch.setattr(module, module_name, implementation)
    dubber = _dubber()
    dubber.speakers_audio_dir = Path("speakers")

    assert getattr(dubber, facade_name)(*args, **kwargs) is sentinel
    assert len(calls) == 1


def test_cache_key_module_matches_exact_facade_results(tmp_path):
    module = _cache_keys_module()
    source = tmp_path / "source.bin"
    source.write_bytes(b"stable cache content")
    raw_key_args = {
        "base_cache_prefix": "base",
        "tts_system": "fake",
        "segment": {
            "start": 1.0,
            "end": 2.0,
            "_timing_available_window": 1.25,
            "semantic_unit_id": "unit-1",
            "semantic_plan_fingerprint": "plan-1",
        },
        "speaker": "SPEAKER_00",
        "translation": "hello",
        "style_prompt": "softly",
        "reference_audio_path": str(source),
        "reference_mode": "configured",
        "reference_text": "sample",
        "client_pool_settings": ("fake",),
    }

    assert module.cache_fingerprint({"b": 2, "a": 1}) == (
        SmartDubbing._cache_fingerprint({"b": 2, "a": 1})
    )
    assert module.file_content_identity(str(source)) == (
        SmartDubbing._file_content_identity(str(source))
    )
    assert module.segment_cache_metadata_path(Path("segment.wav")) == (
        SmartDubbing._segment_cache_metadata_path(Path("segment.wav"))
    )
    assert module.raw_tts_segment_cache_key(
        **raw_key_args,
        cache_fingerprint=SmartDubbing._cache_fingerprint,
        file_content_identity=SmartDubbing._file_content_identity,
    ) == SmartDubbing._raw_tts_segment_cache_key(**raw_key_args)


def test_translation_key_uses_current_facade_dependencies(monkeypatch):
    dubber = _dubber()
    calls = []
    monkeypatch.setattr(
        dubber,
        "_build_dubbing_text_snapshot_key",
        lambda audio_file: calls.append(("snapshot", audio_file)) or "snapshot-live",
    )
    monkeypatch.setattr(
        dubber,
        "_effective_translation_cache_dimensions",
        lambda: calls.append(("settings",)) or {"live": True},
    )
    monkeypatch.setattr(
        SmartDubbing,
        "_cache_fingerprint",
        staticmethod(
            lambda dimensions: calls.append(("fingerprint", dimensions))
            or "fingerprint-live"
        ),
    )

    assert dubber._build_translation_cache_key("source.wav") == (
        "translation_v2_snapshot-live_fingerprint-live"
    )
    assert calls == [
        ("snapshot", "source.wav"),
        ("settings",),
        ("fingerprint", {"live": True}),
    ]


def test_cache_fingerprint_is_canonical_and_rejects_unsupported_values():
    assert SmartDubbing._cache_fingerprint({"b": 2, "a": 1}) == (
        SmartDubbing._cache_fingerprint({"a": 1, "b": 2})
    )

    with pytest.raises(TypeError):
        SmartDubbing._cache_fingerprint({"unstable": object()})

    with pytest.raises(ValueError):
        SmartDubbing._cache_fingerprint({"not_json": float("nan")})


def test_effective_tts_fingerprint_tracks_strict_reference_contract():
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "tts_system": "higgs",
        "tts_model": None,
    }
    dubber.voice_profiles = {
        "SPEAKER_00": VoiceProfile(
            tts_system="higgs",
            voice_name="clone",
            reference_mode="configured",
            reference_audio="D:/voice.wav",
            reference_text="sample",
            params={"top_p": 0.95, "temperature": 0.7},
        )
    }

    baseline = dubber._effective_tts_cache_fingerprint(["SPEAKER_00"])
    dubber.voice_profiles["SPEAKER_00"].reference_mode = "speaker"
    assert dubber._effective_tts_cache_fingerprint(["SPEAKER_00"]) != baseline
    dubber.voice_profiles["SPEAKER_00"].reference_mode = "configured"
    dubber.voice_profiles["SPEAKER_00"].params["temperature"] = 0.8
    assert dubber._effective_tts_cache_fingerprint(["SPEAKER_00"]) != baseline


def test_effective_tts_fingerprint_includes_global_voice_and_omnivoice_bootstrap():
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "tts_system": "omnivoice",
        "voice_name": "global-a",
        "omnivoice_num_steps": 32,
        "omnivoice_speed": 1.0,
    }
    dubber.voice_profiles = {
        "SPEAKER_00": VoiceProfile(tts_system="omnivoice"),
    }

    baseline = dubber._effective_tts_cache_fingerprint(["SPEAKER_00"])
    dubber.config["voice_name"] = "global-b"
    assert dubber._effective_tts_cache_fingerprint(["SPEAKER_00"]) != baseline
    dubber.config["voice_name"] = "global-a"
    dubber.config["omnivoice_num_steps"] = 64
    assert dubber._effective_tts_cache_fingerprint(["SPEAKER_00"]) != baseline


def test_segment_cache_identity_includes_resolved_reference_and_client_settings():
    common = dict(
        base_cache_prefix="base",
        tts_system="higgs",
        segment={},
        speaker="SPEAKER_00",
        translation="hello",
        style_prompt="",
        reference_audio_path="D:/voice.wav",
    )

    baseline = SmartDubbing._raw_tts_segment_cache_key(
        **common,
        reference_mode="configured",
        reference_text="sample",
        client_pool_settings=("higgs", "", "", (("temperature", 0.7),)),
    )
    assert baseline != SmartDubbing._raw_tts_segment_cache_key(
        **common,
        reference_mode="speaker",
        reference_text="sample",
        client_pool_settings=("higgs", "", "", (("temperature", 0.7),)),
    )
    assert baseline != SmartDubbing._raw_tts_segment_cache_key(
        **common,
        reference_mode="configured",
        reference_text="different",
        client_pool_settings=("higgs", "", "", (("temperature", 0.7),)),
    )
    assert baseline != SmartDubbing._raw_tts_segment_cache_key(
        **common,
        reference_mode="configured",
        reference_text="sample",
        client_pool_settings=("higgs", "", "", (("temperature", 0.8),)),
    )


def test_glossary_rendering_is_deterministic():
    first = LLMTranslator(
        enable_cache=False,
        glossary={"zeta": "last", "alpha": "first"},
    )
    second = LLMTranslator(
        enable_cache=False,
        glossary={"alpha": "first", "zeta": "last"},
    )

    assert first._render_glossary_entries() == second._render_glossary_entries()
    assert first._render_glossary_entries().splitlines() == [
        '- "alpha" → "first"',
        '- "zeta" → "last"',
    ]


def test_snapshot_key_tracks_effective_prompt_but_not_semantic_plan():
    dubber = _dubber()
    baseline = dubber._build_dubbing_text_snapshot_key("source.wav")

    dubber._semantic_plan_fingerprint = "plan-a"
    assert dubber._build_dubbing_text_snapshot_key("source.wav") == baseline

    dubber.config["translation_prompt_prefix"] = "Audience: children"
    assert dubber._build_dubbing_text_snapshot_key("source.wav") != baseline


@pytest.mark.parametrize(
    ("dimension", "changed"),
    [
        ("translator_type", "other"),
        ("llm_provider", "openrouter"),
        ("llm_model_name", "model-c"),
        ("llm_temperature", 0.2),
        ("llm_max_tokens", 8192),
        ("refinement_llm_provider", "openrouter"),
        ("refinement_model_name", "model-d"),
        ("refinement_temperature", 0.7),
        ("refinement_max_tokens", 4096),
        ("refinement_persona", "child"),
        ("glossary", {"API": "праграмны інтэрфейс"}),
        ("translation_prompt_prefix", "Audience: children"),
    ],
)
def test_translation_key_tracks_every_output_dimension(dimension, changed):
    dubber = _dubber()
    baseline = dubber._build_translation_cache_key("source.wav")

    dubber.config[dimension] = changed

    assert dubber._build_translation_cache_key("source.wav") != baseline


def test_translation_key_tracks_semantic_plan_and_resolves_translator_defaults():
    minimal = _dubber()
    minimal.config = {
        "source_language": "en",
        "target_language": "be",
        "translation_prompt_prefix": None,
    }
    explicit = _dubber(
        llm_provider="gemini",
        llm_model_name="models/gemini-2.5-flash-preview-04-17",
        llm_temperature=0.5,
        llm_max_tokens=16384,
        refinement_llm_provider="gemini",
        refinement_model_name="models/gemini-2.5-flash-preview-04-17",
        refinement_temperature=1.0,
        refinement_max_tokens=16384,
        refinement_persona="normal",
        glossary={},
        translation_prompt_prefix=None,
    )
    for key in ("whisper_model", "start_time", "duration"):
        explicit.config.pop(key, None)

    assert minimal._build_translation_cache_key("source.wav") == (
        explicit._build_translation_cache_key("source.wav")
    )

    baseline = minimal._build_translation_cache_key("source.wav")
    minimal._semantic_plan_fingerprint = "plan-a"
    assert minimal._build_translation_cache_key("source.wav") != baseline


def test_initialized_smart_dubbing_uses_the_same_effective_translation_defaults(
    tmp_path, monkeypatch
):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    monkeypatch.setattr(
        config_module, "DEFAULT_PROJECTS_ROOT", tmp_path / "projects", raising=False
    )
    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "config": "",
            "llm_model_name": None,
            "refinement_llm_provider": None,
            "refinement_model_name": None,
            "refinement_max_tokens": None,
        }
    )
    captured = {}

    def fake_create_translator(**kwargs):
        captured.update(kwargs)
        return object()

    monkeypatch.setattr(
        smart_dubbing_module.TranslatorFactory,
        "create_translator",
        staticmethod(fake_create_translator),
    )
    monkeypatch.setattr(
        SmartDubbing,
        "_initialize_tts_systems",
        lambda self: setattr(self, "tts_systems", {}),
    )
    monkeypatch.setattr(
        SmartDubbing,
        "_initialize_transcriber",
        lambda self: setattr(self, "transcriber", None),
    )

    dubber = SmartDubbing(config)
    dimensions = dubber._effective_translation_cache_dimensions()

    assert captured["translator_type"] == dimensions["translator_type"]
    assert captured["llm_provider"] == dimensions["primary"]["provider"]
    assert captured["model_name"] == dimensions["primary"]["model"]
    assert captured["temperature"] == dimensions["primary"]["temperature"]
    assert captured["max_tokens"] == dimensions["primary"]["max_tokens"]
    assert captured["refinement_llm_provider"] == dimensions["refinement"]["provider"]
    assert captured["refinement_model_name"] == dimensions["refinement"]["model"]
    assert captured["refinement_temperature"] == dimensions["refinement"]["temperature"]
    assert captured["refinement_max_tokens"] == dimensions["refinement"]["max_tokens"]
    assert captured["refinement_persona"] == dimensions["refinement"]["persona"]


def _chunk_key(translator, **overrides):
    dimensions = {
        "chunk_text": "SPEAKER_00: Hello",
        "source_language": "en",
        "target_language": "be",
        "context_before": "Earlier translation",
        "context_after": "Next source chunk",
        "source_summary": "A discussion",
        "domain": "technology",
        "tone": "friendly",
        "themes": ["AI", "work"],
        "terminology": ["API", "LLM"],
    }
    dimensions.update(overrides)
    return translator._generate_cache_key(**dimensions)


@pytest.mark.parametrize(
    ("dimension", "changed"),
    [
        ("chunk_text", "SPEAKER_00: Goodbye"),
        ("source_language", "de"),
        ("target_language", "uk"),
        ("context_before", "Different earlier translation"),
        ("context_after", "Different next chunk"),
        ("source_summary", "A different discussion"),
        ("domain", "medicine"),
        ("tone", "formal"),
        ("themes", ["work", "AI"]),
        ("terminology", ["LLM", "API"]),
    ],
)
def test_inner_chunk_key_tracks_every_prompt_context(dimension, changed):
    translator = LLMTranslator(
        enable_cache=False,
        glossary={"API": "інтэрфейс"},
        prompt_prefix="Audience: engineers",
    )
    assert _chunk_key(translator, **{dimension: changed}) != _chunk_key(translator)


@pytest.mark.parametrize(
    ("attribute", "changed"),
    [
        ("llm_provider", "openrouter"),
        ("model_name", "another-model"),
        ("temperature", 0.2),
        ("max_tokens", 8192),
        ("glossary", {"API": "праграмны інтэрфейс"}),
        ("prompt_prefix", "Audience: children"),
    ],
)
def test_inner_chunk_key_tracks_primary_translator_settings(attribute, changed):
    translator = LLMTranslator(enable_cache=False, glossary={"API": "інтэрфейс"})
    baseline = _chunk_key(translator)
    setattr(translator, attribute, changed)
    assert _chunk_key(translator) != baseline


def test_inner_chunk_key_ignores_refinement_only_settings():
    translator = LLMTranslator(enable_cache=False)
    baseline = _chunk_key(translator)
    translator.refinement_model_name = "different-refiner"
    translator.refinement_temperature = 0.1
    translator.refinement_persona = "child"
    assert _chunk_key(translator) == baseline


def test_emotion_key_tracks_full_ordered_segment_payload_and_semantic_plan():
    dubber = _dubber()
    segments = [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 1.0,
            "text": "Hello",
            "translation": "Прывітанне",
        }
    ]
    baseline = dubber._build_emotions_cache_key(
        "source.wav", segments, "gemini", "emotion-a"
    )

    changed_text = [{**segments[0], "translation": "Добры дзень"}]
    assert dubber._build_emotions_cache_key(
        "source.wav", changed_text, "gemini", "emotion-a"
    ) != baseline

    reordered = [segments[0], {**segments[0], "speaker": "SPEAKER_01"}]
    assert dubber._build_emotions_cache_key(
        "source.wav", list(reversed(reordered)), "gemini", "emotion-a"
    ) != dubber._build_emotions_cache_key(
        "source.wav", reordered, "gemini", "emotion-a"
    )

    dubber._semantic_plan_fingerprint = "plan-a"
    assert dubber._build_emotions_cache_key(
        "source.wav", segments, "gemini", "emotion-a"
    ) != baseline


def test_emotion_key_tracks_provider_algorithm_identity_without_duplicate_names():
    dubber = _dubber()
    segments = [{"start": 0.0, "end": 1.0, "translation": "Прывітанне"}]

    gemini = dubber._build_emotions_cache_key(
        "source.wav", segments, "gemini", "emotion-a"
    )
    speechbrain = dubber._build_emotions_cache_key(
        "source.wav", segments, "speechbrain", "ignored"
    )

    assert gemini != speechbrain
    assert gemini.count("gemini") == 1
    assert gemini.count("emotion-a") == 1
    assert speechbrain.count("speechbrain") == 1
