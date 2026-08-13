import inspect
import importlib.util
from types import SimpleNamespace

import pytest

from dubbing.core.pipeline import emotions, transcription, translation
from dubbing.core.pipeline.context import PipelineRunContext
from dubbing.core.smart_dubbing import SmartDubbing


TRANSCRIPTION_METHODS = {
    "_initialize_transcriber": "initialize_transcriber",
    "_require_transcriber": "require_transcriber",
    "_load_cached_diarize_and_transcribe": "load_cached_diarize_and_transcribe",
    "diarize_and_transcribe": "diarize_and_transcribe",
    "_semantic_classifier": "semantic_classifier",
    "_write_semantic_boundary_diagnostics": "write_semantic_boundary_diagnostics",
    "_diarize_and_transcribe_isolated": "diarize_and_transcribe_isolated",
    "_isolated_inner_kwargs": "isolated_inner_kwargs",
    "_restore_semantic_plan_fingerprint": "restore_semantic_plan_fingerprint",
}

TRANSLATION_METHODS = {
    "_initialize_translator": "initialize_translator",
    "_require_translator": "require_translator",
    "translate_segments": "translate_segments",
    "_build_translation_prompt_prefix": "build_translation_prompt_prefix",
    "_persist_dubbing_text_snapshot": "persist_dubbing_text_snapshot",
    "_persist_synthesis_results": "persist_synthesis_results",
}

EMOTIONS_METHODS = {
    "analyze_emotions": "analyze_emotions",
    "_analyze_emotions_gemini": "analyze_emotions_gemini",
    "_analyze_emotions_speechbrain": "analyze_emotions_speechbrain",
}


def test_translation_pipeline_module_exists():
    assert importlib.util.find_spec("dubbing.core.pipeline.translation") is not None


def test_emotions_pipeline_module_exists():
    assert importlib.util.find_spec("dubbing.core.pipeline.emotions") is not None


@pytest.mark.parametrize(
    ("module", "owned_methods"),
    [
        (translation, TRANSLATION_METHODS),
        (emotions, EMOTIONS_METHODS),
    ],
)
def test_translation_and_emotion_services_export_every_owned_callable(
    module, owned_methods
):
    for helper_name in owned_methods.values():
        assert callable(getattr(module, helper_name, None)), helper_name


@pytest.mark.parametrize(
    ("module_alias", "facade_name", "helper_name"),
    [
        *(
            ("translation_helpers", facade_name, helper_name)
            for facade_name, helper_name in TRANSLATION_METHODS.items()
        ),
        *(
            ("emotion_helpers", facade_name, helper_name)
            for facade_name, helper_name in EMOTIONS_METHODS.items()
        ),
    ],
)
def test_translation_and_emotion_facade_methods_are_thin_delegates(
    module_alias, facade_name, helper_name
):
    source = inspect.getsource(SmartDubbing.__dict__[facade_name])

    assert f"{module_alias}.{helper_name}(" in source
    assert len(source.splitlines()) <= 16


def test_translation_facade_uses_module_callable_patched_at_call_time(monkeypatch):
    dubber = SmartDubbing.__new__(SmartDubbing)
    sentinel = object()
    calls = []

    def replacement(facade, base_prompt_prefix, stress_marks_requirement):
        calls.append((facade, base_prompt_prefix, stress_marks_requirement))
        return sentinel

    monkeypatch.setattr(translation, "build_translation_prompt_prefix", replacement)

    assert dubber._build_translation_prompt_prefix("custom") is sentinel
    assert calls[0][:2] == (dubber, "custom")
    assert "U+0301" in calls[0][2]


def test_emotions_facade_uses_module_callable_patched_at_call_time(monkeypatch):
    dubber = SmartDubbing.__new__(SmartDubbing)
    sentinel = object()
    calls = []

    def replacement(facade, segments, audio_file):
        calls.append((facade, segments, audio_file))
        return sentinel

    monkeypatch.setattr(emotions, "analyze_emotions", replacement)
    segments = [{}]

    assert dubber.analyze_emotions(segments, "audio.wav") is sentinel
    assert calls == [(dubber, segments, "audio.wav")]


def test_direct_translation_propagates_fingerprint_to_active_context_and_facade():
    events = []
    segments = [
        {
            "semantic_unit_id": "unit-1",
            "semantic_plan_fingerprint": "plan-live",
        }
    ]

    class Cache:
        def cache_exists(self, step, key):
            events.append(("exists", step, key))
            return False

        def save_to_cache(self, step, key, payload):
            events.append(("save", step, key, payload))

    class Translator:
        prompt_prefix = "base"

        def is_available(self):
            return True

        def translate(self, **kwargs):
            events.append(("translate", kwargs["segments"]))
            return kwargs["segments"]

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"source_language": "en", "target_language": "fr"}
    dubber.cache_manager = Cache()
    dubber.performance_tracker = SimpleNamespace(
        start_timing=lambda name: events.append(("start", name)),
        end_timing=lambda name: events.append(("end", name)) or 1.0,
    )
    dubber.debug_data = {}
    dubber._pipeline_run_context = PipelineRunContext()
    dubber._build_translation_cache_key = lambda _audio: "translation-key"
    dubber._require_translator = lambda: Translator()
    dubber._build_translation_prompt_prefix = lambda value: value
    dubber._persist_dubbing_text_snapshot = (
        lambda value, audio: events.append(("snapshot", value, audio))
    )

    result = translation.translate_segments(dubber, segments, "audio.wav")

    assert result is segments
    assert dubber._pipeline_run_context.semantic_plan_fingerprint == "plan-live"
    assert dubber._semantic_plan_fingerprint == "plan-live"
    assert events[-1] == ("snapshot", segments, "audio.wav")
    assert events.index(("save", "translation", "translation-key", segments)) < events.index(
        ("snapshot", segments, "audio.wav")
    )


def test_direct_translation_validates_cached_segments_against_active_context():
    cached = [{"semantic_plan_fingerprint": "plan-active"}]
    validated = []

    class Cache:
        def cache_exists(self, *_args):
            return True

        def load_from_cache(self, *_args):
            return cached

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = Cache()
    dubber.performance_tracker = SimpleNamespace(
        record_metric=lambda *args: None
    )
    dubber.debug_data = {}
    dubber._pipeline_run_context = PipelineRunContext(
        semantic_plan_fingerprint="plan-active"
    )
    dubber._build_translation_cache_key = lambda _audio: "translation-key"
    dubber._validate_plan_dependent_segments = (
        lambda value: validated.append(
            (dubber._pipeline_run_context.semantic_plan_fingerprint, value)
        )
    )
    dubber._persist_dubbing_text_snapshot = lambda *_args: None

    result = translation.translate_segments(dubber, [], "audio.wav")

    assert result is cached
    assert validated == [("plan-active", cached)]


def test_synthesis_persistence_keeps_snapshot_before_reusable_translation_cache():
    events = []
    segments = [{}]

    class Cache:
        def save_to_cache(self, step, key, payload):
            events.append(("save", step, key, payload))

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = Cache()
    dubber._semantic_plan_cache_persistable = True
    dubber._persist_dubbing_text_snapshot = (
        lambda value, audio: events.append(("snapshot", value, audio))
    )
    dubber._build_translation_cache_key = lambda _audio: "translation-key"

    translation.persist_synthesis_results(dubber, segments, "audio.wav")

    assert events == [
        ("snapshot", segments, "audio.wav"),
        ("save", "translation", "translation-key", segments),
    ]


def test_direct_emotion_analysis_mutates_input_in_place_and_uses_live_facade_helper():
    segments = [{"start": 0.0, "end": 1.0}]
    saved = []

    class Cache:
        def cache_exists(self, *_args):
            return False

        def save_to_cache(self, step, key, payload):
            saved.append((step, key, payload))

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"emotion_provider": "gemini", "emotion_model": "model-live"}
    dubber.cache_manager = Cache()
    dubber.performance_tracker = SimpleNamespace(
        start_timing=lambda *_args: None,
        end_timing=lambda *_args: None,
    )
    dubber._build_emotions_cache_key = lambda *_args: "emotion-key"

    def analyze(value, audio, model):
        assert value is segments
        assert (audio, model) == ("audio.wav", "model-live")
        value[0]["emotion"] = "Happy"

    dubber._analyze_emotions_gemini = analyze

    result = emotions.analyze_emotions(dubber, segments, "audio.wav")

    assert result is segments
    assert segments == [{"start": 0.0, "end": 1.0, "emotion": "Happy"}]
    assert saved == [("emotions", "emotion-key", segments)]
    assert saved[0][2] is segments


def test_transcription_service_exports_every_owned_internal_callable():
    for helper_name in TRANSCRIPTION_METHODS.values():
        assert callable(getattr(transcription, helper_name, None)), helper_name


@pytest.mark.parametrize(
    ("facade_name", "helper_name"), TRANSCRIPTION_METHODS.items()
)
def test_every_owned_transcription_facade_method_is_a_thin_delegate(
    facade_name, helper_name
):
    source = inspect.getsource(SmartDubbing.__dict__[facade_name])

    assert f"transcription_helpers.{helper_name}(" in source
    assert len(source.splitlines()) <= 16


def test_transcription_facade_delegates_with_live_facade_and_arguments(monkeypatch):
    dubber = SmartDubbing.__new__(SmartDubbing)
    sentinel = object()
    calls = []

    def replacement(facade, inner_system):
        calls.append((facade, inner_system))
        return sentinel

    monkeypatch.setattr(transcription, "isolated_inner_kwargs", replacement)

    assert dubber._isolated_inner_kwargs("deepgram") is sentinel
    assert calls == [(dubber, "deepgram")]


def test_direct_transcription_helper_uses_live_config_at_call_time():
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"deepgram_model": "first"}

    assert transcription.isolated_inner_kwargs(dubber, "deepgram") == {
        "deepgram_model": "first"
    }

    dubber.config = {"deepgram_model": "patched"}
    assert transcription.isolated_inner_kwargs(dubber, "deepgram") == {
        "deepgram_model": "patched"
    }


def test_restore_semantic_plan_propagates_fingerprint_to_context_and_facade():
    segments = [
        {"semantic_plan_fingerprint": "plan-live"},
        {"semantic_plan_fingerprint": "plan-live"},
    ]

    class Cache:
        def cache_exists(self, step, key):
            return (step, key) == ("isolated_tracks_semantic_plan", "plan-key")

        def load_from_cache(self, step, key):
            return {"transcription": segments}

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "isolated_tracks": {"SPEAKER_00": "track.wav"},
        "semantic_split_enabled": True,
    }
    dubber.cache_manager = Cache()
    dubber._isolated_tracks_cache_key = lambda *_args: "plan-key"

    transcription.restore_semantic_plan_fingerprint(dubber, "audio.wav")

    assert dubber._semantic_plan_fingerprint == "plan-live"
    assert dubber._semantic_plan_cache_persistable is True


def test_cached_semantic_plan_keeps_segment_list_identity_and_restores_fingerprint():
    segments = [{"semantic_plan_fingerprint": "plan-live"}]
    diarization = {(0.0, 1.0): "SPEAKER_00"}

    class Cache:
        use_cache = True

        def cache_exists(self, *_args):
            return True

        def load_from_cache(self, *_args):
            return {
                "diarization": diarization,
                "transcription": segments,
                "semantic_diagnostics": [],
            }

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "semantic_split_enabled": True,
        "debug_info": False,
        "inner_transcription_system": "deepgram",
    }
    dubber.cache_manager = Cache()
    dubber.debug_data = {}
    dubber._isolated_tracks_cache_key = lambda *_args: "plan-key"
    saved = []
    dubber._save_transcription_file = lambda value: saved.append(value)

    speakers, result = transcription.diarize_and_transcribe_isolated(
        dubber, "audio.wav", {"SPEAKER_00": "track.wav"}
    )

    assert speakers is diarization
    assert result is segments
    assert saved == [segments]
    assert saved[0] is segments
    assert dubber._semantic_plan_fingerprint == "plan-live"
