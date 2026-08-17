import inspect
import importlib.util
import importlib
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

ARTIFACT_METHODS = {
    "_prepare_audio_inputs": "prepare_audio_inputs",
    "_load_required_cached_step": "load_required_cached_step",
    "_build_speaker_rolls_from_segments": "build_speaker_rolls_from_segments",
    "_save_requested_subtitles": "save_requested_subtitles",
    "_combine_final_video": "combine_final_video",
    "_reset_input_cache": "reset_input_cache",
    "_save_transcription_file": "save_transcription_file",
    "_get_subtitle_path": "get_subtitle_path",
    "adjust_subtitle_timestamps": "adjust_subtitle_timestamps",
}


def _artifacts_module():
    return importlib.import_module("dubbing.core.pipeline.artifacts")


def test_translation_pipeline_module_exists():
    assert importlib.util.find_spec("dubbing.core.pipeline.translation") is not None


def test_emotions_pipeline_module_exists():
    assert importlib.util.find_spec("dubbing.core.pipeline.emotions") is not None


def test_artifacts_pipeline_module_exists():
    assert importlib.util.find_spec("dubbing.core.pipeline.artifacts") is not None


def test_artifact_service_exports_every_owned_internal_callable():
    artifacts = _artifacts_module()

    for helper_name in ARTIFACT_METHODS.values():
        assert callable(getattr(artifacts, helper_name, None)), helper_name


@pytest.mark.parametrize(("facade_name", "helper_name"), ARTIFACT_METHODS.items())
def test_every_owned_artifact_facade_method_is_a_thin_delegate(
    facade_name, helper_name
):
    source = inspect.getsource(SmartDubbing.__dict__[facade_name])

    assert f"artifact_helpers.{helper_name}(" in source
    assert len(source.splitlines()) <= 16


def test_artifact_facade_uses_module_callable_patched_at_call_time(monkeypatch):
    artifacts = _artifacts_module()
    dubber = SmartDubbing.__new__(SmartDubbing)
    sentinel = object()
    calls = []

    def replacement(facade):
        calls.append(facade)
        return sentinel

    monkeypatch.setattr(artifacts, "prepare_audio_inputs", replacement)

    assert dubber._prepare_audio_inputs() is sentinel
    assert calls == [dubber]


def test_prepare_audio_inputs_uses_live_processor_and_exact_source_selection():
    artifacts = _artifacts_module()
    calls = []

    class Processor:
        def extract_audio(self, *args):
            calls.append(("extract", args))
            return "source.wav"

        def separate_background_and_vocals(self, audio_file):
            calls.append(("separate", audio_file))
            return "background.wav", "vocals.wav"

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "input": "movie.mp4",
        "start_time": 1.25,
        "duration": 8.5,
        "keep_background": True,
    }
    dubber.audio_processor = Processor()

    assert artifacts.prepare_audio_inputs(dubber) == (
        "source.wav",
        "background.wav",
        "vocals.wav",
    )
    assert calls == [
        ("extract", ("movie.mp4", 1.25, 8.5)),
        ("separate", "source.wav"),
    ]

    dubber.config["keep_background"] = False
    calls.clear()
    assert artifacts.prepare_audio_inputs(dubber) == (
        "source.wav",
        None,
        "source.wav",
    )
    assert calls == [("extract", ("movie.mp4", 1.25, 8.5))]


@pytest.mark.parametrize(
    ("cache", "message"),
    [
        (
            SimpleNamespace(use_cache=False),
            "run_step=tts_to_end requires cached translation artifacts from a previous full dubbing run, but caching is currently disabled. Re-enable cache or run the full pipeline first.",
        ),
        (
            SimpleNamespace(
                use_cache=True, cache_exists=lambda *_args: False
            ),
            "run_step=tts_to_end requires cached translation artifacts from a previous full dubbing run in the same project directory, but no cache entry was found for step 'translation'.",
        ),
        (
            SimpleNamespace(
                use_cache=True,
                cache_exists=lambda *_args: True,
                load_from_cache=lambda *_args: None,
            ),
            "run_step=tts_to_end found step 'translation' but could not load cached translation artifacts. Re-run the full pipeline to rebuild them.",
        ),
    ],
)
def test_load_required_cached_step_preserves_exact_errors(cache, message):
    artifacts = _artifacts_module()
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = cache

    with pytest.raises(FileNotFoundError) as exc_info:
        artifacts.load_required_cached_step(
            dubber,
            step_name="translation",
            cache_key="cache-key",
            hint="translation",
        )

    assert str(exc_info.value) == message


def test_load_required_cached_step_returns_exact_cached_object():
    artifacts = _artifacts_module()
    cached = [{"translation": "Bonjour"}]
    calls = []
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = SimpleNamespace(
        use_cache=True,
        cache_exists=lambda *args: calls.append(("exists", args)) or True,
        load_from_cache=lambda *args: calls.append(("load", args)) or cached,
    )

    result = artifacts.load_required_cached_step(
        dubber,
        step_name="translation",
        cache_key="cache-key",
        hint="translation",
    )

    assert result is cached
    assert calls == [
        ("exists", ("translation", "cache-key")),
        ("load", ("translation", "cache-key")),
    ]


def test_build_speaker_rolls_preserves_exact_filtering_and_coercion():
    artifacts = _artifacts_module()
    segments = [
        {"start": "1.25", "end": 2, "speaker": 7},
        {"start": None, "end": 3, "speaker": "missing-start"},
        {"start": 3, "end": None, "speaker": "missing-end"},
        {"start": 4, "end": 5, "speaker": None},
    ]

    assert artifacts.build_speaker_rolls_from_segments(
        SmartDubbing.__new__(SmartDubbing), segments
    ) == {
        (1.25, 2.0): "7"
    }
    assert segments[0] == {"start": "1.25", "end": 2, "speaker": 7}


def test_save_requested_subtitles_uses_live_facade_helpers_and_exact_paths():
    artifacts = _artifacts_module()
    segments = [{"start": 1.0, "end": 2.0}]
    adjusted = [{"start": 0.5, "end": 1.5}]
    calls = []
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "remove_pauses": True,
        "input": "movie.mp4",
        "source_language": "en",
        "target_language": "fr",
    }
    dubber.adjust_subtitle_timestamps = (
        lambda value, pauses: calls.append(("adjust", value, pauses)) or adjusted
    )
    dubber._get_subtitle_path = (
        lambda kind, path, language: calls.append(
            ("path", kind, path, language)
        )
        or f"C:/{kind}-{language}.srt"
    )
    dubber.subtitle_manager = SimpleNamespace(
        save_subtitles=lambda *args: calls.append(("save", args))
    )
    pauses = [{"time_removed": 0.5}]

    artifacts.save_requested_subtitles(
        dubber,
        segments,
        save_original_subtitles=True,
        save_translated_subtitles=True,
        pause_adjustments=pauses,
    )

    assert calls == [
        ("adjust", segments, pauses),
        ("path", "original", "movie.mp4", "en"),
        ("save", (adjusted, "original", "C:/original-en.srt")),
        ("path", "original", "movie.mp4", "en"),
        ("path", "translation", "movie.mp4", "fr"),
        ("save", (adjusted, "translation", "C:/translation-fr.srt")),
        ("path", "translation", "movie.mp4", "fr"),
    ]
    assert segments == [{"start": 1.0, "end": 2.0}]


def test_save_requested_subtitles_without_pause_removal_keeps_segment_identity():
    artifacts = _artifacts_module()
    segments = [{"text": "hello"}]
    saved = []
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "remove_pauses": False,
        "input": "movie.mp4",
        "source_language": "en",
        "target_language": "fr",
    }
    dubber._get_subtitle_path = (
        lambda kind, _path, language: f"{kind}-{language}.srt"
    )
    dubber.subtitle_manager = SimpleNamespace(
        save_subtitles=lambda *args: saved.append(args)
    )

    artifacts.save_requested_subtitles(
        dubber,
        segments,
        save_original_subtitles=False,
        save_translated_subtitles=True,
        pause_adjustments=[{"ignored": True}],
    )

    assert saved == [(segments, "translation", "translation-fr.srt")]
    assert saved[0][0] is segments


def test_combine_final_video_forwards_exact_options_and_computed_ranges():
    artifacts = _artifacts_module()
    received = []
    result = ("dubbed.mp4", [{"time_removed": 0.5}])
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "input": "movie.mp4",
        "include_original_audio": True,
        "output": "output.mp4",
        "source_language": "en",
        "target_language": "fr",
    }
    dubber.muted_speakers = {"MUTED"}
    dubber.video_processor = SimpleNamespace(
        combine_audio_with_video=lambda **kwargs: received.append(kwargs) or result
    )
    rolls = {(0.0, 1.0): "KEPT", (1.0, 2.0): "MUTED"}

    actual = artifacts.combine_final_video(
        dubber,
        translated_audio_path="translated.wav",
        background_audio_path="background.wav",
        speakers_rolls=rolls,
    )

    assert actual is result
    assert received == [
        {
            "video_path": "movie.mp4",
            "translated_audio_path": "translated.wav",
            "background_audio_path": "background.wav",
            "watermark_path": None,
            "watermark_text": None,
            "include_original_audio": True,
            "output_file": "output.mp4",
            "start_time": None,
            "duration": None,
            "keep_original_audio_ranges": [(0.0, 1.0)],
            "source_language": "en",
            "target_language": "fr",
            "normalize_audio": True,
            "use_two_pass_encoding": True,
            "remove_pauses": False,
            "min_pause_duration": 300,
            "preserve_pause_duration": 1.5,
            "keyframe_buffer": 0.2,
            "ffmpeg_batch_size": 50,
            "dubbed_volume": 1.0,
            "background_volume": 0.562341,
            "upscale_factor": 1.0,
            "upscale_sharpen": True,
        }
    ]


def test_reset_input_cache_preserves_exact_deletion_scope(tmp_path):
    artifacts = _artifacts_module()
    audio_chunks = tmp_path / "audio_chunks"
    su_chunks = tmp_path / "su_chunks"
    audio_chunks.mkdir()
    su_chunks.mkdir()
    (audio_chunks / "0.wav").write_bytes(b"wav")
    (audio_chunks / "keep.mp3").write_bytes(b"mp3")
    (su_chunks / "1.wav").write_bytes(b"wav")
    cache_root = tmp_path / "cache"
    removable = cache_root / "translation"
    untouched = cache_root / "unrelated"
    removable.mkdir(parents=True)
    untouched.mkdir()
    (removable / "value.pkl").write_bytes(b"cache")
    (untouched / "value.pkl").write_bytes(b"cache")
    cleared = []
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"input": "movie.mp4"}
    dubber.audio_chunks_dir = audio_chunks
    dubber.su_audio_chunks_dir = su_chunks
    dubber.cache_manager = SimpleNamespace(
        cache_root=cache_root,
        clear_input_cache=lambda path: cleared.append(path),
    )

    artifacts.reset_input_cache(dubber, "test reset")

    assert cleared == ["movie.mp4"]
    assert not (audio_chunks / "0.wav").exists()
    assert (audio_chunks / "keep.mp3").exists()
    assert not (su_chunks / "1.wav").exists()
    assert not removable.exists()
    assert untouched.exists()


def test_save_transcription_file_preserves_exact_utf8_format(tmp_path):
    artifacts = _artifacts_module()
    transcription_path = tmp_path / "nested" / "transcription.txt"
    segments = [
        {
            "speaker": "SPEAKER_00",
            "start": 0.096,
            "end": 11.853,
            "text": "Precise timing — café",
        }
    ]
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"transcription_path": str(transcription_path)}

    artifacts.save_transcription_file(dubber, segments)

    assert transcription_path.read_text(encoding="utf-8") == (
        "[00.00.00.096-00.00.11.853] SPEAKER_00: Precise timing — café\n"
    )
    assert segments[0]["start"] == 0.096


def test_get_subtitle_path_preserves_exact_project_layout_and_disambiguation(
    tmp_path,
):
    artifacts = _artifacts_module()
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "project_dir": str(tmp_path / "project"),
        "source_language": "en",
        "target_language": "fr",
    }

    assert artifacts.get_subtitle_path(
        dubber, "original", r"C:\media\Film.mp4", "en"
    ) == str(tmp_path / "project" / "Film_en.srt")

    dubber.config["target_language"] = "en"
    assert artifacts.get_subtitle_path(
        dubber, "original", r"C:\media\Film.mp4", "en"
    ) == str(tmp_path / "project" / "source_Film_en.srt")
    assert artifacts.get_subtitle_path(
        dubber, "translation", r"C:\media\Film.mp4", "en"
    ) == str(tmp_path / "project" / "target_Film_en.srt")


def test_adjust_subtitle_timestamps_returns_copies_with_exact_values():
    artifacts = _artifacts_module()
    segments = [
        {"start": 1.5, "end": 2.5, "text": "during"},
        {"start": 4.0, "end": 5.0, "text": "after"},
    ]
    original = [segment.copy() for segment in segments]
    adjustments = [
        {
            "original_start": 1.0,
            "original_end": 3.0,
            "time_removed": 1.0,
            "cumulative_offset": 1.0,
        }
    ]

    adjusted = artifacts.adjust_subtitle_timestamps(
        SmartDubbing.__new__(SmartDubbing), segments, adjustments
    )

    assert adjusted == [
        {"start": 1.5, "end": 2.0, "text": "during"},
        {"start": 3.0, "end": 4.0, "text": "after"},
    ]
    assert segments == original
    assert adjusted is not segments
    assert adjusted[0] is not segments[0]


def test_adjust_subtitle_timestamps_without_adjustments_preserves_identity():
    artifacts = _artifacts_module()
    segments = [{"start": 1.0, "end": 2.0}]

    assert (
        artifacts.adjust_subtitle_timestamps(
            SmartDubbing.__new__(SmartDubbing), segments, []
        )
        is segments
    )


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


def test_dubbing_snapshot_uses_active_context_when_facade_mirror_is_stale():
    saved = []
    segments = [{}]

    class Cache:
        def save_to_cache(self, step, key, payload):
            saved.append((step, key, payload))

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = Cache()
    dubber._pipeline_run_context = PipelineRunContext(
        semantic_plan_cache_persistable=False
    )
    dubber._semantic_plan_cache_persistable = True
    dubber._build_dubbing_text_snapshot_key = lambda _audio: "snapshot-key"
    dubber._build_translation_cache_key = lambda _audio: pytest.fail(
        "active context forbids a reusable translation cache key"
    )

    translation.persist_dubbing_text_snapshot(dubber, segments, "audio.wav")

    assert saved == [
        (
            "dubbing_texts",
            "snapshot-key",
            {
                "version": 1,
                "segments": segments,
                "translation_cache_reusable": False,
                "translation_cache_key": None,
            },
        )
    ]


def test_synthesis_persistence_uses_active_context_when_facade_mirror_is_stale():
    events = []
    segments = [{}]

    class Cache:
        def save_to_cache(self, step, key, payload):
            events.append(("save", step, key, payload))

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = Cache()
    dubber._pipeline_run_context = PipelineRunContext(
        semantic_plan_cache_persistable=False
    )
    dubber._semantic_plan_cache_persistable = True
    dubber._persist_dubbing_text_snapshot = (
        lambda value, audio: events.append(("snapshot", value, audio))
    )
    dubber._build_translation_cache_key = lambda _audio: pytest.fail(
        "active context forbids reusable translation persistence"
    )

    translation.persist_synthesis_results(dubber, segments, "audio.wav")

    assert events == [("snapshot", segments, "audio.wav")]


def test_fresh_translation_cache_uses_active_context_when_facade_mirror_is_stale():
    events = []
    segments = [
        {
            "semantic_unit_id": "unit-1",
            "semantic_plan_fingerprint": "plan-live",
        }
    ]

    class Cache:
        def cache_exists(self, *_args):
            return False

        def save_to_cache(self, step, key, payload):
            events.append(("save", step, key, payload))

    class Translator:
        def is_available(self):
            return True

        def translate(self, **_kwargs):
            return segments

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {}
    dubber.cache_manager = Cache()
    dubber.performance_tracker = SimpleNamespace(
        start_timing=lambda *_args: None,
        end_timing=lambda *_args: 1.0,
    )
    dubber.debug_data = {}
    dubber._pipeline_run_context = PipelineRunContext(
        semantic_plan_cache_persistable=False
    )
    dubber._semantic_plan_cache_persistable = True
    dubber._build_translation_cache_key = lambda _audio: "translation-key"
    dubber._require_translator = lambda: Translator()
    dubber._persist_dubbing_text_snapshot = (
        lambda value, audio: events.append(("snapshot", value, audio))
    )

    assert translation.translate_segments(dubber, segments, "audio.wav") is segments
    assert events == [("snapshot", segments, "audio.wav")]


@pytest.mark.parametrize(
    "translated_segments",
    [
        [{"semantic_unit_id": "unit-1"}],
        [
            {
                "semantic_unit_id": "unit-1",
                "semantic_plan_fingerprint": "plan-stale",
            }
        ],
    ],
    ids=["missing", "mismatched"],
)
def test_fresh_translation_validates_plan_fingerprint_before_persistence(
    translated_segments,
):
    persisted = []
    transcription = [
        {
            "semantic_unit_id": "unit-1",
            "semantic_plan_fingerprint": "plan-active",
        }
    ]

    class Cache:
        def cache_exists(self, *_args):
            return False

        def save_to_cache(self, *args):
            persisted.append(("cache", args))

    class Translator:
        def is_available(self):
            return True

        def translate(self, **_kwargs):
            return translated_segments

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {}
    dubber.cache_manager = Cache()
    dubber.performance_tracker = SimpleNamespace(
        start_timing=lambda *_args: None,
        end_timing=lambda *_args: 1.0,
    )
    dubber.debug_data = {}
    dubber._pipeline_run_context = PipelineRunContext(
        semantic_plan_fingerprint="plan-active"
    )
    dubber._build_translation_cache_key = lambda _audio: "translation-key"
    dubber._require_translator = lambda: Translator()
    dubber._persist_dubbing_text_snapshot = (
        lambda *args: persisted.append(("snapshot", args))
    )

    with pytest.raises(
        ValueError,
        match=(
            "Cached artifact semantic_plan_fingerprint is absent or does not "
            "match the active semantic plan"
        ),
    ):
        translation.translate_segments(dubber, transcription, "audio.wav")

    assert persisted == []


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
