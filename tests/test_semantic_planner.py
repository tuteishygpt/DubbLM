import math
import time
import json
from pathlib import Path

import pytest


def _word(text, start, end, confidence=0.9):
    return {
        "word": text,
        "start": start,
        "end": end,
        "confidence": confidence,
    }


def _segment(text, start, end, words=None, speaker="SPEAKER_00"):
    result = {"text": text, "start": start, "end": end, "speaker": speaker}
    if words is not None:
        result["words"] = words
    return result


def test_semantic_config_normalizes_all_entry_point_values():
    from dubbing.core.timing import normalize_timing_config

    config = {
        "semantic_split_enabled": "false",
        "tts_preferred_segment_duration": math.nan,
        "tts_hard_segment_duration": 10,
        "semantic_split_search_window": -1,
    }
    warnings = []

    normalize_timing_config(config, warn=warnings.append)

    assert config["semantic_split_enabled"] is False
    assert config["tts_preferred_segment_duration"] == 15.0
    assert config["tts_hard_segment_duration"] == 35.0
    assert config["semantic_split_search_window"] == 10.0
    assert len(warnings) == 3


def test_semantic_config_cli_and_gradio_surface_all_fields(tmp_path):
    from dubbing.core.config import create_argument_parser
    from dubbing.ui.gradio_app import build_app, save_settings
    import yaml

    args = create_argument_parser().parse_args(
        [
            "--input", "clip.mp4",
            "--source_language", "en",
            "--target_language", "be",
            "--semantic_split_enabled", "false",
            "--tts_preferred_segment_duration", "12.5",
            "--tts_hard_segment_duration", "30",
            "--semantic_split_search_window", "7.5",
        ]
    )
    assert args.semantic_split_enabled is False
    assert args.tts_preferred_segment_duration == 12.5
    assert args.tts_hard_segment_duration == 30.0
    assert args.semantic_split_search_window == 7.5

    app = build_app(config_path=str(tmp_path / "missing.yml"))
    components = {
        component.get("props", {}).get("label"): component.get("props", {})
        for component in app.config["components"]
    }
    assert components["Semantic splitting"]["value"] is True
    assert components["Preferred TTS segment duration"]["value"] == 15.0
    assert components["Hard TTS segment duration"]["value"] == 35.0
    assert components["Semantic split search window"]["value"] == 10.0

    config_path = tmp_path / "settings.yml"
    save_settings(
        {
            "semantic_split_enabled": False,
            "tts_preferred_segment_duration": 12.5,
            "tts_hard_segment_duration": 30.0,
            "semantic_split_search_window": 7.5,
        },
        config_path=str(config_path),
    )
    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["semantic_split_enabled"] is False
    assert saved["tts_preferred_segment_duration"] == 12.5
    assert saved["tts_hard_segment_duration"] == 30.0
    assert saved["semantic_split_search_window"] == 7.5


def test_sentence_boundary_is_considered_below_old_gap_threshold():
    from dubbing.audio.semantic_planner import SemanticPlannerConfig, plan_semantic_segments

    segments = [
        _segment(
            "This is complete. Next thought starts",
            0.0,
            20.0,
            [
                _word("This", 0.0, 4.0),
                _word("is", 4.1, 8.0),
                _word("complete.", 8.1, 15.0),
                _word("Next", 15.05, 17.0),
                _word("thought", 17.1, 19.0),
                _word("starts", 19.1, 20.0),
            ],
        )
    ]

    result = plan_semantic_segments(
        segments,
        vad_regions=[(0.0, 20.0)],
        speaker="SPEAKER_00",
        source_language="en",
        config=SemanticPlannerConfig(preferred_duration=15.0, hard_duration=35.0, search_window=10.0),
    )

    assert [unit["text"] for unit in result.units] == [
        "This is complete.",
        "Next thought starts",
    ]
    chosen = [candidate for candidate in result.diagnostics if candidate["chosen"]]
    assert chosen[0]["source_pause"] == pytest.approx(0.05)
    assert chosen[0]["reason_code"] == "sentence_final"


def test_incomplete_tail_is_hard_continue_even_when_classifier_says_cut():
    from dubbing.audio.semantic_planner import SemanticPlannerConfig, plan_semantic_segments

    words = [
        _word("I", 0.0, 3.0),
        _word("think", 3.1, 7.0),
        _word("that", 7.1, 10.0),
        _word("this", 11.0, 15.0),
        _word("works.", 15.1, 18.0),
    ]

    def classifier(request):
        return {
            "boundaries": [
                {"id": item["id"], "decision": "CUT", "confidence": 0.99}
                for item in request["candidates"]
            ]
        }

    result = plan_semantic_segments(
        [_segment("I think that this works.", 0.0, 18.0, words)],
        vad_regions=[(0.0, 10.0), (11.0, 18.0)],
        speaker="SPEAKER_00",
        source_language="en",
        config=SemanticPlannerConfig(preferred_duration=10.0, hard_duration=35.0, search_window=2.0),
        classifier=classifier,
        classifier_status="ready",
    )

    boundary = next(item for item in result.diagnostics if item["left_context"].endswith("I think that"))
    assert boundary["local_decision"] == "HARD_CONTINUE"
    assert boundary["final_decision"] == "CONTINUE"
    assert boundary["chosen"] is False
    assert " ".join(unit["text"] for unit in result.units) == "I think that this works."
    assert all(not unit["text"].endswith("I think that") for unit in result.units[:-1])


def test_forced_cut_prefers_safe_boundary_and_marks_shared_continuation():
    from dubbing.audio.semantic_planner import SemanticPlannerConfig, plan_semantic_segments

    words = [
        _word("Opening", 0.0, 8.0),
        _word("because", 8.1, 16.0),
        _word("middle", 16.1, 24.0),
        _word("safe", 24.1, 32.0),
        _word("ending", 32.1, 40.0),
    ]
    result = plan_semantic_segments(
        [_segment("Opening because middle safe ending", 0.0, 40.0, words)],
        vad_regions=[(0.0, 40.0)],
        speaker="SPEAKER_00",
        source_language="en",
        config=SemanticPlannerConfig(preferred_duration=15.0, hard_duration=35.0, search_window=0.0),
    )

    assert all(unit["end"] - unit["start"] <= 35.0 for unit in result.units)
    assert len(result.units) == 2
    assert result.units[0]["continuation_id"] == result.units[1]["continuation_id"]
    assert result.units[1]["boundary_before"]["type"] == "technical_continuation"


def test_forced_cut_treats_high_confidence_llm_continue_as_a_veto():
    from dubbing.audio.semantic_planner import SemanticPlannerConfig, plan_semantic_segments

    words = [
        _word("alpha", 0.0, 10.0),
        _word("vetoed", 10.1, 20.0),
        _word("safe", 25.0, 30.0),
        _word("ending", 30.1, 45.0),
    ]

    def classifier(request):
        return {
            "boundaries": [
                {
                    "id": item["id"],
                    "decision": "CONTINUE" if item["left"].endswith("vetoed") else "UNCERTAIN",
                    "confidence": 0.95,
                }
                for item in request["candidates"]
            ]
        }

    result = plan_semantic_segments(
        [_segment("alpha vetoed safe ending", 0.0, 45.0, words)],
        vad_regions=[(0.0, 45.0)],
        speaker="SPEAKER_00",
        source_language="en",
        config=SemanticPlannerConfig(
            preferred_duration=15.0,
            hard_duration=35.0,
            search_window=0.0,
        ),
        classifier=classifier,
        classifier_status="ready",
    )

    chosen = next(item for item in result.diagnostics if item["chosen"])
    vetoed = next(item for item in result.diagnostics if item["left_context"].endswith("vetoed"))
    assert vetoed["llm_decision"] == "CONTINUE"
    assert vetoed["chosen"] is False
    assert chosen["candidate_time"] == pytest.approx(30.0)


def test_planner_is_deterministic_under_input_permutation_and_deduplicates_words():
    from dubbing.audio.semantic_planner import plan_semantic_segments

    duplicate_low = _segment("Hello world.", 0.0, 2.0, [_word("Hello", 0.0, 0.8, 0.3), _word("world.", 1.0, 2.0)])
    duplicate_high = _segment("Hello world.", 0.0, 2.0, [_word("Hello", 0.0, 0.8, 0.9), _word("world.", 1.0, 2.0)])
    later = _segment("Next.", 3.0, 4.0, [_word("Next.", 3.0, 4.0)])

    first = plan_semantic_segments(
        [duplicate_low, later, duplicate_high],
        vad_regions=[(0.0, 2.0), (3.0, 4.0)],
        speaker="SPEAKER_00",
        source_language="en",
    )
    second = plan_semantic_segments(
        [later, duplicate_high, duplicate_low],
        vad_regions=[(3.0, 4.0), (0.0, 2.0)],
        speaker="SPEAKER_00",
        source_language="en",
    )

    assert first.fingerprint == second.fingerprint
    assert " ".join(unit["text"] for unit in first.units) == "Hello world. Next."
    assert sum(len(unit["words"]) for unit in first.units) == 3


@pytest.mark.parametrize(
    "bad",
    [
        {"start": -1.0, "end": 1.0, "text": "bad"},
        {"start": 2.0, "end": 1.0, "text": "bad"},
        {"start": math.nan, "end": 1.0, "text": "bad"},
        {"start": 0.0, "end": math.inf, "text": "bad"},
    ],
)
def test_semantic_planner_rejects_invalid_timestamps(bad):
    from dubbing.audio.semantic_planner import plan_semantic_segments

    with pytest.raises(ValueError, match="SPEAKER_00.*segment 0"):
        plan_semantic_segments([bad], vad_regions=[(0.0, 2.0)], speaker="SPEAKER_00")


def test_indivisible_wordless_segment_over_hard_limit_is_actionable():
    from dubbing.audio.semantic_planner import SemanticSegmentationError, plan_semantic_segments

    with pytest.raises(SemanticSegmentationError, match="word-timestamp backend"):
        plan_semantic_segments(
            [_segment("one indivisible segment", 0.0, 40.0)],
            vad_regions=[(0.0, 40.0)],
            speaker="SPEAKER_00",
        )


def test_translation_optimizer_respects_locked_boundary_and_metadata():
    from translation.llm_translator import LLMTranslator

    translator = LLMTranslator.__new__(LLMTranslator)
    translator_segments = [
        {
            "start": 0.0,
            "end": 1.0,
            "speaker": "SPEAKER_00",
            "text": "First",
            "semantic_unit_id": "unit-a",
            "vad_region_ids": ["SPEAKER_00:v000000"],
        },
        {
            "start": 1.1,
            "end": 2.0,
            "speaker": "SPEAKER_00",
            "text": "Second",
            "semantic_unit_id": "unit-b",
            "vad_region_ids": ["SPEAKER_00:v000001"],
            "lock_boundary_before": True,
        },
    ]

    optimized = translator._optimize_segments(translator_segments, max_gap_seconds=1.0)

    assert [item["semantic_unit_id"] for item in optimized] == ["unit-a", "unit-b"]
    assert optimized[1]["vad_region_ids"] == ["SPEAKER_00:v000001"]


def test_isolated_track_path_uses_semantic_planner_by_default(tmp_path, monkeypatch):
    import dubbing.audio.isolated_tracks as isolated

    audio_path = tmp_path / "speaker.wav"
    audio_path.write_bytes(b"placeholder")
    words = [
        _word("I", 0.0, 2.0),
        _word("think", 2.1, 5.0),
        _word("that", 5.1, 8.0),
        _word("the", 9.0, 11.0),
        _word("answer", 11.1, 14.0),
        _word("works.", 14.1, 17.0),
    ]

    class Transcriber:
        def diarize_and_transcribe(self, **_kwargs):
            return {}, [_segment("I think that the answer works.", 0.0, 17.0, words)]

    monkeypatch.setattr(isolated, "_run_vad", lambda *_args, **_kwargs: [(0.0, 8.0), (9.0, 17.0)])
    monkeypatch.setattr(isolated, "_build_inner_transcriber", lambda **_kwargs: Transcriber())

    diagnostics = []
    _, transcription = isolated.run_isolated_tracks(
        {"SPEAKER_00": str(audio_path)},
        "deepgram",
        "en",
        semantic_split_enabled=True,
        tts_preferred_segment_duration=8.0,
        tts_hard_segment_duration=35.0,
        semantic_split_search_window=2.0,
        semantic_debug_path=str(tmp_path / "semantic_boundaries.jsonl"),
        semantic_diagnostics_out=diagnostics,
    )

    assert transcription
    assert all("semantic_unit_id" in segment for segment in transcription)
    assert all(not segment["text"].endswith("I think that") for segment in transcription[:-1])
    assert transcription[0]["semantic_plan_fingerprint"]
    assert diagnostics
    assert (tmp_path / "semantic_boundaries.jsonl").exists()


def test_isolated_track_legacy_mode_uses_preferred_duration(tmp_path, monkeypatch):
    import dubbing.audio.isolated_tracks as isolated

    audio_path = tmp_path / "speaker.wav"
    audio_path.write_bytes(b"placeholder")
    observed = {}

    class Transcriber:
        def diarize_and_transcribe(self, **_kwargs):
            return {}, [_segment("hello", 0.0, 1.0, [_word("hello", 0.0, 1.0)])]

    monkeypatch.setattr(isolated, "_run_vad", lambda *_args, **_kwargs: [(0.0, 1.0)])
    monkeypatch.setattr(isolated, "_build_inner_transcriber", lambda **_kwargs: Transcriber())
    monkeypatch.setattr(
        isolated,
        "_merge_close_segments",
        lambda segments, max_duration: observed.setdefault("merge", max_duration) and segments,
    )
    monkeypatch.setattr(
        isolated,
        "_split_long_segments",
        lambda segments, max_duration: observed.setdefault("split", max_duration) and segments,
    )

    _, transcription = isolated.run_isolated_tracks(
        {"SPEAKER_00": str(audio_path)},
        "deepgram",
        "en",
        semantic_split_enabled=False,
        tts_preferred_segment_duration=12.0,
    )

    assert observed == {"merge": 12.0, "split": 12.0}
    assert "semantic_unit_id" not in transcription[0]


def test_llm_translator_semantic_classifier_uses_existing_primary_llm():
    from translation.llm_translator import LLMTranslator

    observed = {}

    class LLM:
        def complete(self, prompt):
            observed["prompt"] = prompt
            return '{"boundaries":[{"id":"abc","decision":"CUT","confidence":0.9,"reason_code":"complete"}]}'

    translator = LLMTranslator.__new__(LLMTranslator)
    translator.llm = LLM()

    result = translator.classify_semantic_boundaries(
        {
            "source_language": "en",
            "candidates": [{"id": "abc", "left": "Done.", "right": "Next", "pause": 0.1}],
        }
    )

    assert result["boundaries"][0]["id"] == "abc"
    assert "semantic_boundary_prompt_v1" in observed["prompt"]
    assert "Do not rewrite" in observed["prompt"]


@pytest.mark.parametrize("classifier_status", ["deterministic-only", "ready"])
def test_matthew_fixture_is_lossless_and_preserves_expected_boundaries(classifier_status):
    from dubbing.audio.semantic_planner import plan_semantic_segments

    fixture = json.loads(
        (Path(__file__).parent / "fixtures" / "semantic_boundaries_matthew.json").read_text(
            encoding="utf-8"
        )
    )

    def classifier(request):
        return {
            "boundaries": [
                {
                    "id": candidate["id"],
                    "decision": "CUT",
                    "confidence": 0.9,
                    "reason_code": "complete_thought",
                }
                for candidate in request["candidates"]
            ]
        }

    all_units = []
    all_diagnostics = []
    for speaker in ("SPEAKER_01", "SPEAKER_00"):
        segments = [item for item in fixture["segments"] if item["speaker"] == speaker]
        regions = [item for item in fixture["vad_regions"] if item["speaker"] == speaker]
        result = plan_semantic_segments(
            segments,
            vad_regions=regions,
            speaker=speaker,
            source_language="en",
            classifier=classifier if classifier_status == "ready" else None,
            classifier_status=classifier_status,
        )
        all_units.extend(result.units)
        all_diagnostics.extend(result.diagnostics)

        if classifier_status == "deterministic-only":
            expected_units = fixture["expected_units"][speaker]
            actual_units = [
                {key: unit[key] for key in expected_units[index]}
                for index, unit in enumerate(result.units)
            ]
            assert actual_units == expected_units

        expected_words = " ".join(
            word["word"] for segment in segments for word in segment["words"]
        )
        assert " ".join(unit["text"] for unit in result.units) == expected_words
        assert all(unit["end"] - unit["start"] <= 35.0 for unit in result.units)
        source_word_times = {
            round(value, 6)
            for segment in segments
            for word in segment["words"]
            for value in (word["start"], word["end"])
        }
        assert all(round(unit["start"], 6) in source_word_times for unit in result.units)
        assert all(round(unit["end"], 6) in source_word_times for unit in result.units)

    by_row = {item["source_row"]: item for item in fixture["segments"]}
    for relation, expected in fixture["expected_source_boundary_decisions"].items():
        left_row = int(relation.split("->")[0])
        boundary = next(
            item
            for item in all_diagnostics
            if item["speaker"] == by_row[left_row]["speaker"]
            and item["candidate_time"] == pytest.approx(by_row[left_row]["end"])
        )
        assert boundary["final_decision"] == expected

    assert all(
        not unit["text"].endswith("I think that")
        for unit in all_units[:-1]
    )
    expected_source_ids = {item["source_segment_id"] for item in fixture["segments"]}
    actual_source_ids = [value for unit in all_units for value in unit["source_segment_ids"]]
    expected_vad_ids = {item["vad_region_id"] for item in fixture["vad_regions"]}
    actual_vad_ids = [value for unit in all_units for value in unit["vad_region_ids"]]
    assert len(actual_source_ids) == len(set(actual_source_ids))
    assert set(actual_source_ids) == expected_source_ids
    assert len(actual_vad_ids) == len(set(actual_vad_ids))
    assert set(actual_vad_ids) == expected_vad_ids


def test_isolated_semantic_cache_key_covers_every_planner_and_classifier_setting(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    track = tmp_path / "speaker.wav"
    track.write_bytes(b"audio")

    class Cache:
        def generate_cache_key(self, *_args):
            return "raw"

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = Cache()
    dubber.config = {
        "source_language": "en",
        "target_language": "be",
        "semantic_split_enabled": True,
        "tts_preferred_segment_duration": 15.0,
        "tts_hard_segment_duration": 35.0,
        "semantic_split_search_window": 10.0,
        "llm_provider": "gemini",
        "llm_model_name": "model-a",
        "llm_temperature": 0.5,
        "llm_max_tokens": 1000,
    }
    tracks = {"SPEAKER_00": str(track)}
    baseline = dubber._isolated_tracks_cache_key("source.wav", tracks)
    raw_baseline = dubber._isolated_tracks_raw_cache_key(tracks)

    for key, changed in (
        ("tts_preferred_segment_duration", 14.0),
        ("tts_hard_segment_duration", 34.0),
        ("semantic_split_search_window", 9.0),
        ("source_language", "de"),
        ("llm_provider", "openrouter"),
        ("llm_model_name", "model-b"),
        ("llm_temperature", 0.2),
        ("llm_max_tokens", 2000),
    ):
        original = dubber.config[key]
        dubber.config[key] = changed
        assert dubber._isolated_tracks_cache_key("source.wav", tracks) != baseline
        if key in {
            "tts_preferred_segment_duration",
            "tts_hard_segment_duration",
            "semantic_split_search_window",
        }:
            assert dubber._isolated_tracks_raw_cache_key(tracks) == raw_baseline
        dubber.config[key] = original


def test_isolated_raw_cache_key_tracks_audio_content_not_only_file_metadata(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    track = tmp_path / "speaker.wav"
    track.write_bytes(b"first")

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "source_language": "en",
        "inner_transcription_system": "whisper_timestamped",
        "whisper_model": "large-v3",
    }
    tracks = {"SPEAKER_00": str(track)}

    first_key = dubber._isolated_tracks_raw_cache_key(tracks)
    track.write_bytes(b"other")

    assert dubber._isolated_tracks_raw_cache_key(tracks) != first_key


def test_semantic_plan_cache_separates_llm_and_deterministic_classifier_modes(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    track = tmp_path / "speaker.wav"
    track.write_bytes(b"audio")

    class Cache:
        def generate_cache_key(self, *_args):
            return "raw"

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = Cache()
    dubber.config = {
        "source_language": "en",
        "target_language": "be",
        "semantic_split_enabled": True,
        "tts_preferred_segment_duration": 15.0,
        "tts_hard_segment_duration": 35.0,
        "semantic_split_search_window": 10.0,
        "translator_type": "llm",
        "llm_provider": "gemini",
        "llm_model_name": "model-a",
    }
    tracks = {"SPEAKER_00": str(track)}

    llm_key = dubber._isolated_tracks_cache_key("source.wav", tracks)
    dubber.config["translator_type"] = "identity"
    deterministic_key = dubber._isolated_tracks_cache_key("source.wav", tracks)
    dubber.config["llm_provider"] = "openrouter"
    dubber.config["llm_model_name"] = "unused-model"

    assert deterministic_key != llm_key
    assert dubber._isolated_tracks_cache_key("source.wav", tracks) == deterministic_key


def test_translation_cache_key_includes_semantic_plan_fingerprint():
    from dubbing.core.smart_dubbing import SmartDubbing

    class Cache:
        def generate_cache_key(self, *_args):
            return "base"

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = Cache()
    dubber.config = {
        "source_language": "en",
        "target_language": "be",
        "translation_prompt_prefix": "",
    }
    without_plan = dubber._build_translation_cache_key("source.wav")
    dubber._semantic_plan_fingerprint = "plan-a"
    with_plan_a = dubber._build_translation_cache_key("source.wav")
    dubber._semantic_plan_fingerprint = "plan-b"
    with_plan_b = dubber._build_translation_cache_key("source.wav")

    assert without_plan != with_plan_a != with_plan_b
    assert "plan-a" in with_plan_a


def test_raw_tts_cache_key_includes_semantic_plan_fingerprint():
    from dubbing.core.smart_dubbing import SmartDubbing

    common = {
        "base_cache_prefix": "base",
        "tts_system": "fake",
        "speaker": "SPEAKER_00",
        "translation": "target",
        "style_prompt": "",
        "reference_audio_path": "",
    }

    first = SmartDubbing._raw_tts_segment_cache_key(
        segment={"semantic_unit_id": "unit-a", "semantic_plan_fingerprint": "plan-a"},
        **common,
    )
    second = SmartDubbing._raw_tts_segment_cache_key(
        segment={"semantic_unit_id": "unit-a", "semantic_plan_fingerprint": "plan-b"},
        **common,
    )

    assert first != second


@pytest.mark.parametrize(
    "translated",
    [
        [{"semantic_unit_id": "a", "speaker": "S", "text": "one"}],
        [
            {"semantic_unit_id": "a", "speaker": "S", "text": "one"},
            {"semantic_unit_id": "a", "speaker": "S", "text": "two"},
        ],
        [
            {"semantic_unit_id": "a", "speaker": "S", "text": "one"},
            {"semantic_unit_id": "unknown", "speaker": "S", "text": "two"},
        ],
    ],
)
def test_translation_semantic_ids_must_be_one_to_one(translated):
    from translation.llm_translator import LLMTranslator

    original = [
        {"semantic_unit_id": "a", "speaker": "S", "text": "first"},
        {"semantic_unit_id": "b", "speaker": "S", "text": "second"},
    ]
    with pytest.raises(ValueError, match="semantic_unit_id"):
        LLMTranslator._validate_semantic_translation_pairs(original, translated)


def test_translation_semantic_ids_are_reordered_to_stable_source_order():
    from translation.llm_translator import LLMTranslator

    original = [
        {"semantic_unit_id": "a", "speaker": "S", "text": "first"},
        {"semantic_unit_id": "b", "speaker": "S", "text": "second"},
    ]
    translated = [
        {"semantic_unit_id": "b", "speaker": "S", "text": "two"},
        {"semantic_unit_id": "a", "speaker": "S", "text": "one"},
    ]

    ordered = LLMTranslator._validate_semantic_translation_pairs(original, translated)

    assert [item["semantic_unit_id"] for item in ordered] == ["a", "b"]


def test_successful_boundary_classification_is_cached_by_context():
    from dubbing.audio.semantic_planner import plan_semantic_segments

    cache = {}
    calls = []
    segments = [
        _segment(
            "Alpha middle omega",
            0.0,
            20.0,
            [
                _word("Alpha", 0.0, 6.0),
                _word("middle", 6.1, 14.0),
                _word("omega", 14.1, 20.0),
            ],
        )
    ]

    def classifier(request):
        calls.append(request)
        return {
            "boundaries": [
                {"id": item["id"], "decision": "CONTINUE", "confidence": 0.9}
                for item in request["candidates"]
            ]
        }

    kwargs = {
        "vad_regions": [(0.0, 20.0)],
        "speaker": "SPEAKER_00",
        "source_language": "en",
        "classifier_status": "ready",
        "classification_cache_get": cache.get,
        "classification_cache_set": cache.__setitem__,
        "classifier_cache_context": {"provider": "gemini", "model": "a", "temperature": 0.2},
    }
    first = plan_semantic_segments(segments, classifier=classifier, **kwargs)
    second = plan_semantic_segments(
        segments,
        classifier=lambda _request: pytest.fail("cached classifications should be reused"),
        **kwargs,
    )

    assert calls
    assert cache
    assert first.fingerprint == second.fingerprint
    assert second.cache_persistable is True


def test_boundary_classifier_timeout_falls_back_without_blocking_the_plan():
    from dubbing.audio.semantic_planner import SemanticPlannerConfig, plan_semantic_segments

    segments = [
        _segment(
            "Alpha middle omega",
            0.0,
            20.0,
            [
                _word("Alpha", 0.0, 6.0),
                _word("middle", 6.1, 14.0),
                _word("omega", 14.1, 20.0),
            ],
        )
    ]

    def stalled_classifier(_request):
        time.sleep(0.25)
        return {"boundaries": []}

    started = time.monotonic()
    result = plan_semantic_segments(
        segments,
        vad_regions=[(0.0, 20.0)],
        speaker="SPEAKER_00",
        source_language="en",
        config=SemanticPlannerConfig(classifier_timeout=0.01),
        classifier=stalled_classifier,
        classifier_status="ready",
    )

    assert time.monotonic() - started < 0.20
    assert result.cache_persistable is False
    assert any(
        item["fallback_reason"] == "classifier_error:TimeoutError"
        for item in result.diagnostics
    )


def test_oversized_classifier_candidate_is_not_sent_past_character_limit():
    from dubbing.audio.semantic_planner import SemanticPlannerConfig, plan_semantic_segments

    huge = "x" * 500
    calls = []
    result = plan_semantic_segments(
        [
            _segment(
                f"{huge} tail",
                0.0,
                20.0,
                [_word(huge, 0.0, 10.0), _word("tail", 10.1, 20.0)],
            )
        ],
        vad_regions=[(0.0, 20.0)],
        speaker="SPEAKER_00",
        source_language="en",
        config=SemanticPlannerConfig(classifier_batch_characters=200),
        classifier=lambda request: calls.append(request) or {"boundaries": []},
        classifier_status="ready",
    )

    assert calls == []
    assert result.cache_persistable is False
    assert result.diagnostics[0]["fallback_reason"] == "classifier_request_too_large"


def test_translate_segments_accepts_one_consistent_semantic_fingerprint():
    from dubbing.core.smart_dubbing import SmartDubbing

    segments = [
        {
            "semantic_unit_id": "unit-a",
            "semantic_plan_fingerprint": "plan-a",
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 1.0,
            "text": "hello",
            "translation": "вітаю",
        }
    ]

    class Cache:
        def generate_cache_key(self, *_args):
            return "base"

        def cache_exists(self, *_args):
            return True

        def load_from_cache(self, *_args):
            return segments

    class Performance:
        def record_metric(self, *_args):
            pass

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"source_language": "en", "target_language": "be"}
    dubber.cache_manager = Cache()
    dubber.performance_tracker = Performance()
    dubber.debug_data = {}

    assert dubber.translate_segments(segments, "source.wav") == segments
    assert dubber._semantic_plan_fingerprint == "plan-a"


def test_plan_dependent_cache_rejects_a_partially_missing_fingerprint():
    from dubbing.core.smart_dubbing import SmartDubbing

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber._semantic_plan_fingerprint = "plan-a"

    with pytest.raises(ValueError, match="absent or does not match"):
        dubber._validate_plan_dependent_segments(
            [
                {"semantic_unit_id": "a", "semantic_plan_fingerprint": "plan-a"},
                {"semantic_unit_id": "b"},
            ]
        )


def test_translate_segments_rejects_a_partially_missing_semantic_fingerprint():
    from dubbing.core.smart_dubbing import SmartDubbing

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"source_language": "en", "target_language": "be"}

    with pytest.raises(ValueError, match="missing or inconsistent"):
        dubber.translate_segments(
            [
                {"semantic_unit_id": "a", "semantic_plan_fingerprint": "plan-a"},
                {"semantic_unit_id": "b"},
            ],
            "source.wav",
        )
