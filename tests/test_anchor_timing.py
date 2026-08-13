import json
import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from dubbing.core.voice_profiles import VoiceProfile


def _segment(start, end, *, speaker="SPEAKER_00", audio_file=None):
    return {
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": "source",
        "translation": "target",
        "synthesized_speech_file": str(audio_file) if audio_file else None,
    }


def test_segment_timing_uses_recognized_bounds_not_next_anchor():
    from dubbing.core.timing import TimingPolicy, calculate_segment_timing

    result = calculate_segment_timing(
        start=2.0,
        end=2.5,
        next_start=3.1,
        source_duration=10.0,
        audio_duration=0.8,
        policy=TimingPolicy(),
    )

    assert result.available_window == pytest.approx(0.5)
    assert result.tempo == pytest.approx(0.8 / 0.75)
    assert result.expected_duration == pytest.approx(0.75)
    assert result.residual_overflow == pytest.approx(0.25)
    assert result.within_policy is True


def test_synthesis_measures_normal_then_long_and_selects_closer_wav(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    source_path = tmp_path / "source.wav"
    AudioSegment.silent(duration=4000).export(source_path, format="wav")
    output_path = tmp_path / "output.wav"

    class PerformanceStub:
        def start_timing(self, *_args):
            pass

        def end_timing(self, *_args):
            pass

    class CacheStub:
        use_cache = False

        def generate_cache_key(self, *_args):
            return "base"

        def cache_exists(self, *_args):
            return False

        def get_cache_path(self, step):
            path = tmp_path / "cache" / step
            path.mkdir(parents=True, exist_ok=True)
            return path

    observed = {}

    class TTSStub:
        def estimate_audio_segment_length(self, segment_data, language):
            pytest.fail("Provider duration estimates must not select candidates")

        def synthesize(self, segments_data, language):
            assert len(segments_data) == 1
            observed.setdefault("synthesized_targets", []).append(segments_data[0].target_duration)
            observed.setdefault("synthesized_texts", []).append(segments_data[0].text)
            observed.setdefault("segment_indexes", []).append(segments_data[0].segment_index)
            for item in segments_data:
                duration = 450 if item.text == "long target" else 300
                Sine(440).to_audio_segment(duration=duration).export(item.output_path, format="wav")
            return []

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "source_language": "en",
        "target_language": "be",
        "whisper_model": "large-v3",
        "tts_system": "fake",
        "translated_audio_path": str(output_path),
        "segment_reference_min_duration": 0,
        "timing_short_segment_threshold": 1.5,
        "timing_short_segment_max_speed": 1.08,
        "timing_max_speed": 1.15,
        "timing_max_overflow": 0.25,
    }
    dubber.performance_tracker = PerformanceStub()
    dubber.cache_manager = CacheStub()
    dubber.audio_chunks_dir = tmp_path / "chunks"
    dubber.su_audio_chunks_dir = tmp_path / "timed"
    dubber.audio_chunks_dir.mkdir()
    dubber.su_audio_chunks_dir.mkdir()
    dubber.speakers_audio_dir = tmp_path / "speakers"
    dubber.speakers_audio_dir.mkdir()
    dubber.debug_data = {"voices": {}}
    profile = SimpleNamespace(
        style_prompt="",
        voice_name=None,
        reference_audio=None,
        reference_text=None,
        reference_mode=None,
        tts_system="fake",
        model=None,
        params={},
    )
    dubber._resolve_voice_profile = lambda _speaker: profile
    dubber._profile_pool_key = lambda _profile: ("fake",)
    dubber._default_tts_system = lambda: "fake"
    dubber.tts_clients = {("fake",): TTSStub()}
    dubber.default_tts = dubber.tts_clients[("fake",)]
    dubber._semantic_plan_cache_persistable = False
    dubber._adjust_and_combine_audio_grouped = lambda _segments: (
        AudioSegment.silent(duration=4000),
        [
            {
                "start": 2.0,
                "end": 2.8,
                "original_start": 2.0,
                "original_end": 2.5,
            }
        ],
    )
    first = _segment(2.0, 2.5)
    first["long_translation"] = "long target"
    segments = [first]

    dubber.synthesize_speech(segments, {}, str(source_path))

    assert observed["synthesized_targets"] == [pytest.approx(0.5), pytest.approx(0.5)]
    assert observed["synthesized_texts"] == ["target", "long target"]
    assert observed["segment_indexes"] == [0, 0]
    assert segments[0]["synthesized_text"] == "long target"
    assert segments[0]["selected_variant"] == "long_translation"


def _run_candidate_state_machine(tmp_path, segment, durations):
    from dubbing.core.smart_dubbing import SmartDubbing
    from dubbing.core.timing import TimingPolicy

    dubber = SmartDubbing.__new__(SmartDubbing)
    tmp_path.mkdir(parents=True, exist_ok=True)
    dubber.audio_chunks_dir = tmp_path
    calls = []

    def load_candidate(_metadata, *, variant, text, attempts):
        calls.append((variant, text, attempts))
        duration = durations.get(variant)
        if duration is None:
            return None
        path = tmp_path / f"{variant}.wav"
        path.write_bytes(variant.encode("utf-8"))
        return {
            "variant": variant,
            "text": text,
            "path": str(path),
            "duration": duration,
        }

    dubber._load_or_synthesize_candidate = load_candidate
    dubber._synthesize_measured_candidates(
        {
            "index": 0,
            "segment": segment,
            "final_path": str(tmp_path / "0.wav"),
        },
        TimingPolicy(),
    )
    return calls


def test_long_candidate_branch_generates_short_then_conditional_very_short(tmp_path):
    segment = _segment(0.0, 0.5)
    segment.update(
        {
            "_timing_available_window": 0.5,
            "short_translation": "short",
            "very_short_translation": "very short",
        }
    )

    calls = _run_candidate_state_machine(
        tmp_path,
        segment,
        {"translation": 1.2, "short_translation": 0.9, "very_short_translation": 0.6},
    )

    assert [call[0] for call in calls] == [
        "translation",
        "short_translation",
        "very_short_translation",
    ]
    assert segment["selected_variant"] == "very_short_translation"


def test_short_within_overflow_stops_before_very_short(tmp_path):
    segment = _segment(0.0, 0.5)
    segment.update(
        {
            "_timing_available_window": 0.5,
            "short_translation": "short",
            "very_short_translation": "very short",
        }
    )

    calls = _run_candidate_state_machine(
        tmp_path,
        segment,
        {"translation": 1.2, "short_translation": 0.7, "very_short_translation": 0.4},
    )

    assert [call[0] for call in calls] == ["translation", "short_translation"]
    assert segment["selected_variant"] == "short_translation"


def test_candidate_tie_and_duplicate_text_preserve_generation_order(tmp_path):
    segment = _segment(0.0, 0.5)
    segment.update(
        {
            "_timing_available_window": 0.5,
            "long_translation": "long",
            "short_translation": "target",
        }
    )

    calls = _run_candidate_state_machine(
        tmp_path,
        segment,
        {"translation": 0.3, "long_translation": 0.7},
    )

    assert [call[0] for call in calls] == ["translation", "long_translation"]
    assert segment["selected_variant"] == "translation"


def test_blank_duplicate_and_failed_optional_candidates_keep_normal(tmp_path):
    segment = _segment(0.0, 0.5)
    segment.update(
        {
            "_timing_available_window": 0.5,
            "long_translation": "   ",
            "short_translation": "target",
        }
    )
    blank_calls = _run_candidate_state_machine(
        tmp_path / "blank",
        segment,
        {"translation": 0.3},
    )

    failed_segment = _segment(0.0, 0.5)
    failed_segment.update(
        {
            "_timing_available_window": 0.5,
            "long_translation": "long",
        }
    )
    failed_calls = _run_candidate_state_machine(
        tmp_path / "failed",
        failed_segment,
        {"translation": 0.3, "long_translation": None},
    )

    assert [call[0] for call in blank_calls] == ["translation"]
    assert segment["selected_variant"] == "translation"
    assert [call[0] for call in failed_calls] == [
        "translation",
        "long_translation",
    ]
    assert failed_segment["selected_variant"] == "translation"


def test_failed_initial_candidate_removes_stale_chunk(tmp_path):
    segment = _segment(0.0, 0.5)
    segment["_timing_available_window"] = 0.5
    stale_path = tmp_path / "0.wav"
    stale_path.write_bytes(b"stale")

    calls = _run_candidate_state_machine(tmp_path, segment, {"translation": None})

    assert [call[0] for call in calls] == ["translation"]
    assert segment["synthesized_speech_file"] is None
    assert stale_path.exists() is False


@pytest.mark.parametrize(
    ("normal_duration", "expected_calls"),
    [(0.25, ["translation"]), (0.251, ["translation", "short_translation"])],
)
def test_zero_duration_target_respects_overflow_boundary(
    tmp_path, normal_duration, expected_calls
):
    segment = _segment(0.0, 0.0)
    segment.update(
        {
            "_timing_available_window": 0.0,
            "short_translation": "short",
        }
    )

    calls = _run_candidate_state_machine(
        tmp_path,
        segment,
        {"translation": normal_duration, "short_translation": 0.1},
    )

    assert [call[0] for call in calls] == expected_calls


@pytest.mark.parametrize(
    ("end", "expected_tempo"),
    [(1.0, 1.08), (2.0, 1.15)],
)
def test_overflowing_clip_uses_duration_class_speed_cap(end, expected_tempo):
    from dubbing.core.timing import TimingPolicy, calculate_segment_timing

    result = calculate_segment_timing(
        start=0.0,
        end=end,
        next_start=end,
        source_duration=10.0,
        audio_duration=4.0,
        policy=TimingPolicy(),
    )

    assert result.tempo == expected_tempo
    assert result.residual_overflow > 0.25
    assert result.within_policy is False


def test_exact_allowed_overflow_boundary_needs_no_tempo():
    from dubbing.core.timing import TimingPolicy, calculate_segment_timing

    result = calculate_segment_timing(
        start=0.0,
        end=0.5,
        next_start=1.0,
        source_duration=5.0,
        audio_duration=0.75,
        policy=TimingPolicy(),
    )

    assert result.tempo == 1.0
    assert result.residual_overflow == pytest.approx(0.25)
    assert result.within_policy is True


def test_assembly_applies_tempo_only_after_segment_local_selection(tmp_path):
    dubber = _dubber_for_assembly(tmp_path, source_duration=2.0)
    first_path = tmp_path / "first.wav"
    second_path = tmp_path / "second.wav"
    Sine(440).to_audio_segment(duration=800).export(first_path, format="wav")
    Sine(550).to_audio_segment(duration=400).export(second_path, format="wav")

    _, positions = dubber._adjust_and_combine_audio_grouped(
        [
            _segment(0.0, 0.5, audio_file=first_path),
            _segment(1.1, 1.5, audio_file=second_path),
        ]
    )

    assert positions[0]["tempo"] == pytest.approx(0.8 / 0.75)
    assert positions[0]["end"] == pytest.approx(0.75, abs=0.03)


def test_anchor_plan_sorts_and_keeps_equal_start_bounds_segment_local():
    from dubbing.core.timing import plan_anchor_windows

    later = _segment(4.0, 4.5)
    equal_b = _segment(1.0, 2.0, speaker="SPEAKER_01")
    equal_a = _segment(1.0, 1.5)

    planned = plan_anchor_windows([later, equal_b, equal_a], source_duration=6.0)

    assert [item.segment for item in planned] == [equal_a, equal_b, later]
    assert [item.original_index for item in planned] == [2, 1, 0]
    assert planned[0].available_window == 0.5
    assert planned[1].available_window == 1.0
    assert planned[2].available_window == 0.5


def test_replanning_preserves_stable_original_indexes():
    from dubbing.core.timing import plan_anchor_windows

    segments = [_segment(3.0, 3.5), _segment(1.0, 1.5), _segment(2.0, 2.5)]
    first_plan = plan_anchor_windows(segments, source_duration=5.0)
    for item in first_plan:
        item.segment["_timing_original_index"] = item.original_index
    chronological = [item.segment for item in first_plan]

    second_plan = plan_anchor_windows(chronological, source_duration=5.0)

    assert [item.original_index for item in second_plan] == [1, 2, 0]


@pytest.mark.parametrize(
    ("start", "end"),
    [
        (-0.1, 1.0),
        (2.0, 1.0),
        (math.nan, 1.0),
        (0.0, math.nan),
        (math.inf, math.inf),
    ],
)
def test_anchor_plan_rejects_invalid_timestamps_atomically(start, end):
    from dubbing.core.timing import plan_anchor_windows

    valid = _segment(0.0, 0.5)
    invalid = _segment(start, end)

    with pytest.raises(ValueError, match="timestamp"):
        plan_anchor_windows([valid, invalid], source_duration=3.0)

    assert "_timing_original_index" not in valid
    assert "_timing_original_index" not in invalid


def test_zero_length_range_has_zero_target_duration():
    from dubbing.core.timing import plan_anchor_windows

    planned = plan_anchor_windows(
        [_segment(1.0, 1.0), _segment(2.0, 2.4)],
        source_duration=4.0,
    )

    assert planned[0].available_window == 0.0


def test_last_segment_does_not_inherit_remaining_source_duration():
    from dubbing.core.timing import plan_anchor_windows

    planned = plan_anchor_windows([_segment(1.0, 1.4)], source_duration=9.0)

    assert planned[0].available_window == pytest.approx(0.4)


def test_edge_trimming_keeps_head_and_tail_safety_margins(tmp_path):
    from dubbing.core.timing import trim_audio_edges

    raw_path = tmp_path / "raw.wav"
    trimmed_path = tmp_path / "timed.wav"
    raw = (
        AudioSegment.silent(duration=300)
        + Sine(440).to_audio_segment(duration=500).apply_gain(-6)
        + AudioSegment.silent(duration=400)
    )
    raw.export(raw_path, format="wav")

    result = trim_audio_edges(raw_path, trimmed_path)

    assert result.raw_duration == pytest.approx(1.2, abs=0.01)
    assert result.leading_removed == pytest.approx(0.25, abs=0.02)
    assert result.trailing_removed == pytest.approx(0.30, abs=0.02)
    assert result.trimmed_duration == pytest.approx(0.65, abs=0.03)
    assert trimmed_path.exists()
    assert raw_path.stat().st_size > 0


def test_all_silent_clip_is_reported_unusable(tmp_path):
    from dubbing.core.timing import trim_audio_edges

    raw_path = tmp_path / "silent.wav"
    output_path = tmp_path / "timed.wav"
    AudioSegment.silent(duration=800).export(raw_path, format="wav")
    Sine(440).to_audio_segment(duration=200).export(output_path, format="wav")

    result = trim_audio_edges(raw_path, output_path)

    assert result.usable is False
    assert output_path.exists() is False


def test_edge_trim_io_failure_keeps_readable_raw_clip(tmp_path, monkeypatch):
    import dubbing.core.timing as timing_module

    dubber = _dubber_for_assembly(tmp_path, source_duration=2.0, debug=False)
    raw_path = tmp_path / "raw.wav"
    Sine(440).to_audio_segment(duration=1200).export(raw_path, format="wav")
    segment = _segment(0.0, 0.5, audio_file=raw_path)
    segment["_tts_cache_contract"] = "anchor_raw_v1"
    monkeypatch.setattr(
        timing_module,
        "trim_audio_edges",
        lambda *_args, **_kwargs: timing_module.EdgeTrimResult(
            raw_duration=1.2,
            trimmed_duration=1.2,
            leading_removed=0.0,
            trailing_removed=0.0,
            usable=False,
            error="timing output is locked",
        ),
    )

    combined, positions = dubber._adjust_and_combine_audio_grouped([segment])

    assert combined.dBFS != float("-inf")
    assert positions[0]["end"] > 1.0
    assert dubber.debug_data["timing_alignment"][0]["trim_error"] == "timing output is locked"


def _dubber_for_assembly(tmp_path, *, source_duration=3.0, debug=True):
    from dubbing.core.smart_dubbing import SmartDubbing

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "timing_short_segment_threshold": 1.5,
        "timing_short_segment_max_speed": 1.08,
        "timing_max_speed": 1.15,
        "timing_max_overflow": 0.25,
        "debug_info": debug,
        "debug_dir": str(tmp_path / "debug"),
    }
    dubber.audio_chunks_dir = tmp_path / "audio_chunks"
    dubber.su_audio_chunks_dir = tmp_path / "su_audio_chunks"
    dubber.audio_chunks_dir.mkdir()
    dubber.su_audio_chunks_dir.mkdir()
    dubber.debug_data = {}
    dubber._timing_source_duration = source_duration
    return dubber


def test_assembly_keeps_recognized_starts_overlap_and_exact_source_duration(tmp_path):
    dubber = _dubber_for_assembly(tmp_path, source_duration=3.0)
    first = tmp_path / "first.wav"
    second = tmp_path / "second.wav"
    third = tmp_path / "third.wav"
    Sine(330).to_audio_segment(duration=800).export(first, format="wav")
    Sine(440).to_audio_segment(duration=700).export(second, format="wav")
    Sine(550).to_audio_segment(duration=400).export(third, format="wav")
    segments = [
        _segment(0.2, 0.7, audio_file=first),
        _segment(1.3, 2.0, speaker="SPEAKER_01", audio_file=second),
        _segment(1.5, 1.9, speaker="SPEAKER_00", audio_file=third),
    ]

    combined, positions = dubber._adjust_and_combine_audio_grouped(segments)

    assert len(combined) == 3000
    assert [position["start"] for position in positions] == [0.2, 1.3, 1.5]
    assert positions[1]["end"] > positions[2]["start"]
    assert (tmp_path / "debug" / "timing_alignment.tsv").exists()


def test_missing_unsorted_segment_does_not_reuse_another_rows_chunk(tmp_path):
    dubber = _dubber_for_assembly(tmp_path, source_duration=2.0)
    first_chunk = dubber.audio_chunks_dir / "0.wav"
    Sine(440).to_audio_segment(duration=300).export(first_chunk, format="wav")
    later_missing = _segment(1.0, 1.5)
    earlier = _segment(0.0, 0.5, audio_file=first_chunk)

    combined, _positions = dubber._adjust_and_combine_audio_grouped(
        [later_missing, earlier]
    )

    assert combined[1000:1300].dBFS == float("-inf")


def test_ffmpeg_failure_uses_unmodified_duration_in_diagnostics(tmp_path, monkeypatch):
    import dubbing.core.smart_dubbing as smart_dubbing_module

    dubber = _dubber_for_assembly(tmp_path, source_duration=3.0)
    clip_path = tmp_path / "long.wav"
    Sine(440).to_audio_segment(duration=2200).export(clip_path, format="wav")
    segments = [_segment(0.0, 1.0, audio_file=clip_path), _segment(1.0, 1.2)]

    class FailedProcess:
        returncode = 1
        stderr = b"tempo failed"

    monkeypatch.setattr(smart_dubbing_module.subprocess, "run", lambda *args, **kwargs: FailedProcess())

    _, positions = dubber._adjust_and_combine_audio_grouped(segments)

    assert positions[0]["tempo"] == 1.0
    assert positions[0]["end"] == pytest.approx(2.2, abs=0.01)
    assert positions[0]["within_policy"] is False


def test_source_boundary_truncation_is_explicit(tmp_path):
    dubber = _dubber_for_assembly(tmp_path, source_duration=1.0)
    clip_path = tmp_path / "crosses-boundary.wav"
    Sine(440).to_audio_segment(duration=1200).export(clip_path, format="wav")

    combined, positions = dubber._adjust_and_combine_audio_grouped(
        [_segment(0.5, 0.8, audio_file=clip_path)]
    )

    assert len(combined) == 1000
    assert positions[0]["boundary_truncated_ms"] > 0
    assert positions[0]["end"] == 1.0


def test_timing_policy_config_normalization():
    from dubbing.core.config import DubbingConfig

    config = DubbingConfig()
    config.config.update(
        {
            "timing_short_segment_threshold": -1,
            "timing_short_segment_max_speed": math.nan,
            "timing_max_speed": 0.9,
            "timing_max_stretch": 0.9,
            "timing_max_overflow": math.inf,
        }
    )
    config.process_special_parameters()

    assert config.get("timing_short_segment_threshold") == 1.5
    assert config.get("timing_short_segment_max_speed") == 1.08
    assert config.get("timing_max_speed") == 1.15
    assert config.get("timing_max_stretch") == 1.15
    assert config.get("timing_max_overflow") == 0.25

def test_final_cache_fingerprint_includes_policy_and_algorithm_version():
    from dubbing.core.timing import TimingPolicy, timing_cache_fingerprint

    baseline = timing_cache_fingerprint(TimingPolicy())

    assert "segment_bounded_timing_v3" in baseline
    assert baseline != timing_cache_fingerprint(TimingPolicy(max_overflow=0.1))
    assert baseline != timing_cache_fingerprint(TimingPolicy(max_speed=1.2))
    assert baseline != timing_cache_fingerprint(TimingPolicy(max_stretch=1.1))
    assert baseline != timing_cache_fingerprint(TimingPolicy(short_segment_max_speed=1.04))
    assert baseline != timing_cache_fingerprint(TimingPolicy(short_segment_threshold=1.0))


def test_aggregate_selection_fingerprint_includes_bounds_and_all_candidates():
    from dubbing.core.smart_dubbing import SmartDubbing

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {}
    baseline_segment = _segment(0.0, 1.0)
    baseline = dubber._tts_selection_cache_fingerprint([baseline_segment])

    changed_end = dict(baseline_segment, end=1.1)
    with_long = dict(baseline_segment, long_translation="long target")
    with_short = dict(baseline_segment, short_translation="short target")
    with_very_short = dict(
        baseline_segment, very_short_translation="very short target"
    )

    assert baseline != dubber._tts_selection_cache_fingerprint([changed_end])
    assert baseline != dubber._tts_selection_cache_fingerprint([with_long])
    assert baseline != dubber._tts_selection_cache_fingerprint([with_short])
    assert baseline != dubber._tts_selection_cache_fingerprint([with_very_short])

    reassigned = dict(baseline_segment, speaker="SPEAKER_01")
    emotional = dict(baseline_segment, emotion="Happy")
    styled = dict(baseline_segment, style_prompt="Speak softly")
    assert baseline != dubber._tts_selection_cache_fingerprint([reassigned])
    assert baseline != dubber._tts_selection_cache_fingerprint([emotional])
    assert baseline != dubber._tts_selection_cache_fingerprint([styled])

    dubber.config["tts_prompt_prefix"] = "Keep the voice stable"
    assert baseline != dubber._tts_selection_cache_fingerprint([baseline_segment])


def test_aggregate_tts_fingerprint_includes_reference_file_contents(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"first voice")
    profile = SimpleNamespace(
        tts_system="fake",
        model=None,
        voice_name=None,
        style_prompt="",
        reference_mode="configured",
        reference_audio=str(reference_path),
        reference_text="sample",
        params={},
    )
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"tts_system": "fake"}
    dubber._resolve_voice_profile = lambda _speaker: profile
    dubber._default_tts_system = lambda: "fake"

    baseline = dubber._effective_tts_cache_fingerprint(["SPEAKER_00"])
    reference_path.write_bytes(b"second voice")

    assert baseline != dubber._effective_tts_cache_fingerprint(["SPEAKER_00"])


def test_synthesis_hits_policy_specific_wav_cache_without_pickle_marker(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing
    from dubbing.core.timing import TimingPolicy, timing_cache_fingerprint

    source_path = tmp_path / "source.wav"
    output_path = tmp_path / "output.wav"
    AudioSegment.silent(duration=2000).export(source_path, format="wav")

    class PerformanceStub:
        def start_timing(self, *_args):
            pass

        def end_timing(self, *_args):
            pass

    class CacheStub:
        use_cache = True

        def generate_cache_key(self, *_args):
            return "base"

        def cache_exists(self, *_args):
            return False

        def get_cache_path(self, step):
            path = tmp_path / "cache" / step
            path.mkdir(parents=True, exist_ok=True)
            return path

    cache = CacheStub()
    fingerprint = timing_cache_fingerprint(TimingPolicy())
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "source_language": "en",
        "target_language": "be",
        "whisper_model": "large-v3",
        "tts_system": "fake",
        "translated_audio_path": str(output_path),
        "timing_short_segment_threshold": 1.5,
        "timing_short_segment_max_speed": 1.08,
        "timing_max_speed": 1.15,
        "timing_max_overflow": 0.25,
    }
    dubber.performance_tracker = PerformanceStub()
    dubber.cache_manager = cache
    dubber.voice_profiles = {
        "SPEAKER_00": VoiceProfile(tts_system="fake"),
    }
    dubber.tts_clients = {}

    tts_fingerprint = dubber._effective_tts_cache_fingerprint(["SPEAKER_00"])
    segment = _segment(0.0, 1.0)
    selection_fingerprint = dubber._tts_selection_cache_fingerprint([segment])
    cached_path = (
        cache.get_cache_path("synthesized_speech")
        / f"base_be_fake_{fingerprint}_{tts_fingerprint}_{selection_fingerprint}.wav"
    )
    Sine(440).to_audio_segment(duration=2000).export(cached_path, format="wav")

    result = dubber.synthesize_speech([segment], {}, str(source_path))

    assert result == str(output_path)
    assert output_path.exists()
    assert len(AudioSegment.from_file(output_path)) == 2000


def test_raw_segment_cache_sidecar_distinguishes_new_and_legacy_audio(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    source_path = tmp_path / "raw.wav"
    cache_path = tmp_path / "cache" / "segment.wav"
    cache_path.parent.mkdir()
    Sine(440).to_audio_segment(duration=300).export(source_path, format="wav")
    dubber = SmartDubbing.__new__(SmartDubbing)

    assert dubber._cached_segment_contract(cache_path) == "legacy"

    dubber._cache_raw_tts_segment(
        str(source_path), cache_path, synthesized_text="long target"
    )

    assert dubber._cached_segment_contract(cache_path) == "anchor_raw_v2"
    metadata = json.loads(
        dubber._segment_cache_metadata_path(cache_path).read_text(encoding="utf-8")
    )
    assert metadata["synthesized_text"] == "long target"
    assert cache_path.exists()


def test_raw_candidate_cache_reuse_skips_provider_call(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    class CacheStub:
        use_cache = True

    class TTSStub:
        def synthesize(self, *_args, **_kwargs):
            pytest.fail("Cached raw candidate must skip the provider")

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = CacheStub()
    dubber._plan_dependent_cache_allowed = True
    dubber.config = {"target_language": "be"}
    dubber.audio_chunks_dir = tmp_path / "chunks"
    dubber.su_audio_chunks_dir = tmp_path / "timed"
    dubber.audio_chunks_dir.mkdir()
    dubber.su_audio_chunks_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    segment = _segment(0.0, 0.5)
    segment["_timing_original_index"] = 0
    segment["_timing_available_window"] = 0.5
    base_args = {
        "speaker": "SPEAKER_00",
        "text": "target",
        "segment_index": 0,
        "target_duration": 0.5,
    }
    metadata = {
        "index": 0,
        "segment": segment,
        "tts_instance": TTSStub(),
        "tts_system": "fake",
        "base_args": base_args,
        "base_cache_prefix": "base",
        "segment_cache_dir": cache_dir,
        "pool_key": ("fake",),
        "style_prompt": "",
    }
    key = dubber._raw_tts_segment_cache_key(
        base_cache_prefix="base",
        tts_system="fake",
        segment=segment,
        speaker="SPEAKER_00",
        translation="target",
        style_prompt="",
        reference_audio_path=None,
        reference_mode=None,
        reference_text=None,
        client_pool_settings=("fake",),
        legacy_index=0,
    )
    cached_path = cache_dir / f"{key}.wav"
    Sine(440).to_audio_segment(duration=400).export(cached_path, format="wav")

    result = dubber._load_or_synthesize_candidate(
        metadata,
        variant="translation",
        text="target",
        attempts=3,
    )

    assert result is not None
    assert result["duration"] == pytest.approx(0.4, abs=0.02)


def test_raw_candidate_cache_uses_active_context_not_stale_facade_mirror(tmp_path):
    from dubbing.core.pipeline.context import PipelineRunContext
    from dubbing.core.smart_dubbing import SmartDubbing

    class CacheStub:
        use_cache = True

    class TTSStub:
        calls = 0

        def synthesize(self, segments_data, **_kwargs):
            self.calls += 1
            Sine(440).to_audio_segment(duration=200).export(
                segments_data[0].output_path, format="wav"
            )

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.cache_manager = CacheStub()
    dubber._plan_dependent_cache_allowed = True
    dubber._pipeline_run_context = PipelineRunContext(
        plan_dependent_cache_allowed=False
    )
    dubber.config = {"target_language": "be"}
    dubber.audio_chunks_dir = tmp_path / "chunks"
    dubber.su_audio_chunks_dir = tmp_path / "timed"
    dubber.audio_chunks_dir.mkdir()
    dubber.su_audio_chunks_dir.mkdir()
    cache_dir = tmp_path / "cache"
    cache_dir.mkdir()
    segment = _segment(0.0, 0.5)
    segment["_timing_original_index"] = 0
    segment["_timing_available_window"] = 0.5
    tts = TTSStub()
    metadata = {
        "index": 0,
        "segment": segment,
        "tts_instance": tts,
        "tts_system": "fake",
        "base_args": {
            "speaker": "SPEAKER_00",
            "text": "target",
            "segment_index": 0,
            "target_duration": 0.5,
        },
        "base_cache_prefix": "base",
        "segment_cache_dir": cache_dir,
        "pool_key": ("fake",),
        "style_prompt": "",
    }
    key = dubber._raw_tts_segment_cache_key(
        base_cache_prefix="base",
        tts_system="fake",
        segment=segment,
        speaker="SPEAKER_00",
        translation="target",
        style_prompt="",
        reference_audio_path=None,
        client_pool_settings=("fake",),
    )
    cached_path = cache_dir / f"{key}.wav"
    Sine(440).to_audio_segment(duration=400).export(cached_path, format="wav")

    result = dubber._load_or_synthesize_candidate(
        metadata, variant="translation", text="target", attempts=1
    )

    assert tts.calls == 1
    assert result["duration"] == pytest.approx(0.2, abs=0.02)
    assert len(AudioSegment.from_file(cached_path)) == 400


def test_raw_candidate_key_includes_prompts_emotion_and_reference_contents(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    reference_path = tmp_path / "reference.wav"
    reference_path.write_bytes(b"first reference")
    common = {
        "base_cache_prefix": "base",
        "tts_system": "fake",
        "segment": {"_timing_available_window": 0.5},
        "speaker": "SPEAKER_00",
        "translation": "target",
        "style_prompt": "",
        "reference_audio_path": str(reference_path),
    }
    baseline = SmartDubbing._raw_tts_segment_cache_key(**common)

    assert baseline != SmartDubbing._raw_tts_segment_cache_key(
        **common, emotion="Happy"
    )
    assert baseline != SmartDubbing._raw_tts_segment_cache_key(
        **common, tts_prompt_prefix="Preserve identity"
    )
    assert baseline != SmartDubbing._raw_tts_segment_cache_key(
        **common, voice_prompt={"SPEAKER_00": "Speak softly"}
    )

    reference_path.write_bytes(b"second reference")
    assert baseline != SmartDubbing._raw_tts_segment_cache_key(**common)


def test_single_row_resynthesis_generates_only_selected_text(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    calls = []

    class TTSStub:
        reference_capability = "unsupported"

        def synthesize(self, segments_data, language):
            calls.extend(item.text for item in segments_data)
            for item in segments_data:
                Sine(440).to_audio_segment(duration=300).export(
                    item.output_path, format="wav"
                )

    profile = SimpleNamespace(
        style_prompt="",
        voice_name=None,
        reference_audio=None,
        reference_text=None,
        reference_mode=None,
        tts_system="fake",
        model=None,
        params={},
    )
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"target_language": "be", "segment_reference_min_duration": 0}
    dubber.audio_chunks_dir = tmp_path / "chunks"
    dubber.su_audio_chunks_dir = tmp_path / "timed"
    dubber.speakers_audio_dir = tmp_path / "speakers"
    dubber.su_audio_chunks_dir.mkdir()
    dubber.speakers_audio_dir.mkdir()
    dubber._resolve_voice_profile = lambda _speaker: profile
    dubber._profile_pool_key = lambda _profile: ("fake",)
    dubber._default_tts_system = lambda: "fake"
    dubber.tts_clients = {("fake",): TTSStub()}
    dubber.default_tts = dubber.tts_clients[("fake",)]
    segment = _segment(0.0, 0.5)
    segment.update(
        {
            "long_translation": "long",
            "short_translation": "short",
            "very_short_translation": "very short",
        }
    )

    updated = dubber.resynthesize_one_segment(
        [segment], 0, override_text="manual override"
    )

    assert calls == ["manual override"]
    assert updated["synthesized_text"] == "manual override"


def test_gradio_exposes_and_persists_anchor_timing_settings(tmp_path):
    from dubbing.ui.gradio_app import build_app, save_settings

    app = build_app(config_path=str(tmp_path / "missing.yml"))
    components = {
        component.get("props", {}).get("label"): component.get("props", {})
        for component in app.config["components"]
    }

    assert components["Short segment threshold"]["value"] == 1.5
    assert components["Short segment max speed"]["value"] == 1.08
    assert components["Maximum timing speed"]["value"] == 1.15
    assert components["Maximum timing stretch"]["value"] == 1.15
    assert components["Maximum timing overflow"]["value"] == 0.25
    assert "Group overflow tolerance" not in components

    config_path = tmp_path / "saved.yml"
    save_settings(
        {
            "timing_short_segment_threshold": 1.2,
            "timing_short_segment_max_speed": 1.04,
            "timing_max_speed": 1.12,
            "timing_max_stretch": 1.10,
            "timing_max_overflow": 0.15,
            "group_overflow_tolerance": 0.5,
        },
        config_path=str(config_path),
    )
    import yaml

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved == {
        "timing_short_segment_threshold": 1.2,
        "timing_short_segment_max_speed": 1.04,
        "timing_max_speed": 1.12,
        "timing_max_stretch": 1.10,
        "timing_max_overflow": 0.15,
        "semantic_split_enabled": True,
        "tts_preferred_segment_duration": 15.0,
        "tts_hard_segment_duration": 35.0,
        "semantic_split_search_window": 10.0,
    }


def test_gradio_save_normalizes_invalid_timing_values(tmp_path):
    from dubbing.ui.gradio_app import save_settings
    import yaml

    config_path = tmp_path / "saved.yml"
    save_settings(
        {
            "timing_short_segment_threshold": -1,
            "timing_short_segment_max_speed": math.nan,
            "timing_max_speed": 0.8,
            "timing_max_stretch": 0.8,
            "timing_max_overflow": math.inf,
        },
        config_path=str(config_path),
    )

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["timing_short_segment_threshold"] == 1.5
    assert saved["timing_short_segment_max_speed"] == 1.08
    assert saved["timing_max_speed"] == 1.15
    assert saved["timing_max_stretch"] == 1.15
    assert saved["timing_max_overflow"] == 0.25


def test_transcription_artifact_preserves_millisecond_timestamps(tmp_path):
    from dubbing.core.smart_dubbing import SmartDubbing

    transcription_path = tmp_path / "transcription.txt"
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"transcription_path": str(transcription_path)}

    dubber._save_transcription_file(
        [
            {
                "speaker": "SPEAKER_00",
                "start": 0.096,
                "end": 11.853,
                "text": "Precise timing",
            }
        ]
    )

    assert transcription_path.read_text(encoding="utf-8") == (
        "[00.00.00.096-00.00.11.853] SPEAKER_00: Precise timing\n"
    )
