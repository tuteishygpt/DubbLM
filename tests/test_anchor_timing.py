import math
from pathlib import Path
from types import SimpleNamespace

import pytest
from pydub import AudioSegment
from pydub.generators import Sine


def _segment(start, end, *, speaker="SPEAKER_00", audio_file=None):
    return {
        "speaker": speaker,
        "start": start,
        "end": end,
        "text": "source",
        "translation": "target",
        "synthesized_speech_file": str(audio_file) if audio_file else None,
    }


def test_short_clip_uses_anchor_slack_without_speed_change():
    from dubbing.core.timing import TimingPolicy, calculate_segment_timing

    result = calculate_segment_timing(
        start=2.0,
        end=2.5,
        next_start=3.1,
        source_duration=10.0,
        audio_duration=0.8,
        policy=TimingPolicy(),
    )

    assert result.available_window == pytest.approx(1.1)
    assert result.tempo == 1.0
    assert result.expected_duration == pytest.approx(0.8)
    assert result.residual_overflow == 0.0
    assert result.within_policy is True


def test_synthesis_uses_anchor_window_and_does_not_resynthesize_a_fitting_clip(tmp_path):
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
            observed.setdefault("estimated_targets", []).append(segment_data.target_duration)
            return 0.8

        def synthesize(self, segments_data, language):
            observed["synthesized_targets"] = [item.target_duration for item in segments_data]
            for item in segments_data:
                Sine(440).to_audio_segment(duration=800).export(item.output_path, format="wav")
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
        tts_system="fake",
        model=None,
    )
    dubber._resolve_voice_profile = lambda _speaker: profile
    dubber._profile_pool_key = lambda _profile: ("fake",)
    dubber._default_tts_system = lambda: "fake"
    dubber._apply_reference_fallbacks = lambda **kwargs: (
        kwargs["tts_segment_data_args"],
        kwargs["original_audio_segment"],
    )
    dubber.tts_clients = {("fake",): TTSStub()}
    dubber.default_tts = dubber.tts_clients[("fake",)]
    dubber._semantic_plan_cache_persistable = False
    dubber._resynthesize_segment = lambda *_args, **_kwargs: pytest.fail(
        "A clip that fits its anchor window must not be resynthesized"
    )
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
    segments = [
        _segment(2.0, 2.5),
        _segment(3.1, 3.5),
    ]

    dubber.synthesize_speech(segments, {}, str(source_path))

    assert observed["estimated_targets"][0] == pytest.approx(1.1)
    assert observed["synthesized_targets"][0] == pytest.approx(1.1)


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
        audio_duration=1.25,
        policy=TimingPolicy(),
    )

    assert result.tempo == 1.0
    assert result.residual_overflow == pytest.approx(0.25)
    assert result.within_policy is True


def test_anchor_plan_sorts_and_equal_starts_share_next_distinct_anchor():
    from dubbing.core.timing import plan_anchor_windows

    later = _segment(4.0, 4.5)
    equal_b = _segment(1.0, 2.0, speaker="SPEAKER_01")
    equal_a = _segment(1.0, 1.5)

    planned = plan_anchor_windows([later, equal_b, equal_a], source_duration=6.0)

    assert [item.segment for item in planned] == [equal_a, equal_b, later]
    assert [item.original_index for item in planned] == [2, 1, 0]
    assert planned[0].next_start == 4.0
    assert planned[1].next_start == 4.0
    assert planned[0].available_window == 3.0
    assert planned[1].available_window == 3.0
    assert planned[2].available_window == 2.0


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


def test_zero_length_range_uses_next_anchor_as_slack():
    from dubbing.core.timing import plan_anchor_windows

    planned = plan_anchor_windows(
        [_segment(1.0, 1.0), _segment(2.0, 2.4)],
        source_duration=4.0,
    )

    assert planned[0].available_window == 1.0


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


def test_timing_policy_config_normalization_and_cli_flags():
    from dubbing.core.config import DubbingConfig, create_argument_parser

    config = DubbingConfig()
    config.config.update(
        {
            "timing_short_segment_threshold": -1,
            "timing_short_segment_max_speed": math.nan,
            "timing_max_speed": 0.9,
            "timing_max_overflow": math.inf,
        }
    )
    config.process_special_parameters()

    assert config.get("timing_short_segment_threshold") == 1.5
    assert config.get("timing_short_segment_max_speed") == 1.08
    assert config.get("timing_max_speed") == 1.15
    assert config.get("timing_max_overflow") == 0.25

    args = create_argument_parser().parse_args(
        [
            "--input", "clip.mp4",
            "--source_language", "en",
            "--target_language", "be",
            "--timing_short_segment_threshold", "1.2",
            "--timing_short_segment_max_speed", "1.04",
            "--timing_max_speed", "1.12",
            "--timing_max_overflow", "0.15",
        ]
    )
    assert args.timing_short_segment_threshold == 1.2
    assert args.timing_short_segment_max_speed == 1.04
    assert args.timing_max_speed == 1.12
    assert args.timing_max_overflow == 0.15


def test_final_cache_fingerprint_includes_policy_and_algorithm_version():
    from dubbing.core.timing import TimingPolicy, timing_cache_fingerprint

    baseline = timing_cache_fingerprint(TimingPolicy())

    assert "anchor_timing_v1" in baseline
    assert baseline != timing_cache_fingerprint(TimingPolicy(max_overflow=0.1))
    assert baseline != timing_cache_fingerprint(TimingPolicy(max_speed=1.2))
    assert baseline != timing_cache_fingerprint(TimingPolicy(short_segment_max_speed=1.04))
    assert baseline != timing_cache_fingerprint(TimingPolicy(short_segment_threshold=1.0))


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
    cached_path = cache.get_cache_path("synthesized_speech") / f"base_be_fake_{fingerprint}.wav"
    Sine(440).to_audio_segment(duration=2000).export(cached_path, format="wav")

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
    dubber.tts_clients = {}

    result = dubber.synthesize_speech([_segment(0.0, 1.0)], {}, str(source_path))

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

    dubber._cache_raw_tts_segment(str(source_path), cache_path)

    assert dubber._cached_segment_contract(cache_path) == "anchor_raw_v1"
    assert cache_path.exists()


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
    assert components["Maximum timing overflow"]["value"] == 0.25
    assert "Group overflow tolerance" not in components

    config_path = tmp_path / "saved.yml"
    save_settings(
        {
            "timing_short_segment_threshold": 1.2,
            "timing_short_segment_max_speed": 1.04,
            "timing_max_speed": 1.12,
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
            "timing_max_overflow": math.inf,
        },
        config_path=str(config_path),
    )

    saved = yaml.safe_load(config_path.read_text(encoding="utf-8"))
    assert saved["timing_short_segment_threshold"] == 1.5
    assert saved["timing_short_segment_max_speed"] == 1.08
    assert saved["timing_max_speed"] == 1.15
    assert saved["timing_max_overflow"] == 0.25
