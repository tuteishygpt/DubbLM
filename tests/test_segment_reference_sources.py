import importlib
from pathlib import Path

import pytest
from pydub import AudioSegment
from pydub.generators import Sine

from dubbing.core.smart_dubbing import SmartDubbing
from dubbing.core.voice_profiles import VoiceProfile


def _audio(path: Path, frequency: int, duration_ms: int = 5000) -> Path:
    Sine(frequency).to_audio_segment(duration=duration_ms).export(path, format="wav")
    return path


def _dubber(tmp_path: Path, **config) -> SmartDubbing:
    audio_dir = tmp_path / "audio"
    speakers_dir = tmp_path / "speakers"
    audio_dir.mkdir()
    speakers_dir.mkdir()
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "audio_artifacts_dir": str(audio_dir),
        "keep_background": False,
        "isolated_tracks": None,
        "start_time": None,
        "segment_reference_min_duration": 0.5,
        **config,
    }
    dubber.speakers_audio_dir = speakers_dir
    return dubber


def _references_module():
    return importlib.import_module("dubbing.core.pipeline.references")


def _segment(**overrides):
    return {
        "speaker": "SPEAKER_00",
        "start": 1.0,
        "end": 2.0,
        "text": "recognized words",
        **overrides,
    }


def _prepared_audio(dubber: SmartDubbing, segment=None, **kwargs) -> AudioSegment:
    path, text = dubber._prepare_segment_reference(
        segment_dict=segment or _segment(),
        speaker="SPEAKER_00",
        chronological_index=3,
        reuse_existing=False,
        **kwargs,
    )
    assert text == "recognized words"
    return AudioSegment.from_file(path)


@pytest.mark.parametrize(
    ("facade_name", "module_name", "args", "kwargs"),
    [
        (
            "_attach_segment_reference",
            "attach_segment_reference",
            (),
            {
                "tts_segment_data_args": {},
                "segment_dict": _segment(),
                "speaker": "SPEAKER_00",
                "segment_index": 0,
                "original_audio_segment": None,
                "segment_reference_min_duration": 0.5,
                "segment_reference_min_duration_ms": 500,
            },
        ),
        ("_canonical_segment_index", "canonical_segment_index", (_segment(), 0), {}),
        ("_segment_reference_artifact_paths", "segment_reference_artifact_paths", (), {}),
        (
            "_segment_reference_error",
            "segment_reference_error",
            ("SPEAKER_00", 0, Path("source.wav"), "reason"),
            {},
        ),
        (
            "_prepare_segment_reference",
            "prepare_segment_reference",
            (),
            {
                "segment_dict": _segment(),
                "speaker": "SPEAKER_00",
                "chronological_index": 0,
                "reuse_existing": False,
            },
        ),
        (
            "_resolve_segment_reference",
            "resolve_segment_reference",
            (),
            {
                "tts_segment_data_args": {},
                "segment_dict": _segment(),
                "profile": VoiceProfile(tts_system="fake", reference_mode="none"),
                "provider_capability": "unsupported",
                "speaker": "SPEAKER_00",
                "segment_index": 0,
                "original_audio_segment": None,
                "segment_reference_min_duration": 0.5,
            },
        ),
    ],
)
def test_reference_facade_methods_delegate_to_module_at_call_time(
    tmp_path, monkeypatch, facade_name, module_name, args, kwargs
):
    module = _references_module()
    sentinel = object()
    calls = []

    def implementation(*implementation_args, **implementation_kwargs):
        calls.append((implementation_args, implementation_kwargs))
        return sentinel

    monkeypatch.setattr(module, module_name, implementation)
    dubber = _dubber(tmp_path)

    assert getattr(dubber, facade_name)(*args, **kwargs) is sentinel
    assert len(calls) == 1


def test_reference_module_matches_exact_facade_results(tmp_path):
    module = _references_module()
    dubber = _dubber(tmp_path, keep_background=True)
    processed = tmp_path / "processed" / "vocals.wav"
    segment = _segment(_timing_original_index=7)

    assert module.canonical_segment_index(segment, 3) == (
        SmartDubbing._canonical_segment_index(segment, 3)
    )
    assert module.segment_reference_artifact_paths(
        config=dubber.config,
        processed_source_path=str(processed),
    ) == dubber._segment_reference_artifact_paths(str(processed))
    module_error = module.segment_reference_error(
        "SPEAKER_00", 7, processed, "unreadable"
    )
    facade_error = SmartDubbing._segment_reference_error(
        "SPEAKER_00", 7, processed, "unreadable"
    )
    assert type(module_error) is type(facade_error)
    assert str(module_error) == str(facade_error)


def test_prepare_segment_reference_uses_current_facade_helpers(tmp_path, monkeypatch):
    dubber = _dubber(tmp_path)
    reference = dubber.speakers_audio_dir / "segments" / "SPEAKER_00_17.wav"
    reference.parent.mkdir()
    _audio(reference, 440, 1000)
    calls = []
    monkeypatch.setattr(
        dubber,
        "_canonical_segment_index",
        lambda segment, index: calls.append((segment, index)) or 17,
    )

    path, text = dubber._prepare_segment_reference(
        segment_dict=_segment(),
        speaker="SPEAKER_00",
        chronological_index=3,
        reuse_existing=True,
    )

    assert path == str(reference)
    assert text == "recognized words"
    assert calls == [(_segment(), 3)]


def test_normal_segment_reference_uses_mapped_isolated_track(tmp_path):
    dubber = _dubber(tmp_path)
    source = _audio(Path(dubber.config["audio_artifacts_dir"]) / "source.wav", 220)
    isolated = _audio(tmp_path / "isolated.wav", 880)
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(isolated)}

    prepared = _prepared_audio(dubber, processed_source_path=str(source))

    expected = AudioSegment.from_file(isolated)[1000:2000]
    assert prepared.raw_data == expected.raw_data


def test_normal_segment_reference_applies_start_time_to_isolated_track(tmp_path):
    dubber = _dubber(tmp_path, start_time=2.0)
    source = _audio(Path(dubber.config["audio_artifacts_dir"]) / "source.wav", 220)
    isolated = AudioSegment.silent(duration=2000) + Sine(990).to_audio_segment(duration=3000)
    isolated_path = tmp_path / "isolated.wav"
    isolated.export(isolated_path, format="wav")
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(isolated_path)}

    prepared = _prepared_audio(dubber, processed_source_path=str(source))

    assert prepared.raw_data == isolated[3000:4000].raw_data


def test_normal_segment_reference_rebuilds_existing_file(tmp_path):
    dubber = _dubber(tmp_path)
    source = _audio(Path(dubber.config["audio_artifacts_dir"]) / "source.wav", 660)
    reference = dubber.speakers_audio_dir / "segments" / "SPEAKER_00_3.wav"
    reference.parent.mkdir()
    _audio(reference, 110, 1000)

    prepared = _prepared_audio(dubber, processed_source_path=str(source))

    assert prepared.raw_data == AudioSegment.from_file(source)[1000:2000].raw_data


def test_segment_reference_prefers_vocals_then_falls_back_to_source(tmp_path):
    dubber = _dubber(tmp_path, keep_background=True)
    audio_dir = Path(dubber.config["audio_artifacts_dir"])
    source = _audio(audio_dir / "source.wav", 330)
    vocals = _audio(audio_dir / "vocals.wav", 770)

    prepared = _prepared_audio(dubber, processed_source_path=str(vocals))
    assert prepared.raw_data == AudioSegment.from_file(vocals)[1000:2000].raw_data

    vocals.write_bytes(b"unreadable")
    prepared = _prepared_audio(dubber, processed_source_path=str(vocals))
    assert prepared.raw_data == AudioSegment.from_file(source)[1000:2000].raw_data


@pytest.mark.parametrize("broken", ["missing", "unreadable", "too-short"])
def test_mapped_isolated_track_failure_never_falls_back(tmp_path, broken):
    dubber = _dubber(tmp_path, keep_background=True)
    audio_dir = Path(dubber.config["audio_artifacts_dir"])
    _audio(audio_dir / "source.wav", 330)
    _audio(audio_dir / "vocals.wav", 770)
    isolated = tmp_path / "isolated.wav"
    if broken == "unreadable":
        isolated.write_bytes(b"not wav")
    elif broken == "too-short":
        _audio(isolated, 990, 1200)
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(isolated)}

    with pytest.raises(ValueError) as error:
        _prepared_audio(dubber, processed_source_path=str(audio_dir / "vocals.wav"))

    message = str(error.value)
    assert "speaker=SPEAKER_00" in message
    assert "segment=3" in message
    assert "mode=segment" in message
    assert str(isolated) in message
    assert not (dubber.speakers_audio_dir / "segments" / "SPEAKER_00_3.wav").exists()


def test_button_builds_missing_reference_and_reuses_valid_reference_without_source(tmp_path):
    dubber = _dubber(tmp_path, start_time=1.0)
    isolated = _audio(tmp_path / "isolated.wav", 900)
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(isolated)}

    path, text = dubber._prepare_segment_reference(
        segment_dict=_segment(_timing_original_index=8),
        speaker="SPEAKER_00",
        chronological_index=3,
        reuse_existing=True,
    )
    assert Path(path).name == "SPEAKER_00_8.wav"
    assert text == "recognized words"
    assert AudioSegment.from_file(path).raw_data == AudioSegment.from_file(isolated)[2000:3000].raw_data

    isolated.unlink()
    reused_path, reused_text = dubber._prepare_segment_reference(
        segment_dict=_segment(_timing_original_index=8),
        speaker="SPEAKER_00",
        chronological_index=3,
        reuse_existing=True,
    )
    assert reused_path == path
    assert reused_text == "recognized words"


@pytest.mark.parametrize("existing", ["unreadable", "empty", "too-short"])
def test_button_rebuilds_invalid_existing_reference(tmp_path, existing):
    dubber = _dubber(tmp_path)
    source = _audio(Path(dubber.config["audio_artifacts_dir"]) / "source.wav", 550)
    reference = dubber.speakers_audio_dir / "segments" / "SPEAKER_00_3.wav"
    reference.parent.mkdir()
    if existing == "unreadable":
        reference.write_bytes(b"bad")
    elif existing == "empty":
        AudioSegment.empty().export(reference, format="wav")
    else:
        _audio(reference, 100, 100)

    path, _ = dubber._prepare_segment_reference(
        segment_dict=_segment(),
        speaker="SPEAKER_00",
        chronological_index=3,
        reuse_existing=True,
        processed_source_path=str(source),
    )

    assert AudioSegment.from_file(path).raw_data == AudioSegment.from_file(source)[1000:2000].raw_data


@pytest.mark.parametrize("original_index, expected", [(7, 7), (-1, 3), (1.5, 3), (True, 3)])
def test_segment_reference_uses_only_non_negative_integer_canonical_index(
    tmp_path, original_index, expected
):
    dubber = _dubber(tmp_path)
    source = _audio(Path(dubber.config["audio_artifacts_dir"]) / "source.wav", 440)

    path, _ = dubber._prepare_segment_reference(
        segment_dict=_segment(_timing_original_index=original_index),
        speaker="SPEAKER_00",
        chronological_index=3,
        reuse_existing=False,
        processed_source_path=str(source),
    )

    assert Path(path).name == f"SPEAKER_00_{expected}.wav"


def test_segment_reference_cache_fingerprint_changes_with_isolated_content(tmp_path):
    dubber = _dubber(tmp_path, tts_system="higgs", tts_model=None)
    isolated = _audio(tmp_path / "isolated.wav", 440)
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(isolated)}
    dubber.voice_profiles = {
        "SPEAKER_00": __import__(
            "dubbing.core.voice_profiles", fromlist=["VoiceProfile"]
        ).VoiceProfile(tts_system="higgs", reference_mode="segment")
    }

    baseline = dubber._effective_tts_cache_fingerprint(["SPEAKER_00"])
    _audio(isolated, 880)

    assert dubber._effective_tts_cache_fingerprint(["SPEAKER_00"]) != baseline


def test_effective_tts_fingerprint_hashes_shared_segment_source_once(tmp_path, monkeypatch):
    dubber = _dubber(tmp_path, tts_system="higgs", tts_model=None)
    isolated = _audio(tmp_path / "isolated.wav", 440)
    dubber.config["isolated_tracks"] = {
        "SPEAKER_00": str(isolated),
        "SPEAKER_01": str(isolated),
    }
    dubber.voice_profiles = {
        speaker: VoiceProfile(tts_system="higgs", reference_mode="segment")
        for speaker in dubber.config["isolated_tracks"]
    }
    original = SmartDubbing._file_content_identity
    calls = []

    def tracked(path):
        calls.append(path)
        return original(path)

    monkeypatch.setattr(SmartDubbing, "_file_content_identity", staticmethod(tracked))

    dubber._effective_tts_cache_fingerprint(dubber.voice_profiles)

    assert calls.count(str(isolated)) == 1


def test_single_row_missing_segment_reference_is_prepared_before_tts(tmp_path):
    dubber = _dubber(tmp_path, target_language="be", start_time=1.0)
    isolated = _audio(tmp_path / "isolated.wav", 900)
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(isolated)}
    dubber.audio_chunks_dir = tmp_path / "chunks"
    dubber.su_audio_chunks_dir = tmp_path / "timed"
    dubber.su_audio_chunks_dir.mkdir()
    profile = VoiceProfile(tts_system="fake", reference_mode="segment")
    received = []

    class TTSStub:
        reference_capability = "required"

        def validate_segments(self, segments_data):
            return []

        def synthesize(self, segments_data, language):
            received.extend(segments_data)
            for item in segments_data:
                Sine(440).to_audio_segment(duration=400).export(
                    item.output_path, format="wav"
                )

    client = TTSStub()
    dubber._resolve_voice_profile = lambda _speaker: profile
    dubber._profile_pool_key = lambda _profile: ("fake",)
    dubber._default_tts_system = lambda: "fake"
    dubber.tts_clients = {("fake",): client}
    dubber.default_tts = client
    segment = {
        **_segment(_timing_original_index=8),
        "translation": "translated",
        "style_prompt": "",
    }

    dubber.resynthesize_one_segment([segment], 0)

    assert len(received) == 1
    assert Path(received[0].reference_audio_path).name == "SPEAKER_00_8.wav"
    assert received[0].reference_text == "recognized words"
    assert AudioSegment.from_file(received[0].reference_audio_path).raw_data == (
        AudioSegment.from_file(isolated)[2000:3000].raw_data
    )


def test_single_row_reference_failure_stops_before_tts(tmp_path):
    dubber = _dubber(tmp_path, target_language="be")
    missing = tmp_path / "missing-isolated.wav"
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(missing)}
    dubber.audio_chunks_dir = tmp_path / "chunks"
    dubber.su_audio_chunks_dir = tmp_path / "timed"
    dubber.su_audio_chunks_dir.mkdir()
    profile = VoiceProfile(tts_system="fake", reference_mode="segment")
    synthesize_calls = []

    class TTSStub:
        reference_capability = "required"

        def synthesize(self, segments_data, language):
            synthesize_calls.append(segments_data)

    client = TTSStub()
    dubber._resolve_voice_profile = lambda _speaker: profile
    dubber._profile_pool_key = lambda _profile: ("fake",)
    dubber._default_tts_system = lambda: "fake"
    dubber.tts_clients = {("fake",): client}
    dubber.default_tts = client
    segment = {**_segment(), "translation": "translated", "style_prompt": ""}

    with pytest.raises(ValueError, match="missing-isolated.wav"):
        dubber.resynthesize_one_segment([segment], 0)

    assert synthesize_calls == []


def test_invalid_segment_timestamp_error_names_authoritative_source(tmp_path):
    dubber = _dubber(tmp_path)
    isolated = tmp_path / "isolated.wav"
    dubber.config["isolated_tracks"] = {"SPEAKER_00": str(isolated)}

    with pytest.raises(ValueError) as error:
        dubber._prepare_segment_reference(
            segment_dict=_segment(start=float("nan")),
            speaker="SPEAKER_00",
            chronological_index=3,
            reuse_existing=False,
        )

    assert f"source={isolated}" in str(error.value)
