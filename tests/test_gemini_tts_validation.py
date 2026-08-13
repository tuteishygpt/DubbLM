import numpy as np
import pytest

from tts.gemini_tts_wrapper import GeminiTTSWrapper, SAMPLE_RATE
from tts.models import TTSSegmentData


def _pcm_with_trailing_silence(
    *, speech_seconds: float = 1.2, silence_seconds: float = 0.8
) -> bytes:
    time_axis = np.arange(int(SAMPLE_RATE * speech_seconds)) / SAMPLE_RATE
    speech = (0.2 * np.sin(2 * np.pi * 140 * time_axis) * 32767).astype("<i2")
    silence = np.zeros(int(SAMPLE_RATE * silence_seconds), dtype="<i2")
    return np.concatenate([speech, silence]).tobytes()


def test_segment_trailing_silence_is_accepted_without_retry(tmp_path):
    wrapper = GeminiTTSWrapper(enable_voice_matching=False)
    wrapper.api_client.client = object()
    calls = 0

    def synthesize_chunk(_content, _speech_config):
        nonlocal calls
        calls += 1
        return _pcm_with_trailing_silence()

    wrapper.api_client.synthesize_chunk = synthesize_chunk
    segment = TTSSegmentData(
        speaker="SPEAKER_00",
        text="Test speech with a recoverable silent tail.",
        voice="Achird",
    )

    success, silence_ratio, best_path = wrapper._attempt_segment_synthesis(
        segment,
        str(tmp_path / "segment.wav"),
        "en",
        max_retries=3,
        max_silence_ratio=0.04,
    )

    assert success is True
    assert calls == 1
    assert silence_ratio > 0.04
    assert best_path is not None


def test_subsecond_segment_with_trailing_silence_is_accepted_without_retry(tmp_path):
    wrapper = GeminiTTSWrapper(enable_voice_matching=False)
    wrapper.api_client.client = object()
    calls = 0

    def synthesize_chunk(_content, _speech_config):
        nonlocal calls
        calls += 1
        return _pcm_with_trailing_silence(
            speech_seconds=0.2,
            silence_seconds=1.8,
        )

    wrapper.api_client.synthesize_chunk = synthesize_chunk
    segment = TTSSegmentData(
        speaker="SPEAKER_00",
        text="This take does not contain enough usable speech.",
        voice="Achird",
    )

    success, _silence_ratio, best_path = wrapper._attempt_segment_synthesis(
        segment,
        str(tmp_path / "segment.wav"),
        "en",
        max_retries=3,
        max_silence_ratio=0.04,
    )

    assert success is True
    assert calls == 1
    assert best_path is not None


def test_silence_only_segment_is_still_rejected(tmp_path):
    wrapper = GeminiTTSWrapper(enable_voice_matching=False)
    wrapper.api_client.client = object()
    calls = 0

    def synthesize_chunk(_content, _speech_config):
        nonlocal calls
        calls += 1
        return np.zeros(int(SAMPLE_RATE * 0.4), dtype="<i2").tobytes()

    wrapper.api_client.synthesize_chunk = synthesize_chunk
    segment = TTSSegmentData(
        speaker="SPEAKER_00",
        text="Silence must not be accepted.",
        voice="Achird",
    )

    success, _silence_ratio, best_path = wrapper._attempt_segment_synthesis(
        segment,
        str(tmp_path / "segment.wav"),
        "en",
        max_retries=3,
        max_silence_ratio=0.04,
    )

    assert success is False
    assert calls == 3
    assert best_path is not None


def test_single_segment_never_promotes_structurally_invalid_best_attempt(tmp_path):
    wrapper = GeminiTTSWrapper(enable_voice_matching=False)
    wrapper.api_client.client = object()
    rejected_path = tmp_path / "rejected.wav"
    rejected_path.write_bytes(b"invalid audio")
    output_path = tmp_path / "segment.wav"
    wrapper._attempt_segment_synthesis = (
        lambda *_args, **_kwargs: (False, 1.0, str(rejected_path))
    )
    segment = TTSSegmentData(
        speaker="SPEAKER_00",
        text="Silence must not reach the final output.",
        voice="Achird",
    )

    with pytest.raises(RuntimeError, match="Failed to synthesize segment"):
        wrapper._synthesize_single_segment(segment, str(output_path), "en")

    assert output_path.exists() is False
    assert rejected_path.exists() is False


def test_nonflat_audio_without_non_silent_frames_is_not_recoverable(
    tmp_path, monkeypatch
):
    import tts.gemini_tts_wrapper as gemini_module

    wrapper = GeminiTTSWrapper(enable_voice_matching=False)
    wrapper.api_client.client = object()
    calls = 0
    stationary_noise = np.random.default_rng(7).normal(
        0.0, 0.01, int(SAMPLE_RATE * 0.4)
    )
    no_non_silent_frames = np.clip(
        stationary_noise * 32767, -32768, 32767
    ).astype("<i2").tobytes()
    monkeypatch.setattr(
        gemini_module.librosa.feature,
        "rms",
        lambda **_kwargs: np.full((1, 5), 0.01, dtype=np.float32),
    )

    def synthesize_chunk(_content, _speech_config):
        nonlocal calls
        calls += 1
        return no_non_silent_frames

    wrapper.api_client.synthesize_chunk = synthesize_chunk
    segment = TTSSegmentData(
        speaker="SPEAKER_00",
        text="Low-energy noise must not be accepted as speech.",
        voice="Achird",
    )

    success, _silence_ratio, _best_path = wrapper._attempt_segment_synthesis(
        segment,
        str(tmp_path / "segment.wav"),
        "en",
        max_retries=2,
        max_silence_ratio=0.04,
    )

    assert success is False
    assert calls == 2
