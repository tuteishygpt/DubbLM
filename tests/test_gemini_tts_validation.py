import numpy as np

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


def test_segment_with_too_little_audible_speech_is_still_retried(tmp_path):
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

    assert success is False
    assert calls == 3
    assert best_path is not None
