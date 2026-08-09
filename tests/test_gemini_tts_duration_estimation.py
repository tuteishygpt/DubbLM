import pytest

from tts.gemini_tts_wrapper import GeminiTTSWrapper, TextAnalysisUtils
from tts.models import TTSSegmentData


def test_duration_estimate_ignores_combining_stress_marks():
    wrapper = GeminiTTSWrapper(enable_voice_matching=False)
    plain = TTSSegmentData(
        speaker="SPEAKER_00",
        text="Сейчас в индустрии настоящий кризис.",
    )
    stressed = TTSSegmentData(
        speaker="SPEAKER_00",
        text="Сейча́с в инду́стрии настоя́щий кри́зис.",
    )
    original_stressed_text = stressed.text

    plain_duration = wrapper.estimate_audio_segment_length(plain, language="ru")
    stressed_duration = wrapper.estimate_audio_segment_length(stressed, language="ru")

    assert stressed_duration == pytest.approx(plain_duration)
    assert stressed.text == original_stressed_text


def test_duration_estimate_preserves_unrelated_unicode_characters():
    wrapper = GeminiTTSWrapper(enable_voice_matching=False)
    text = "안녕하세요 세계"
    segment = TTSSegmentData(speaker="SPEAKER_00", text=text)
    expected_duration = (
        TextAnalysisUtils.count_words(text)
        / 150.0
        * 60
        * TextAnalysisUtils.estimate_speech_complexity(text)
    )

    actual_duration = wrapper.estimate_audio_segment_length(segment, language="ko")

    assert actual_duration == pytest.approx(expected_duration)
    assert segment.text == text
