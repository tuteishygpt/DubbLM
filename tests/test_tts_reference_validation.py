
import pytest

from tts.models import TTSSegmentData
from tts.tts_interface import TTSInterface


class _ValidationTTS(TTSInterface):
    provider_name = "test-provider"

    def __init__(self, capability):
        self.reference_capability = capability

    def initialize(self):
        pass

    def synthesize(self, segments_data, language="en", **kwargs):
        return []

    def estimate_audio_segment_length(self, segment_data, language="en"):
        return None

    def is_available(self):
        return True

    def set_voice_mapping(self, mapping):
        pass

    def set_voice_prompt_mapping(self, mapping):
        pass

    def cleanup(self):
        pass


def _segment(*, mode=None, reference=None):
    return TTSSegmentData(
        speaker="SPEAKER_02",
        text="hello",
        segment_index=7,
        reference_mode=mode,
        reference_audio_path=str(reference) if reference else None,
    )


@pytest.mark.parametrize(
    ("capability", "mode", "reason"),
    [
        ("unsupported", "configured", "does not support voice-cloning references"),
        ("optional", None, "explicit reference_mode is required; add reference_mode under the new-style voices: profile"),
        ("required", None, "explicit reference_mode is required; add reference_mode under the new-style voices: profile"),
        ("required", "none", "reference_mode 'none' is not allowed"),
        ("required", "mystery", "unknown reference_mode 'mystery'"),
    ],
)
def test_validate_segments_reports_mode_capability_errors(capability, mode, reason):
    issues = _ValidationTTS(capability).validate_segments([_segment(mode=mode)])

    assert issues == [
        (
            7,
            f"provider=test-provider speaker=SPEAKER_02 segment=7 mode={mode or '<missing>'}: "
            f"{reason}",
        )
    ]


def test_optional_provider_accepts_none_without_reference():
    assert _ValidationTTS("optional").validate_segments([_segment(mode="none")]) == []


def test_missing_mode_error_explains_new_style_voices_migration():
    issues = _ValidationTTS("required").validate_segments([_segment()])

    assert "add reference_mode under the new-style voices: profile" in issues[0][1]


def test_non_none_mode_requires_an_existing_local_file(tmp_path):
    missing = tmp_path / "missing.wav"
    issues = _ValidationTTS("required").validate_segments(
        [_segment(mode="configured", reference=missing)]
    )

    assert issues == [
        (
            7,
            "provider=test-provider speaker=SPEAKER_02 segment=7 mode=configured: "
            f"reference file does not exist: {missing}",
        )
    ]

    existing = tmp_path / "voice.wav"
    existing.write_bytes(b"wav")
    assert _ValidationTTS("required").validate_segments(
        [_segment(mode="configured", reference=existing)]
    ) == []


def test_cloning_wrapper_capabilities_are_classified():
    from tts.bextts_wrapper import BexTTSWrapper
    from tts.f5_tts_wrapper import F5TTSWrapper
    from tts.omnivoice_wrapper import OmniVoiceWrapper
    from tts.xtts_local_wrapper import XTTSLocalWrapper

    assert BexTTSWrapper.reference_capability == "optional"
    assert F5TTSWrapper.reference_capability == "required"
    assert OmniVoiceWrapper.reference_capability == "required"
    assert XTTSLocalWrapper.reference_capability == "required"
