import pytest
from pathlib import Path
import tomllib


def test_gemini_transcriber_requires_google_api_key(monkeypatch):
    from transcription.gemini_transcriber import GeminiTranscriber

    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_API_KEY"):
        GeminiTranscriber(source_language="en")


def test_gemini_transcriber_accepts_legacy_gemini_api_key_env(monkeypatch):
    from transcription.gemini_transcriber import GeminiTranscriber
    import transcription.gemini_transcriber as gemini_transcriber

    created = {}

    class FakeClient:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(gemini_transcriber, "GEMINI_GENAI_AVAILABLE", True)
    monkeypatch.setattr(gemini_transcriber.genai, "Client", FakeClient)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.setenv("GEMINI_API_KEY", "legacy-key")

    transcriber = GeminiTranscriber(source_language="en")

    assert transcriber.api_key == "legacy-key"
    assert created["api_key"] == "legacy-key"


def test_gemini_transcriber_normalizes_segments_into_internal_format():
    from transcription.gemini_transcriber import GeminiTranscriber

    payload = {
        "segments": [
            {
                "start": "00.00.00",
                "end": "00.00.07",
                "speaker": "SPEAKER_A",
                "text": "What have you done?",
            },
            {
                "start": "00.00.07",
                "end": "00.00.10",
                "speaker": "SPEAKER_B",
                "text": "That's the whole point of it.",
            },
            {
                "start": "00.00.11",
                "end": "00.00.33",
                "speaker": "SPEAKER_A",
                "text": "Yes. This rear spoiler",
            },
        ]
    }

    speakers_rolls, transcription = GeminiTranscriber._normalize_segments(payload)

    assert speakers_rolls == {
        (0.0, 7.0): "SPEAKER_00",
        (7.0, 10.0): "SPEAKER_01",
        (11.0, 33.0): "SPEAKER_00",
    }
    assert transcription == [
        {
            "text": "What have you done?",
            "start": 0.0,
            "end": 7.0,
            "speaker": "SPEAKER_00",
        },
        {
            "text": "That's the whole point of it.",
            "start": 7.0,
            "end": 10.0,
            "speaker": "SPEAKER_01",
        },
        {
            "text": "Yes. This rear spoiler",
            "start": 11.0,
            "end": 33.0,
            "speaker": "SPEAKER_00",
        },
    ]


def test_gemini_transcriber_rejects_invalid_payload():
    from transcription.gemini_transcriber import GeminiTranscriber

    with pytest.raises(ValueError, match="segments"):
        GeminiTranscriber._normalize_segments({"segments": [{"start": "00.00.00"}]})


def test_pyproject_declares_google_genai_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("google-genai") for dep in dependencies)
