from pathlib import Path
import tomllib

import pytest


def test_deepgram_transcriber_requires_api_key(monkeypatch):
    from transcription.deepgram_transcriber import DeepgramTranscriber

    monkeypatch.delenv("DEEPGRAM_API_KEY", raising=False)

    with pytest.raises(ValueError, match="DEEPGRAM_API_KEY"):
        DeepgramTranscriber(source_language="ru")


def test_deepgram_transcriber_normalizes_utterances_into_internal_format():
    from transcription.deepgram_transcriber import DeepgramTranscriber

    payload = {
        "results": {
            "utterances": [
                {
                    "start": 0.0,
                    "end": 1.25,
                    "speaker": 1,
                    "transcript": "Привет.",
                    "confidence": 0.98,
                    "words": [
                        {"word": "Привет.", "start": 0.0, "end": 1.25, "confidence": 0.98}
                    ],
                },
                {
                    "start": 1.5,
                    "end": 3.0,
                    "speaker": 2,
                    "transcript": "Как дела?",
                    "confidence": 0.95,
                    "words": [
                        {"word": "Как", "start": 1.5, "end": 2.0, "confidence": 0.96},
                        {"word": "дела?", "start": 2.05, "end": 3.0, "confidence": 0.94},
                    ],
                },
                {
                    "start": 3.2,
                    "end": 4.0,
                    "speaker": 1,
                    "transcript": "Хорошо.",
                    "confidence": 0.97,
                },
            ]
        }
    }

    speakers_rolls, transcription = DeepgramTranscriber._normalize_response(payload)

    assert speakers_rolls == {
        (0.0, 1.25): "SPEAKER_00",
        (1.5, 3.0): "SPEAKER_01",
        (3.2, 4.0): "SPEAKER_00",
    }
    assert transcription == [
        {
            "text": "Привет.",
            "start": 0.0,
            "end": 1.25,
            "speaker": "SPEAKER_00",
            "confidence": 0.98,
            "words": [
                {"word": "Привет.", "start": 0.0, "end": 1.25, "confidence": 0.98}
            ],
        },
        {
            "text": "Как дела?",
            "start": 1.5,
            "end": 3.0,
            "speaker": "SPEAKER_01",
            "confidence": 0.95,
            "words": [
                {"word": "Как", "start": 1.5, "end": 2.0, "confidence": 0.96},
                {"word": "дела?", "start": 2.05, "end": 3.0, "confidence": 0.94},
            ],
        },
        {
            "text": "Хорошо.",
            "start": 3.2,
            "end": 4.0,
            "speaker": "SPEAKER_00",
            "confidence": 0.97,
        },
    ]


def test_deepgram_transcriber_rejects_missing_utterances():
    from transcription.deepgram_transcriber import DeepgramTranscriber

    with pytest.raises(ValueError, match="utterances"):
        DeepgramTranscriber._normalize_response({"results": {}})


def test_deepgram_transcriber_uses_utterances_without_resplitting(tmp_path, monkeypatch):
    from transcription.deepgram_transcriber import DeepgramTranscriber
    import transcription.deepgram_transcriber as deepgram_transcriber

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")

    captured = {}

    class FakeResponse:
        def to_dict(self):
            return {
                "results": {
                    "utterances": [
                        {
                            "start": 0.0,
                            "end": 2.5,
                            "speaker": 0,
                            "transcript": "Это уже естественная фраза.",
                        }
                    ]
                }
            }

    class FakeMediaClient:
        def transcribe_file(self, payload=None, **kwargs):
            captured["payload"] = payload
            captured["kwargs"] = kwargs
            return FakeResponse()

    class FakeDeepgramClient:
        def __init__(self, api_key):
            captured["api_key"] = api_key
            self.listen = type(
                "FakeListen",
                (),
                {"v1": type("FakeV1", (), {"media": FakeMediaClient()})()},
            )()

    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram-test-key")
    monkeypatch.setattr(deepgram_transcriber, "DeepgramClient", FakeDeepgramClient)

    transcriber = DeepgramTranscriber(source_language="ru")
    speakers_rolls, transcription = transcriber.diarize_and_transcribe(str(audio_path), use_cache=False)

    assert captured["api_key"] == "deepgram-test-key"
    assert captured["kwargs"]["request"] == b"fake-audio"
    assert captured["kwargs"]["model"] == "nova-3"
    assert captured["kwargs"]["language"] == "ru"
    assert captured["kwargs"]["utterances"] is True
    assert captured["kwargs"]["diarize"] is True
    assert transcription == [
        {
            "text": "Это уже естественная фраза.",
            "start": 0.0,
            "end": 2.5,
            "speaker": "SPEAKER_00",
        }
    ]
    assert speakers_rolls == {(0.0, 2.5): "SPEAKER_00"}


def test_deepgram_transcriber_surfaces_actionable_auth_errors(tmp_path, monkeypatch):
    from transcription.deepgram_transcriber import DeepgramTranscriber
    import transcription.deepgram_transcriber as deepgram_transcriber

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio")

    class FakeApiError(Exception):
        def __init__(self):
            super().__init__("Invalid credentials.")
            self.status_code = 401
            self.body = {"err_code": "INVALID_AUTH", "err_msg": "Invalid credentials."}

    class FakeMediaClient:
        def transcribe_file(self, payload=None, request=None, **kwargs):
            raise FakeApiError()

    class FakeDeepgramClient:
        def __init__(self, api_key):
            self.listen = type(
                "FakeListen",
                (),
                {"v1": type("FakeV1", (), {"media": FakeMediaClient()})()},
            )()

    monkeypatch.setenv("DEEPGRAM_API_KEY", "deepgram-test-key")
    monkeypatch.setattr(deepgram_transcriber, "DeepgramClient", FakeDeepgramClient)

    transcriber = DeepgramTranscriber(source_language="ru")

    with pytest.raises(RuntimeError, match="DEEPGRAM_API_KEY"):
        transcriber.diarize_and_transcribe(str(audio_path), use_cache=False)


def test_pyproject_declares_deepgram_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("deepgram-sdk") for dep in dependencies)
