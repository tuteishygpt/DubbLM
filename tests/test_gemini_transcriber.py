import pytest
from pathlib import Path
import tomllib


def test_gemini_transcriber_requires_vertex_ai_configuration(monkeypatch):
    from transcription.gemini_transcriber import GeminiTranscriber

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_PROJECT", raising=False)
    monkeypatch.delenv("GOOGLE_CLOUD_LOCATION", raising=False)

    with pytest.raises(ValueError, match="GOOGLE_GENAI_USE_VERTEXAI"):
        GeminiTranscriber(source_language="en")


def test_gemini_transcriber_uses_vertex_ai_client_configuration(monkeypatch):
    from transcription.gemini_transcriber import GeminiTranscriber
    import transcription.gemini_transcriber as gemini_transcriber

    created = {}

    class FakeClient:
        def __init__(self, **kwargs):
            created.update(kwargs)

    monkeypatch.setattr(gemini_transcriber, "GEMINI_GENAI_AVAILABLE", True)
    monkeypatch.setattr(gemini_transcriber.genai, "Client", FakeClient)
    monkeypatch.delenv("GOOGLE_API_KEY", raising=False)
    monkeypatch.delenv("GEMINI_API_KEY", raising=False)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "vertex-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    transcriber = GeminiTranscriber(source_language="en")

    assert transcriber.vertex_ai_settings.project == "vertex-project"
    assert created == {
        "vertexai": True,
        "project": "vertex-project",
        "location": "global",
    }


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


def test_gemini_transcriber_prompt_requests_millisecond_precision():
    from transcription.gemini_transcriber import GeminiTranscriber

    prompt = GeminiTranscriber._build_prompt()

    assert "HH:MM:SS.mmm" in prompt
    assert "millisecond" in prompt.lower()


def test_gemini_transcriber_normalizes_subsecond_timestamps():
    from transcription.gemini_transcriber import GeminiTranscriber

    payload = {
        "segments": [
            {
                "start": "00:00:10.250",
                "end": "00:00:12.000",
                "speaker": "SPEAKER_A",
                "text": "First line",
            },
            {
                "start": "00.00.12.125",
                "end": "00.00.13.375",
                "speaker": "SPEAKER_B",
                "text": "Second line",
            },
        ]
    }

    speakers_rolls, transcription = GeminiTranscriber._normalize_segments(payload)

    assert speakers_rolls == {
        (10.25, 12.0): "SPEAKER_00",
        (12.125, 13.375): "SPEAKER_01",
    }
    assert transcription == [
        {
            "text": "First line",
            "start": 10.25,
            "end": 12.0,
            "speaker": "SPEAKER_00",
        },
        {
            "text": "Second line",
            "start": 12.125,
            "end": 13.375,
            "speaker": "SPEAKER_01",
        },
    ]


def test_gemini_transcriber_rejects_invalid_payload():
    from transcription.gemini_transcriber import GeminiTranscriber

    with pytest.raises(ValueError, match="segments"):
        GeminiTranscriber._normalize_segments({"segments": [{"start": "00.00.00"}]})


def test_gemini_transcriber_sends_vertex_audio_as_inline_bytes(tmp_path, monkeypatch):
    from transcription.gemini_transcriber import GeminiTranscriber
    import transcription.gemini_transcriber as gemini_transcriber

    audio_path = tmp_path / "sample.wav"
    audio_path.write_bytes(b"fake-audio-bytes")

    created = {"part": None, "contents": None}

    class FakeFiles:
        def upload(self, *args, **kwargs):
            raise AssertionError("Vertex path must not call files.upload")

    class FakeModels:
        def generate_content(self, **kwargs):
            created["contents"] = kwargs["contents"]
            return type(
                "FakeResponse",
                (),
                {"parsed": {"segments": [{"start": "00.00.00", "end": "00.00.01", "speaker": "SPEAKER_A", "text": "Hi"}]}},
            )()

    class FakeClient:
        def __init__(self, **kwargs):
            self.files = FakeFiles()
            self.models = FakeModels()

    def fake_from_bytes(*, data, mime_type):
        created["part"] = {"data": data, "mime_type": mime_type}
        return {"inline_data": data, "mime_type": mime_type}

    monkeypatch.setattr(gemini_transcriber, "GEMINI_GENAI_AVAILABLE", True)
    monkeypatch.setattr(gemini_transcriber.genai, "Client", FakeClient)
    monkeypatch.setattr(gemini_transcriber.genai_types.Part, "from_bytes", fake_from_bytes)
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "vertex-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    transcriber = GeminiTranscriber(source_language="en")
    speakers_rolls, transcription = transcriber.diarize_and_transcribe(str(audio_path), use_cache=False)

    assert created["part"] == {"data": b"fake-audio-bytes", "mime_type": "audio/wav"}
    assert created["contents"][0] == {"inline_data": b"fake-audio-bytes", "mime_type": "audio/wav"}
    assert created["contents"][1] == transcriber._build_prompt()
    assert speakers_rolls == {(0.0, 1.0): "SPEAKER_00"}
    assert transcription[0]["text"] == "Hi"


def test_gemini_transcriber_cache_key_includes_timestamp_precision_version(monkeypatch):
    from transcription.gemini_transcriber import GeminiTranscriber

    captured = {}

    class CacheStub:
        def cache_exists(self, step_name, cache_key):
            assert step_name == "gemini_diarization_transcription"
            assert cache_key == "cache-key"
            return True

        def load_from_cache(self, step_name, cache_key):
            return {
                "diarization": {(0.0, 1.25): "SPEAKER_00"},
                "transcription": [
                    {"start": 0.0, "end": 1.25, "speaker": "SPEAKER_00", "text": "Hi"}
                ],
            }

    transcriber = object.__new__(GeminiTranscriber)
    transcriber.gemini_transcription_model = "gemini-3-flash-preview"
    transcriber.cache_manager = CacheStub()
    transcriber.debug_data = {"diarization": None, "transcription": None}

    def fake_generate_cache_key(_self, audio_file, additional_params=""):
        captured["audio_file"] = audio_file
        captured["additional_params"] = additional_params
        return "cache-key"

    monkeypatch.setattr(GeminiTranscriber, "_generate_cache_key", fake_generate_cache_key)

    speakers_rolls, transcription = transcriber.diarize_and_transcribe("sample.wav")

    assert speakers_rolls == {(0.0, 1.25): "SPEAKER_00"}
    assert transcription[0]["end"] == 1.25
    assert captured["additional_params"].endswith("_ts_v2")


def test_pyproject_declares_google_genai_dependency():
    pyproject_path = Path(__file__).resolve().parents[1] / "pyproject.toml"
    pyproject = tomllib.loads(pyproject_path.read_text(encoding="utf-8"))

    dependencies = pyproject["project"]["dependencies"]

    assert any(dep.startswith("google-genai") for dep in dependencies)
