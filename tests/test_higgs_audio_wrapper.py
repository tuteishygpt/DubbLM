import importlib
from pathlib import Path

import pytest


def _new_wrapper(module, tmp_path):
    wrapper = object.__new__(module.HiggsAudioWrapper)
    wrapper.space_id = "custom/higgs"
    wrapper.api_name = "/synthesize"
    wrapper.temperature = 0.6
    wrapper.top_p = 0.9
    wrapper.top_k = 42
    wrapper.max_new_tokens = 1024
    wrapper.seed = 123
    wrapper.hf_token_env = "CUSTOM_HF_TOKEN"
    wrapper.client = None
    wrapper._temp_dir = str(tmp_path / "private")
    Path(wrapper._temp_dir).mkdir()
    return wrapper


def test_higgs_defaults_and_required_capability(monkeypatch):
    module = importlib.import_module("tts.higgs_audio_wrapper")
    monkeypatch.setattr(module, "GRADIO_AVAILABLE", True)
    monkeypatch.setattr(module, "PYDUB_AVAILABLE", True)

    wrapper = module.HiggsAudioWrapper()

    assert wrapper.reference_capability == "required"
    assert wrapper.space_id == "archivartaunik/higgs-audio-v3-tts"
    assert wrapper.api_name == "/synthesize"
    assert (wrapper.temperature, wrapper.top_p, wrapper.top_k) == (0.7, 0.95, 50)
    assert (wrapper.max_new_tokens, wrapper.seed, wrapper.hf_token_env) == (2048, -1, "HF_TOKEN")


def test_higgs_initialize_uses_token_and_legacy_headers_fallback(tmp_path, monkeypatch):
    module = importlib.import_module("tts.higgs_audio_wrapper")
    wrapper = _new_wrapper(module, tmp_path)
    wrapper._temp_dir = None
    calls = []

    class ClientStub:
        def __init__(self, space_id, **kwargs):
            calls.append((space_id, kwargs))
            if "hf_token" in kwargs:
                raise TypeError("old client")

    monkeypatch.setattr(module, "Client", ClientStub)
    monkeypatch.setenv("CUSTOM_HF_TOKEN", "secret")

    wrapper.initialize()

    assert calls == [
        ("custom/higgs", {"hf_token": "secret"}),
        ("custom/higgs", {"headers": {"Authorization": "Bearer secret"}}),
    ]
    assert Path(wrapper._temp_dir).is_dir()
    wrapper.cleanup()


@pytest.mark.parametrize(
    "prediction_factory",
    [
        lambda path: str(path),
        lambda path: {"path": str(path)},
        lambda path: ({"name": str(path)}, "metadata"),
    ],
)
def test_higgs_predict_contract_output_copy_and_alignment(
    tmp_path, monkeypatch, prediction_factory
):
    module = importlib.import_module("tts.higgs_audio_wrapper")
    wrapper = _new_wrapper(module, tmp_path)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    downloaded = tmp_path / "downloaded.wav"
    downloaded.write_bytes(b"generated")
    output = tmp_path / "output" / "segment.wav"
    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return prediction_factory(downloaded)

    class FakeAudio:
        def __len__(self):
            return 1250

    wrapper.client = ClientStub()
    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(
        module,
        "AudioSegment",
        type("AudioStub", (), {"from_file": staticmethod(lambda path: FakeAudio())}),
    )
    segment = module.TTSSegmentData(
        speaker="SPEAKER_00",
        text="hello world",
        segment_index=3,
        reference_mode="configured",
        reference_audio_path=str(reference),
        reference_text="original words",
        output_path=str(output),
    )

    alignments = wrapper.synthesize([segment])

    assert captured == {
        "text": "hello world",
        "reference_audio": f"handled:{reference}",
        "reference_text": "original words",
        "temperature": 0.6,
        "top_p": 0.9,
        "top_k": 42,
        "max_new_tokens": 1024,
        "seed": 123,
        "api_name": "/synthesize",
    }
    assert output.read_bytes() == b"generated"
    assert alignments[0].diarized_segment.end_time == 1.25
    assert not downloaded.exists()


def test_higgs_validates_every_reference_before_first_predict(tmp_path):
    module = importlib.import_module("tts.higgs_audio_wrapper")
    wrapper = _new_wrapper(module, tmp_path)

    class ClientStub:
        calls = 0

        def predict(self, **kwargs):
            self.calls += 1

    wrapper.client = ClientStub()
    segment = module.TTSSegmentData(
        speaker="SPEAKER_00",
        text="hello",
        segment_index=2,
        reference_mode="configured",
        reference_audio_path=str(tmp_path / "missing.wav"),
    )

    with pytest.raises(ValueError, match="reference file does not exist"):
        wrapper.synthesize([segment])
    assert wrapper.client.calls == 0


def test_higgs_runtime_failure_removes_downloaded_temporary_file(tmp_path, monkeypatch):
    module = importlib.import_module("tts.higgs_audio_wrapper")
    wrapper = _new_wrapper(module, tmp_path)
    reference = tmp_path / "reference.wav"
    reference.write_bytes(b"reference")
    downloaded = tmp_path / "broken.wav"
    downloaded.write_bytes(b"broken")

    class ClientStub:
        def predict(self, **kwargs):
            return str(downloaded)

    wrapper.client = ClientStub()
    monkeypatch.setattr(module, "handle_file", lambda path: path)
    monkeypatch.setattr(
        module,
        "AudioSegment",
        type(
            "AudioStub",
            (),
            {"from_file": staticmethod(lambda path: (_ for _ in ()).throw(ValueError("bad audio")))},
        ),
    )
    segment = module.TTSSegmentData(
        speaker="SPEAKER_00",
        text="hello",
        reference_mode="configured",
        reference_audio_path=str(reference),
    )

    assert wrapper.synthesize([segment]) == []
    assert not downloaded.exists()


def test_higgs_estimator_and_cleanup(tmp_path):
    module = importlib.import_module("tts.higgs_audio_wrapper")
    wrapper = _new_wrapper(module, tmp_path)
    wrapper.client = object()
    (Path(wrapper._temp_dir) / "part.wav").write_bytes(b"audio")

    assert wrapper.estimate_audio_segment_length(
        module.TTSSegmentData(speaker="S", text="one two three")
    ) == 1.35
    assert wrapper.estimate_audio_segment_length(
        module.TTSSegmentData(speaker="S", text="")
    ) == 0.0

    private_dir = Path(wrapper._temp_dir)
    wrapper.cleanup()
    assert not private_dir.exists()
    assert wrapper.client is None
    assert wrapper._temp_dir is None
