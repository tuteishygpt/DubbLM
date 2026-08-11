import importlib
import logging
import sys
from pathlib import Path


ROOT = Path(__file__).resolve().parents[1]
SRC = ROOT / "src"
if str(SRC) not in sys.path:
    sys.path.insert(0, str(SRC))


def test_omnivoice_prepare_text_ignores_style_prompt():
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.voice_prompt_mapping = {"SPEAKER_00": "warm and soft"}

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Толькі гэты тэкст.",
        style_prompt="do not use",
    )

    assert wrapper._prepare_text(segment) == "Толькі гэты тэкст."


def test_omnivoice_uses_segment_reference_text_for_predict(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.instruct = ""
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = "fallback text"
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_mode="configured",
        reference_audio_path=str(reference_audio),
        reference_text="Recognized original speech",
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["ref_text"] == "Recognized original speech"


def test_omnivoice_does_not_fallback_to_default_reference_text(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.instruct = ""
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = "fallback text"
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_mode="configured",
        reference_audio_path=str(reference_audio),
        reference_text=None,
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["ref_text"] == ""


def test_omnivoice_sends_empty_reference_text_when_value_is_blank(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.instruct = ""
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = "fallback text"
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_mode="configured",
        reference_audio_path=str(reference_audio),
        reference_text="   ",
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["ref_text"] == ""


def test_omnivoice_uses_segment_target_duration_for_du(tmp_path, monkeypatch):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    captured = {}

    class ClientStub:
        def predict(self, **kwargs):
            captured.update(kwargs)
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.instruct = ""
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = None
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        reference_mode="configured",
        reference_audio_path=str(reference_audio),
        reference_text="Recognized original speech",
        target_duration=1.75,
    )

    alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert captured["du"] == 1.75


def test_omnivoice_logs_reference_audio_name_and_segment_text(tmp_path, monkeypatch, caplog):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "speaker_ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    class ClientStub:
        def predict(self, **kwargs):
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1000

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.instruct = ""
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = None
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(module, "AudioSegment", type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}))

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Generated Belarusian text",
        reference_mode="configured",
        reference_audio_path=str(reference_audio),
        reference_text="Recognized original speech",
    )

    with caplog.at_level(logging.INFO, logger=module.logger.name):
        alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert "speaker_ref.wav" in caplog.text
    assert "Generated Belarusian text" in caplog.text


def test_omnivoice_logs_target_and_actual_duration_when_debug_tts_enabled(tmp_path, monkeypatch, caplog):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    reference_audio = tmp_path / "speaker_ref.wav"
    reference_audio.write_bytes(b"fake-audio")
    generated_audio = tmp_path / "generated.wav"
    generated_audio.write_bytes(b"RIFFfakeWAVE")

    class ClientStub:
        def predict(self, **kwargs):
            return str(generated_audio)

    class FakeAudioSegment:
        def __len__(self):
            return 1234

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.lang = "Belarusian"
    wrapper.instruct = ""
    wrapper.num_steps = 32
    wrapper.guidance_scale = 2.0
    wrapper.denoise = True
    wrapper.speed = 1.0
    wrapper.duration = 3.0
    wrapper.preprocess_prompt = True
    wrapper.postprocess_output = True
    wrapper.api_name = "/_clone_fn"
    wrapper.default_reference_audio = None
    wrapper.default_reference_text = None
    wrapper.voice_mapping = {}
    wrapper.voice_prompt_mapping = {}
    wrapper.debug_tts = True
    wrapper._temp_dir = str(tmp_path)

    monkeypatch.setattr(module, "handle_file", lambda path: f"handled:{path}")
    monkeypatch.setattr(
        module,
        "AudioSegment",
        type("AudioSegmentStub", (), {"from_file": staticmethod(lambda _path: FakeAudioSegment())}),
    )

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Generated Belarusian text",
        reference_mode="configured",
        reference_audio_path=str(reference_audio),
        reference_text="Recognized original speech",
        target_duration=1.75,
    )

    with caplog.at_level(logging.INFO, logger=module.logger.name):
        alignments = wrapper.synthesize([segment])

    assert len(alignments) == 1
    assert "target_duration=1.75s" in caplog.text
    assert "actual_duration=1.23s" in caplog.text


def test_omnivoice_uses_only_pipeline_resolved_reference_and_validates_before_predict(tmp_path):
    module = importlib.import_module("tts.omnivoice_wrapper")
    models = importlib.import_module("tts.models")

    class ClientStub:
        calls = 0

        def predict(self, **kwargs):
            self.calls += 1

    wrapper = object.__new__(module.OmniVoiceWrapper)
    wrapper.client = ClientStub()
    wrapper.voice_mapping = {"SPEAKER_00": str(tmp_path / "mapped.wav")}
    wrapper.default_reference_audio = str(tmp_path / "default.wav")

    segment = models.TTSSegmentData(
        speaker="SPEAKER_00",
        text="Synth text",
        segment_index=4,
        reference_mode="configured",
    )

    assert wrapper._resolve_reference_audio(segment) is None
    try:
        wrapper.synthesize([segment])
    except ValueError as exc:
        assert "reference file does not exist: <missing>" in str(exc)
    else:
        raise AssertionError("missing resolved reference must fail before predict")
    assert wrapper.client.calls == 0
