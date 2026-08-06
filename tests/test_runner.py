from pathlib import Path

from pydub import AudioSegment

import dubbing.core.config as config_module
import dubbing.core.runner as runner
from dubbing.core.config import create_argument_parser
from dubbing.core.smart_dubbing import SmartDubbing
from dubbing.core.runner import build_config_from_overrides, run_dubbing_job


def _patch_projects_root(monkeypatch, tmp_path):
    projects_root = tmp_path / "prj"
    monkeypatch.setattr(config_module, "DEFAULT_PROJECTS_ROOT", projects_root, raising=False)
    return projects_root


def test_build_config_from_overrides_parses_structured_fields(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    projects_root = _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "tts_system_mapping": '{"SPEAKER_00": "gemini"}',
            "glossary": '{"term": "translation"}',
            "voice_prompt": '{"SPEAKER_00": "calm"}',
            "keep_original_audio_ranges": ["00:01-00:03", "10-12"],
            "output": "",
        }
    )

    assert config.get("tts_system_mapping") == {"SPEAKER_00": "gemini"}
    assert config.get("glossary") == {"term": "translation"}
    assert config.get("voice_prompt") == {"SPEAKER_00": "calm"}
    assert config.get("keep_original_audio_ranges") == [(1.0, 3.0), (10.0, 12.0)]
    assert config.get("output") == str(projects_root / "clip" / "clip_be.mp4")


def test_build_config_from_overrides_treats_zero_duration_as_unset(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "start_time": 0,
            "duration": 0,
        }
    )

    assert config.get("start_time") == 0
    assert config.get("duration") is None


def test_build_config_from_overrides_clears_zero_duration_from_yaml(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    config_path = tmp_path / "config.yml"
    config_path.write_text(
        "source_language: en\n"
        "target_language: be\n"
        "start_time: 0\n"
        "duration: 0\n",
        encoding="utf-8",
    )
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "config": str(config_path),
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
        }
    )

    assert config.get("start_time") == 0
    assert config.get("duration") is None


def test_run_dubbing_job_returns_pipeline_output(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "dubbed.mp4"

    class FakeDubber:
        def __init__(self, config):
            self.config = config

        def run_pipeline(self, save_original_subtitles=False, save_translated_subtitles=False):
            return str(output_path)

    result = run_dubbing_job(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
        },
        dubbing_factory=FakeDubber,
    )

    assert result.status == "Completed"
    assert result.output_file == str(output_path)
    assert result.report_file is None


def test_run_dubbing_job_loads_dotenv_before_constructing_dubber(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "dubbed.mp4"
    calls = []

    def fake_load_dotenv(**kwargs):
        calls.append(("dotenv", kwargs))

    class FakeDubber:
        def __init__(self, config):
            calls.append("dubber")
            self.config = config

        def run_pipeline(self, save_original_subtitles=False, save_translated_subtitles=False):
            return str(output_path)

    monkeypatch.setattr(runner, "load_dotenv", fake_load_dotenv, raising=False)

    result = run_dubbing_job(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
        },
        dubbing_factory=FakeDubber,
    )

    assert result.status == "Completed"
    assert calls == [("dotenv", {"override": True}), "dubber"]


def test_run_dubbing_job_returns_speaker_report_paths(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    report_path = tmp_path / "report.txt"
    samples_path = tmp_path / "samples"
    samples_path.mkdir()

    class FakeDubber:
        def __init__(self, config):
            self.config = config

        def generate_diarization_report(self):
            return str(report_path), str(samples_path)

    result = run_dubbing_job(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "generate_speaker_report": True,
        },
        dubbing_factory=FakeDubber,
    )

    assert result.status == "Speaker report generated"
    assert result.output_file == str(samples_path)
    assert result.report_file == str(report_path)


def test_run_dubbing_job_explains_missing_combine_video_artifacts(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")

    class FakeDubber:
        def __init__(self, config):
            self.config = config
            self.video_processor = object()

    result = run_dubbing_job(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "run_step": "combine_video",
        },
        dubbing_factory=FakeDubber,
    )

    assert result.status.startswith("Failed:")
    assert "combine_video" in result.status
    assert "Run step" in result.status


def test_run_dubbing_job_extracts_file_path_from_combine_video_tuple(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "clip_be.mp4"

    class FakeDubber:
        def __init__(self, config):
            self.config = config

    monkeypatch.setattr(
        runner,
        "_run_combine_video_step",
        lambda dubber, config: (str(output_path), []),
    )

    result = run_dubbing_job(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "run_step": "combine_video",
        },
        dubbing_factory=FakeDubber,
    )

    assert result.status == "Combine step completed"
    assert result.output_file == str(output_path)


def test_run_dubbing_job_routes_tts_to_end_step(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "clip_be.mp4"

    class FakeDubber:
        def __init__(self, config):
            self.config = config

        def run_from_tts(self, save_original_subtitles=False, save_translated_subtitles=False):
            assert save_original_subtitles is True
            assert save_translated_subtitles is False
            return str(output_path)

    result = run_dubbing_job(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "run_step": "tts_to_end",
            "save_original_subtitles": True,
        },
        dubbing_factory=FakeDubber,
    )

    assert result.status == "TTS resume step completed"
    assert result.output_file == str(output_path)


def test_run_dubbing_job_treats_full_pipeline_step_as_normal_run(tmp_path):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    output_path = tmp_path / "clip_be.mp4"

    class FakeDubber:
        def __init__(self, config):
            self.config = config

        def run_pipeline(self, save_original_subtitles=False, save_translated_subtitles=False):
            return str(output_path)

    result = run_dubbing_job(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "run_step": "full_pipeline",
        },
        dubbing_factory=FakeDubber,
    )

    assert result.status == "Completed"
    assert result.output_file == str(output_path)


def test_argument_parser_accepts_omnivoice_tts_system():
    parser = create_argument_parser()

    args = parser.parse_args(
        [
            "--input",
            "clip.mp4",
            "--source_language",
            "en",
            "--target_language",
            "be",
            "--tts_system",
            "omnivoice",
        ]
    )

    assert args.tts_system == "omnivoice"


def test_argument_parser_accepts_gemini_transcription_backend_and_model():
    parser = create_argument_parser()

    args = parser.parse_args(
        [
            "--input",
            "clip.mp4",
            "--source_language",
            "en",
            "--target_language",
            "be",
            "--transcription_system",
            "gemini",
            "--gemini_transcription_model",
            "gemini-2.5-flash",
        ]
    )

    assert args.transcription_system == "gemini"
    assert args.gemini_transcription_model == "gemini-2.5-flash"


def test_argument_parser_accepts_deepgram_transcription_backend():
    parser = create_argument_parser()

    args = parser.parse_args(
        [
            "--input",
            "clip.mp4",
            "--source_language",
            "ru",
            "--target_language",
            "be",
            "--transcription_system",
            "deepgram",
        ]
    )

    assert args.transcription_system == "deepgram"


def test_argument_parser_accepts_tts_to_end_run_step():
    parser = create_argument_parser()

    args = parser.parse_args(
        [
            "--input",
            "clip.mp4",
            "--source_language",
            "en",
            "--target_language",
            "be",
            "--run_step",
            "tts_to_end",
        ]
    )

    assert args.run_step == "tts_to_end"


def test_argument_parser_accepts_full_pipeline_run_step():
    parser = create_argument_parser()

    args = parser.parse_args(
        [
            "--input",
            "clip.mp4",
            "--source_language",
            "en",
            "--target_language",
            "be",
            "--run_step",
            "full_pipeline",
        ]
    )

    assert args.run_step == "full_pipeline"


def test_build_config_from_overrides_preserves_gemini_transcription_model(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "transcription_system": "gemini",
            "gemini_transcription_model": "gemini-2.5-flash",
        }
    )

    assert config.get("transcription_system") == "gemini"
    assert config.get("gemini_transcription_model") == "gemini-2.5-flash"


def test_build_config_from_overrides_defaults_gemini_transcription_model(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "transcription_system": "gemini",
        }
    )

    assert config.get("gemini_transcription_model") == "gemini-3-flash-preview"


def test_build_config_from_overrides_defaults_omnivoice_language_to_belarusian(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "tts_system": "omnivoice",
        }
    )

    assert config.get("omnivoice_lang") == "Belarusian"


def test_build_config_from_overrides_places_outputs_inside_project_dir(tmp_path, monkeypatch):
    video_path = tmp_path / "Are.mp4"
    video_path.write_bytes(b"video")
    projects_root = _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
        }
    )

    project_dir = projects_root / "Are"

    assert config.get("project_dir") == str(project_dir)
    assert config.get("artifacts_dir") == str(project_dir / "artifacts")
    assert config.get("output") == str(project_dir / "Are_be.mp4")
    assert project_dir.is_dir()


def test_build_config_from_overrides_appends_missing_output_extension(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)
    custom_output = tmp_path / "катс2"

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "output": str(custom_output),
        }
    )

    assert config.get("output") == str(custom_output.with_suffix(".mp4"))


def test_smart_dubbing_subtitles_default_to_project_dir(tmp_path, monkeypatch):
    video_path = tmp_path / "Are.mp4"
    video_path.write_bytes(b"video")
    projects_root = _patch_projects_root(monkeypatch, tmp_path)
    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
        }
    )

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config

    assert dubber._get_subtitle_path("original", str(video_path), "en") == str(projects_root / "Are" / "Are_en.srt")
    assert dubber._get_subtitle_path("translation", str(video_path), "be") == str(projects_root / "Are" / "Are_be.srt")


def test_translate_segments_passes_project_debug_paths_to_translator(tmp_path, monkeypatch):
    video_path = tmp_path / "Are.mp4"
    video_path.write_bytes(b"video")
    projects_root = _patch_projects_root(monkeypatch, tmp_path)
    audio_path = projects_root / "Are" / "artifacts" / "audio" / "source.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
        }
    )

    class TranslatorStub:
        def __init__(self):
            self.kwargs = None

        def is_available(self):
            return True

        def translate(self, **kwargs):
            self.kwargs = kwargs
            return []

    class PerfStub:
        def start_timing(self, *_args, **_kwargs):
            return None

        def end_timing(self, *_args, **_kwargs):
            return 0.0

        def record_metric(self, *_args, **_kwargs):
            return None

        def write_performance_summary(self, *_args, **_kwargs):
            return None

        def record_metric(self, *_args, **_kwargs):
            return None

        def record_metric(self, *_args, **_kwargs):
            return None

        def record_metric(self, *_args, **_kwargs):
            return None

        def record_metric(self, *_args, **_kwargs):
            return None

    class CacheStub:
        def generate_cache_key(self, *_args, **_kwargs):
            return "cache-key"

        def cache_exists(self, *_args, **_kwargs):
            return False

        def save_to_cache(self, *_args, **_kwargs):
            return None

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.cache_manager = CacheStub()
    dubber.performance_tracker = PerfStub()
    dubber.debug_data = {}
    translator = TranslatorStub()
    dubber._require_translator = lambda: translator

    dubber.translate_segments([], str(audio_path))

    assert translator.kwargs["debug_dir"] == str(projects_root / "Are" / "artifacts" / "debug" / "translation")
    assert translator.kwargs["timecodes_report_path"] == str(projects_root / "Are" / "artifacts" / "timecodes.txt")


def test_translate_segments_temporarily_adds_stress_marks_requirement_to_prompt(tmp_path, monkeypatch):
    video_path = tmp_path / "Are.mp4"
    video_path.write_bytes(b"video")
    projects_root = _patch_projects_root(monkeypatch, tmp_path)
    audio_path = projects_root / "Are" / "artifacts" / "audio" / "source.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    audio_path.write_bytes(b"audio")

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "translation_prompt_prefix": "Keep idioms natural.",
        }
    )

    class TranslatorStub:
        def __init__(self):
            self.kwargs = None
            self.prompt_prefix = "Keep idioms natural."
            self.prompt_prefix_at_call = None

        def is_available(self):
            return True

        def translate(self, **kwargs):
            self.kwargs = kwargs
            self.prompt_prefix_at_call = self.prompt_prefix
            return []

    class PerfStub:
        def start_timing(self, *_args, **_kwargs):
            return None

        def end_timing(self, *_args, **_kwargs):
            return 0.0

        def record_metric(self, *_args, **_kwargs):
            return None

        def write_performance_summary(self, *_args, **_kwargs):
            return None

    class CacheStub:
        def generate_cache_key(self, *_args, **_kwargs):
            return "cache-key"

        def cache_exists(self, *_args, **_kwargs):
            return False

        def save_to_cache(self, *_args, **_kwargs):
            return None

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.cache_manager = CacheStub()
    dubber.performance_tracker = PerfStub()
    dubber.debug_data = {}
    translator = TranslatorStub()
    dubber._require_translator = lambda: translator

    dubber.translate_segments([], str(audio_path))

    assert translator.prompt_prefix == "Keep idioms natural."
    assert "Keep idioms natural." in translator.prompt_prefix_at_call
    assert "Use the combining acute accent symbol U+0301" in translator.prompt_prefix_at_call
    assert "каса́" in translator.prompt_prefix_at_call


def test_segment_reference_clip_uses_segment_transcription_as_reference_text(tmp_path):
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.speakers_audio_dir = tmp_path / "speakers_audio"

    base_args = {
        "speaker": "SPEAKER_00",
        "text": "Translated text",
        "reference_audio_path": None,
        "reference_text": None,
    }
    segment_dict = {
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 1.5,
        "text": "Recognized original speech",
        "translation": "Translated text",
    }
    original_audio = AudioSegment.silent(duration=2000)

    updated_args, returned_audio = dubber._attach_segment_reference(
        tts_segment_data_args=base_args,
        segment_dict=segment_dict,
        speaker="SPEAKER_00",
        segment_index=0,
        original_audio_segment=original_audio,
        segment_reference_min_duration=1.0,
        segment_reference_min_duration_ms=1000,
    )

    assert updated_args["reference_text"] == "Recognized original speech"
    assert Path(updated_args["reference_audio_path"]).is_file()
    assert returned_audio is original_audio


def test_manual_speaker_reference_mapping_overrides_auto_segment_reference(tmp_path):
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "reference_audio_mapping": {"SPEAKER_00": "D:/voices/manual.wav"},
        "reference_text_mapping": {"SPEAKER_00": "Manual reference text"},
    }

    base_args = {
        "speaker": "SPEAKER_00",
        "text": "Translated text",
        "reference_audio_path": str(tmp_path / "auto.wav"),
        "reference_text": "Auto reference text",
    }

    updated_args = dubber._apply_configured_reference_mapping(base_args, "SPEAKER_00")

    assert updated_args["reference_audio_path"] == "D:/voices/manual.wav"
    assert updated_args["reference_text"] == "Manual reference text"


def test_adjust_and_combine_audio_grouped_handles_none_synthesized_speech_file(tmp_path):
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"group_overflow_tolerance": 1.0}
    dubber.audio_chunks_dir = tmp_path / "audio_chunks"
    dubber.su_audio_chunks_dir = tmp_path / "su_audio_chunks"
    dubber.audio_chunks_dir.mkdir(parents=True, exist_ok=True)
    dubber.su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)
    dubber.debug_data = {}

    segments = [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 2.0,
            "text": "Hello",
            "translation": "Привет",
            "synthesized_speech_file": None,
        },
    ]

    combined_audio, positions = dubber._adjust_and_combine_audio_grouped(segments)
    assert len(combined_audio) >= 2000
    assert len(positions) == 1


def test_segment_reference_clip_takes_priority_over_speaker_wav_and_keeps_matching_text(tmp_path):
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "reference_audio_mapping": {},
        "reference_text_mapping": {},
    }
    dubber.reference_audio_mapping = {}
    dubber.reference_text_mapping = {}
    dubber.speakers_audio_dir = tmp_path / "speakers_audio"
    dubber.speakers_audio_dir.mkdir(parents=True, exist_ok=True)
    speaker_wav = dubber.speakers_audio_dir / "SPEAKER_00.wav"
    speaker_wav.write_bytes(b"speaker-audio")

    base_args = {
        "speaker": "SPEAKER_00",
        "text": "Translated text",
        "reference_audio_path": None,
        "reference_text": None,
    }
    segment_dict = {
        "speaker": "SPEAKER_00",
        "start": 0.0,
        "end": 1.5,
        "text": "Recognized original speech",
        "translation": "Translated text",
    }
    original_audio = AudioSegment.silent(duration=2000)

    updated_args, returned_audio = dubber._apply_reference_fallbacks(
        tts_segment_data_args=base_args,
        segment_dict=segment_dict,
        speaker="SPEAKER_00",
        segment_index=0,
        original_audio_segment=original_audio,
        segment_reference_min_duration=1.0,
        segment_reference_min_duration_ms=1000,
    )

    assert updated_args["reference_audio_path"] != str(speaker_wav)
    assert Path(updated_args["reference_audio_path"]).is_file()
    assert Path(updated_args["reference_audio_path"]).name == "SPEAKER_00_0.wav"
    assert updated_args["reference_text"] == "Recognized original speech"
    assert returned_audio is original_audio


def test_run_pipeline_uses_separated_vocals_for_segment_references_when_keep_background_enabled(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)
    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "keep_background": True,
        }
    )

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.muted_speakers = set()
    dubber.debug_data = {}
    dubber.pause_adjustments = []
    call_order = []

    class PerfStub:
        def start_timing(self, *_args, **_kwargs):
            return None

        def end_timing(self, *_args, **_kwargs):
            return 0.0

        def record_metric(self, *_args, **_kwargs):
            return None

        def write_performance_summary(self, *_args, **_kwargs):
            return None

    class AudioProcessorStub:
        def extract_audio(self, *_args, **_kwargs):
            call_order.append("extract")
            return str(tmp_path / "source.wav")

        def separate_background_and_vocals(self, audio_file):
            call_order.append(("separate", audio_file))
            return str(tmp_path / "background.wav"), str(tmp_path / "vocals.wav")

        def get_total_duration(self):
            return 2.0

    class SpeakerProcessorStub:
        def extract_speaker_audio(self, *_args, **_kwargs):
            call_order.append("extract_speaker_audio")
            return {"SPEAKER_00": str(tmp_path / "speaker.wav")}

        def save_translated_samples(self, *_args, **_kwargs):
            call_order.append("save_samples")
            return None

    class SubtitleManagerStub:
        def save_debug_tsv(self, *_args, **_kwargs):
            return None

        def save_subtitles(self, *_args, **_kwargs):
            return None

    class VideoProcessorStub:
        def combine_audio_with_video(self, **kwargs):
            call_order.append(("combine", kwargs["background_audio_path"]))
            return str(tmp_path / "output.mp4"), []

    dubber.performance_tracker = PerfStub()
    dubber.audio_processor = AudioProcessorStub()
    dubber.speaker_processor = SpeakerProcessorStub()
    dubber.subtitle_manager = SubtitleManagerStub()
    dubber.video_processor = VideoProcessorStub()
    dubber.debug_generator = object()
    dubber.tts_systems = {}
    dubber.audio_chunks_dir = tmp_path / "audio_chunks"
    dubber.su_audio_chunks_dir = tmp_path / "su_audio_chunks"

    segments = [
        {
            "speaker": "SPEAKER_00",
            "start": 0.0,
            "end": 1.5,
            "text": "hello",
            "translation": "priviet",
        }
    ]

    dubber.diarize_and_transcribe = lambda _audio_file: ({(0.0, 1.5): "SPEAKER_00"}, segments)
    dubber.translate_segments = lambda transcription, _audio_file: transcription

    def synthesize_speech(_segments, _speakers_rolls, reference_audio_file):
        call_order.append(("synthesize", reference_audio_file))
        return str(tmp_path / "dubbed.wav")

    dubber.synthesize_speech = synthesize_speech

    output_path = dubber.run_pipeline()

    assert output_path == str(tmp_path / "output.mp4")
    assert call_order.index(("separate", str(tmp_path / "source.wav"))) < call_order.index(
        ("synthesize", str(tmp_path / "vocals.wav"))
    )


def test_segment_target_duration_matches_original_segment_length():
    segment_dict = {
        "start": 10.25,
        "end": 12.0,
        "translation": "Translated text",
        "speaker": "SPEAKER_00",
    }

    tts_segment_data_args = {
        "speaker": segment_dict["speaker"],
        "text": segment_dict["translation"],
        "target_duration": max(segment_dict["end"] - segment_dict["start"], 0.0),
    }

    assert tts_segment_data_args["target_duration"] == 1.75


def test_run_from_tts_uses_cached_translation_and_creates_video(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)
    config = build_config_from_overrides(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "enable_emotion_analysis": False,
        }
    )

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.muted_speakers = set()
    dubber.debug_data = {}
    dubber.pause_adjustments = []

    class PerfStub:
        def start_timing(self, *_args, **_kwargs):
            return None

        def end_timing(self, *_args, **_kwargs):
            return 0.0

        def record_metric(self, *_args, **_kwargs):
            return None

        def write_performance_summary(self, *_args, **_kwargs):
            return None

    class AudioProcessorStub:
        def extract_audio(self, *_args, **_kwargs):
            return str(tmp_path / "source.wav")

        def get_total_duration(self):
            return 3.0

    class SpeakerProcessorStub:
        def save_translated_samples(self, segments, audio_file):
            assert len(segments) == 1
            assert audio_file == str(tmp_path / "source.wav")
            return None

    class SubtitleManagerStub:
        def save_debug_tsv(self, segments, output_dir=None):
            assert len(segments) == 1
            return None

        def save_subtitles(self, *_args, **_kwargs):
            return None

    class VideoProcessorStub:
        def combine_audio_with_video(self, **kwargs):
            assert kwargs["translated_audio_path"] == str(tmp_path / "dubbed.wav")
            assert kwargs["video_path"] == str(video_path)
            return str(tmp_path / "output.mp4"), []

    class CacheStub:
        def __init__(self):
            self.seen = []
            self.cleared = []
            self.use_cache = True

        def generate_cache_key(self, *_args, **_kwargs):
            return "cache-key"

        def cache_exists(self, step_name, cache_key):
            self.seen.append((step_name, cache_key))
            return step_name == "translation"

        def load_from_cache(self, step_name, cache_key):
            self.seen.append((step_name, cache_key, "load"))
            assert step_name == "translation"
            return [
                {
                    "speaker": "SPEAKER_00",
                    "start": 0.0,
                    "end": 1.2,
                    "text": "hello",
                    "translation": "прывітанне",
                }
            ]

        def clear_cache(self, step_name=None):
            self.cleared.append(step_name)

    dubber.performance_tracker = PerfStub()
    dubber.audio_processor = AudioProcessorStub()
    dubber.speaker_processor = SpeakerProcessorStub()
    dubber.subtitle_manager = SubtitleManagerStub()
    dubber.video_processor = VideoProcessorStub()
    dubber.cache_manager = CacheStub()

    synth_calls = []

    def synthesize_speech(segments, speakers_rolls, audio_file):
        assert dubber.cache_manager.use_cache is False
        synth_calls.append((segments, speakers_rolls, audio_file))
        return str(tmp_path / "dubbed.wav")

    dubber.synthesize_speech = synthesize_speech

    output_path = dubber.run_from_tts()

    assert output_path == str(tmp_path / "output.mp4")
    assert synth_calls[0][1] == {(0.0, 1.2): "SPEAKER_00"}
    assert synth_calls[0][2] == str(tmp_path / "source.wav")
    assert dubber.cache_manager.seen[0][0] == "translation"
    assert "synthesized_speech" in dubber.cache_manager.cleared
    assert "segment_synthesis" in dubber.cache_manager.cleared
    assert dubber.cache_manager.use_cache is True


def test_run_translate_only_requires_cached_transcription(tmp_path, monkeypatch):
    """translate_only is a resume step: it must fail rather than silently
    re-run the transcriber when the transcription cache is missing.

    Regression guard for the previous behaviour where translate_only wiped the
    cache and re-ran diarize+transcribe from scratch, burning API credits.
    """
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)
    config = build_config_from_overrides(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
        }
    )

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.debug_data = {"diarization": None, "transcription": None}
    dubber.muted_speakers = set()
    dubber.pause_adjustments = []

    class PerfStub:
        def start_timing(self, *_args, **_kwargs):
            return None

        def end_timing(self, *_args, **_kwargs):
            return 0.0

        def record_metric(self, *_args, **_kwargs):
            return None

    class AudioProcessorStub:
        def extract_audio(self, *_args, **_kwargs):
            return str(tmp_path / "source.wav")

    class CacheStub:
        use_cache = True

        def generate_cache_key(self, *_args, **_kwargs):
            return "cache-key"

        def cache_exists(self, step_name, cache_key):
            return False

    class TranscriberStub:
        name = "AssemblyAI"
        cache_step_name = "assemblyai_diarization_transcription"

        def __init__(self):
            self.called = False

        def diarize_and_transcribe(self, *_args, **_kwargs):
            self.called = True
            raise AssertionError("translate_only must not re-run the transcriber")

    stub_transcriber = TranscriberStub()
    dubber.performance_tracker = PerfStub()
    dubber.audio_processor = AudioProcessorStub()
    dubber.cache_manager = CacheStub()
    dubber.transcriber = stub_transcriber
    dubber.transcriber_init_error = None
    dubber._cleanup = lambda: None

    raised = False
    try:
        dubber.run_translate_only()
    except FileNotFoundError as exc:
        raised = True
        assert "translate_only" in str(exc)
        assert "transcribe_only" in str(exc)
    assert raised, "Expected FileNotFoundError when transcription cache is missing"
    assert stub_transcriber.called is False


def test_run_from_tts_requires_cached_translation(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)
    config = build_config_from_overrides(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "enable_emotion_analysis": False,
        }
    )

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config

    class AudioProcessorStub:
        def extract_audio(self, *_args, **_kwargs):
            return str(tmp_path / "source.wav")

    class CacheStub:
        use_cache = True

        def generate_cache_key(self, *_args, **_kwargs):
            return "cache-key"

        def cache_exists(self, *_args, **_kwargs):
            return False

    dubber.audio_processor = AudioProcessorStub()
    dubber.cache_manager = CacheStub()

    try:
        dubber.run_from_tts()
    except FileNotFoundError as exc:
        assert "translation" in str(exc)
        assert "full dubbing run" in str(exc)
    else:
        raise AssertionError("Expected FileNotFoundError when translation cache is missing")
