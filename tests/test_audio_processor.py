import builtins
import logging
import sys
import types
from pathlib import Path
from types import SimpleNamespace

from dubbing.core.cache_manager import CacheManager
from dubbing.debug.performance_tracker import PerformanceTracker


def test_process_background_audio_skips_when_audio_separator_missing(
    tmp_path, caplog, monkeypatch
):
    audio_file = tmp_path / "source.wav"
    audio_file.write_bytes(b"fake wav bytes")

    real_import = builtins.__import__

    def import_without_audio_separator(name, *args, **kwargs):
        if name == "audio_separator.separator":
            raise ModuleNotFoundError(
                "No module named 'audio_separator'", name="audio_separator"
            )
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", import_without_audio_separator)

    from dubbing.audio.audio_processor import AudioProcessor

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(audio_file)),
        PerformanceTracker(),
    )

    with caplog.at_level(logging.WARNING):
        background_audio = processor.process_background_audio(str(audio_file))

    assert background_audio is None
    assert "audio_separator" in caplog.text


def test_separate_background_and_vocals_returns_both_tracks(tmp_path):
    audio_file = tmp_path / "source.wav"
    audio_file.write_bytes(b"fake wav bytes")
    separated_background = tmp_path / "source_(Instrumental)_2_HP-UVR.wav"
    separated_vocals = tmp_path / "source_(Vocals)_2_HP-UVR.wav"
    separated_background.write_bytes(b"background")
    separated_vocals.write_bytes(b"vocals")

    class SeparatorStub:
        def load_model(self, model_filename):
            assert model_filename == "2_HP-UVR.pth"

        def separate(self, source_path):
            assert source_path == str(audio_file)
            return [str(separated_background), str(separated_vocals)]

    separator_module = types.ModuleType("audio_separator.separator")
    separator_module.Separator = SeparatorStub
    package_module = types.ModuleType("audio_separator")
    package_module.separator = separator_module
    sys.modules["audio_separator"] = package_module
    sys.modules["audio_separator.separator"] = separator_module

    from dubbing.audio.audio_processor import AudioProcessor

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(audio_file)),
        PerformanceTracker(),
        artifacts_root=str(tmp_path / "artifacts"),
    )

    background_audio, vocals_audio = processor.separate_background_and_vocals(str(audio_file))

    assert Path(background_audio).is_file()
    assert Path(vocals_audio).is_file()
    assert Path(background_audio).name == "background.wav"
    assert Path(vocals_audio).name == "vocals.wav"


def test_separate_background_and_vocals_disables_vr_progress_bars(tmp_path, monkeypatch):
    audio_file = tmp_path / "source.wav"
    audio_file.write_bytes(b"fake wav bytes")
    separated_background = tmp_path / "source_(Instrumental)_2_HP-UVR.wav"
    separated_background.write_bytes(b"background")
    progress = {"disabled": False}

    def tqdm_stub(iterable, *args, **kwargs):
        progress["disabled"] = kwargs.get("disable", False)
        return iterable

    vr_separator_module = types.ModuleType(
        "audio_separator.separator.architectures.vr_separator"
    )
    vr_separator_module.tqdm = tqdm_stub
    architectures_module = types.ModuleType("audio_separator.separator.architectures")
    architectures_module.vr_separator = vr_separator_module

    class SeparatorStub:
        def load_model(self, model_filename):
            assert model_filename == "2_HP-UVR.pth"

        def separate(self, source_path):
            assert source_path == str(audio_file)
            list(vr_separator_module.tqdm(range(1)))
            return [str(separated_background)]

    separator_module = types.ModuleType("audio_separator.separator")
    separator_module.Separator = SeparatorStub
    package_module = types.ModuleType("audio_separator")
    package_module.separator = separator_module
    monkeypatch.setitem(sys.modules, "audio_separator", package_module)
    monkeypatch.setitem(sys.modules, "audio_separator.separator", separator_module)
    monkeypatch.setitem(
        sys.modules, "audio_separator.separator.architectures", architectures_module
    )
    monkeypatch.setitem(
        sys.modules,
        "audio_separator.separator.architectures.vr_separator",
        vr_separator_module,
    )

    from dubbing.audio.audio_processor import AudioProcessor

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(audio_file)),
        PerformanceTracker(),
        artifacts_root=str(tmp_path / "artifacts"),
    )

    processor.separate_background_and_vocals(str(audio_file))

    assert progress["disabled"] is True


def test_audio_processor_creates_directories_inside_custom_artifacts_root(tmp_path):
    artifacts_root = tmp_path / "Are" / "artifacts"

    from dubbing.audio.audio_processor import AudioProcessor

    AudioProcessor(
        CacheManager(use_cache=False, input_file=str(tmp_path / "clip.mp4")),
        PerformanceTracker(),
        artifacts_root=str(artifacts_root),
    )

    expected_dirs = [
        artifacts_root / "audio",
        artifacts_root / "speakers_audio",
        artifacts_root / "audio_chunks",
        artifacts_root / "su_audio_chunks",
    ]

    assert all(path.is_dir() for path in expected_dirs)


def test_run_ffmpeg_command_uses_safe_text_decoding(tmp_path, monkeypatch):
    from dubbing.audio.audio_processor import AudioProcessor

    audio_file = tmp_path / "clip.mp4"
    audio_file.write_bytes(b"fake")

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(audio_file)),
        PerformanceTracker(),
    )

    captured = {}

    def fake_run(*args, **kwargs):
        captured["args"] = args
        captured["kwargs"] = kwargs
        return SimpleNamespace(stderr="")

    monkeypatch.setattr("dubbing.audio.audio_processor.subprocess.run", fake_run)

    processor._run_ffmpeg_command("ffmpeg -version")

    assert captured["args"] == ("ffmpeg -version",)
    assert captured["kwargs"]["text"] is True
    assert captured["kwargs"]["encoding"] == "utf-8"
    assert captured["kwargs"]["errors"] == "replace"


def test_extract_audio_handles_same_input_and_output_file(tmp_path, monkeypatch):
    from dubbing.audio.audio_processor import AudioProcessor

    artifacts_audio = tmp_path / "artifacts" / "audio"
    artifacts_audio.mkdir(parents=True)
    source_wav = artifacts_audio / "source.wav"
    source_wav.write_bytes(b"existing wav audio data")

    processor = AudioProcessor(
        CacheManager(use_cache=False, input_file=str(source_wav)),
        PerformanceTracker(),
        artifacts_root=str(tmp_path / "artifacts"),
    )

    # 1. When start_time is None and duration is None, should return source.wav directly without ffmpeg
    result_path = processor.extract_audio(str(source_wav), start_time=None, duration=None)
    assert Path(result_path).resolve() == source_wav.resolve()
    assert (artifacts_audio / "source.wav.json").exists()

    # 2. When start_time or duration is provided, ffmpeg should output to temp file first
    executed_commands = []

    def fake_subprocess_run(cmd, *args, **kwargs):
        executed_commands.append(cmd)
        # Create temp file so os.replace works
        temp_file = artifacts_audio / "source_extract_tmp.wav"
        temp_file.write_bytes(b"trimmed wav audio data")
        return SimpleNamespace(returncode=0, stdout=b"", stderr=b"")

    monkeypatch.setattr("dubbing.audio.audio_processor.subprocess.run", fake_subprocess_run)

    result_path = processor.extract_audio(str(source_wav), start_time=0.0, duration=120.0)
    assert Path(result_path).resolve() == source_wav.resolve()
    assert len(executed_commands) == 1
    # Verify input is source_wav and output is source_extract_tmp.wav (not same file)
    assert f'"{source_wav}"' in executed_commands[0]
    assert "source_extract_tmp.wav" in executed_commands[0]

