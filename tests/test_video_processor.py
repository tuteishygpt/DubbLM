import inspect
from types import SimpleNamespace

from dubbing.debug.performance_tracker import PerformanceTracker
from dubbing.video.video_processor import VideoProcessor


def test_combine_audio_with_video_uses_safe_pause_defaults():
    parameters = inspect.signature(VideoProcessor.combine_audio_with_video).parameters

    assert parameters["remove_pauses"].default is False
    assert parameters["min_pause_duration"].default == 300


def test_combine_audio_with_video_two_pass_maps_only_primary_video_stream(tmp_path, monkeypatch):
    video_path = tmp_path / "input.mp4"
    translated_audio_path = tmp_path / "dub.wav"
    background_audio_path = tmp_path / "bg.wav"
    output_path = tmp_path / "output.mp4"

    video_path.write_bytes(b"video")
    translated_audio_path.write_bytes(b"dub")
    background_audio_path.write_bytes(b"bg")

    processor = VideoProcessor(PerformanceTracker())
    captured = {}

    monkeypatch.setattr(
        "dubbing.video.video_processor.AudioProcessor.normalize_audio",
        lambda self, path: str(path),
    )
    monkeypatch.setattr(
        processor,
        "_get_video_info",
        lambda path: {
            "video_bitrate": "1659000",
            "video_codec": "h264",
            "audio_codec": "aac",
            "audio_bitrate": "128000",
            "video_profile": "high",
            "video_pix_fmt": "yuv420p",
        },
    )
    monkeypatch.setattr(processor, "_can_use_stream_copy_for_trim", lambda *args, **kwargs: False)

    def fake_two_pass(base_cmd, output_file, video_info):
        captured["base_cmd"] = base_cmd
        captured["output_file"] = output_file
        captured["video_info"] = video_info

    monkeypatch.setattr(processor, "_encode_with_two_pass", fake_two_pass)

    result_path, pause_adjustments = processor.combine_audio_with_video(
        video_path=str(video_path),
        translated_audio_path=str(translated_audio_path),
        background_audio_path=str(background_audio_path),
        output_file=str(output_path),
        start_time=0,
        normalize_audio=True,
        use_two_pass_encoding=True,
        remove_pauses=False,
        target_language="be",
        dubbed_volume=2.0,
    )

    assert result_path == str(output_path)
    assert pause_adjustments == []
    assert captured["output_file"] == str(output_path)
    assert captured["video_info"]["video_bitrate"] == "1659000"

    map_indexes = [i for i, token in enumerate(captured["base_cmd"]) if token == "-map"]
    mapped_streams = [captured["base_cmd"][i + 1] for i in map_indexes]
    assert "0:v:0" in mapped_streams
    assert "0:v" not in mapped_streams


def test_encode_with_two_pass_uses_safe_text_decoding(monkeypatch, tmp_path):
    processor = VideoProcessor(PerformanceTracker())
    calls = []

    def fake_run(*args, **kwargs):
        calls.append({"args": args, "kwargs": kwargs})
        return SimpleNamespace(stderr="")

    monkeypatch.setattr("dubbing.video.video_processor.subprocess.run", fake_run)

    processor._encode_with_two_pass(
        base_cmd=["ffmpeg", "-i", str(tmp_path / "input.mp4")],
        output_path=str(tmp_path / "output.mp4"),
        video_info={
            "video_bitrate": "1659000",
            "video_profile": "high",
            "video_pix_fmt": "yuv420p",
            "audio_codec": "aac",
            "audio_bitrate": "128000",
        },
    )

    assert len(calls) == 2
    for call in calls:
        assert call["kwargs"]["text"] is True
        assert call["kwargs"]["encoding"] == "utf-8"
        assert call["kwargs"]["errors"] == "replace"


def test_build_reencoding_cuts_command_remaps_original_audio_track_for_pause_removed_video(
    tmp_path, monkeypatch
):
    artifacts_root = tmp_path / "artifacts"
    video_path = tmp_path / "input.mp4"
    translated_audio_path = tmp_path / "dub.wav"
    background_audio_path = tmp_path / "bg.wav"
    output_path = tmp_path / "output.mp4"

    video_path.write_bytes(b"video")
    translated_audio_path.write_bytes(b"dub")
    background_audio_path.write_bytes(b"bg")

    processor = VideoProcessor(PerformanceTracker(), artifacts_root=str(artifacts_root))

    cut_translated_audio_path = tmp_path / "dub_cut.wav"
    cut_background_audio_path = tmp_path / "bg_cut.wav"
    cut_original_audio_path = tmp_path / "orig_cut.wav"
    cut_translated_audio_path.write_bytes(b"dub-cut")
    cut_background_audio_path.write_bytes(b"bg-cut")
    cut_original_audio_path.write_bytes(b"orig-cut")

    monkeypatch.setattr(
        "dubbing.video.video_processor.subprocess.run",
        lambda *args, **kwargs: SimpleNamespace(stdout="audio", stderr=""),
    )

    def fake_create_cut_audio_file(audio_path, cuts_to_keep):
        mapping = {
            str(translated_audio_path): str(cut_translated_audio_path),
            str(background_audio_path): str(cut_background_audio_path),
            str(video_path): str(cut_original_audio_path),
        }
        return mapping.get(str(audio_path))

    monkeypatch.setattr(processor, "_create_cut_audio_file", fake_create_cut_audio_file)

    original_command = [
        "ffmpeg",
        "-y",
        "-i",
        str(video_path),
        "-i",
        str(translated_audio_path),
        "-i",
        str(background_audio_path),
        "-filter_complex",
        "[1:a:0]volume=2[dubbed_vol_adj];[2:a:0]volume=0.762341[bg_audio_reduced];"
        "[dubbed_vol_adj][bg_audio_reduced]amix=inputs=2:duration=longest[dub_mixed_with_bg]",
        "-map",
        "0:v:0",
        "-map",
        "[dub_mixed_with_bg]",
        "-map",
        "0:a:0",
        "-shortest",
        str(output_path),
    ]

    modified_command, temp_files_to_cleanup = processor._build_reencoding_cuts_command(
        original_command=original_command,
        video_path=str(video_path),
        cuts_to_keep=[(0.0, 10.0)],
        output_path=str(output_path),
        use_two_pass_encoding=False,
        video_info={},
    )

    map_indexes = [i for i, token in enumerate(modified_command) if token == "-map"]
    mapped_streams = [modified_command[i + 1] for i in map_indexes]

    assert str(artifacts_root / "temp_video_with_cuts.mp4") in modified_command
    assert str(cut_original_audio_path) in modified_command
    assert "0:a:0" not in mapped_streams
    assert "3:a:0" in mapped_streams
    assert str(cut_original_audio_path) in temp_files_to_cleanup
