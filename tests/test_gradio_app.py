import pickle
from pathlib import Path
from types import SimpleNamespace

import gradio as gr
from pydub import AudioSegment
import yaml

import dubbing.ui.gradio_app as gradio_app
from dubbing.ui.gradio_app import DEFAULT_CONFIG_PATH, build_app, load_ui_defaults, save_settings
from dubbing.core.cache_manager import CacheManager
from dubbing.core.runner import build_config_from_overrides
from dubbing.core.smart_dubbing import SmartDubbing
import dubbing.core.config as config_module


def test_ui_defaults_keep_two_speaker_labels_after_page_load(tmp_path):
    config_path = tmp_path / "config.yaml"
    config_path.write_text("{}\n", encoding="utf-8")

    defaults = load_ui_defaults(str(config_path))

    assert defaults["isolated_tracks_labels"] == "SPEAKER_00, SPEAKER_01"


def test_build_app_returns_gradio_blocks():
    app = build_app()

    assert isinstance(app, gr.Blocks)


def _component_value_by_label(app, label):
    for component in app.config["components"]:
        props = component.get("props", {})
        if props.get("label") == label:
            value = props.get("value")
            if isinstance(value, dict) and "data" in value:
                return value["data"]
            return value
    raise AssertionError(f"Component with label {label!r} not found")


def _component_props_by_label(app, label):
    for component in app.config["components"]:
        props = component.get("props", {})
        if props.get("label") == label:
            return props
    raise AssertionError(f"Component with label {label!r} not found")


def test_build_app_uses_yaml_defaults(tmp_path):
    config_path = tmp_path / "ui_defaults.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "source_language": "pl",
                "target_language": "be",
                "save_original_subtitles": True,
                "keep_background": True,
                "transcription_system": "assemblyai",
                "tts_system": "bextts",
                "debug_tts": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    app = build_app(config_path=str(config_path))

    assert _component_value_by_label(app, "Source language") == "pl"
    assert _component_value_by_label(app, "Target language") == "be"
    assert _component_value_by_label(app, "Save original subtitles") is True
    assert _component_value_by_label(app, "Keep background audio") is True
    assert _component_value_by_label(app, "Transcription system") == "assemblyai"
    assert _component_value_by_label(app, "TTS system") == "bextts"
    assert _component_value_by_label(app, "Debug TTS") is True


def test_build_app_lists_omnivoice_in_tts_system_choices():
    app = build_app()

    tts_props = _component_props_by_label(app, "TTS system")
    choice_values = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in tts_props["choices"]]

    assert "omnivoice" in choice_values


def test_build_app_lists_gemini_in_transcription_system_choices():
    app = build_app()

    transcription_props = _component_props_by_label(app, "Transcription system")
    choice_values = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in transcription_props["choices"]]

    assert "gemini" in choice_values


def test_build_app_lists_deepgram_in_transcription_system_choices():
    app = build_app()

    transcription_props = _component_props_by_label(app, "Transcription system")
    choice_values = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in transcription_props["choices"]]

    assert "deepgram" in choice_values


def test_build_app_exposes_transcription_model_field(tmp_path):
    config_path = tmp_path / "ui_defaults.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "transcription_system": "gemini",
                "transcription_model": "gemini-2.5-flash",
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    app = build_app(config_path=str(config_path))

    assert _component_value_by_label(app, "Model") == "gemini-2.5-flash"


def test_build_app_explains_run_step_requires_existing_artifacts():
    app = build_app()

    run_step_props = _component_props_by_label(app, "Run step")

    assert "existing artifacts" in run_step_props["info"]


def test_build_app_lists_tts_to_end_in_run_step_choices():
    app = build_app()

    run_step_props = _component_props_by_label(app, "Run step")
    choice_values = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in run_step_props["choices"]]

    assert "tts_to_end" in choice_values
    assert "TTS" in run_step_props["info"]


def test_build_app_defaults_run_step_to_full_pipeline():
    app = build_app()

    run_step_props = _component_props_by_label(app, "Run step")
    choice_values = [choice[1] if isinstance(choice, (list, tuple)) else choice for choice in run_step_props["choices"]]

    assert run_step_props["value"] == "full_pipeline"
    for expected in [
        "full_pipeline",
        "from_scratch",
        "transcribe_only",
        "translate_only",
        "combine_video",
        "tts_to_end",
    ]:
        assert expected in choice_values


def test_build_app_exposes_dubbing_texts_editor():
    app = build_app()

    dubbing_texts_props = _component_props_by_label(app, "Dubbing texts")

    assert dubbing_texts_props["headers"] == [
        "Speaker",
        "Start",
        "End",
        "Original",
        "Translation",
        "Synthesized text",
        "Audio file",
    ]
    assert dubbing_texts_props["column_widths"] == ["9%", "8%", "8%", "20%", "25%", "22%", "8%"]
    assert dubbing_texts_props["static_columns"] == [6]
    assert dubbing_texts_props["show_search"] == "search"
    assert dubbing_texts_props["pinned_columns"] == 3


def test_save_settings_writes_to_default_config_and_preserves_other_keys(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)
    initial_config_path = tmp_path / DEFAULT_CONFIG_PATH
    initial_config_path.write_text(
        yaml.safe_dump(
            {
                "source_language": "en",
                "target_language": "be",
                "normalize_audio": True,
            },
            sort_keys=False,
        ),
        encoding="utf-8",
    )

    status = save_settings(
        {
            "input": "clip.mp4",
            "output": "result.mp4",
            "config": "custom.yml",
            "run_step": "combine_video",
            "generate_speaker_report": True,
            "source_language": "de",
            "target_language": "uk",
            "keep_background": True,
            "save_translated_subtitles": True,
            "tts_system": "bextts",
        }
    )

    saved_data = yaml.safe_load(initial_config_path.read_text(encoding="utf-8"))

    assert status == f"Settings saved to {DEFAULT_CONFIG_PATH}"
    assert saved_data["source_language"] == "de"
    assert saved_data["target_language"] == "uk"
    assert saved_data["keep_background"] is True
    assert saved_data["save_translated_subtitles"] is True
    assert saved_data["tts_system"] == "bextts"
    assert saved_data["normalize_audio"] is True
    assert "input" not in saved_data
    assert "output" not in saved_data
    assert "config" not in saved_data
    assert "run_step" not in saved_data
    assert "generate_speaker_report" not in saved_data
    assert not (tmp_path / "custom.yml").exists()


def test_build_app_formats_structured_yaml_values_for_text_inputs(tmp_path):
    config_path = tmp_path / "structured_defaults.yml"
    config_path.write_text(
        yaml.safe_dump(
            {
                "glossary": {"AI": "ШІ"},
                "voice_prompt": {"SPEAKER_00": "warm"},
                "tts_system_mapping": {"SPEAKER_00": "gemini"},
                "reference_audio_mapping": {
                    "SPEAKER_00": "D:/voices/speaker_00.wav",
                    "SPEAKER_01": "D:/voices/speaker_01.wav",
                },
                "reference_text_mapping": {
                    "SPEAKER_00": "First speaker reference",
                    "SPEAKER_01": "Second speaker reference",
                },
                "keep_original_audio_ranges": ["00:10-00:15", "01:02-01:08"],
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )

    app = build_app(config_path=str(config_path))

    assert _component_value_by_label(app, "Glossary JSON") == '{\n  "AI": "ШІ"\n}'
    assert _component_value_by_label(app, "Voice prompt JSON") == '{\n  "SPEAKER_00": "warm"\n}'
    assert _component_value_by_label(app, "TTS system mapping JSON") == '{\n  "SPEAKER_00": "gemini"\n}'
    assert _component_value_by_label(app, "Speaker reference mappings") == [
        ["SPEAKER_00", "D:/voices/speaker_00.wav", "First speaker reference"],
        ["SPEAKER_01", "D:/voices/speaker_01.wav", "Second speaker reference"],
    ]
    assert _component_value_by_label(app, "Keep original audio ranges") == "00:10-00:15\n01:02-01:08"


def test_save_settings_parses_structured_text_fields(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    status = save_settings(
        {
            "glossary": '{\n  "AI": "ШІ"\n}',
            "voice_prompt": '{"SPEAKER_00": "warm"}',
            "tts_system_mapping": '{"SPEAKER_00": "gemini"}',
            "speaker_reference_rows": [
                ["SPEAKER_00", "D:/voices/speaker_00.wav", "First speaker reference"],
                ["SPEAKER_01", "D:/voices/speaker_01.wav", "Second speaker reference"],
                ["", "", ""],
            ],
            "keep_original_audio_ranges": "00:10-00:15\n01:02-01:08",
        }
    )

    saved_data = yaml.safe_load((tmp_path / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))

    assert status == f"Settings saved to {DEFAULT_CONFIG_PATH}"
    assert saved_data["glossary"] == {"AI": "ШІ"}
    assert saved_data["voice_prompt"] == {"SPEAKER_00": "warm"}
    assert saved_data["tts_system_mapping"] == {"SPEAKER_00": "gemini"}
    assert saved_data["reference_audio_mapping"] == {
        "SPEAKER_00": "D:/voices/speaker_00.wav",
        "SPEAKER_01": "D:/voices/speaker_01.wav",
    }
    assert saved_data["reference_text_mapping"] == {
        "SPEAKER_00": "First speaker reference",
        "SPEAKER_01": "Second speaker reference",
    }
    assert saved_data["keep_original_audio_ranges"] == ["00:10-00:15", "01:02-01:08"]


def test_save_settings_drops_zero_duration(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    save_settings(
        {
            "source_language": "en",
            "target_language": "be",
            "start_time": 0,
            "duration": 0,
        }
    )

    saved_data = yaml.safe_load((tmp_path / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))

    assert saved_data["start_time"] == 0
    assert "duration" not in saved_data


def test_save_settings_persists_transcription_model(tmp_path, monkeypatch):
    monkeypatch.chdir(tmp_path)

    save_settings(
        {
            "transcription_system": "gemini",
            "transcription_model": "gemini-2.5-flash",
        }
    )

    saved_data = yaml.safe_load((tmp_path / DEFAULT_CONFIG_PATH).read_text(encoding="utf-8"))

    assert saved_data["transcription_system"] == "gemini"
    assert saved_data["transcription_model"] == "gemini-2.5-flash"


def test_save_speaker_reference_to_library_copies_audio_and_writes_metadata(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    source_audio = tmp_path / "speaker01.wav"
    source_audio.write_bytes(b"fake-audio")

    saved_audio_path = gradio_app.save_speaker_reference_to_library(
        speaker_id="SPEAKER_01",
        source_audio_path=str(source_audio),
        reference_text="Second speaker reference",
    )

    meta_path = library_dir / "SPEAKER_01" / "meta.yml"

    assert saved_audio_path == str(library_dir / "SPEAKER_01" / "reference.wav")
    assert Path(saved_audio_path).read_bytes() == b"fake-audio"
    assert yaml.safe_load(meta_path.read_text(encoding="utf-8")) == {
        "speaker_id": "SPEAKER_01",
        "reference_audio_path": str(library_dir / "SPEAKER_01" / "reference.wav"),
        "reference_text": "Second speaker reference",
    }


def test_save_library_reference_updates_current_mapping_rows(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    source_audio = tmp_path / "speaker00.wav"
    source_audio.write_bytes(b"fake-audio")

    status, updated_rows, library_rows, cleared_file, cleared_text, cleared_speaker = gradio_app._save_library_reference(
        "SPEAKER_00",
        str(source_audio),
        "First speaker reference",
        [["SPEAKER_99", "D:/voices/other.wav", "Other"]],
    )

    assert "saved" in status.lower()
    assert updated_rows == [
        ["SPEAKER_99", "D:/voices/other.wav", "Other"],
        [str("SPEAKER_00"), str(library_dir / "SPEAKER_00" / "reference.wav"), "First speaker reference"],
    ]
    assert library_rows == [
        ["SPEAKER_00", str(library_dir / "SPEAKER_00" / "reference.wav"), "First speaker reference"]
    ]
    assert cleared_file is None
    assert cleared_text == ""
    assert cleared_speaker == ""


def test_build_app_lists_saved_speaker_reference_library_entries(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    speaker_dir = library_dir / "SPEAKER_01"
    speaker_dir.mkdir(parents=True, exist_ok=True)
    (speaker_dir / "reference.wav").write_bytes(b"fake-audio")
    (speaker_dir / "meta.yml").write_text(
        yaml.safe_dump(
            {
                "speaker_id": "SPEAKER_01",
                "reference_audio_path": str(speaker_dir / "reference.wav"),
                "reference_text": "Second speaker reference",
            },
            sort_keys=False,
            allow_unicode=True,
        ),
        encoding="utf-8",
    )
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    app = build_app()

    assert _component_value_by_label(app, "Speaker reference library") == [
        ["SPEAKER_01", str(speaker_dir / "reference.wav"), "Second speaker reference"]
    ]


def test_delete_speaker_reference_from_library(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    source_audio = tmp_path / "speaker01.wav"
    source_audio.write_bytes(b"fake-audio")
    gradio_app.save_speaker_reference_to_library(
        speaker_id="SPEAKER_01",
        source_audio_path=str(source_audio),
        reference_text="Ref text",
    )
    assert (library_dir / "SPEAKER_01").exists()

    deleted = gradio_app.delete_speaker_reference_from_library("SPEAKER_01")
    assert deleted is True
    assert not (library_dir / "SPEAKER_01").exists()
    assert gradio_app.load_speaker_reference_library() == []


def test_delete_selected_library_reference_ui_handler(tmp_path, monkeypatch):
    library_dir = tmp_path / "speaker_reference_library"
    monkeypatch.setattr(gradio_app, "DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH", str(library_dir))

    source_audio = tmp_path / "speaker01.wav"
    source_audio.write_bytes(b"fake-audio")
    saved_path = gradio_app.save_speaker_reference_to_library(
        speaker_id="SPEAKER_01",
        source_audio_path=str(source_audio),
        reference_text="Ref text",
    )

    status, library_rows = gradio_app._delete_selected_library_reference(["SPEAKER_01", saved_path, "Ref text"])
    assert "Deleted speaker 'SPEAKER_01'" in status
    assert library_rows == [["", "", ""]]


def test_mapping_row_operations_delete_move_swap():
    initial_rows = [
        ["SPEAKER_00", "path/0.wav", "text 0"],
        ["SPEAKER_01", "path/1.wav", "text 1"],
        ["SPEAKER_02", "path/2.wav", "text 2"],
    ]

    # Test Move Down
    msg, rows, hist = gradio_app._move_mapping_row("down", [0], initial_rows)
    assert "Moved row 1" in msg
    assert rows[0][0] == "SPEAKER_01"
    assert rows[1][0] == "SPEAKER_00"

    # Test Move Up
    msg, rows, hist = gradio_app._move_mapping_row("up", [1], rows)
    assert "Moved row 2" in msg
    assert rows[0][0] == "SPEAKER_00"
    assert rows[1][0] == "SPEAKER_01"

    # Test Swap
    msg, rows, hist = gradio_app._swap_selected_mappings([0, 2], initial_rows)
    assert "Swapped row 1" in msg
    assert rows[0][0] == "SPEAKER_02"
    assert rows[2][0] == "SPEAKER_00"

    # Test Delete mapping
    msg, rows, hist = gradio_app._delete_selected_mapping([1], initial_rows)
    assert "Deleted mapping for 'SPEAKER_01'" in msg
    assert len(rows) == 2
    assert [r[0] for r in rows] == ["SPEAKER_00", "SPEAKER_02"]


def test_select_library_row_stores_only_one_selected_entry():
    selected_row = gradio_app._store_selected_library_row(
        SimpleNamespace(
            row_value=["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
            selected=True,
        )
    )

    assert selected_row == ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"]


def test_deselect_library_row_clears_selected_entry():
    selected_row = gradio_app._store_selected_library_row(
        SimpleNamespace(
            row_value=["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
            selected=False,
        )
    )

    assert selected_row is None


def test_use_selected_library_row_updates_current_mappings():
    status, updated_rows = gradio_app._use_selected_library_reference(
        ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
        [["SPEAKER_99", "D:/voices/other.wav", "Other"]],
    )

    assert "added" in status.lower()
    assert updated_rows == [
        ["SPEAKER_99", "D:/voices/other.wav", "Other"],
        ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Second speaker reference"],
    ]


def test_use_selected_library_row_replaces_existing_speaker_mapping():
    status, updated_rows = gradio_app._use_selected_library_reference(
        ["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Library text"],
        [["SPEAKER_01", "D:/old.wav", "Old text"]],
    )

    assert "updated" in status.lower()
    assert updated_rows == [["SPEAKER_01", "D:/lib/SPEAKER_01/reference.wav", "Library text"]]


def _patch_projects_root(monkeypatch, tmp_path):
    projects_root = tmp_path / "prj"
    monkeypatch.setattr(config_module, "DEFAULT_PROJECTS_ROOT", projects_root, raising=False)
    return projects_root


def _create_translation_cache(tmp_path, monkeypatch, *, segments):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    overrides = {
        "input": str(video_path),
        "source_language": "en",
        "target_language": "be",
        "config": "",
    }
    config = build_config_from_overrides(overrides)
    audio_path = Path(config.get("audio_artifacts_dir")) / "source.wav"
    audio_path.parent.mkdir(parents=True, exist_ok=True)
    AudioSegment.silent(duration=1200).export(audio_path, format="wav")

    cache_manager = CacheManager(use_cache=True, input_file=config.get("input"))
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.cache_manager = cache_manager
    cache_key = dubber._build_translation_cache_key(str(audio_path))
    cache_path = cache_manager.get_cache_path("translation") / f"{cache_key}.pkl"
    with cache_path.open("wb") as handle:
        pickle.dump(segments, handle)

    return overrides, config, cache_path


def test_load_dubbing_text_rows_reads_translation_cache(tmp_path, monkeypatch):
    overrides, _config, _cache_path = _create_translation_cache(
        tmp_path,
        monkeypatch,
        segments=[
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 1.2,
                "text": "Hello there",
                "translation": "Прывітанне",
            }
        ],
    )

    status, rows = gradio_app.load_dubbing_text_rows(overrides)

    assert status == "Loaded 1 dubbing text row(s)."
    assert rows == [[
        "SPEAKER_00",
        "0.000",
        "1.200",
        "Hello there",
        "Прывітанне",
        "",
        "",
    ]]


def test_load_dubbing_text_rows_reads_current_transcription_after_transcribe_only(
    tmp_path, monkeypatch
):
    overrides, config, cache_path = _create_translation_cache(
        tmp_path,
        monkeypatch,
        segments=[],
    )
    cache_path.unlink()
    Path(config.get("transcription_path")).write_text(
        "[00.00.00-00.00.01] SPEAKER_00: Hello there\n"
        "[00.00.02-00.00.03] SPEAKER_01: Second line\n",
        encoding="utf-8",
    )

    status, rows = gradio_app.load_dubbing_text_rows(overrides)

    assert status.startswith(
        "Loaded 2 row(s) from current transcription — no translations yet."
    )
    assert rows == [
        ["SPEAKER_00", "0.000", "1.000", "Hello there", "Hello there", "", "", ""],
        ["SPEAKER_01", "2.000", "3.000", "Second line", "Second line", "", "", ""],
    ]
    assert not cache_path.exists()


def test_save_dubbing_text_rows_updates_cache_and_tsv(tmp_path, monkeypatch):
    overrides, config, cache_path = _create_translation_cache(
        tmp_path,
        monkeypatch,
        segments=[
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 1.2,
                "text": "Hello there",
                "translation": "Стары тэкст",
                "short_translation": "Стары кароткі",
                "very_short_translation": "Стары вельмі кароткі",
                "long_translation": "Стары доўгі",
            }
        ],
    )

    edited_row = [
        "SPEAKER_00",
        "0.000",
        "1.200",
        "Hello there",
        "Новы тэкст",
        "Новы сінтэз",
        "",
    ]
    status, saved_rows = gradio_app.save_dubbing_text_rows([edited_row], overrides)

    artifact_path = Path(config.get("artifacts_dir")) / "dubbing_texts.tsv"
    with cache_path.open("rb") as handle:
        saved_segments = pickle.load(handle)

    assert status == "Saved 1 dubbing text row(s)."
    assert saved_rows == [edited_row]
    assert artifact_path.is_file()
    assert "Новы тэкст" in artifact_path.read_text(encoding="utf-8")
    assert saved_segments[0]["translation"] == "Новы тэкст"
    assert saved_segments[0]["short_translation"] == "Новы тэкст"
    assert saved_segments[0]["very_short_translation"] == "Новы тэкст"
    assert saved_segments[0]["long_translation"] == "Новы тэкст"
    assert saved_segments[0]["synthesized_text"] == "Новы сінтэз"
    assert saved_segments[0]["start"] == 0.0
    assert saved_segments[0]["end"] == 1.2


def test_save_dubbing_text_rows_rejects_row_count_mismatch(tmp_path, monkeypatch):
    overrides, _config, cache_path = _create_translation_cache(
        tmp_path,
        monkeypatch,
        segments=[
            {
                "speaker": "SPEAKER_00",
                "start": 0.0,
                "end": 1.2,
                "text": "Hello there",
                "translation": "Прывітанне",
            }
        ],
    )

    status, saved_rows = gradio_app.save_dubbing_text_rows([], overrides)

    with cache_path.open("rb") as handle:
        saved_segments = pickle.load(handle)

    assert "row count" in status.lower()
    assert saved_rows == []
    assert saved_segments[0]["translation"] == "Прывітанне"
