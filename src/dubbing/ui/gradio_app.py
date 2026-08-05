"""Gradio UI for the DubbLM pipeline."""

from __future__ import annotations

import csv
import json
import pickle
import shutil
from pathlib import Path

import gradio as gr
import yaml

from src.utils.time_utils import format_seconds_to_hms

from ..core.cache_manager import CacheManager
from ..core.runner import build_config_from_overrides, run_dubbing_job
from ..core.smart_dubbing import SmartDubbing
from ..core.config import DubbingConfig


DEFAULT_CONFIG_PATH = "dubbing_config.yml"
DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH = str(Path(__file__).resolve().parents[3] / "speaker_reference_library")


WORKFLOW_FIELDS = [
    "input",
    "source_language",
    "target_language",
    "output",
    "config",
    "run_step",
    "generate_speaker_report",
    "save_original_subtitles",
    "save_translated_subtitles",
    "keep_background",
    "include_original_audio",
    "remove_pauses",
]

SETTINGS_FIELDS = [
    "whisper_model",
    "gemini_transcription_model",
    "transcription_system",
    "start_time",
    "duration",
    "no_cache",
    "translator_type",
    "llm_provider",
    "llm_model_name",
    "llm_temperature",
    "translation_prompt_prefix",
    "glossary",
    "refinement_llm_provider",
    "refinement_model_name",
    "refinement_temperature",
    "refinement_max_tokens",
    "refinement_persona",
    "tts_system",
    "tts_model",
    "tts_fallback_model",
    "voice_name",
    "voice_auto_selection",
    "reference_audio",
    "reference_text",
    "speaker_reference_rows",
    "tts_system_mapping",
    "tts_prompt_prefix",
    "voice_prompt",
    "enable_emotion_analysis",
    "segment_reference_min_duration",
    "watermark_path",
    "watermark_text",
    "keep_original_audio_ranges",
    "min_pause_duration",
    "keyframe_buffer",
    "use_two_pass_encoding",
    "dubbed_volume",
    "background_volume",
    "group_overflow_tolerance",
    "debug_info",
    "debug_tts",
    "debug_diarize_only",
]

ALL_FIELDS = WORKFLOW_FIELDS + SETTINGS_FIELDS
NON_PERSISTED_FIELDS = {"input", "output", "config", "run_step", "generate_speaker_report"}
PERSISTED_FIELDS = [field for field in ALL_FIELDS if field not in NON_PERSISTED_FIELDS]
JSON_TEXT_FIELDS = {"glossary", "voice_prompt", "tts_system_mapping"}
LIST_TEXT_FIELDS = {"keep_original_audio_ranges"}
SPEAKER_REFERENCE_FIELD = "speaker_reference_rows"
SPEAKER_REFERENCE_HEADERS = ["Speaker ID", "Reference audio path", "Reference text"]
SPEAKER_REFERENCE_LIBRARY_HEADERS = ["Speaker ID", "Saved audio path", "Reference text"]
DUBBING_TEXT_HEADERS = ["Speaker", "Time", "Translation", "Original"]
DUBBING_TEXT_COLUMN_WIDTHS = ["12%", "14%", "54%", "20%"]
TRANSLATION_TRACK_FIELDS = (
    "translation",
    "short_translation",
    "very_short_translation",
    "long_translation",
)


def _load_yaml_mapping(config_path: str | Path) -> dict:
    path = Path(config_path)
    if not path.is_file():
        return {}

    with path.open("r", encoding="utf-8") as config_file:
        data = yaml.safe_load(config_file) or {}

    return data if isinstance(data, dict) else {}


def _get_speaker_reference_library_dir(library_path: str | Path | None = None) -> Path:
    return Path(library_path or DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH)


def _sanitize_speaker_directory_name(speaker_id: str) -> str:
    cleaned = "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in speaker_id.strip())
    return cleaned or "UNKNOWN_SPEAKER"


def load_speaker_reference_library(
    library_path: str | Path | None = None,
) -> list[list[str]]:
    library_dir = _get_speaker_reference_library_dir(library_path)
    if not library_dir.exists():
        return []

    rows: list[list[str]] = []
    for speaker_dir in sorted(path for path in library_dir.iterdir() if path.is_dir()):
        meta_path = speaker_dir / "meta.yml"
        if not meta_path.is_file():
            continue

        metadata = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
        speaker_id = str(metadata.get("speaker_id") or speaker_dir.name).strip()
        reference_audio_path = str(metadata.get("reference_audio_path") or "").strip()
        reference_text = str(metadata.get("reference_text") or "").strip()
        if not speaker_id or not reference_audio_path:
            continue

        rows.append([speaker_id, reference_audio_path, reference_text])

    return rows


def save_speaker_reference_to_library(
    speaker_id: str,
    source_audio_path: str,
    reference_text: str,
    library_path: str | Path | None = None,
) -> str:
    speaker_id = str(speaker_id or "").strip()
    source_path = Path(str(source_audio_path or "").strip())
    if not speaker_id:
        raise ValueError("Speaker ID is required.")
    if not source_path.is_file():
        raise FileNotFoundError(f"Reference audio file not found: {source_path}")

    library_dir = _get_speaker_reference_library_dir(library_path)
    speaker_dir = library_dir / _sanitize_speaker_directory_name(speaker_id)
    speaker_dir.mkdir(parents=True, exist_ok=True)

    for existing_file in speaker_dir.glob("reference.*"):
        existing_file.unlink(missing_ok=True)

    suffix = source_path.suffix or ".wav"
    saved_audio_path = speaker_dir / f"reference{suffix}"
    shutil.copy2(source_path, saved_audio_path)

    metadata = {
        "speaker_id": speaker_id,
        "reference_audio_path": str(saved_audio_path),
        "reference_text": str(reference_text or "").strip(),
    }
    (speaker_dir / "meta.yml").write_text(
        yaml.safe_dump(metadata, sort_keys=False, allow_unicode=True),
        encoding="utf-8",
    )
    return str(saved_audio_path)


def _normalize_table_rows(rows: object) -> list[list[str]]:
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    elif hasattr(rows, "to_numpy"):
        rows = rows.to_numpy().tolist()
    elif hasattr(rows, "values"):
        rows = rows.values.tolist()

    normalized_rows: list[list[str]] = []
    if not isinstance(rows, (list, tuple)):
        return normalized_rows

    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        normalized_rows.append(
            [
                str(row[0]).strip() if len(row) > 0 and row[0] is not None else "",
                str(row[1]).strip() if len(row) > 1 and row[1] is not None else "",
                str(row[2]).strip() if len(row) > 2 and row[2] is not None else "",
            ]
        )
    return normalized_rows


def _upsert_speaker_reference_row(
    rows: object,
    speaker_id: str,
    reference_audio_path: str,
    reference_text: str,
) -> list[list[str]]:
    normalized_rows = [
        row for row in _normalize_table_rows(rows) if any(cell for cell in row)
    ]
    updated_row = [speaker_id, reference_audio_path, reference_text]

    for index, row in enumerate(normalized_rows):
        if row[0] == speaker_id:
            normalized_rows[index] = updated_row
            return normalized_rows

    normalized_rows.append(updated_row)
    return normalized_rows


def _save_library_reference(
    speaker_id: str,
    reference_audio_file: str,
    reference_text: str,
    current_rows: object,
):
    speaker_id = str(speaker_id or "").strip()
    reference_text = str(reference_text or "").strip()
    if not speaker_id:
        return (
            "Speaker ID is required to save a library reference.",
            _normalize_table_rows(current_rows),
            load_speaker_reference_library(),
            None,
            reference_text,
            speaker_id,
        )
    if not reference_audio_file:
        return (
            "Reference audio file is required to save a library reference.",
            _normalize_table_rows(current_rows),
            load_speaker_reference_library(),
            None,
            reference_text,
            speaker_id,
        )

    saved_audio_path = save_speaker_reference_to_library(
        speaker_id=speaker_id,
        source_audio_path=str(reference_audio_file),
        reference_text=reference_text,
    )
    updated_rows = _upsert_speaker_reference_row(current_rows, speaker_id, saved_audio_path, reference_text)
    library_rows = load_speaker_reference_library()
    return (
        f"Saved speaker reference for {speaker_id} to library.",
        updated_rows,
        library_rows,
        None,
        "",
        "",
    )


def _store_selected_library_row(evt: gr.SelectData):
    if not getattr(evt, "selected", True):
        return None

    row_value = getattr(evt, "row_value", None)
    if isinstance(row_value, (list, tuple)) and len(row_value) >= 3:
        return [
            str(row_value[0]).strip(),
            str(row_value[1]).strip(),
            str(row_value[2]).strip(),
        ]
    return None


def _use_selected_library_reference(selected_row: object, current_rows: object):
    if not isinstance(selected_row, (list, tuple)) or len(selected_row) < 3:
        return "Select one library row first.", _normalize_table_rows(current_rows)

    speaker_id = str(selected_row[0]).strip()
    reference_audio_path = str(selected_row[1]).strip()
    reference_text = str(selected_row[2]).strip()
    if not speaker_id or not reference_audio_path:
        return "Selected library row is incomplete.", _normalize_table_rows(current_rows)

    normalized_rows = _normalize_table_rows(current_rows)
    existing_speakers = {row[0] for row in normalized_rows if row[0]}
    updated_rows = _upsert_speaker_reference_row(
        normalized_rows,
        speaker_id,
        reference_audio_path,
        reference_text,
    )
    action = "updated" if speaker_id in existing_speakers else "added"
    return f"{speaker_id} {action} from library.", updated_rows


def delete_speaker_reference_from_library(
    speaker_id: str,
    library_path: str | Path | None = None,
) -> bool:
    speaker_id = str(speaker_id or "").strip()
    if not speaker_id:
        return False

    library_dir = _get_speaker_reference_library_dir(library_path)
    if not library_dir.exists():
        return False

    deleted = False
    speaker_dir = library_dir / _sanitize_speaker_directory_name(speaker_id)
    if speaker_dir.exists() and speaker_dir.is_dir():
        shutil.rmtree(speaker_dir)
        deleted = True

    for s_dir in library_dir.iterdir():
        if s_dir.is_dir():
            meta_path = s_dir / "meta.yml"
            if meta_path.is_file():
                try:
                    meta = yaml.safe_load(meta_path.read_text(encoding="utf-8")) or {}
                    if str(meta.get("speaker_id") or "").strip() == speaker_id:
                        shutil.rmtree(s_dir)
                        deleted = True
                except Exception:
                    pass

    return deleted


def _delete_selected_library_reference(selected_row: object):
    if not isinstance(selected_row, (list, tuple)) or len(selected_row) < 1:
        return "Select one library row first.", load_speaker_reference_library() or [["", "", ""]]

    speaker_id = str(selected_row[0]).strip()
    if not speaker_id:
        return "Selected library row has no speaker ID.", load_speaker_reference_library() or [["", "", ""]]

    deleted = delete_speaker_reference_from_library(speaker_id)
    library_rows = load_speaker_reference_library()
    status_msg = f"Deleted speaker '{speaker_id}' from library." if deleted else f"Speaker '{speaker_id}' not found in library."
    return status_msg, library_rows or [["", "", ""]]


def _store_selected_mapping_info(evt: gr.SelectData, history: object):
    if not getattr(evt, "selected", True):
        return history
    row_idx = evt.index[0] if isinstance(evt.index, (list, tuple)) else 0

    history_list = list(history) if isinstance(history, list) else []
    if history_list and history_list[-1] == row_idx:
        return history_list

    history_list.append(row_idx)
    if len(history_list) > 2:
        history_list = history_list[-2:]
    return history_list


def _delete_selected_mapping(mapping_history: object, current_rows: object):
    normalized_rows = [
        row for row in _normalize_table_rows(current_rows) if any(cell for cell in row)
    ]
    if not normalized_rows:
        return "No mappings to delete.", [["", "", ""]], []

    history_list = list(mapping_history) if isinstance(mapping_history, list) else []
    idx = history_list[-1] if history_list else None

    if idx is not None and 0 <= idx < len(normalized_rows):
        deleted_spk = normalized_rows[idx][0]
        del normalized_rows[idx]
        msg = f"Deleted mapping for '{deleted_spk}'."
    else:
        msg = "Select a mapping row to delete first."

    if not normalized_rows:
        normalized_rows = [["", "", ""]]
    return msg, normalized_rows, []


def _move_mapping_row(direction: str, mapping_history: object, current_rows: object):
    normalized_rows = [
        row for row in _normalize_table_rows(current_rows) if any(cell for cell in row)
    ]
    if not normalized_rows or len(normalized_rows) < 2:
        return "Need at least 2 rows to reorder.", normalized_rows or [["", "", ""]], mapping_history

    history_list = list(mapping_history) if isinstance(mapping_history, list) else []
    idx = history_list[-1] if history_list else None

    if idx is None or not (0 <= idx < len(normalized_rows)):
        return "Select a mapping row first.", normalized_rows, mapping_history

    target_idx = idx - 1 if direction == "up" else idx + 1
    if 0 <= target_idx < len(normalized_rows):
        normalized_rows[idx], normalized_rows[target_idx] = (
            normalized_rows[target_idx],
            normalized_rows[idx],
        )
        msg = f"Moved row {idx + 1} ({normalized_rows[target_idx][0]}) {direction}."
        return msg, normalized_rows, [target_idx]
    else:
        msg = f"Row is already at the {'top' if direction == 'up' else 'bottom'}."
        return msg, normalized_rows, mapping_history


def _swap_selected_mappings(mapping_history: object, current_rows: object):
    normalized_rows = [
        row for row in _normalize_table_rows(current_rows) if any(cell for cell in row)
    ]
    if len(normalized_rows) < 2:
        return "Need at least 2 rows to swap.", normalized_rows or [["", "", ""]], mapping_history

    history_list = list(mapping_history) if isinstance(mapping_history, list) else []
    valid_indices = [i for i in history_list if isinstance(i, int) and 0 <= i < len(normalized_rows)]

    if len(valid_indices) >= 2:
        i1, i2 = valid_indices[-2], valid_indices[-1]
        normalized_rows[i1], normalized_rows[i2] = (
            normalized_rows[i2],
            normalized_rows[i1],
        )
        msg = f"Swapped row {i1 + 1} ({normalized_rows[i2][0]}) and row {i2 + 1} ({normalized_rows[i1][0]})."
        return msg, normalized_rows, [i1, i2]
    elif len(valid_indices) == 1:
        i1 = valid_indices[0]
        i2 = i1 + 1 if i1 + 1 < len(normalized_rows) else i1 - 1
        normalized_rows[i1], normalized_rows[i2] = (
            normalized_rows[i2],
            normalized_rows[i1],
        )
        msg = f"Swapped row {i1 + 1} ({normalized_rows[i2][0]}) with row {i2 + 1} ({normalized_rows[i1][0]})."
        return msg, normalized_rows, [i1, i2]
    else:
        return "Select mapping row(s) to swap first.", normalized_rows, mapping_history


def _speaker_reference_rows_from_mappings(
    reference_audio_mapping: object,
    reference_text_mapping: object,
) -> list[list[str]]:
    audio_mapping = reference_audio_mapping if isinstance(reference_audio_mapping, dict) else {}
    text_mapping = reference_text_mapping if isinstance(reference_text_mapping, dict) else {}
    rows: list[list[str]] = []
    seen: set[str] = set()

    for mapping in (audio_mapping, text_mapping):
        for speaker in mapping:
            speaker_id = str(speaker).strip()
            if not speaker_id or speaker_id in seen:
                continue
            rows.append(
                [
                    speaker_id,
                    str(audio_mapping.get(speaker_id, "") or ""),
                    str(text_mapping.get(speaker_id, "") or ""),
                ]
            )
            seen.add(speaker_id)

    return rows or [["", "", ""]]


def _speaker_reference_rows_to_mappings(rows: object) -> tuple[dict[str, str] | None, dict[str, str] | None]:
    if isinstance(rows, dict) and "data" in rows:
        rows = rows["data"]
    elif hasattr(rows, "to_numpy"):
        rows = rows.to_numpy().tolist()
    elif hasattr(rows, "values"):
        rows = rows.values.tolist()

    if not isinstance(rows, (list, tuple)):
        return None, None

    reference_audio_mapping: dict[str, str] = {}
    reference_text_mapping: dict[str, str] = {}

    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue

        speaker_id = str(row[0]).strip() if len(row) > 0 and row[0] is not None else ""
        reference_audio = str(row[1]).strip() if len(row) > 1 and row[1] is not None else ""
        reference_text = str(row[2]).strip() if len(row) > 2 and row[2] is not None else ""

        if not speaker_id:
            continue
        if reference_audio:
            reference_audio_mapping[speaker_id] = reference_audio
        if reference_text:
            reference_text_mapping[speaker_id] = reference_text

    return reference_audio_mapping or None, reference_text_mapping or None


def _expand_speaker_reference_rows(overrides: dict[str, object]) -> dict[str, object]:
    expanded = dict(overrides)
    if SPEAKER_REFERENCE_FIELD in expanded:
        reference_audio_mapping, reference_text_mapping = _speaker_reference_rows_to_mappings(
            expanded.pop(SPEAKER_REFERENCE_FIELD, None)
        )
        expanded["reference_audio_mapping"] = reference_audio_mapping or {}
        expanded["reference_text_mapping"] = reference_text_mapping or {}

    return expanded


def load_ui_defaults(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    config = DubbingConfig()
    config.load_from_yaml(config_path)

    defaults = config.to_dict()
    for field in JSON_TEXT_FIELDS:
        value = defaults.get(field)
        if value is not None:
            defaults[field] = json.dumps(value, ensure_ascii=False, indent=2)

    for field in LIST_TEXT_FIELDS:
        value = defaults.get(field)
        if isinstance(value, list):
            defaults[field] = "\n".join(str(item) for item in value)

    defaults[SPEAKER_REFERENCE_FIELD] = _speaker_reference_rows_from_mappings(
        defaults.get("reference_audio_mapping"),
        defaults.get("reference_text_mapping"),
    )
    defaults["config"] = config_path
    return defaults


def save_settings(
    overrides: dict[str, object],
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> str:
    config_data = _load_yaml_mapping(config_path)
    reference_audio_mapping, reference_text_mapping = _speaker_reference_rows_to_mappings(
        overrides.get(SPEAKER_REFERENCE_FIELD)
    )

    for field in PERSISTED_FIELDS:
        if field not in overrides:
            continue

        if field == SPEAKER_REFERENCE_FIELD:
            if reference_audio_mapping is None:
                config_data.pop("reference_audio_mapping", None)
            else:
                config_data["reference_audio_mapping"] = reference_audio_mapping

            if reference_text_mapping is None:
                config_data.pop("reference_text_mapping", None)
            else:
                config_data["reference_text_mapping"] = reference_text_mapping
            continue

        value = overrides[field]
        if isinstance(value, str):
            value = value.strip()
            if value == "":
                value = None
        elif field == "duration":
            try:
                if value is not None and float(value) <= 0:
                    value = None
            except (TypeError, ValueError):
                value = None

        if isinstance(value, str) and field in JSON_TEXT_FIELDS:
            value = json.loads(value)
        elif isinstance(value, str) and field in LIST_TEXT_FIELDS:
            value = [line.strip() for line in value.splitlines() if line.strip()]

        if value is None:
            config_data.pop(field, None)
        else:
            config_data[field] = value

    with Path(config_path).open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config_data, config_file, sort_keys=False, allow_unicode=True)

    return f"Settings saved to {DEFAULT_CONFIG_PATH}"


def _collect_overrides(*values) -> dict[str, object]:
    overrides = dict(zip(ALL_FIELDS, values))
    overrides = _expand_speaker_reference_rows(overrides)
    return overrides


def _build_dubbing_text_context(overrides: dict[str, object]) -> tuple[DubbingConfig, Path, Path]:
    config = build_config_from_overrides(overrides)
    audio_path = Path(config.get("audio_artifacts_dir")) / "source.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Expected extracted source audio at {audio_path}. "
            "Run the full pipeline once before editing dubbing texts."
        )

    cache_manager = CacheManager(use_cache=True, input_file=config.get("input"))
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.cache_manager = cache_manager
    cache_key = dubber._build_translation_cache_key(str(audio_path))
    cache_path = cache_manager.get_cache_path("translation") / f"{cache_key}.pkl"
    artifact_path = Path(config.get("artifacts_dir")) / "dubbing_texts.tsv"
    return config, cache_path, artifact_path


def _format_dubbing_text_time(start: object, end: object) -> str:
    start_seconds = float(start or 0.0)
    end_seconds = float(end or 0.0)
    return f"{format_seconds_to_hms(start_seconds)} - {format_seconds_to_hms(end_seconds)}"


def _segments_to_dubbing_text_rows(segments: object) -> list[list[str]]:
    rows: list[list[str]] = []
    if not isinstance(segments, list):
        return rows

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        rows.append(
            [
                str(segment.get("speaker", "") or ""),
                _format_dubbing_text_time(segment.get("start"), segment.get("end")),
                str(segment.get("translation", "") or ""),
                str(segment.get("text", "") or ""),
            ]
        )
    return rows


def _normalize_dubbing_text_rows(rows: object) -> list[list[str]]:
    normalized_rows: list[list[str]] = []
    if not isinstance(rows, list):
        return normalized_rows

    for row in rows:
        if not isinstance(row, (list, tuple)):
            continue
        normalized_rows.append(
            [
                str(row[0]).strip() if len(row) > 0 and row[0] is not None else "",
                str(row[1]).strip() if len(row) > 1 and row[1] is not None else "",
                str(row[2]) if len(row) > 2 and row[2] is not None else "",
                str(row[3]) if len(row) > 3 and row[3] is not None else "",
            ]
        )
    return normalized_rows


def _load_cached_translation_segments(cache_path: Path) -> list[dict[str, object]]:
    if not cache_path.is_file():
        raise FileNotFoundError(
            f"Translation cache not found: {cache_path}. "
            "Run the full pipeline first so `tts_to_end` has cached translations to edit."
        )

    with cache_path.open("rb") as handle:
        cached_segments = pickle.load(handle)
    if not isinstance(cached_segments, list):
        raise ValueError(f"Unexpected translation cache payload in {cache_path}")
    return cached_segments


def _write_dubbing_text_artifact(artifact_path: Path, rows: list[list[str]]) -> None:
    artifact_path.parent.mkdir(parents=True, exist_ok=True)
    with artifact_path.open("w", encoding="utf-8", newline="") as handle:
        writer = csv.writer(handle, delimiter="\t")
        writer.writerow(["speaker", "time", "translation", "original"])
        for row in rows:
            writer.writerow(row)


def load_dubbing_text_rows(overrides: dict[str, object]) -> tuple[str, list[list[str]]]:
    try:
        _config, cache_path, _artifact_path = _build_dubbing_text_context(overrides)
        cached_segments = _load_cached_translation_segments(cache_path)
        rows = _segments_to_dubbing_text_rows(cached_segments)
        return f"Loaded {len(rows)} dubbing text row(s).", rows
    except Exception as exc:
        return f"Failed: {exc}", []


def save_dubbing_text_rows(rows: object, overrides: dict[str, object]) -> tuple[str, list[list[str]]]:
    normalized_rows = _normalize_dubbing_text_rows(rows)
    try:
        _config, cache_path, artifact_path = _build_dubbing_text_context(overrides)
        cached_segments = _load_cached_translation_segments(cache_path)
        if len(normalized_rows) != len(cached_segments):
            raise ValueError(
                f"Edited row count ({len(normalized_rows)}) does not match cached segment count ({len(cached_segments)})."
            )

        saved_rows: list[list[str]] = []
        for segment, row in zip(cached_segments, normalized_rows):
            translation = row[2].strip()
            if not translation:
                raise ValueError("Translation text cannot be empty.")

            for field in TRANSLATION_TRACK_FIELDS:
                segment[field] = translation

            saved_rows.append(
                [
                    str(segment.get("speaker", "") or ""),
                    _format_dubbing_text_time(segment.get("start"), segment.get("end")),
                    translation,
                    str(segment.get("text", "") or ""),
                ]
            )

        with cache_path.open("wb") as handle:
            pickle.dump(cached_segments, handle)
        _write_dubbing_text_artifact(artifact_path, saved_rows)
        return f"Saved {len(saved_rows)} dubbing text row(s).", saved_rows
    except Exception as exc:
        return f"Failed: {exc}", normalized_rows


def _load_dubbing_text_values(*values):
    return load_dubbing_text_rows(_collect_overrides(*values))


def _save_dubbing_text_values(rows, *values):
    return save_dubbing_text_rows(rows, _collect_overrides(*values))


def _collect_values(*values):
    overrides = _collect_overrides(*values)
    result = run_dubbing_job(overrides)

    output_file = result.output_file
    if output_file and not Path(output_file).is_file():
        artifacts_path = output_file
        output_file = None
    else:
        artifacts_path = None

    report_file = result.report_file if result.report_file and Path(result.report_file).is_file() else None
    return result.status, result.logs, output_file, report_file, artifacts_path


def _save_values(*values):
    overrides = dict(zip(ALL_FIELDS, values))
    status = save_settings(overrides)
    return status, f"Saved current settings to {DEFAULT_CONFIG_PATH}"


def build_app(config_path: str = DEFAULT_CONFIG_PATH) -> gr.Blocks:
    """Build the Gradio interface."""
    defaults = load_ui_defaults(config_path)
    library_rows = load_speaker_reference_library()

    with gr.Blocks(title="DubbLM", theme=gr.themes.Soft()) as app:
        selected_library_row = gr.State(None)
        selected_mapping_history = gr.State([])
        gr.Markdown(
            """
            # DubbLM
            Python Gradio wrapper around the internal dubbing pipeline.
            Use `Workflow` for the main run and `Settings` for the full option set.
            """
        )

        input_components = []

        with gr.Tabs():
            with gr.Tab("Workflow"):
                with gr.Row():
                    input_file = gr.File(label="Input video", type="filepath")
                    source_language = gr.Textbox(label="Source language", placeholder="en", value=defaults.get("source_language"))
                    target_language = gr.Textbox(label="Target language", placeholder="be", value=defaults.get("target_language"))
                with gr.Row():
                    output = gr.Textbox(label="Output path", placeholder="Leave empty for auto-generated file name")
                    config = gr.Textbox(label="Config path", value=defaults.get("config", DEFAULT_CONFIG_PATH))
                    run_step = gr.Dropdown(
                        label="Run step",
                        choices=["full_pipeline", "combine_video", "tts_to_end"],
                        value=defaults.get("run_step") or "full_pipeline",
                        allow_custom_value=False,
                        info="Choose `full_pipeline` for the normal end-to-end run, then click `Run DubbLM`. Resume options require existing artifacts from a previous full run in the same project directory: use `combine_video` to rebuild the final video from an existing dubbed audio file, or `tts_to_end` to restart at cached translation data, regenerate TTS, replace the generated audio artifacts, and finish a new video.",
                    )
                with gr.Row():
                    generate_speaker_report = gr.Checkbox(label="Generate speaker report only")
                    save_original_subtitles = gr.Checkbox(
                        label="Save original subtitles",
                        value=bool(defaults.get("save_original_subtitles", False)),
                    )
                    save_translated_subtitles = gr.Checkbox(
                        label="Save translated subtitles",
                        value=bool(defaults.get("save_translated_subtitles", False)),
                    )
                with gr.Row():
                    keep_background = gr.Checkbox(
                        label="Keep background audio",
                        value=bool(defaults.get("keep_background", False)),
                    )
                    include_original_audio = gr.Checkbox(
                        label="Include original audio in final video",
                        value=bool(defaults.get("include_original_audio", False)),
                    )
                    remove_pauses = gr.Checkbox(
                        label="Remove pauses",
                        value=bool(defaults.get("remove_pauses", False)),
                    )

                input_components.extend(
                    [
                        input_file,
                        source_language,
                        target_language,
                        output,
                        config,
                        run_step,
                        generate_speaker_report,
                        save_original_subtitles,
                        save_translated_subtitles,
                        keep_background,
                        include_original_audio,
                        remove_pauses,
                    ]
                )

            with gr.Tab("Settings"):
                gr.Markdown("## Transcription")
                with gr.Row():
                    whisper_model = gr.Textbox(label="Whisper model", value=defaults.get("whisper_model", "large-v3"))
                    gemini_transcription_model = gr.Textbox(
                        label="Gemini transcription model",
                        value=defaults.get("gemini_transcription_model", "gemini-3-flash-preview"),
                    )
                    transcription_system = gr.Dropdown(
                        label="Transcription system",
                        choices=["whisper", "openai", "whisperx", "assemblyai", "gemini", "deepgram"],
                        value=defaults.get("transcription_system", "whisper"),
                    )
                    start_time = gr.Number(label="Start time (seconds)", precision=2, value=defaults.get("start_time"))
                    duration = gr.Number(label="Duration (seconds)", precision=2, value=defaults.get("duration"))
                    no_cache = gr.Checkbox(label="Disable cache", value=bool(defaults.get("no_cache", False)))

                gr.Markdown("## Translation")
                with gr.Row():
                    translator_type = gr.Textbox(label="Translator type", value=defaults.get("translator_type", "llm"))
                    llm_provider = gr.Dropdown(
                        label="LLM provider",
                        choices=["gemini", "openrouter"],
                        value=defaults.get("llm_provider", "gemini"),
                    )
                    llm_model_name = gr.Textbox(label="LLM model", value=defaults.get("llm_model_name"))
                    llm_temperature = gr.Number(label="LLM temperature", value=defaults.get("llm_temperature", 0.5), precision=2)
                translation_prompt_prefix = gr.Textbox(
                    label="Translation prompt prefix",
                    lines=3,
                    value=defaults.get("translation_prompt_prefix"),
                )
                glossary = gr.Textbox(
                    label="Glossary JSON",
                    lines=4,
                    placeholder='{"term": "translation"}',
                    value=defaults.get("glossary"),
                )

                gr.Markdown("## Refinement")
                with gr.Row():
                    refinement_llm_provider = gr.Dropdown(
                        label="Refinement LLM provider",
                        choices=["gemini", "openrouter"],
                        value=defaults.get("refinement_llm_provider"),
                    )
                    refinement_model_name = gr.Textbox(label="Refinement model", value=defaults.get("refinement_model_name"))
                    refinement_temperature = gr.Number(
                        label="Refinement temperature",
                        value=defaults.get("refinement_temperature", 1.0),
                        precision=2,
                    )
                    refinement_max_tokens = gr.Number(
                        label="Refinement max tokens",
                        precision=0,
                        value=defaults.get("refinement_max_tokens"),
                    )
                refinement_persona = gr.Dropdown(
                    label="Refinement persona",
                    choices=[
                        "normal",
                        "casual_manager",
                        "child",
                        "housewife",
                        "science_popularizer",
                        "it_buddy",
                        "ai_buddy",
                    ],
                    value=defaults.get("refinement_persona", "normal"),
                    allow_custom_value=True,
                )

                gr.Markdown("## TTS")
                with gr.Row():
                    tts_system = gr.Dropdown(
                        label="TTS system",
                        choices=["coqui", "xtts", "openai", "f5_tts", "gemini", "bextts", "omnivoice"],
                        value=defaults.get("tts_system", "coqui"),
                    )
                    tts_model = gr.Textbox(label="TTS model", value=defaults.get("tts_model"))
                    tts_fallback_model = gr.Textbox(label="Fallback TTS model", value=defaults.get("tts_fallback_model"))
                    voice_name = gr.Textbox(label="Voice name or speaker mapping", value=defaults.get("voice_name"))
                with gr.Row():
                    voice_auto_selection = gr.Checkbox(
                        label="Automatic voice selection",
                        value=bool(defaults.get("voice_auto_selection", True)),
                    )
                    reference_audio = gr.Textbox(label="Reference audio path", value=defaults.get("reference_audio"))
                    reference_text = gr.Textbox(label="Reference text", value=defaults.get("reference_text"))
                speaker_reference_rows = gr.Dataframe(
                    headers=SPEAKER_REFERENCE_HEADERS,
                    datatype=["str", "str", "str"],
                    row_count=(1, "dynamic"),
                    col_count=(3, "fixed"),
                    label="Speaker reference mappings",
                    value=defaults.get(SPEAKER_REFERENCE_FIELD),
                    type="array",
                    interactive=True,
                )
                with gr.Row():
                    move_up_button = gr.Button("Move Up", size="sm")
                    move_down_button = gr.Button("Move Down", size="sm")
                    swap_mappings_button = gr.Button("Swap selected rows", size="sm")
                    delete_mapping_button = gr.Button("Delete selected mapping", size="sm", variant="stop")
                gr.Markdown(f"Speaker reference library path: `{DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH}`")
                with gr.Row():
                    library_speaker_id = gr.Textbox(label="Library speaker ID", placeholder="SPEAKER_01")
                    library_reference_audio_file = gr.File(label="Library reference audio file", type="filepath")
                    library_reference_text = gr.Textbox(label="Library reference text", lines=2)
                save_library_button = gr.Button("Save to library")
                speaker_reference_library = gr.Dataframe(
                    headers=SPEAKER_REFERENCE_LIBRARY_HEADERS,
                    datatype=["str", "str", "str"],
                    row_count=(max(1, len(library_rows)), "fixed"),
                    col_count=(3, "fixed"),
                    label="Speaker reference library",
                    value=library_rows or [["", "", ""]],
                    interactive=False,
                    type="array",
                )
                with gr.Row():
                    use_selected_library_button = gr.Button("Use selected from library")
                    delete_library_button = gr.Button("Delete selected from library", variant="stop")
                with gr.Row():
                    enable_emotion_analysis = gr.Checkbox(
                        label="Enable emotion analysis",
                        value=bool(defaults.get("enable_emotion_analysis", False)),
                    )
                    segment_reference_min_duration = gr.Number(
                        label="Min segment reference duration",
                        value=defaults.get("segment_reference_min_duration", 2.0),
                        precision=2,
                    )
                tts_system_mapping = gr.Textbox(
                    label="TTS system mapping JSON",
                    lines=4,
                    placeholder='{"SPEAKER_00": "gemini"}',
                    value=defaults.get("tts_system_mapping"),
                )
                tts_prompt_prefix = gr.Textbox(label="TTS prompt prefix", lines=3, value=defaults.get("tts_prompt_prefix"))
                voice_prompt = gr.Textbox(
                    label="Voice prompt JSON",
                    lines=4,
                    placeholder='{"SPEAKER_00": "calm, friendly"}',
                    value=defaults.get("voice_prompt"),
                )

                gr.Markdown("## Video / Audio")
                with gr.Row():
                    watermark_path = gr.Textbox(label="Watermark image path", value=defaults.get("watermark_path"))
                    watermark_text = gr.Textbox(label="Watermark text", value=defaults.get("watermark_text"))
                keep_original_audio_ranges = gr.Textbox(
                    label="Keep original audio ranges",
                    lines=4,
                    placeholder="00:10-00:15\n01:02-01:08",
                    value=defaults.get("keep_original_audio_ranges"),
                )
                with gr.Row():
                    min_pause_duration = gr.Number(
                        label="Min pause duration",
                        value=defaults.get("min_pause_duration", 3.0),
                        precision=2,
                    )
                    keyframe_buffer = gr.Number(
                        label="Keyframe buffer",
                        value=defaults.get("keyframe_buffer", 0.2),
                        precision=2,
                    )
                    use_two_pass_encoding = gr.Checkbox(
                        label="Use two-pass encoding",
                        value=bool(defaults.get("use_two_pass_encoding", True)),
                    )
                with gr.Row():
                    dubbed_volume = gr.Number(
                        label="Dubbed volume",
                        value=defaults.get("dubbed_volume", 1.0),
                        precision=3,
                    )
                    background_volume = gr.Number(
                        label="Background volume",
                        value=defaults.get("background_volume", 0.562341),
                        precision=6,
                    )
                    group_overflow_tolerance = gr.Number(
                        label="Group overflow tolerance",
                        value=defaults.get("group_overflow_tolerance", 1.0),
                        precision=2,
                    )

                gr.Markdown("## Debug / Advanced")
                with gr.Row():
                    debug_info = gr.Checkbox(label="Debug info", value=bool(defaults.get("debug_info", False)))
                    debug_tts = gr.Checkbox(label="Debug TTS", value=bool(defaults.get("debug_tts", False)))
                    debug_diarize_only = gr.Checkbox(
                        label="Debug diarize only",
                        value=bool(defaults.get("debug_diarize_only", False)),
                    )

                input_components.extend(
                    [
                        whisper_model,
                        gemini_transcription_model,
                        transcription_system,
                        start_time,
                        duration,
                        no_cache,
                        translator_type,
                        llm_provider,
                        llm_model_name,
                        llm_temperature,
                        translation_prompt_prefix,
                        glossary,
                        refinement_llm_provider,
                        refinement_model_name,
                        refinement_temperature,
                        refinement_max_tokens,
                        refinement_persona,
                        tts_system,
                        tts_model,
                        tts_fallback_model,
                        voice_name,
                        voice_auto_selection,
                        reference_audio,
                        reference_text,
                        speaker_reference_rows,
                        tts_system_mapping,
                        tts_prompt_prefix,
                        voice_prompt,
                        enable_emotion_analysis,
                        segment_reference_min_duration,
                        watermark_path,
                        watermark_text,
                        keep_original_audio_ranges,
                        min_pause_duration,
                        keyframe_buffer,
                        use_two_pass_encoding,
                        dubbed_volume,
                        background_volume,
                        group_overflow_tolerance,
                        debug_info,
                        debug_tts,
                        debug_diarize_only,
                    ]
                )

            with gr.Tab("Dubbing Texts"):
                gr.Markdown(
                    "Load cached translation segments, edit the dubbing text in the `Translation` column, then save. "
                    "Saved edits are written both to `artifacts/dubbing_texts.tsv` and to the translation cache used by `tts_to_end`."
                )
                with gr.Row():
                    load_dubbing_texts_button = gr.Button("Load texts")
                    save_dubbing_texts_button = gr.Button("Save texts")
                dubbing_text_status = gr.Textbox(label="Dubbing text status", interactive=False)
                dubbing_text_rows = gr.Dataframe(
                    headers=DUBBING_TEXT_HEADERS,
                    datatype=["str", "str", "str", "str"],
                    row_count=(1, "dynamic"),
                    col_count=(4, "fixed"),
                    label="Dubbing texts",
                    value=[["", "", "", ""]],
                    type="array",
                    interactive=True,
                    wrap=True,
                    line_breaks=True,
                    show_search="search",
                    show_row_numbers=True,
                    pinned_columns=2,
                    static_columns=[0, 1, 3],
                    column_widths=DUBBING_TEXT_COLUMN_WIDTHS,
                )

        with gr.Row():
            save_button = gr.Button("Save settings")
            run_button = gr.Button("Run DubbLM", variant="primary")
        status = gr.Textbox(label="Status", interactive=False)
        logs = gr.Textbox(label="Logs", lines=16, interactive=False)
        with gr.Row():
            output_file = gr.File(label="Output file")
            report_file = gr.File(label="Report file")
        artifacts_path = gr.Textbox(label="Artifacts path", interactive=False)

        save_button.click(
            fn=_save_values,
            inputs=input_components,
            outputs=[status, logs],
        )
        save_library_button.click(
            fn=_save_library_reference,
            inputs=[
                library_speaker_id,
                library_reference_audio_file,
                library_reference_text,
                speaker_reference_rows,
            ],
            outputs=[
                status,
                speaker_reference_rows,
                speaker_reference_library,
                library_reference_audio_file,
                library_reference_text,
                library_speaker_id,
            ],
        )
        speaker_reference_rows.select(
            fn=_store_selected_mapping_info,
            inputs=[selected_mapping_history],
            outputs=selected_mapping_history,
        )
        move_up_button.click(
            fn=lambda hist, rows: _move_mapping_row("up", hist, rows),
            inputs=[selected_mapping_history, speaker_reference_rows],
            outputs=[status, speaker_reference_rows, selected_mapping_history],
        )
        move_down_button.click(
            fn=lambda hist, rows: _move_mapping_row("down", hist, rows),
            inputs=[selected_mapping_history, speaker_reference_rows],
            outputs=[status, speaker_reference_rows, selected_mapping_history],
        )
        swap_mappings_button.click(
            fn=_swap_selected_mappings,
            inputs=[selected_mapping_history, speaker_reference_rows],
            outputs=[status, speaker_reference_rows, selected_mapping_history],
        )
        delete_mapping_button.click(
            fn=_delete_selected_mapping,
            inputs=[selected_mapping_history, speaker_reference_rows],
            outputs=[status, speaker_reference_rows, selected_mapping_history],
        )

        speaker_reference_library.select(
            fn=_store_selected_library_row,
            outputs=selected_library_row,
        )
        use_selected_library_button.click(
            fn=_use_selected_library_reference,
            inputs=[selected_library_row, speaker_reference_rows],
            outputs=[status, speaker_reference_rows],
        )
        delete_library_button.click(
            fn=_delete_selected_library_reference,
            inputs=[selected_library_row],
            outputs=[status, speaker_reference_library],
        )
        load_dubbing_texts_button.click(
            fn=_load_dubbing_text_values,
            inputs=input_components,
            outputs=[dubbing_text_status, dubbing_text_rows],
        )
        save_dubbing_texts_button.click(
            fn=_save_dubbing_text_values,
            inputs=[dubbing_text_rows, *input_components],
            outputs=[dubbing_text_status, dubbing_text_rows],
        )
        run_button.click(
            fn=_collect_values,
            inputs=input_components,
            outputs=[status, logs, output_file, report_file, artifacts_path],
        )

        def _load_values_on_page_load():
            fresh_defaults = load_ui_defaults(config_path)
            return [fresh_defaults.get(field) for field in ALL_FIELDS]

        app.load(
            fn=_load_values_on_page_load,
            outputs=input_components,
        )

    return app


def main() -> None:
    """Launch the Gradio app."""
    app = build_app()
    app.launch()


if __name__ == "__main__":
    main()
