"""Gradio UI for the DubbLM pipeline."""

from __future__ import annotations

import csv
import json
import logging
import pickle
import re
import shutil
from pathlib import Path
from typing import Mapping

import gradio as gr
import yaml

from src.utils.time_utils import format_seconds_to_hms

from ..core.cache_manager import CacheManager
from ..core.runner import build_config_from_overrides, run_dubbing_job, run_dubbing_job_streaming
from ..core.smart_dubbing import SmartDubbing
from ..core.config import DubbingConfig
from ..core.voice_profiles import VoiceProfile, normalize_voices, resolve_profile
from tts.gemini_tts_wrapper import ALL_GEMINI_VOICES, GeminiTTSConfig
from tts.openai_tts_wrapper import ALL_OPENAI_VOICES
from tts.tts_factory import TTSFactory


DEFAULT_CONFIG_PATH = "dubbing_config.yml"
DEFAULT_SPEAKER_REFERENCE_LIBRARY_PATH = str(Path(__file__).resolve().parents[3] / "speaker_reference_library")
LOGGER = logging.getLogger(__name__)


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
    # Isolated per-speaker tracks — UI-only intermediate fields. Combined
    # into ``isolated_tracks`` (dict) inside ``_collect_overrides``. Not
    # persisted; the file list changes per run.
    "isolated_tracks_files",
    "isolated_tracks_labels",
    "inner_transcription_system",
]

SETTINGS_FIELDS = [
    "transcription_model",
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
    "voice_auto_selection",
    "voices",
    "tts_prompt_prefix",
    "enable_emotion_analysis",
    "emotion_provider",
    "emotion_model",
    "segment_reference_min_duration",
    "watermark_path",
    "watermark_text",
    "keep_original_audio_ranges",
    "min_pause_duration",
    "keyframe_buffer",
    "use_two_pass_encoding",
    "dubbed_volume",
    "background_volume",
    "timing_short_segment_threshold",
    "timing_short_segment_max_speed",
    "timing_max_speed",
    "timing_max_stretch",
    "timing_max_overflow",
    "semantic_split_enabled",
    "tts_preferred_segment_duration",
    "tts_hard_segment_duration",
    "semantic_split_search_window",
    "debug_info",
    "debug_tts",
    "debug_diarize_only",
]

ALL_FIELDS = WORKFLOW_FIELDS + SETTINGS_FIELDS
NON_PERSISTED_FIELDS = {
    "input",
    "output",
    "config",
    "run_step",
    "generate_speaker_report",
    "isolated_tracks_files",
    "isolated_tracks_labels",
}
PERSISTED_FIELDS = [field for field in ALL_FIELDS if field not in NON_PERSISTED_FIELDS]
JSON_TEXT_FIELDS = {"glossary"}
YAML_TEXT_FIELDS: set[str] = set()
LIST_TEXT_FIELDS = {"keep_original_audio_ranges"}
SPEAKER_REFERENCE_LIBRARY_HEADERS = ["Speaker ID", "Saved audio path", "Reference text"]
VOICE_PROFILE_HEADERS = ["Speaker ID", "TTS system", "Model", "Voice name", "Reference mode"]
VOICE_PROFILE_FIELDS = (
    "tts_system",
    "model",
    "voice_name",
    "style_prompt",
    "reference_audio",
    "reference_text",
    "reference_mode",
)

_TTS_MODEL_CHOICES = {
    "gemini": [GeminiTTSConfig.model_fields["model"].default],
    "openai": ["tts-1", "tts-1-hd"],
}
_TTS_VOICE_CHOICES = {
    "gemini": list(ALL_GEMINI_VOICES),
    "openai": list(ALL_OPENAI_VOICES),
}
_TTS_REFERENCE_CAPABILITIES = {
    "coqui": "required",
    "xtts": "required",
    "f5": "required",
    "f5_tts": "required",
    "omnivoice": "required",
    "higgs": "required",
    "bextts": "optional",
    "gemini": "unsupported",
    "openai": "unsupported",
}
OBSOLETE_TTS_KEYS = {
    "tts_system_mapping",
    "voice_prompt",
    "reference_audio_mapping",
    "reference_text_mapping",
    "tts_fallback_model",
    "tts_system",
    "tts_model",
    "voice_name",
    "reference_audio",
    "reference_text",
}
DUBBING_TEXT_HEADERS = [
    "Speaker",
    "Start",
    "End",
    "Original",
    "Translation",
    "Synthesized text",
    "Style instructions",
    "Audio file",
]
DUBBING_TEXT_COLUMN_WIDTHS = ["8%", "7%", "7%", "18%", "22%", "18%", "13%", "7%"]
DUBBING_TEXT_COLUMN_COUNT = len(DUBBING_TEXT_HEADERS)
DUBBING_TEXT_EMPTY_ROW = [""] * DUBBING_TEXT_COLUMN_COUNT
TRANSLATION_TRACK_FIELDS = (
    "translation",
    "short_translation",
    "very_short_translation",
    "long_translation",
)

_TRANSCRIPTION_MODEL_CHOICES: dict[str, list[str]] = {
    "whisper": ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
    "openai": ["whisper-1"],
    "pyannote_openai": ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
    "whisperx": ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
    "assemblyai": ["best", "nano"],
    "gemini": ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "deepgram": ["nova-3", "nova-2", "nova", "enhanced", "base", "whisper"],
}

_EMOTION_MODEL_CHOICES: list[str] = [
    "gemini-3.1-flash-lite",
    "gemini-3-flash-preview",
    "gemini-2.5-flash",
    "gemma-3-27b-it",
    "gemma-3-12b-it",
    "gemma-3-4b-it",
]

_TRANSCRIPTION_MODEL_DEFAULTS: dict[str, str] = {
    "whisper": "large-v3",
    "openai": "whisper-1",
    "pyannote_openai": "large-v3",
    "whisperx": "large-v3",
    "assemblyai": "best",
    "gemini": "gemini-3-flash-preview",
    "deepgram": "nova-3",
}


def _get_model_choices(system: str) -> list[str]:
    return _TRANSCRIPTION_MODEL_CHOICES.get(system, [])


def _get_initial_transcription_model(defaults: dict) -> tuple[str, str, list[str]]:
    system = str(defaults.get("transcription_system") or "whisper")
    model = (
        defaults.get("transcription_model")
        or defaults.get("whisper_model")
        or defaults.get("gemini_transcription_model")
        or defaults.get("deepgram_model")
        or _TRANSCRIPTION_MODEL_DEFAULTS.get(system, "")
    )
    choices = _get_model_choices(system)
    return system, str(model), choices


def _update_transcription_model_choices(system: str, current_model: str | None = None):
    choices = _get_model_choices(system)
    default = _TRANSCRIPTION_MODEL_DEFAULTS.get(system, "")
    new_val = current_model if current_model and current_model in choices else default
    return gr.update(choices=choices, value=new_val)


def _profile_to_dict(profile: VoiceProfile | Mapping[str, object]) -> dict[str, object]:
    if isinstance(profile, VoiceProfile):
        source = profile
    else:
        source = VoiceProfile(
            **{
                field: profile.get(field)
                for field in VOICE_PROFILE_FIELDS
                if profile.get(field) is not None
            },
            params=dict(profile.get("params") or {}),
        )
    return {
        "tts_system": source.tts_system,
        "model": source.model,
        "voice_name": source.voice_name,
        "style_prompt": source.style_prompt,
        "reference_audio": source.reference_audio,
        "reference_text": source.reference_text,
        "reference_mode": source.reference_mode,
        "params": dict(source.params),
    }


def voice_profiles_to_state(config: Mapping[str, object]) -> dict[str, dict[str, object]]:
    """Convert compatible config shapes to plain Gradio profile state."""
    return {
        speaker: _profile_to_dict(profile)
        for speaker, profile in normalize_voices(config).items()
    }


def _normalize_profile_state(state: object) -> dict[str, dict[str, object]]:
    if not isinstance(state, Mapping):
        return {}
    normalized: dict[str, dict[str, object]] = {}
    for speaker, profile in state.items():
        if isinstance(profile, (VoiceProfile, Mapping)):
            normalized[str(speaker)] = _profile_to_dict(profile)
    return normalized


def voice_profile_table_rows(state: object) -> list[list[str]]:
    profiles = _normalize_profile_state(state)
    return [
        [
            speaker,
            str(profile.get("tts_system") or ""),
            str(profile.get("model") or ""),
            str(profile.get("voice_name") or ""),
            str(profile.get("reference_mode") or ""),
        ]
        for speaker, profile in profiles.items()
    ]


def get_tts_profile_choices(
    tts_system: str | None,
    current_model: str | None = None,
    current_voice: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    provider = str(tts_system or "").lower()
    models = list(_TTS_MODEL_CHOICES.get(provider, []))
    voices = list(_TTS_VOICE_CHOICES.get(provider, []))
    if current_model and current_model not in models:
        models.append(str(current_model))
    if current_voice and current_voice not in voices:
        voices.append(str(current_voice))
    capability = _TTS_REFERENCE_CAPABILITIES.get(provider, "unsupported")
    modes = {
        "required": ["configured", "segment", "speaker"],
        "optional": ["none", "configured", "segment", "speaker"],
    }.get(capability, [])
    return models, voices, modes


def _profile_objects(state: object) -> dict[str, VoiceProfile]:
    return {
        speaker: VoiceProfile(
            **{
                field: profile.get(field)
                for field in VOICE_PROFILE_FIELDS
                if profile.get(field) is not None
            },
            params=dict(profile.get("params") or {}),
        )
        for speaker, profile in _normalize_profile_state(state).items()
    }


def _validate_voice_profile_state(
    state: object,
    speaker_id: str,
) -> None:
    if speaker_id != "*" and not re.fullmatch(r"SPEAKER_\d+", speaker_id):
        raise ValueError("Speaker ID must be '*' or match SPEAKER_XX.")
    profiles = _profile_objects(state)
    effective = resolve_profile(profiles, speaker_id)
    provider = str(effective.tts_system or "").lower()
    if not provider:
        raise ValueError("The effective profile must define tts_system.")
    if provider not in TTSFactory.get_available_providers() and provider != "f5_tts":
        raise ValueError(f"Unknown TTS system: {provider}.")
    if provider in {"gemini", "openai"} and not effective.model:
        raise ValueError(f"A model is required for {provider}.")

    capability = _TTS_REFERENCE_CAPABILITIES.get(provider, "unsupported")
    mode = effective.reference_mode
    if capability == "unsupported" and mode:
        raise ValueError(f"{provider} does not support reference_mode.")
    if capability in {"required", "optional"} and not mode:
        raise ValueError(f"An explicit reference_mode is required for {provider}.")
    if capability == "required" and mode == "none":
        raise ValueError(f"reference_mode 'none' is not allowed for {provider}.")
    if mode == "configured":
        configured_path = str(effective.reference_audio or "").strip()
        path = Path(configured_path).expanduser() if configured_path else None
        if path is None or not path.is_file():
            raise ValueError(
                f"Reference file does not exist: {configured_path or '<missing>'}."
            )


def save_voice_profile(
    state: object,
    selected_speaker: str | None,
    speaker_id: str,
    tts_system: str,
    model: str,
    voice_name: str,
    style_prompt: str,
    reference_mode: str,
    reference_audio: str,
    reference_text: str,
    params_yaml: str,
) -> tuple[str, dict[str, dict[str, object]], list[list[str]], str | None]:
    current = _normalize_profile_state(state)
    speaker = str(speaker_id or "").strip()
    if speaker != "*" and not re.fullmatch(r"SPEAKER_\d+", speaker):
        return (
            "Speaker ID must be '*' or match SPEAKER_XX.",
            current,
            voice_profile_table_rows(current),
            selected_speaker,
        )
    if speaker in current and speaker != selected_speaker:
        return (
            f"Profile {speaker} already exists.",
            current,
            voice_profile_table_rows(current),
            selected_speaker,
        )
    try:
        parsed_params = yaml.safe_load(params_yaml) if str(params_yaml or "").strip() else {}
    except yaml.YAMLError as exc:
        return (
            f"Invalid params YAML: {exc}",
            current,
            voice_profile_table_rows(current),
            selected_speaker,
        )
    if not isinstance(parsed_params, dict):
        return (
            "Profile params must decode to a mapping.",
            current,
            voice_profile_table_rows(current),
            selected_speaker,
        )

    candidate = dict(current)
    if selected_speaker and selected_speaker != speaker:
        candidate.pop(selected_speaker, None)
    candidate[speaker] = {
        "tts_system": str(tts_system or "").strip() or None,
        "model": str(model or "").strip() or None,
        "voice_name": str(voice_name or "").strip() or None,
        "style_prompt": str(style_prompt or "").strip() or None,
        "reference_audio": str(reference_audio or "").strip() or None,
        "reference_text": str(reference_text or "").strip() or None,
        "reference_mode": str(reference_mode or "").strip() or None,
        "params": parsed_params,
    }
    affected_speakers = candidate if speaker == "*" or selected_speaker == "*" else [speaker]
    try:
        for affected_speaker in affected_speakers:
            _validate_voice_profile_state(candidate, affected_speaker)
    except ValueError as exc:
        return str(exc), current, voice_profile_table_rows(current), selected_speaker
    return f"Saved profile {speaker}.", candidate, voice_profile_table_rows(candidate), speaker


def delete_voice_profile(
    state: object,
    selected_speaker: str | None,
) -> tuple[str, dict[str, dict[str, object]], list[list[str]], str | None]:
    current = _normalize_profile_state(state)
    if not selected_speaker or selected_speaker not in current:
        return "Select a profile first.", current, voice_profile_table_rows(current), None
    updated = dict(current)
    updated.pop(selected_speaker)
    if selected_speaker == "*":
        try:
            for speaker in updated:
                _validate_voice_profile_state(updated, speaker)
        except ValueError as exc:
            return str(exc), current, voice_profile_table_rows(current), selected_speaker
    return (
        f"Deleted profile {selected_speaker}.",
        updated,
        voice_profile_table_rows(updated),
        None,
    )


def assign_library_reference_to_profile(
    selected_row: object,
    state: object,
    selected_speaker: str | None,
) -> tuple[str, dict[str, dict[str, object]], list[list[str]]]:
    current = _normalize_profile_state(state)
    if not selected_speaker or selected_speaker not in current:
        return "Select a voice profile first.", current, voice_profile_table_rows(current)
    if not isinstance(selected_row, (list, tuple)) or len(selected_row) < 3:
        return "Select one library row first.", current, voice_profile_table_rows(current)
    label = str(selected_row[0] or "").strip()
    audio_path = str(selected_row[1] or "").strip()
    reference_text = str(selected_row[2] or "").strip()
    if not label or not audio_path:
        return "Selected library row is incomplete.", current, voice_profile_table_rows(current)
    updated = {speaker: dict(profile) for speaker, profile in current.items()}
    updated[selected_speaker]["reference_audio"] = audio_path
    updated[selected_speaker]["reference_text"] = reference_text or None
    updated[selected_speaker]["reference_mode"] = "configured"
    try:
        _validate_voice_profile_state(updated, selected_speaker)
    except ValueError as exc:
        return str(exc), current, voice_profile_table_rows(current)
    return (
        f"Assigned library entry '{label}' to {selected_speaker}.",
        updated,
        voice_profile_table_rows(updated),
    )


def _assign_library_reference_in_ui(
    selected_row: object,
    state: object,
    selected_speaker: str | None,
):
    status, updated, rows = assign_library_reference_to_profile(
        selected_row, state, selected_speaker
    )
    profile = updated.get(selected_speaker or "", {})
    mode = str(profile.get("reference_mode") or "")
    provider = str(profile.get("tts_system") or "")
    modes = get_tts_profile_choices(provider)[2]
    return (
        status,
        updated,
        rows,
        profile.get("reference_audio") or "",
        profile.get("reference_text") or "",
        gr.update(choices=modes, value=mode if mode in modes else None),
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


def _save_library_reference(
    speaker_id: str,
    reference_audio_file: str,
    reference_text: str,
):
    speaker_id = str(speaker_id or "").strip()
    reference_text = str(reference_text or "").strip()
    if not speaker_id:
        return (
            "Speaker ID is required to save a library reference.",
            load_speaker_reference_library(),
            None,
            reference_text,
            speaker_id,
        )
    if not reference_audio_file:
        return (
            "Reference audio file is required to save a library reference.",
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
    # Library entries accept any label ("MaleDeep", "Anchor", "SPEAKER_01",
    # …). The mapping table below is what has to match the diarization IDs,
    # not the library.
    library_rows = load_speaker_reference_library()
    return (
        f"Saved speaker reference for {speaker_id} to library.",
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


def load_ui_defaults(config_path: str = DEFAULT_CONFIG_PATH) -> dict[str, object]:
    persisted_config = _load_yaml_mapping(config_path)
    config = DubbingConfig()
    config.load_from_yaml(config_path)
    from ..core.timing import normalize_timing_config
    normalize_timing_config(config.config, warn=LOGGER.warning)

    defaults = config.to_dict()
    for field in JSON_TEXT_FIELDS:
        value = defaults.get(field)
        if value is not None:
            defaults[field] = json.dumps(value, ensure_ascii=False, indent=2)

    defaults["voices"] = voice_profiles_to_state(persisted_config)

    for field in LIST_TEXT_FIELDS:
        value = defaults.get(field)
        if isinstance(value, list):
            defaults[field] = "\n".join(str(item) for item in value)

    # These are UI-only fields and are not stored in YAML.  Keep the same
    # initial two-speaker value when ``app.load`` refreshes component values.
    defaults["isolated_tracks_labels"] = "SPEAKER_00, SPEAKER_01"
    defaults["config"] = config_path
    return defaults


def save_settings(
    overrides: dict[str, object],
    *,
    config_path: str = DEFAULT_CONFIG_PATH,
) -> str:
    config_data = _load_yaml_mapping(config_path)
    config_data.pop("group_overflow_tolerance", None)
    for obsolete_key in OBSOLETE_TTS_KEYS:
        config_data.pop(obsolete_key, None)

    for field in PERSISTED_FIELDS:
        if field not in overrides:
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

        if field == "voices":
            value = _normalize_profile_state(value)
            for speaker in value:
                _validate_voice_profile_state(value, speaker)
        elif isinstance(value, str) and field in JSON_TEXT_FIELDS:
            value = json.loads(value)
        elif isinstance(value, str) and field in YAML_TEXT_FIELDS:
            parsed = yaml.safe_load(value)
            value = parsed if isinstance(parsed, dict) else None
        elif isinstance(value, str) and field in LIST_TEXT_FIELDS:
            value = [line.strip() for line in value.splitlines() if line.strip()]

        if value is None:
            config_data.pop(field, None)
        else:
            config_data[field] = value

    from ..core.timing import normalize_timing_config
    normalize_timing_config(config_data, warn=LOGGER.warning)
    with Path(config_path).open("w", encoding="utf-8") as config_file:
        yaml.safe_dump(config_data, config_file, sort_keys=False, allow_unicode=True)

    return f"Settings saved to {config_path}"


def _collect_overrides(*values) -> dict[str, object]:
    overrides = dict(zip(ALL_FIELDS, values))
    overrides = _expand_isolated_tracks(overrides)
    return overrides


def _expand_isolated_tracks(overrides: dict[str, object]) -> dict[str, object]:
    """Fold the two UI-only inputs (multi-file upload + comma-separated labels)
    into the single ``isolated_tracks`` mapping the pipeline expects.

    The `inner_transcription_system` value stays as-is (it's a regular config
    key). UI-only fields are removed so they don't reach ``DubbingConfig``.
    """
    files = overrides.pop("isolated_tracks_files", None)
    labels_raw = overrides.pop("isolated_tracks_labels", None)

    if not files:
        # Nothing uploaded — leave `isolated_tracks` alone (falls back to
        # whatever YAML or previous state provided).
        return overrides

    # `files` is a list from gr.File(file_count="multiple"); paths are
    # tempfile._TemporaryFileWrapper-like objects or plain strings.
    file_paths: list[str] = []
    for item in files:
        if isinstance(item, str):
            file_paths.append(item)
        elif hasattr(item, "name"):
            file_paths.append(str(item.name))
        else:
            file_paths.append(str(item))

    labels: list[str]
    if isinstance(labels_raw, str) and labels_raw.strip():
        labels = [label.strip() for label in labels_raw.split(",") if label.strip()]
    else:
        labels = []

    if not labels:
        # Fall back to file stem so the mapping is still useful.
        labels = [Path(p).stem for p in file_paths]

    if len(labels) != len(file_paths):
        raise ValueError(
            f"Isolated tracks: got {len(file_paths)} files but "
            f"{len(labels)} speaker labels. Provide one label per file, "
            f"comma-separated, in the same order."
        )

    overrides["isolated_tracks"] = dict(zip(labels, file_paths))
    return overrides


def _build_dubbing_text_context(
    overrides: dict[str, object],
) -> tuple[DubbingConfig, Path, Path, Path, Path]:
    config = build_config_from_overrides(overrides)
    audio_path = Path(config.get("audio_artifacts_dir")) / "source.wav"
    if not audio_path.is_file():
        raise FileNotFoundError(
            f"Expected extracted source audio at {audio_path}. "
            "Run at least `transcribe_only` (or the full pipeline) once before editing dubbing texts."
        )

    cache_manager = CacheManager(use_cache=True, input_file=config.get("input"))
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    dubber.cache_manager = cache_manager
    cache_key = dubber._build_translation_cache_key(str(audio_path))
    cache_path = cache_manager.get_cache_path("translation") / f"{cache_key}.pkl"
    snapshot_key = dubber._build_dubbing_text_snapshot_key(str(audio_path))
    snapshot_path = cache_manager.get_cache_path("dubbing_texts") / f"{snapshot_key}.pkl"
    artifact_path = Path(config.get("artifacts_dir")) / "dubbing_texts.tsv"
    return config, cache_path, snapshot_path, artifact_path, audio_path


def _seed_segments_from_transcription(
    config: DubbingConfig, _audio_path: Path
) -> list[dict[str, object]]:
    """Build editable segments from the current ``transcribe_only`` artifact."""
    transcription_path = Path(config.get("transcription_path", ""))
    if not transcription_path.is_file():
        raise FileNotFoundError(
            f"Current transcription not found: {transcription_path}. "
            "Run `transcribe_only` or the full pipeline first."
        )

    timestamp_pattern = r"\d{2}\.\d{2}\.\d{2}(?:\.\d{1,3})?"
    line_pattern = re.compile(
        rf"^\[(?P<start>{timestamp_pattern})-"
        rf"(?P<end>{timestamp_pattern})\]\s+"
        r"(?P<speaker>[^:]+):\s?(?P<text>.*)$"
    )

    def parse_timestamp(value: str) -> float:
        parts = value.split(".")
        hours, minutes, seconds = (int(part) for part in parts[:3])
        milliseconds = int(parts[3].ljust(3, "0")) if len(parts) == 4 else 0
        return float(hours * 3600 + minutes * 60 + seconds) + milliseconds / 1000

    transcription: list[dict[str, object]] = []
    for line_number, raw_line in enumerate(
        transcription_path.read_text(encoding="utf-8").splitlines(), start=1
    ):
        line = raw_line.strip()
        if not line:
            continue
        match = line_pattern.match(line)
        if match is None:
            raise ValueError(
                f"Unexpected transcription format at {transcription_path}:{line_number}."
            )
        transcription.append(
            {
                "speaker": match.group("speaker").strip() or "SPEAKER_00",
                "start": parse_timestamp(match.group("start")),
                "end": parse_timestamp(match.group("end")),
                "text": match.group("text"),
            }
        )

    if not transcription:
        raise RuntimeError(f"Current transcription is empty: {transcription_path}")

    segments: list[dict[str, object]] = []
    for entry in transcription:
        text = str(entry["text"] or "").strip()
        speaker = str(entry["speaker"] or "SPEAKER_00")
        start = float(entry["start"] or 0.0)
        end = float(entry["end"] or 0.0)
        segments.append(
            {
                "speaker": speaker,
                "start": start,
                "end": end,
                "text": text,
                "translation": text,
                "short_translation": text,
                "very_short_translation": text,
                "long_translation": text,
                "emotion": "Neutral",
                "style_prompt": "",
            }
        )
    return segments


def _load_or_seed_segments(
    config: DubbingConfig, cache_path: Path, snapshot_path: Path, audio_path: Path
) -> tuple[list[dict[str, object]], str, bool, Path]:
    """Load editable segments from pipeline cache, latest-run snapshot, or transcription."""
    if snapshot_path.is_file():
        with snapshot_path.open("rb") as handle:
            payload = pickle.load(handle)
        if isinstance(payload, list):
            return payload, "snapshot", False, cache_path
        if not isinstance(payload, dict) or payload.get("version") != 1:
            raise ValueError(f"Unexpected Dubbing Texts snapshot in {snapshot_path}")
        segments = payload.get("segments")
        if not isinstance(segments, list):
            raise ValueError(f"Dubbing Texts snapshot has no segment list: {snapshot_path}")
        reusable = payload.get("translation_cache_reusable") is True
        translation_cache_key = payload.get("translation_cache_key")
        if reusable and isinstance(translation_cache_key, str) and translation_cache_key:
            cache_path = cache_path.parent / f"{translation_cache_key}.pkl"
        else:
            reusable = False
        return segments, "snapshot", reusable, cache_path
    if cache_path.is_file():
        return _load_cached_translation_segments(cache_path), "translation", True, cache_path
    transcription_reusable = not (
        config.get("isolated_tracks")
        and config.get("semantic_split_enabled", True)
    )
    return (
        _seed_segments_from_transcription(config, audio_path),
        "transcription",
        transcription_reusable,
        cache_path,
    )


def _write_dubbing_text_snapshot(
    snapshot_path: Path,
    segments: list[dict[str, object]],
    *,
    translation_cache_reusable: bool,
    translation_cache_path: Path,
) -> None:
    snapshot_path.parent.mkdir(parents=True, exist_ok=True)
    payload = {
        "version": 1,
        "segments": segments,
        "translation_cache_reusable": translation_cache_reusable,
        "translation_cache_key": (
            translation_cache_path.stem if translation_cache_reusable else None
        ),
    }
    with snapshot_path.open("wb") as handle:
        pickle.dump(payload, handle)


def _format_seconds(value: object) -> str:
    try:
        return f"{float(value or 0.0):.3f}"
    except (TypeError, ValueError):
        return "0.000"


def _parse_seconds(value: object) -> float:
    if value is None:
        return 0.0
    try:
        return float(str(value).strip() or 0.0)
    except (TypeError, ValueError) as exc:
        raise ValueError(f"Invalid timestamp '{value}': {exc}")


def _segment_to_dubbing_text_row(segment: dict) -> list[str]:
    audio_file = str(segment.get("synthesized_speech_file", "") or "")
    if audio_file:
        try:
            if not Path(audio_file).is_file():
                audio_file = f"(MISSING) {audio_file}"
        except (OSError, ValueError):
            audio_file = f"(MISSING) {audio_file}"
    return [
        str(segment.get("speaker", "") or ""),
        _format_seconds(segment.get("start")),
        _format_seconds(segment.get("end")),
        str(segment.get("text", "") or ""),
        str(segment.get("translation", "") or ""),
        str(segment.get("synthesized_text", "") or ""),
        str(segment.get("style_prompt", "") or ""),
        audio_file,
    ]


def _segments_to_dubbing_text_rows(segments: object) -> list[list[str]]:
    rows: list[list[str]] = []
    if not isinstance(segments, list):
        return rows

    for segment in segments:
        if not isinstance(segment, dict):
            continue
        rows.append(_segment_to_dubbing_text_row(segment))
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
                str(row[c]).strip() if c < len(row) and row[c] is not None else ""
                for c in range(DUBBING_TEXT_COLUMN_COUNT)
            ]
        )
    return normalized_rows


def _apply_row_to_segment(segment: dict, row: list[str]) -> None:
    """Mutate ``segment`` with edited values from a Dubbing Texts row."""
    segment["speaker"] = row[0]
    segment["start"] = _parse_seconds(row[1])
    segment["end"] = _parse_seconds(row[2])
    segment["text"] = row[3]

    translation = row[4]
    if not translation:
        raise ValueError("Translation text cannot be empty.")
    for field in TRANSLATION_TRACK_FIELDS:
        segment[field] = translation

    synthesized_text = row[5]
    if synthesized_text:
        segment["synthesized_text"] = synthesized_text
    segment["style_prompt"] = row[6]
    # Column 7 (audio file) is read-only.


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
        writer.writerow([
            "speaker",
            "start",
            "end",
            "original",
            "translation",
            "synthesized_text",
            "style_prompt",
            "audio_file",
        ])
        for row in rows:
            writer.writerow(row)


def load_dubbing_text_rows(overrides: dict[str, object]) -> tuple[str, list[list[str]]]:
    try:
        (
            config,
            cache_path,
            snapshot_path,
            _artifact_path,
            audio_path,
        ) = _build_dubbing_text_context(overrides)
        cached_segments, source, _reusable, _active_cache_path = _load_or_seed_segments(
            config, cache_path, snapshot_path, audio_path
        )
        rows = _segments_to_dubbing_text_rows(cached_segments)
        missing_indexes = [
            i for i, row in enumerate(rows)
            if not row[7] or row[7].startswith("(MISSING)")
        ]
        missing_note = (
            f" {len(missing_indexes)} segment(s) missing audio (rows: {missing_indexes[:20]}"
            f"{'…' if len(missing_indexes) > 20 else ''}). Select a row and click "
            "`Regenerate selected row` to re-run TTS for it."
            if missing_indexes else ""
        )
        if source == "transcription":
            status = (
                f"Loaded {len(rows)} row(s) from current transcription — no translations yet. "
                "Edit the `Translation` column and click `Save texts` to create the translation cache."
                + missing_note
            )
        elif source == "snapshot":
            status = (
                f"Loaded {len(rows)} dubbing text row(s) from the latest run snapshot."
                + missing_note
            )
        else:
            status = f"Loaded {len(rows)} dubbing text row(s)." + missing_note
        return status, rows
    except Exception as exc:
        return f"Failed: {exc}", []


def save_dubbing_text_rows(rows: object, overrides: dict[str, object]) -> tuple[str, list[list[str]]]:
    normalized_rows = _normalize_dubbing_text_rows(rows)
    try:
        (
            config,
            cache_path,
            snapshot_path,
            artifact_path,
            audio_path,
        ) = _build_dubbing_text_context(overrides)
        cached_segments, source, reusable, active_cache_path = _load_or_seed_segments(
            config, cache_path, snapshot_path, audio_path
        )
        if len(normalized_rows) != len(cached_segments):
            raise ValueError(
                f"Edited row count ({len(normalized_rows)}) does not match cached segment count ({len(cached_segments)})."
            )

        for segment, row in zip(cached_segments, normalized_rows):
            _apply_row_to_segment(segment, row)

        saved_rows = [_segment_to_dubbing_text_row(seg) for seg in cached_segments]

        _write_dubbing_text_snapshot(
            snapshot_path,
            cached_segments,
            translation_cache_reusable=reusable,
            translation_cache_path=active_cache_path,
        )
        if reusable:
            active_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with active_cache_path.open("wb") as handle:
                pickle.dump(cached_segments, handle)
        _write_dubbing_text_artifact(artifact_path, saved_rows)
        prefix = (
            "Created translation cache with"
            if source == "transcription" and reusable
            else "Saved"
        )
        return f"{prefix} {len(saved_rows)} dubbing text row(s).", saved_rows
    except Exception as exc:
        return f"Failed: {exc}", normalized_rows


def regenerate_dubbing_text_row(
    rows: object,
    selected_index: object,
    overrides: dict[str, object],
) -> tuple[str, list[list[str]]]:
    import traceback

    normalized_rows = _normalize_dubbing_text_rows(rows)
    try:
        try:
            row_index = int(selected_index) if selected_index is not None else -1
        except (TypeError, ValueError):
            row_index = -1
        if row_index < 0 or row_index >= len(normalized_rows):
            raise ValueError(
                "Click a cell in the table first so a row is selected, then click Regenerate."
            )

        (
            config,
            cache_path,
            snapshot_path,
            artifact_path,
            audio_path,
        ) = _build_dubbing_text_context(overrides)
        cached_segments, _source, reusable, active_cache_path = _load_or_seed_segments(
            config, cache_path, snapshot_path, audio_path
        )
        if len(normalized_rows) != len(cached_segments):
            raise ValueError(
                f"Edited row count ({len(normalized_rows)}) does not match cached segment count ({len(cached_segments)}). "
                "Save your edits or reload the table before regenerating."
            )

        # Apply every edit from the table first so the segment we're about to
        # resynthesize reflects any recent user changes.
        for segment, row in zip(cached_segments, normalized_rows):
            _apply_row_to_segment(segment, row)

        segment_dict = cached_segments[row_index]
        # Prefer explicit `Synthesized text` override; fall back to edited
        # `Translation` when the user has not filled the synthesized column
        # (typical for the initial regenerate-after-skip flow).
        synthesized_override = (normalized_rows[row_index][5] or "").strip()
        translation_text = (normalized_rows[row_index][4] or "").strip()
        override_text = synthesized_override or translation_text or None

        dubber = SmartDubbing(config)

        # Ensure per-speaker reference audio exists before running TTS —
        # otherwise cloning backends will silently skip the segment again.
        try:
            speakers_dir = Path(dubber.speakers_audio_dir)
            speakers_dir.mkdir(parents=True, exist_ok=True)
            speaker = str(segment_dict.get("speaker", "") or "")
            speaker_wav = speakers_dir / f"{speaker}.wav" if speaker else None
            if speaker_wav is not None and not speaker_wav.is_file():
                speakers_rolls = dubber._build_speaker_rolls_from_segments(cached_segments)
                dubber.speaker_processor.extract_speaker_audio(str(audio_path), speakers_rolls)
        except Exception as ref_exc:
            # Non-fatal — resynthesize_one_segment may still find another
            # reference via the active voice profile / segment ref clip.
            _ = ref_exc

        dubber.resynthesize_one_segment(
            segments=cached_segments,
            segment_index=row_index,
            override_text=override_text,
        )

        _write_dubbing_text_snapshot(
            snapshot_path,
            cached_segments,
            translation_cache_reusable=reusable,
            translation_cache_path=active_cache_path,
        )
        if reusable:
            active_cache_path.parent.mkdir(parents=True, exist_ok=True)
            with active_cache_path.open("wb") as handle:
                pickle.dump(cached_segments, handle)
        saved_rows = [_segment_to_dubbing_text_row(seg) for seg in cached_segments]
        _write_dubbing_text_artifact(artifact_path, saved_rows)
        audio_file = segment_dict.get("synthesized_speech_file") or "(missing)"
        return (
            f"Regenerated row {row_index}: {audio_file}",
            saved_rows,
        )
    except Exception as exc:
        detail = traceback.format_exc(limit=4)
        return f"Failed to regenerate row: {exc}\n{detail}", normalized_rows


def _load_dubbing_text_values(*values):
    return load_dubbing_text_rows(_collect_overrides(*values))


def _save_dubbing_text_values(rows, *values):
    return save_dubbing_text_rows(rows, _collect_overrides(*values))


def _regenerate_dubbing_text_value(rows, selected_index, *values):
    return regenerate_dubbing_text_row(rows, selected_index, _collect_overrides(*values))


def _store_selected_dubbing_row(evt: gr.SelectData) -> int:
    try:
        index = evt.index
    except Exception:
        return -1
    if isinstance(index, (list, tuple)) and index:
        try:
            return int(index[0])
        except (TypeError, ValueError):
            return -1
    try:
        return int(index) if index is not None else -1
    except (TypeError, ValueError):
        return -1


def _collect_values(*values):
    """Streaming variant: yields ``(status, logs, output_file, report_file, artifacts_path)``
    tuples so the Gradio UI updates the log textbox live instead of only after
    the whole pipeline finishes.
    """
    overrides = _collect_overrides(*values)

    for status, logs, result in run_dubbing_job_streaming(overrides):
        if result is None:
            # In-flight update — only status/logs are meaningful.
            yield status, logs, None, None, None
            continue

        output_file = result.output_file
        if output_file and not Path(output_file).is_file():
            artifacts_path = output_file
            output_file = None
        else:
            artifacts_path = None

        report_file = (
            result.report_file
            if result.report_file and Path(result.report_file).is_file()
            else None
        )
        yield status, logs, output_file, report_file, artifacts_path


def _save_values(*values):
    overrides = dict(zip(ALL_FIELDS, values))
    try:
        status = save_settings(overrides)
    except (ValueError, TypeError, yaml.YAMLError, json.JSONDecodeError) as exc:
        return f"Settings not saved: {exc}", "Configuration file was not changed."
    return status, f"Saved current settings to {DEFAULT_CONFIG_PATH}"


def _profile_editor_updates(tts_system: str, model: str, voice_name: str, reference_mode: str):
    models, voices, modes = get_tts_profile_choices(tts_system, model, voice_name)
    return (
        gr.update(choices=models, value=model or None),
        gr.update(choices=voices, value=voice_name or None),
        gr.update(
            choices=modes,
            value=reference_mode if reference_mode in modes else None,
        ),
    )


def _select_voice_profile(evt: gr.SelectData, state: object):
    row = getattr(evt, "row_value", None)
    speaker = str(row[0]).strip() if isinstance(row, (list, tuple)) and row else ""
    profile = _normalize_profile_state(state).get(speaker)
    if not profile:
        return (None, "", "", gr.update(choices=[], value=None), gr.update(choices=[], value=None), "", gr.update(choices=[], value=None), "", "", "")
    model_update, voice_update, mode_update = _profile_editor_updates(
        str(profile.get("tts_system") or ""),
        str(profile.get("model") or ""),
        str(profile.get("voice_name") or ""),
        str(profile.get("reference_mode") or ""),
    )
    params = profile.get("params") or {}
    params_text = yaml.safe_dump(params, sort_keys=False, allow_unicode=True).strip() if params else ""
    return (
        speaker,
        speaker,
        profile.get("tts_system") or None,
        model_update,
        voice_update,
        profile.get("style_prompt") or "",
        mode_update,
        profile.get("reference_audio") or "",
        profile.get("reference_text") or "",
        params_text,
    )


def _start_new_voice_profile():
    return (
        "New profile draft. Save profile to add it.",
        None,
        "",
        None,
        gr.update(choices=[], value=None),
        gr.update(choices=[], value=None),
        "",
        gr.update(choices=[], value=None),
        "",
        "",
        "",
    )


def _update_tts_profile_choice_components(
    tts_system: str,
    current_model: str | None,
    current_voice: str | None,
):
    models, voices, modes = get_tts_profile_choices(
        tts_system, current_model, current_voice
    )
    return (
        gr.update(choices=models, value=current_model or None),
        gr.update(choices=voices, value=current_voice or None),
        gr.update(choices=modes, value=None),
    )


def build_app(config_path: str = DEFAULT_CONFIG_PATH) -> gr.Blocks:
    """Build the Gradio interface."""
    defaults = load_ui_defaults(config_path)
    library_rows = load_speaker_reference_library()

    with gr.Blocks(title="DubbLM", theme=gr.themes.Soft()) as app:
        selected_library_row = gr.State(None)
        selected_voice_profile = gr.State(None)
        voice_profiles_state = gr.State(defaults.get("voices") or {})
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
                        choices=[
                            "full_pipeline",
                            "from_scratch",
                            "transcribe_only",
                            "translate_only",
                            "analyze_emotions_only",
                            "combine_video",
                            "tts_to_end",
                        ],
                        value=defaults.get("run_step") or "full_pipeline",
                        allow_custom_value=False,
                        info="`full_pipeline` — normal end-to-end run. `from_scratch` — clear cached artifacts for this input and rerun everything from zero. `transcribe_only` — stop after diarization + transcription (saves original subtitles when requested). `translate_only` — diarization + transcription + translation only (saves subtitles when requested). `analyze_emotions_only` — re-run emotion analysis over the cached translation and write emotion/style_prompt back into it (requires a prior translate_only or full run). Resume options require existing artifacts from a previous full run: `combine_video` rebuilds the final video from existing dubbed audio; `tts_to_end` restarts at cached translation data, regenerates TTS, and finishes the video.",
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

                gr.Markdown("### Isolated speaker tracks (optional)")
                gr.Markdown(
                    "Upload one clean audio file per speaker to bypass automatic "
                    "diarization. Speaker labels below (comma-separated, in the "
                    "same order as files) must match the keys used in your "
                    "`voices:` mapping. Leave empty for the standard pipeline.\n\n"
                    "**Tip:** to upload several files at once, hold **Ctrl** "
                    "(or **Shift**) in the file picker and select them all, "
                    "or drag multiple files together onto the upload area."
                )
                with gr.Row():
                    isolated_tracks_files = gr.File(
                        label="Isolated per-speaker audio files",
                        file_count="multiple",
                        type="filepath",
                    )
                    isolated_tracks_labels = gr.Textbox(
                        label="Speaker labels (comma-separated)",
                        placeholder="SPEAKER_00, SPEAKER_01",
                        value="SPEAKER_00, SPEAKER_01",
                    )
                    inner_transcription_system = gr.Dropdown(
                        label="Inner transcription (per track)",
                        choices=["deepgram", "assemblyai", "gemini"],
                        value=defaults.get("inner_transcription_system", "deepgram"),
                        info="Backend applied to each isolated track. Only used when files are uploaded above.",
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
                        isolated_tracks_files,
                        isolated_tracks_labels,
                        inner_transcription_system,
                    ]
                )



            with gr.Tab("Settings"):
                gr.Markdown("## Transcription")
                with gr.Row():
                    transcription_system = gr.Dropdown(
                        label="Transcription system",
                        choices=["whisper", "openai", "pyannote_openai", "whisperx", "assemblyai", "gemini", "deepgram"],
                        value=defaults.get("transcription_system", "whisper"),
                    )
                    _init_system, _init_model, _init_choices = _get_initial_transcription_model(defaults)
                    transcription_model = gr.Dropdown(
                        label="Model",
                        choices=_init_choices,
                        value=_init_model,
                        allow_custom_value=True,
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
                voice_profiles_table = gr.Dataframe(
                    headers=VOICE_PROFILE_HEADERS,
                    datatype=["str"] * len(VOICE_PROFILE_HEADERS),
                    row_count=(max(1, len(defaults.get("voices") or {})), "fixed"),
                    col_count=(len(VOICE_PROFILE_HEADERS), "fixed"),
                    label="Voice profiles",
                    value=voice_profile_table_rows(defaults.get("voices")),
                    type="array",
                    interactive=False,
                )
                with gr.Row():
                    profile_speaker_id = gr.Textbox(
                        label="Profile speaker ID",
                        placeholder="SPEAKER_00 or *",
                    )
                    profile_tts_system = gr.Dropdown(
                        label="Profile TTS system",
                        choices=TTSFactory.get_available_providers(),
                    )
                    profile_model = gr.Dropdown(
                        label="Profile model",
                        choices=[],
                        allow_custom_value=True,
                    )
                    profile_voice_name = gr.Dropdown(
                        label="Profile voice name",
                        choices=[],
                        allow_custom_value=True,
                    )
                profile_style_prompt = gr.Textbox(label="Profile style prompt", lines=2)
                with gr.Row():
                    profile_reference_mode = gr.Dropdown(
                        label="Profile reference mode",
                        choices=[],
                    )
                    profile_reference_audio = gr.Textbox(label="Profile reference audio")
                    profile_reference_text = gr.Textbox(label="Profile reference text")
                profile_params = gr.Textbox(
                    label="Profile params (YAML)",
                    lines=6,
                    placeholder="temperature: 0.7",
                )
                with gr.Row():
                    add_profile_button = gr.Button("Add profile")
                    save_profile_button = gr.Button("Save profile", variant="primary")
                    delete_profile_button = gr.Button("Delete profile", variant="stop")
                with gr.Row():
                    voice_auto_selection = gr.Checkbox(
                        label="Automatic voice selection",
                        value=bool(defaults.get("voice_auto_selection", True)),
                    )
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
                    use_selected_library_button = gr.Button("Use in selected profile")
                    delete_library_button = gr.Button("Delete selected from library", variant="stop")
                with gr.Row():
                    enable_emotion_analysis = gr.Checkbox(
                        label="Enable emotion analysis",
                        value=bool(defaults.get("enable_emotion_analysis", False)),
                    )
                    emotion_provider = gr.Dropdown(
                        label="Emotion provider",
                        choices=["gemini", "speechbrain"],
                        value=str(defaults.get("emotion_provider") or "gemini"),
                    )
                    emotion_model = gr.Dropdown(
                        label="Emotion model",
                        choices=_EMOTION_MODEL_CHOICES,
                        value=str(defaults.get("emotion_model") or _EMOTION_MODEL_CHOICES[0]),
                        allow_custom_value=True,
                        info="Model name for the Gemini emotion classifier. Ignored when provider=speechbrain.",
                    )
                    segment_reference_min_duration = gr.Number(
                        label="Min segment reference duration",
                        value=defaults.get("segment_reference_min_duration", 2.0),
                        precision=2,
                    )
                tts_prompt_prefix = gr.Textbox(label="TTS prompt prefix", lines=3, value=defaults.get("tts_prompt_prefix"))

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
                        value=defaults.get("min_pause_duration", 300.0),
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
                    timing_short_segment_threshold = gr.Number(
                        label="Short segment threshold",
                        value=defaults.get("timing_short_segment_threshold", 1.5),
                        precision=2,
                    )
                    timing_short_segment_max_speed = gr.Number(
                        label="Short segment max speed",
                        value=defaults.get("timing_short_segment_max_speed", 1.08),
                        precision=3,
                    )
                with gr.Row():
                    timing_max_speed = gr.Number(
                        label="Maximum timing speed",
                        value=defaults.get("timing_max_speed", 1.15),
                        precision=3,
                    )
                    timing_max_stretch = gr.Number(
                        label="Maximum timing stretch",
                        value=defaults.get("timing_max_stretch", 1.15),
                        precision=3,
                    )
                    timing_max_overflow = gr.Number(
                        label="Maximum timing overflow",
                        value=defaults.get("timing_max_overflow", 0.25),
                        precision=3,
                    )
                with gr.Row():
                    semantic_split_enabled = gr.Checkbox(
                        label="Semantic splitting",
                        value=bool(defaults.get("semantic_split_enabled", True)),
                    )
                    tts_preferred_segment_duration = gr.Number(
                        label="Preferred TTS segment duration",
                        value=defaults.get("tts_preferred_segment_duration", 15.0),
                        precision=2,
                    )
                with gr.Row():
                    tts_hard_segment_duration = gr.Number(
                        label="Hard TTS segment duration",
                        value=defaults.get("tts_hard_segment_duration", 35.0),
                        precision=2,
                    )
                    semantic_split_search_window = gr.Number(
                        label="Semantic split search window",
                        value=defaults.get("semantic_split_search_window", 10.0),
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
                        transcription_model,
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
                        voice_auto_selection,
                        voice_profiles_state,
                        tts_prompt_prefix,
                        enable_emotion_analysis,
                        emotion_provider,
                        emotion_model,
                        segment_reference_min_duration,
                        watermark_path,
                        watermark_text,
                        keep_original_audio_ranges,
                        min_pause_duration,
                        keyframe_buffer,
                        use_two_pass_encoding,
                        dubbed_volume,
                        background_volume,
                        timing_short_segment_threshold,
                        timing_short_segment_max_speed,
                        timing_max_speed,
                        timing_max_stretch,
                        timing_max_overflow,
                        semantic_split_enabled,
                        tts_preferred_segment_duration,
                        tts_hard_segment_duration,
                        semantic_split_search_window,
                        debug_info,
                        debug_tts,
                        debug_diarize_only,
                    ]
                )

            with gr.Tab("Dubbing Texts"):
                gr.Markdown(
                    "Load cached translation segments, edit any column (except the read-only `Audio file`), then save. "
                    "`Synthesized text` shows what was actually spoken by the TTS after best-variant selection and any auto-adjustments; "
                    "editing it and clicking `Regenerate selected row` runs TTS just for that segment. "
                    "`Style instructions` is a free-text style prompt passed to the TTS backend — for Gemini TTS "
                    "it is prepended to the utterance (e.g. `Say cheerfully:`). Populated automatically by emotion analysis "
                    "when enabled; you can override per row. "
                    "Saved edits are written to `artifacts/dubbing_texts.tsv` and to the translation cache used by `tts_to_end`."
                )
                with gr.Row():
                    load_dubbing_texts_button = gr.Button("Load texts")
                    save_dubbing_texts_button = gr.Button("Save texts")
                    regenerate_dubbing_row_button = gr.Button("Regenerate selected row", variant="secondary")
                dubbing_text_status = gr.Textbox(label="Dubbing text status", interactive=False)
                selected_dubbing_row_index = gr.State(-1)
                dubbing_text_rows = gr.Dataframe(
                    headers=DUBBING_TEXT_HEADERS,
                    datatype=["str"] * DUBBING_TEXT_COLUMN_COUNT,
                    row_count=(1, "dynamic"),
                    col_count=(DUBBING_TEXT_COLUMN_COUNT, "fixed"),
                    label="Dubbing texts",
                    value=[list(DUBBING_TEXT_EMPTY_ROW)],
                    type="array",
                    interactive=True,
                    wrap=True,
                    line_breaks=True,
                    show_search="search",
                    show_row_numbers=True,
                    pinned_columns=3,
                    static_columns=[7],
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
            ],
            outputs=[
                status,
                speaker_reference_library,
                library_reference_audio_file,
                library_reference_text,
                library_speaker_id,
            ],
        )
        voice_profiles_table.select(
            fn=_select_voice_profile,
            inputs=[voice_profiles_state],
            outputs=[
                selected_voice_profile,
                profile_speaker_id,
                profile_tts_system,
                profile_model,
                profile_voice_name,
                profile_style_prompt,
                profile_reference_mode,
                profile_reference_audio,
                profile_reference_text,
                profile_params,
            ],
        )
        add_profile_button.click(
            fn=_start_new_voice_profile,
            outputs=[
                status,
                selected_voice_profile,
                profile_speaker_id,
                profile_tts_system,
                profile_model,
                profile_voice_name,
                profile_style_prompt,
                profile_reference_mode,
                profile_reference_audio,
                profile_reference_text,
                profile_params,
            ],
        )
        save_profile_button.click(
            fn=save_voice_profile,
            inputs=[
                voice_profiles_state,
                selected_voice_profile,
                profile_speaker_id,
                profile_tts_system,
                profile_model,
                profile_voice_name,
                profile_style_prompt,
                profile_reference_mode,
                profile_reference_audio,
                profile_reference_text,
                profile_params,
            ],
            outputs=[status, voice_profiles_state, voice_profiles_table, selected_voice_profile],
        )
        delete_profile_button.click(
            fn=delete_voice_profile,
            inputs=[voice_profiles_state, selected_voice_profile],
            outputs=[status, voice_profiles_state, voice_profiles_table, selected_voice_profile],
        )
        profile_tts_system.change(
            fn=_update_tts_profile_choice_components,
            inputs=[profile_tts_system, profile_model, profile_voice_name],
            outputs=[profile_model, profile_voice_name, profile_reference_mode],
        )

        speaker_reference_library.select(
            fn=_store_selected_library_row,
            outputs=selected_library_row,
        )
        use_selected_library_button.click(
            fn=_assign_library_reference_in_ui,
            inputs=[selected_library_row, voice_profiles_state, selected_voice_profile],
            outputs=[
                status,
                voice_profiles_state,
                voice_profiles_table,
                profile_reference_audio,
                profile_reference_text,
                profile_reference_mode,
            ],
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
        dubbing_text_rows.select(
            fn=_store_selected_dubbing_row,
            outputs=selected_dubbing_row_index,
        )
        regenerate_dubbing_row_button.click(
            fn=_regenerate_dubbing_text_value,
            inputs=[dubbing_text_rows, selected_dubbing_row_index, *input_components],
            outputs=[dubbing_text_status, dubbing_text_rows],
        )
        run_button.click(
            fn=_collect_values,
            inputs=input_components,
            outputs=[status, logs, output_file, report_file, artifacts_path],
        )

        transcription_system.change(
            fn=_update_transcription_model_choices,
            inputs=[transcription_system, transcription_model],
            outputs=[transcription_model],
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
