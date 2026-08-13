"""Legacy UI field inventory and option lists, independent of web frameworks."""

from __future__ import annotations

from .provider_metadata import (
    GEMINI_TTS_DEFAULT_MODEL,
    GEMINI_VOICE_CHOICES,
    OPENAI_VOICE_CHOICES,
)


WORKFLOW_FIELDS = [
    "input", "source_language", "target_language", "output", "config", "run_step",
    "generate_speaker_report", "save_original_subtitles", "save_translated_subtitles",
    "keep_background", "include_original_audio", "remove_pauses",
    "isolated_tracks_files", "isolated_tracks_labels", "inner_transcription_system",
]

SETTINGS_FIELDS = [
    "transcription_model", "transcription_system", "start_time", "duration", "no_cache",
    "translator_type", "llm_provider", "llm_model_name", "llm_temperature",
    "translation_prompt_prefix", "glossary", "refinement_llm_provider",
    "refinement_model_name", "refinement_temperature", "refinement_max_tokens",
    "refinement_persona", "voice_auto_selection", "voices", "tts_prompt_prefix",
    "enable_emotion_analysis", "emotion_provider", "emotion_model",
    "segment_reference_min_duration", "watermark_path", "watermark_text",
    "keep_original_audio_ranges", "min_pause_duration", "keyframe_buffer",
    "use_two_pass_encoding", "dubbed_volume", "background_volume",
    "timing_short_segment_threshold", "timing_short_segment_max_speed", "timing_max_speed",
    "timing_max_stretch", "timing_max_overflow", "semantic_split_enabled",
    "tts_preferred_segment_duration", "tts_hard_segment_duration",
    "semantic_split_search_window", "debug_info", "debug_tts", "debug_diarize_only",
]

ALL_FIELDS = WORKFLOW_FIELDS + SETTINGS_FIELDS
SERVER_MANAGED_FIELDS = {"input", "output", "config"}
DEDICATED_UPLOAD_FIELDS = {"isolated_tracks_files", "isolated_tracks_labels"}
PUBLIC_SCHEMA_FIELDS = [
    field for field in ALL_FIELDS
    if field not in SERVER_MANAGED_FIELDS | DEDICATED_UPLOAD_FIELDS
]
NON_PERSISTED_FIELDS = {
    "input", "output", "config", "run_step", "generate_speaker_report",
    "isolated_tracks_files", "isolated_tracks_labels",
}
PERSISTED_FIELDS = [field for field in ALL_FIELDS if field not in NON_PERSISTED_FIELDS]
JSON_TEXT_FIELDS = {"glossary"}
YAML_TEXT_FIELDS: set[str] = set()
LIST_TEXT_FIELDS = {"keep_original_audio_ranges"}
OBSOLETE_TTS_KEYS = {
    "tts_system_mapping", "voice_prompt", "reference_audio_mapping",
    "reference_text_mapping", "tts_fallback_model", "tts_system", "tts_model",
    "voice_name", "reference_audio", "reference_text",
}

RUN_MODES = [
    "full_pipeline", "from_scratch", "transcribe_only", "translate_only",
    "analyze_emotions_only", "combine_video", "tts_to_end",
]
INNER_TRANSCRIPTION_SYSTEM_CHOICES = ["deepgram", "assemblyai", "gemini"]
TRANSCRIPTION_SYSTEM_CHOICES = [
    "whisper", "openai", "pyannote_openai", "whisperx", "assemblyai", "gemini", "deepgram",
]
LLM_PROVIDER_CHOICES = ["gemini", "openrouter"]
EMOTION_PROVIDER_CHOICES = ["gemini", "speechbrain"]
TTS_PROVIDER_CHOICES = [
    "coqui", "xtts", "f5", "openai", "gemini", "bextts", "omnivoice", "higgs",
]
REFINEMENT_PERSONA_CHOICES = [
    "normal", "casual_manager", "child", "housewife", "science_popularizer", "it_buddy", "ai_buddy",
]

BOOLEAN_FIELDS = {
    "generate_speaker_report", "save_original_subtitles", "save_translated_subtitles",
    "keep_background", "include_original_audio", "remove_pauses", "no_cache",
    "voice_auto_selection", "enable_emotion_analysis", "use_two_pass_encoding",
    "semantic_split_enabled", "debug_info", "debug_tts", "debug_diarize_only",
}
NUMBER_FIELDS = {
    "start_time", "duration", "llm_temperature", "refinement_temperature",
    "refinement_max_tokens", "segment_reference_min_duration", "min_pause_duration",
    "keyframe_buffer", "dubbed_volume", "background_volume",
    "timing_short_segment_threshold", "timing_short_segment_max_speed",
    "timing_max_speed", "timing_max_stretch", "timing_max_overflow",
    "tts_preferred_segment_duration", "tts_hard_segment_duration",
    "semantic_split_search_window",
}
OBJECT_FIELDS = {"glossary", "voices"}
LIST_FIELDS = {"keep_original_audio_ranges"}
SELECT_OPTION_KEYS = {
    "run_step": "run_modes",
    "inner_transcription_system": "inner_transcription_systems",
    "transcription_system": "transcription_systems",
    "llm_provider": "llm_providers",
    "refinement_llm_provider": "llm_providers",
    "refinement_persona": "refinement_personas",
    "emotion_provider": "emotion_providers",
    "emotion_model": "emotion_models",
}


def api_field(name: str) -> dict[str, object]:
    """Return the authoritative browser control metadata for one public field."""
    definition: dict[str, object] = {
        "name": name,
        "type": (
            "boolean" if name in BOOLEAN_FIELDS
            else "number" if name in NUMBER_FIELDS
            else "object" if name in OBJECT_FIELDS
            else "list" if name in LIST_FIELDS
            else "select" if name in SELECT_OPTION_KEYS
            else "string"
        ),
        "workflow": name in WORKFLOW_FIELDS,
    }
    options_key = SELECT_OPTION_KEYS.get(name)
    if options_key:
        definition["options_key"] = options_key
    return definition

TTS_MODEL_CHOICES = {
    "gemini": [GEMINI_TTS_DEFAULT_MODEL],
    "openai": ["tts-1", "tts-1-hd"],
}
TTS_VOICE_CHOICES = {
    "gemini": list(GEMINI_VOICE_CHOICES),
    "openai": list(OPENAI_VOICE_CHOICES),
}
TTS_REFERENCE_CAPABILITIES = {
    "coqui": "required", "xtts": "required", "f5": "required", "f5_tts": "required",
    "omnivoice": "required", "higgs": "required", "bextts": "optional",
    "gemini": "unsupported", "openai": "unsupported",
}
TRANSCRIPTION_MODEL_CHOICES = {
    "whisper": ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
    "openai": ["whisper-1"],
    "pyannote_openai": ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
    "whisperx": ["large-v3", "large-v2", "large", "medium", "small", "base", "tiny"],
    "assemblyai": ["best", "nano"],
    "gemini": ["gemini-3-flash-preview", "gemini-2.5-flash", "gemini-2.5-pro", "gemini-2.0-flash"],
    "deepgram": ["nova-3", "nova-2", "nova", "enhanced", "base", "whisper"],
}
EMOTION_MODEL_CHOICES = [
    "gemini-3.1-flash-lite", "gemini-3-flash-preview", "gemini-2.5-flash",
    "gemma-3-27b-it", "gemma-3-12b-it", "gemma-3-4b-it",
]
TRANSCRIPTION_MODEL_DEFAULTS = {
    "whisper": "large-v3", "openai": "whisper-1", "pyannote_openai": "large-v3",
    "whisperx": "large-v3", "assemblyai": "best", "gemini": "gemini-3-flash-preview",
    "deepgram": "nova-3",
}


def get_tts_profile_choices(
    tts_system: str | None,
    current_model: str | None = None,
    current_voice: str | None = None,
) -> tuple[list[str], list[str], list[str]]:
    """Return legacy profile editor options for a provider."""
    provider = str(tts_system or "").lower()
    models = list(TTS_MODEL_CHOICES.get(provider, []))
    voices = list(TTS_VOICE_CHOICES.get(provider, []))
    if current_model and current_model not in models:
        models.append(str(current_model))
    if current_voice and current_voice not in voices:
        voices.append(str(current_voice))
    modes = {
        "required": ["configured", "segment", "speaker"],
        "optional": ["none", "configured", "segment", "speaker"],
    }.get(TTS_REFERENCE_CAPABILITIES.get(provider, "unsupported"), [])
    return models, voices, modes
