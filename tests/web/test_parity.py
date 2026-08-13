"""Committed parity gate for removing the legacy Python UI."""

from dubbing.web import schema


# Extracted from the final legacy ``build_app`` input component order. Keep the
# expected values local to this gate so deleting or accidentally narrowing the
# production inventory cannot make the test pass by construction.
EXPECTED_WORKFLOW_FIELDS = [
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
    "isolated_tracks_files",
    "isolated_tracks_labels",
    "inner_transcription_system",
]

EXPECTED_SETTINGS_FIELDS = [
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

EXPECTED_RUN_MODES = [
    "full_pipeline",
    "from_scratch",
    "transcribe_only",
    "translate_only",
    "analyze_emotions_only",
    "combine_video",
    "tts_to_end",
]


def test_schema_keeps_every_extracted_legacy_field_in_component_order():
    assert schema.WORKFLOW_FIELDS == EXPECTED_WORKFLOW_FIELDS
    assert schema.SETTINGS_FIELDS == EXPECTED_SETTINGS_FIELDS
    assert schema.ALL_FIELDS == EXPECTED_WORKFLOW_FIELDS + EXPECTED_SETTINGS_FIELDS
    assert len(schema.ALL_FIELDS) == len(set(schema.ALL_FIELDS)) == 58


def test_schema_keeps_every_extracted_legacy_run_mode_in_choice_order():
    assert schema.RUN_MODES == EXPECTED_RUN_MODES
