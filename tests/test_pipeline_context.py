import inspect
from dataclasses import fields

import pytest

import dubbing.core.pipeline as pipeline_module
from dubbing.core.pipeline.context import (
    PipelineRunContext,
    commit_context,
    snapshot_context,
    validate_plan_dependent_segments,
)
from dubbing.core.smart_dubbing import SmartDubbing


CONTEXT_FIELDS = [
    "semantic_plan_fingerprint",
    "semantic_plan_cache_persistable",
    "plan_dependent_cache_allowed",
    "timing_source_audio_file",
    "timing_source_duration",
]


def test_pipeline_run_context_has_exactly_the_five_frozen_fields_and_defaults():
    context = PipelineRunContext()

    assert [field.name for field in fields(PipelineRunContext)] == CONTEXT_FIELDS
    assert context == PipelineRunContext(
        semantic_plan_fingerprint=None,
        semantic_plan_cache_persistable=True,
        plan_dependent_cache_allowed=True,
        timing_source_audio_file=None,
        timing_source_duration=None,
    )


def test_snapshot_context_reads_facade_mirrors_and_uses_only_frozen_defaults():
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber._semantic_plan_fingerprint = "plan-a"
    dubber._plan_dependent_cache_allowed = False
    dubber.unrelated_state = object()

    context = snapshot_context(dubber)

    assert context == PipelineRunContext(
        semantic_plan_fingerprint="plan-a",
        semantic_plan_cache_persistable=True,
        plan_dependent_cache_allowed=False,
        timing_source_audio_file=None,
        timing_source_duration=None,
    )
    assert vars(context).keys() == set(CONTEXT_FIELDS)


def test_commit_context_updates_only_fields_changed_by_the_completed_stage():
    dubber = SmartDubbing.__new__(SmartDubbing)
    unrelated_state = object()
    dubber.unrelated_state = unrelated_state
    context = PipelineRunContext(
        semantic_plan_fingerprint="plan-b",
        semantic_plan_cache_persistable=False,
        plan_dependent_cache_allowed=False,
        timing_source_audio_file="processed.wav",
        timing_source_duration=12.5,
    )

    commit_context(
        dubber,
        context,
        changed_fields={"semantic_plan_fingerprint", "timing_source_duration"},
    )

    assert dubber.__dict__ == {
        "unrelated_state": unrelated_state,
        "_semantic_plan_fingerprint": "plan-b",
        "_timing_source_duration": 12.5,
    }


def test_partial_new_facade_can_snapshot_commit_and_validate_context():
    dubber = SmartDubbing.__new__(SmartDubbing)

    context = snapshot_context(dubber)
    context.semantic_plan_fingerprint = "plan-c"
    context.semantic_plan_cache_persistable = False
    commit_context(
        dubber,
        context,
        changed_fields={
            "semantic_plan_fingerprint",
            "semantic_plan_cache_persistable",
        },
    )

    dubber._validate_plan_dependent_segments(
        [{"semantic_plan_fingerprint": "plan-c"}]
    )
    with pytest.raises(
        ValueError,
        match=(
            "Cached artifact semantic_plan_fingerprint is absent or does not "
            "match the active semantic plan"
        ),
    ):
        dubber._validate_plan_dependent_segments([{}])


def test_plan_dependent_validation_is_read_only_and_uses_context_fingerprint():
    context = PipelineRunContext(semantic_plan_fingerprint="plan-d")
    segments = [{"semantic_plan_fingerprint": "plan-d"}]
    original = [segment.copy() for segment in segments]

    validate_plan_dependent_segments(context, segments)

    assert segments == original

    with pytest.raises(ValueError):
        validate_plan_dependent_segments(context, [{}])
    validate_plan_dependent_segments(PipelineRunContext(), [{}])


def test_facade_validation_uses_active_run_context_and_direct_calls_snapshot():
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber._semantic_plan_fingerprint = "facade-plan"
    dubber._pipeline_run_context = PipelineRunContext(
        semantic_plan_fingerprint="active-plan"
    )

    dubber._validate_plan_dependent_segments(
        [{"semantic_plan_fingerprint": "active-plan"}]
    )

    del dubber._pipeline_run_context
    dubber._validate_plan_dependent_segments(
        [{"semantic_plan_fingerprint": "facade-plan"}]
    )


def test_run_transcribe_only_shares_one_context_and_commits_only_its_changes():
    dubber = SmartDubbing.__new__(SmartDubbing)
    seen_contexts = []

    def record_context():
        context = dubber._pipeline_run_context
        seen_contexts.append(context)
        return context

    class PerformanceTracker:
        def start_timing(self, _name):
            record_context()

    class SubtitleManager:
        def save_debug_tsv(self, *_args, **_kwargs):
            record_context()

    dubber.performance_tracker = PerformanceTracker()
    dubber.subtitle_manager = SubtitleManager()
    dubber.config = {
        "debug_dir": "debug",
        "transcription_path": "transcription.tsv",
    }
    dubber._reset_input_cache = lambda _reason: record_context()

    def prepare_audio_inputs():
        record_context().semantic_plan_fingerprint = "plan-from-stage"
        return "audio.wav", None, "audio.wav"

    dubber._prepare_audio_inputs = prepare_audio_inputs
    dubber.diarize_and_transcribe = lambda _audio: (
        record_context() and {(0.0, 1.0): "SPEAKER_00"},
        [{"speaker": "SPEAKER_00"}],
    )
    dubber._apply_speaker_filter = lambda segments: record_context() and segments
    dubber._save_requested_subtitles = lambda *_args, **_kwargs: record_context()
    dubber._cleanup = lambda: record_context()

    assert dubber.run_transcribe_only() == "transcription.tsv"
    assert len({id(context) for context in seen_contexts}) == 1
    assert dubber._semantic_plan_fingerprint == "plan-from-stage"
    assert not hasattr(dubber, "_semantic_plan_cache_persistable")
    assert not hasattr(dubber, "_plan_dependent_cache_allowed")
    assert not hasattr(dubber, "_timing_source_audio_file")
    assert not hasattr(dubber, "_timing_source_duration")
    assert not hasattr(dubber, "_pipeline_run_context")


@pytest.mark.parametrize(
    "method_name",
    [
        "run_transcribe_only",
        "run_translate_only",
        "run_analyze_emotions_only",
        "run_from_scratch",
        "run_from_tts",
        "run_pipeline",
    ],
)
def test_every_top_level_run_method_owns_a_pipeline_context(method_name):
    method = SmartDubbing.__dict__[method_name]

    assert method.__dict__.get("pipeline_context_owner") is True


def test_pipeline_context_callables_remain_internal_to_the_context_module():
    assert not hasattr(pipeline_module, "PipelineRunContext")
    assert not hasattr(pipeline_module, "snapshot_context")
    assert not hasattr(pipeline_module, "commit_context")
    assert not hasattr(pipeline_module, "validate_plan_dependent_segments")


FROZEN_FACADE = """
__init__|function|(self, config: DubbingConfig)
_get_torch_device|function|(self, device_str: Optional[str] = None) -> torch.device
_apply_speaker_filter|function|(self, segments: List[Dict]) -> List[Dict]
_initialize_translator|function|(self) -> None
_default_tts_system|function|(self) -> str
_resolve_voice_profile|function|(self, speaker: str) -> VoiceProfile
_profile_pool_key|function|(self, profile: VoiceProfile) -> tuple
_global_omnivoice_kwargs|function|(self) -> Dict[str, Any]
_build_tts_client|function|(self, profile: VoiceProfile) -> Any
_initialize_tts_systems|function|(self) -> None
_initialize_transcriber|function|(self) -> None
_format_component_init_error|function|(self, component_name: str, configured_backend: str, init_error: Optional[Exception]) -> str
_require_transcriber|function|(self)
_require_translator|function|(self)
_attach_segment_reference|function|(self, *, tts_segment_data_args: Dict[str, Any], segment_dict: Dict[str, Any], speaker: str, segment_index: int, original_audio_segment: Optional[AudioSegment], segment_reference_min_duration: float, segment_reference_min_duration_ms: int) -> tuple[Dict[str, Any], Optional[AudioSegment]]
_canonical_segment_index|staticmethod|(segment_dict: Dict[str, Any], chronological_index: int) -> int
_segment_reference_artifact_paths|function|(self, processed_source_path: Optional[str] = None) -> tuple[Path, Path]
_segment_reference_error|staticmethod|(speaker: str, segment_index: int, source_path: Path, reason: str) -> ValueError
_prepare_segment_reference|function|(self, *, segment_dict: Dict[str, Any], speaker: str, chronological_index: int, reuse_existing: bool, processed_source_path: Optional[str] = None, decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None) -> tuple[str, Optional[str]]
_prepare_audio_inputs|function|(self) -> tuple[str, Optional[str], str]
_cache_fingerprint|staticmethod|(dimensions: Dict[str, Any]) -> str
_effective_tts_cache_fingerprint|function|(self, speakers: Iterable[str]) -> str
_file_content_identity|staticmethod|(path: Optional[str]) -> str
_shared_audio_transcription_identity|function|(self, audio_file: str) -> str
_effective_translation_cache_dimensions|function|(self) -> Dict[str, Any]
_build_dubbing_text_snapshot_key|function|(self, audio_file: str) -> str
_build_translation_cache_key|function|(self, audio_file: str) -> str
_build_emotions_cache_key|function|(self, audio_file: str, segments: List[Dict[str, Any]], provider: Optional[str] = None, model: Optional[str] = None) -> str
_restore_semantic_plan_fingerprint|function|(self, audio_file: str) -> None
_validate_plan_dependent_segments|function|(self, segments: List[Dict[str, Any]]) -> None
_load_required_cached_step|function|(self, *, step_name: str, cache_key: str, hint: str) -> Any
_build_speaker_rolls_from_segments|function|(self, segments: List[Dict]) -> Dict[Tuple[float, float], str]
_save_requested_subtitles|function|(self, segments_for_output: List[Dict], *, save_original_subtitles: bool, save_translated_subtitles: bool, pause_adjustments: Optional[List[Dict[str, float]]] = None) -> None
_combine_final_video|function|(self, *, translated_audio_path: str, background_audio_path: Optional[str], speakers_rolls: Dict[Tuple[float, float], str]) -> tuple[str, List[Dict[str, float]]]
_persist_dubbing_text_snapshot|function|(self, segments: List[Dict], audio_file: str) -> None
_persist_synthesis_results|function|(self, segments: List[Dict], audio_file: str) -> None
_reset_input_cache|function|(self, reason: str) -> None
run_transcribe_only|function|(self, save_original_subtitles: bool = False) -> str
run_translate_only|function|(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
run_analyze_emotions_only|function|(self, save_translated_subtitles: bool = False) -> str
run_from_scratch|function|(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
run_from_tts|function|(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
run_pipeline|function|(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str
_handle_debug_diarize_only|function|(self, audio_file: str, speakers_rolls: Dict) -> str
generate_diarization_report|function|(self) -> Tuple[str, str]
_load_cached_diarize_and_transcribe|function|(self, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]
diarize_and_transcribe|function|(self, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]
_isolated_tracks_cache_key|function|(self, audio_file: str, isolated_tracks: Dict[str, str]) -> str
_isolated_tracks_raw_cache_key|function|(self, isolated_tracks: Dict[str, str]) -> str
_semantic_classifier|function|(self) -> Tuple[Optional[Any], str]
_semantic_classifier_identity|function|(self) -> Dict[str, Any]
_write_semantic_boundary_diagnostics|staticmethod|(path: str, records: List[Dict[str, Any]]) -> None
_diarize_and_transcribe_isolated|function|(self, audio_file: str, isolated_tracks: Dict[str, str]) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]
_isolated_inner_kwargs|function|(self, inner_system: str) -> Dict[str, Any]
translate_segments|function|(self, transcription: List[Dict], audio_file: str) -> List[Dict]
_build_translation_prompt_prefix|function|(self, base_prompt_prefix: Optional[str]) -> str
analyze_emotions|function|(self, segments: List[Dict], audio_file: str) -> List[Dict]
_analyze_emotions_gemini|function|(self, segments: List[Dict], audio_file: str, model: str) -> None
_analyze_emotions_speechbrain|function|(self, segments: List[Dict], audio_file: str) -> None
synthesize_speech|function|(self, segments: List[Dict], speakers_rolls: Dict, audio_file: str) -> str
_tts_selection_cache_fingerprint|function|(self, segments: List[Dict[str, Any]]) -> str
_synthesize_measured_candidates|function|(self, metadata: Dict[str, Any], policy) -> None
_load_or_synthesize_candidate|function|(self, metadata: Dict[str, Any], *, variant: str, text: str, attempts: int) -> Optional[Dict[str, Any]]
resynthesize_one_segment|function|(self, segments: List[Dict], segment_index: int, override_text: Optional[str] = None) -> Dict[str, Any]
rebuild_translated_audio_from_chunks|function|(self) -> Optional[str]
_save_transcription_file|function|(self, transcription: List[Dict]) -> None
_cleanup|function|(self) -> None
_get_tts_system_for_speaker|function|(self, speaker_id: str) -> str
_resolve_segment_reference|function|(self, *, tts_segment_data_args: Dict[str, Any], segment_dict: Dict[str, Any], profile: VoiceProfile, provider_capability: str, speaker: str, segment_index: int, original_audio_segment: Optional[AudioSegment], segment_reference_min_duration: float, for_resynthesis: bool = False, processed_source_path: Optional[str] = None, decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None) -> tuple[Dict[str, Any], Optional[AudioSegment]]
_preflight_tts_pools|staticmethod|(segments_by_pool: Dict[tuple, List[Any]], clients: Dict[tuple, Any], initial_issues: Optional[List[tuple[int, str]]] = None) -> None
_trim_trailing_silence|staticmethod|(audio_path: str, silence_threshold_db: float = -40.0, keep_tail_ms: int = 100, window_ms: int = 10) -> None
_measure_raw_tts_for_timing|function|(self, audio_path: str, segment_index: int) -> float
_segment_cache_metadata_path|staticmethod|(cache_path: Path) -> Path
_raw_tts_segment_cache_key|staticmethod|(*, base_cache_prefix: str, tts_system: str, segment: Dict[str, Any], speaker: str, translation: str, style_prompt: str, reference_audio_path: Optional[str], reference_mode: Optional[str] = None, reference_text: Optional[str] = None, client_pool_settings: Any = None, legacy_index: Any = 0, emotion: Optional[str] = 'Neutral', tts_prompt_prefix: Optional[str] = None, voice_prompt: Any = None) -> str
_cache_raw_tts_segment|function|(self, source_path: str, cache_path: Path, *, synthesized_text: str = '') -> None
_cached_segment_metadata|function|(self, cache_path: Path) -> Dict[str, Any]
_cached_segment_contract|function|(self, cache_path: Path) -> str
_adjust_and_combine_audio_grouped_legacy|function|(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]
_adjust_and_combine_audio_grouped|function|(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]
_get_subtitle_path|function|(self, subtitle_type: str, input_path: str, language: str) -> str
adjust_subtitle_timestamps|function|(self, segments: List[Dict], pause_adjustments: List[Dict[str, float]]) -> List[Dict]
""".strip()


def _normalized_signature(value):
    signature = str(inspect.signature(value))
    for module in (
        "typing.",
        "dubbing.core.config.",
        "dubbing.core.voice_profiles.",
        "pydub.audio_segment.",
        "pathlib._local.",
        "pathlib.",
    ):
        signature = signature.replace(module, "")
    return signature


def test_facade_descriptors_and_signatures_are_frozen():
    expected = {
        name: (descriptor, signature)
        for name, descriptor, signature in (
            line.split("|", 2) for line in FROZEN_FACADE.splitlines()
        )
    }

    facade_descriptors = {
        name
        for name, descriptor in SmartDubbing.__dict__.items()
        if inspect.isfunction(descriptor)
        or isinstance(descriptor, (staticmethod, classmethod))
    }
    assert facade_descriptors == expected.keys()

    actual = {}
    for name in expected:
        descriptor = SmartDubbing.__dict__[name]
        descriptor_name = (
            "staticmethod" if isinstance(descriptor, staticmethod) else "function"
        )
        target = descriptor.__func__ if isinstance(descriptor, staticmethod) else descriptor
        actual[name] = (descriptor_name, _normalized_signature(target))

    assert actual == expected
