"""Main SmartDubbing orchestrator class."""

# --- Suppress noisy library logs BEFORE any imports ---
import os
# Set TensorFlow log level to suppress INFO and WARNING messages.
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Changed to 3 for even more suppression
# Disable oneDNN custom operations log.
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'
# Suppress CUDA registration warnings
os.environ['CUDA_VISIBLE_DEVICES'] = os.environ.get('CUDA_VISIBLE_DEVICES', '0')
# --- End of suppression block ---

import time
import torch
import warnings
from functools import wraps
from typing import Dict, Iterable, List, Tuple, Optional, Any
from pathlib import Path
from pydub import AudioSegment

# Disable all warnings for a cleaner output.
warnings.filterwarnings("ignore")

# Import our components
from .config import DubbingConfig  # noqa: E402
from .cache_manager import CacheManager  # noqa: E402
from .voice_profiles import (  # noqa: E402
    VoiceProfile,
    normalize_voices,
    reject_legacy_voice_config,
)
from ..audio.audio_processor import AudioProcessor  # noqa: E402
from ..audio.speaker_processor import SpeakerProcessor  # noqa: E402
from ..video.video_processor import VideoProcessor  # noqa: E402
from ..debug.performance_tracker import PerformanceTracker  # noqa: E402
from ..debug.debug_generator import DebugGenerator  # noqa: E402
from ..debug.reporter import SpeakerReporter  # noqa: E402
from ..utils.subtitle_utils import SubtitleManager  # noqa: E402
from .log_config import get_logger  # noqa: E402
from .pipeline import artifacts as artifact_helpers  # noqa: E402
from .pipeline import audio_assembly as audio_assembly_helpers  # noqa: E402
from .pipeline import synthesis as synthesis_helpers  # noqa: E402
from .pipeline import cache_keys as cache_key_helpers  # noqa: E402
from .pipeline import emotions as emotion_helpers  # noqa: E402
from .pipeline import references as reference_helpers  # noqa: E402
from .pipeline import transcription as transcription_helpers  # noqa: E402
from .pipeline import translation as translation_helpers  # noqa: E402
from .pipeline.context import (  # noqa: E402
    active_context,
    commit_context,
    snapshot_context,
    validate_plan_dependent_segments,
)

# Import existing factories and interfaces
from translation.llm_translator import DEFAULT_LLM_MODELS  # noqa: E402

# Get logger
logger = get_logger(__name__)


EMOTION_ANALYSIS_PROMPT = (
    "You are annotating a short audio clip for a text-to-speech dubbing engine. "
    "Listen only for the speaker's emotional intensity and prosody. Produce ONE "
    "short imperative instruction (max ~12 words) describing subtle changes to "
    "rhythm, pauses, emphasis, or energy. Preserve the speaker's voice identity, "
    "timbre, pitch range, accent, and vocal character exactly. Do not ask the TTS "
    "model to imitate, transform, deepen, brighten, or otherwise change the voice. "
    "Do not describe the words spoken. Do not add quotes, JSON, or explanations. "
    "Also pick a single dominant emotion tag from: Neutral, Angry, Happy, Sad. "
    "Respond with exactly two lines in this format (no extra text):\n"
    "STYLE: <subtle prosody instruction>\n"
    "EMOTION: <one of Neutral|Angry|Happy|Sad>"
)


SOFT_STYLE_BY_EMOTION = {
    "Neutral": "Keep a natural conversational rhythm with subtle emphasis; preserve the voice, timbre, and pitch.",
    "Angry": "Use subtle firmer emphasis and tighter pauses; preserve the voice, timbre, and pitch.",
    "Happy": "Use a subtle brighter rhythm and gentle emphasis; preserve the voice, timbre, and pitch.",
    "Sad": "Use subtle slower pacing and gentler pauses; preserve the voice, timbre, and pitch.",
}


SMART_DUBBING_STRESS_MARKS_REQUIREMENT = (
    "For dubbing-ready text, add word stress marks where they help pronunciation. "
    "Use the combining acute accent symbol U+0301 directly after the stressed vowel, "
    "for example: каса́. Apply this to the main translation and to alternative "
    "very_short / short / long variants intended for TTS."
)


def _with_pipeline_context(method):
    """Give each outer run one shared context without changing its signature."""

    @wraps(method)
    def wrapper(self, *args, **kwargs):
        if getattr(self, "_pipeline_run_context", None) is not None:
            return method(self, *args, **kwargs)

        context = snapshot_context(self)
        initial_values = vars(context).copy()
        self._pipeline_run_context = context
        completed = False
        try:
            result = method(self, *args, **kwargs)
            completed = True
            return result
        finally:
            if completed:
                changed_fields = {
                    field_name
                    for field_name, initial_value in initial_values.items()
                    if getattr(context, field_name) != initial_value
                }
                commit_context(
                    self, context, changed_fields=changed_fields
                )
            if getattr(self, "_pipeline_run_context", None) is context:
                del self._pipeline_run_context

    wrapper.pipeline_context_owner = True
    return wrapper


def _update_pipeline_context(facade, field_name: str, value: Any) -> None:
    """Keep an active run context and its compatibility mirror synchronized."""
    setattr(active_context(facade), field_name, value)
    setattr(facade, f"_{field_name}", value)


class SmartDubbing:
    """
    A video dubbing system that transcribes, translates, and synthesizes speech for videos.
    Uses context-aware translation to produce more natural-sounding results.
    
    This is the main orchestrator class that coordinates all components.
    """
    
    def __init__(self, config: DubbingConfig):
        """
        Initialize the SmartDubbing system.
        
        Args:
            config: Configuration object containing all settings
        """
        raw_config = config.to_dict() if hasattr(config, "to_dict") else dict(config)
        reject_legacy_voice_config(raw_config, source="SmartDubbing config")
        self.config = config
        self.project_dir = Path(self.config.get("project_dir"))
        self.artifacts_root = Path(self.config.get("artifacts_dir"))
        self.audio_dir = Path(self.config.get("audio_artifacts_dir"))
        self.speakers_audio_dir = Path(self.config.get("speakers_audio_dir"))
        self.audio_chunks_dir = Path(self.config.get("audio_chunks_dir"))
        self.su_audio_chunks_dir = Path(self.config.get("su_audio_chunks_dir"))
        # Normalize plain mapping configs so downstream code always sees VoiceProfile objects.
        voices_cfg = self.config.get('voices')
        if not isinstance(voices_cfg, dict) or not all(
            isinstance(v, VoiceProfile) for v in voices_cfg.values()
        ):
            voices_cfg = normalize_voices(raw_config)
            if hasattr(self.config, "set"):
                self.config.set('voices', voices_cfg)
            else:
                self.config['voices'] = voices_cfg
        self.voice_profiles: Dict[str, VoiceProfile] = voices_cfg
        
        # Speakers to mute (remove entirely from output)
        self.muted_speakers = set()
        mute_cfg = self.config.get('mute_speakers')
        if isinstance(mute_cfg, str) and mute_cfg.strip():
            self.muted_speakers = {mute_cfg.strip()}
        elif isinstance(mute_cfg, (list, tuple, set)):
            self.muted_speakers = {str(s).strip() for s in mute_cfg if isinstance(s, (str,)) and str(s).strip()}
        if self.muted_speakers:
            logger.info(f"Muted speakers: {sorted(self.muted_speakers)}")

        # Initialize core components
        self.cache_manager = CacheManager(
            use_cache=not config.get('no_cache', False),
            input_file=config.get('input')
        )
        self.performance_tracker = PerformanceTracker()
        
        # Initialize processors
        self.audio_processor = AudioProcessor(self.cache_manager, self.performance_tracker, artifacts_root=str(self.artifacts_root))
        self.speaker_processor = SpeakerProcessor(self.cache_manager, self.performance_tracker, artifacts_root=str(self.artifacts_root))
        self.video_processor = VideoProcessor(self.performance_tracker, artifacts_root=str(self.artifacts_root))
        
        # Initialize utilities
        self.subtitle_manager = SubtitleManager()
        self.debug_generator = DebugGenerator(artifacts_root=str(self.artifacts_root))
        self.speaker_reporter = SpeakerReporter(self.performance_tracker, artifacts_root=str(self.artifacts_root))
        
        # Set device
        device = config.get('device', 'cuda' if torch.cuda.is_available() else 'cpu')
        self.device = device
        self.torch_device = self._get_torch_device(device)
        
        # Initialize debug data container
        self.debug_data = {
            "diarization": None,
            "transcription": None,
            "translation": None,
            "speed_ratios": {},
            "voices": {},
            "speaker_groups": {}
        }
        self.translator_init_error = None
        self.transcriber_init_error = None
        self.tts_init_error = None
        
        # Initialize real segment positions for pause removal
        self.real_segment_positions = []
        
        # Initialize pause adjustments for subtitle timing
        self.pause_adjustments = []
        
        # Initialize translator
        self._initialize_translator()
        
        # Initialize TTS systems
        self._initialize_tts_systems()
        
        # Initialize transcriber
        self._initialize_transcriber()
        
        logger.info(f"Initialized SmartDubbing with {self.device} device")
        logger.debug(f"Using {config.get('tts_system', 'coqui')} TTS system")
        if self.transcriber is not None:
            logger.debug(f"Using {self.transcriber.name} transcriber")
        
        if config.get('start_time') is not None or config.get('duration') is not None:
            start_str = f"from {config.get('start_time')}s" if config.get('start_time') is not None else "from beginning"
            duration_str = f"for {config.get('duration')}s" if config.get('duration') is not None else "to the end"
            logger.debug(f"Processing video segment {start_str} {duration_str}")
        
        if config.get('use_cache', True):
            logger.debug("Caching enabled: will use cached results when available")
            
        if config.get('debug_info', False):
            logger.debug("Debug mode enabled: will generate a debug video with speaker labels")
        
        if config.get('debug_diarize_only', False):
            logger.debug("Debug diarize-only mode enabled: will exit after diarization and transcription with debug video")
    
    def _get_torch_device(self, device_str: Optional[str] = None) -> torch.device:
        """Helper method to get a proper torch.device object."""
        if device_str is None:
            device_str = 'cuda' if torch.cuda.is_available() else 'cpu'
        return torch.device(device_str)

    def _apply_speaker_filter(self, segments: List[Dict]) -> List[Dict]:
        """Return segments with muted speakers removed (if configured)."""
        if not segments or not self.muted_speakers:
            return segments
        filtered = [s for s in segments if s.get("speaker") not in self.muted_speakers]
        if self.config.get('debug_info', False):
            removed = len(segments) - len(filtered)
            logger.debug(f"Speaker filter applied: muted={sorted(self.muted_speakers)} removed={removed} kept={len(filtered)}")
        return filtered
    
    def _initialize_translator(self) -> None:
        return translation_helpers.initialize_translator(self)
    
    def _default_tts_system(self) -> str:
        return synthesis_helpers.default_tts_system(self)

    def _resolve_voice_profile(self, speaker: str) -> VoiceProfile:
        return synthesis_helpers.resolve_voice_profile(self, speaker)

    def _profile_pool_key(self, profile: VoiceProfile) -> tuple:
        return synthesis_helpers.profile_pool_key(self, profile)

    def _global_omnivoice_kwargs(self) -> Dict[str, Any]:
        return synthesis_helpers.global_omnivoice_kwargs(self)

    def _build_tts_client(self, profile: VoiceProfile) -> Any:
        return synthesis_helpers.build_tts_client(self, profile)

    def _initialize_tts_systems(self) -> None:
        return synthesis_helpers.initialize_tts_systems(self)

    def _initialize_transcriber(self) -> None:
        return transcription_helpers.initialize_transcriber(self)

    def _format_component_init_error(
        self,
        component_name: str,
        configured_backend: str,
        init_error: Optional[Exception]
    ) -> str:
        """Build a clear runtime error when a required backend failed to initialize."""
        message = f"{component_name} '{configured_backend}' is not available"
        if init_error is not None:
            message = f"{message}: {init_error}"

        error_text = str(init_error or "")
        if "ASSEMBLYAI_API_KEY" in error_text:
            message += " Set ASSEMBLYAI_API_KEY or choose another transcription_system."
        elif "DEEPGRAM_API_KEY" in error_text:
            message += " Set DEEPGRAM_API_KEY or choose another transcription_system."
        elif "HF_TOKEN" in error_text:
            message += " Set HF_TOKEN or choose a backend that does not require Hugging Face authentication."
        elif "json_repair" in error_text:
            message += " Install json_repair in the active Python environment."

        return message

    def _require_transcriber(self):
        return transcription_helpers.require_transcriber(self)

    def _require_translator(self):
        return translation_helpers.require_translator(self)

    def _attach_segment_reference(
        self, *, tts_segment_data_args: Dict[str, Any], segment_dict: Dict[str, Any],
        speaker: str, segment_index: int, original_audio_segment: Optional[AudioSegment],
        segment_reference_min_duration: float, segment_reference_min_duration_ms: int,
    ) -> tuple[Dict[str, Any], Optional[AudioSegment]]:
        return reference_helpers.attach_segment_reference(
            self, tts_segment_data_args=tts_segment_data_args, segment_dict=segment_dict,
            speaker=speaker, segment_index=segment_index,
            original_audio_segment=original_audio_segment,
            segment_reference_min_duration=segment_reference_min_duration,
            segment_reference_min_duration_ms=segment_reference_min_duration_ms,
        )

    @staticmethod
    def _canonical_segment_index(segment_dict: Dict[str, Any], chronological_index: int) -> int:
        return reference_helpers.canonical_segment_index(segment_dict, chronological_index)

    def _segment_reference_artifact_paths(self, processed_source_path: Optional[str] = None) -> tuple[Path, Path]:
        return reference_helpers.segment_reference_artifact_paths(config=self.config, processed_source_path=processed_source_path)

    @staticmethod
    def _segment_reference_error(speaker: str, segment_index: int, source_path: Path, reason: str) -> ValueError:
        return reference_helpers.segment_reference_error(speaker, segment_index, source_path, reason)

    def _prepare_segment_reference(
        self, *, segment_dict: Dict[str, Any], speaker: str, chronological_index: int,
        reuse_existing: bool, processed_source_path: Optional[str] = None,
        decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None,
    ) -> tuple[str, Optional[str]]:
        return reference_helpers.prepare_segment_reference(
            self, segment_dict=segment_dict, speaker=speaker,
            chronological_index=chronological_index, reuse_existing=reuse_existing,
            processed_source_path=processed_source_path, decoded_audio_cache=decoded_audio_cache,
        )

    def _prepare_audio_inputs(self) -> tuple[str, Optional[str], str]:
        return artifact_helpers.prepare_audio_inputs(self)

    @staticmethod
    def _cache_fingerprint(dimensions: Dict[str, Any]) -> str:
        return cache_key_helpers.cache_fingerprint(dimensions)

    def _effective_tts_cache_fingerprint(self, speakers: Iterable[str]) -> str:
        return cache_key_helpers.effective_tts_cache_fingerprint(self, speakers)

    @staticmethod
    def _file_content_identity(path: Optional[str]) -> str:
        return cache_key_helpers.file_content_identity(path)

    def _shared_audio_transcription_identity(self, audio_file: str) -> str:
        return cache_key_helpers.shared_audio_transcription_identity(self, audio_file)

    def _effective_translation_cache_dimensions(self) -> Dict[str, Any]:
        return cache_key_helpers.effective_translation_cache_dimensions(
            self,
            DEFAULT_LLM_MODELS,
            active_context(self).semantic_plan_fingerprint,
        )

    def _build_dubbing_text_snapshot_key(self, audio_file: str) -> str:
        return cache_key_helpers.build_dubbing_text_snapshot_key(self, audio_file)

    def _build_translation_cache_key(self, audio_file: str) -> str:
        return cache_key_helpers.build_translation_cache_key(self, audio_file)

    def _build_emotions_cache_key(
        self, audio_file: str, segments: List[Dict[str, Any]],
        provider: Optional[str] = None, model: Optional[str] = None,
    ) -> str:
        return cache_key_helpers.build_emotions_cache_key(
            self, audio_file, segments, provider, model,
            emotion_analysis_prompt=EMOTION_ANALYSIS_PROMPT,
            soft_style_by_emotion=SOFT_STYLE_BY_EMOTION,
            semantic_plan_fingerprint=active_context(
                self
            ).semantic_plan_fingerprint,
        )

    def _restore_semantic_plan_fingerprint(self, audio_file: str) -> None:
        return transcription_helpers.restore_semantic_plan_fingerprint(self, audio_file)

    def _validate_plan_dependent_segments(self, segments: List[Dict[str, Any]]) -> None:
        validate_plan_dependent_segments(active_context(self), segments)

    def _load_required_cached_step(self, *, step_name: str, cache_key: str, hint: str) -> Any:
        return artifact_helpers.load_required_cached_step(
            self, step_name=step_name, cache_key=cache_key, hint=hint
        )

    def _build_speaker_rolls_from_segments(self, segments: List[Dict]) -> Dict[Tuple[float, float], str]:
        return artifact_helpers.build_speaker_rolls_from_segments(self, segments)

    def _save_requested_subtitles(
        self,
        segments_for_output: List[Dict],
        *,
        save_original_subtitles: bool,
        save_translated_subtitles: bool,
        pause_adjustments: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        return artifact_helpers.save_requested_subtitles(
            self, segments_for_output,
            save_original_subtitles=save_original_subtitles,
            save_translated_subtitles=save_translated_subtitles,
            pause_adjustments=pause_adjustments,
        )

    def _combine_final_video(
        self,
        *,
        translated_audio_path: str,
        background_audio_path: Optional[str],
        speakers_rolls: Dict[Tuple[float, float], str],
    ) -> tuple[str, List[Dict[str, float]]]:
        return artifact_helpers.combine_final_video(
            self, translated_audio_path=translated_audio_path,
            background_audio_path=background_audio_path,
            speakers_rolls=speakers_rolls,
        )

    def _persist_dubbing_text_snapshot(
        self, segments: List[Dict], audio_file: str
    ) -> None:
        return translation_helpers.persist_dubbing_text_snapshot(
            self, segments, audio_file
        )

    def _persist_synthesis_results(
        self, segments: List[Dict], audio_file: str
    ) -> None:
        return translation_helpers.persist_synthesis_results(
            self, segments, audio_file
        )

    def _reset_input_cache(self, reason: str) -> None:
        return artifact_helpers.reset_input_cache(self, reason)

    @_with_pipeline_context
    def run_transcribe_only(self, save_original_subtitles: bool = False) -> str:
        """Run only audio extraction, diarization, and transcription; then exit.

        Cached transcription/translation artifacts for this input are wiped
        before the step runs so the transcriber always produces a fresh
        result. `use_cache` stays enabled so the fresh transcription is
        written back to disk for later `tts_to_end` / `translate_only` runs.

        Returns the path to the saved transcription file.
        """
        logger.info("Running transcription-only step")
        self.performance_tracker.start_timing("total")
        self._reset_input_cache("run_step=transcribe_only")
        try:
            audio_file, _, _ = self._prepare_audio_inputs()
            speakers_rolls, transcription = self.diarize_and_transcribe(audio_file)
            if speakers_rolls is None or len(speakers_rolls) == 0:
                raise ValueError("No speakers found in the video")

            segments_for_output = self._apply_speaker_filter(transcription)
            self.subtitle_manager.save_debug_tsv(
                segments_for_output, output_dir=self.config.get("debug_dir")
            )
            self._save_requested_subtitles(
                segments_for_output,
                save_original_subtitles=save_original_subtitles,
                save_translated_subtitles=False,
            )
            transcription_path = self.config.get("transcription_path")
            logger.info(f"Transcription-only step complete: {transcription_path}")
            return transcription_path
        finally:
            self._cleanup()

    @_with_pipeline_context
    def run_translate_only(
        self,
        save_original_subtitles: bool = False,
        save_translated_subtitles: bool = False,
    ) -> str:
        """Reuse cached diarization+transcription, then translate; then exit.

        Unlike ``run_transcribe_only`` / ``run_from_scratch``, this step
        **requires** a previous ``transcribe_only`` (or full-pipeline) run to
        have left a cached transcription for this input in the same project
        directory. It never re-runs the transcriber — running it again in
        ``translate_only`` would be a waste of AssemblyAI / Deepgram / Gemini
        API credits and would break the promise of the resume mode.

        Only the translation step runs afresh (so re-translating with new
        prompts / glossary is cheap). Fresh translation results are still
        persisted so a follow-up ``tts_to_end`` can pick them up.

        Returns the path to the saved translated subtitles when requested,
        otherwise the transcription file path.

        Raises:
            FileNotFoundError: if no cached diarization+transcription exists
                for this input.
        """
        logger.info("Running translation-only step (reusing cached transcription)")
        self.performance_tracker.start_timing("total")
        try:
            audio_file, _, _ = self._prepare_audio_inputs()
            speakers_rolls, transcription = self._load_cached_diarize_and_transcribe(audio_file)
            if speakers_rolls is None or len(speakers_rolls) == 0:
                raise ValueError("No speakers found in the video")

            translated_segments = self.translate_segments(transcription, audio_file)
            segments_for_output = self._apply_speaker_filter(translated_segments)
            self.subtitle_manager.save_debug_tsv(
                segments_for_output, output_dir=self.config.get("debug_dir")
            )
            self._save_requested_subtitles(
                segments_for_output,
                save_original_subtitles=save_original_subtitles,
                save_translated_subtitles=save_translated_subtitles,
            )
            if save_translated_subtitles:
                return self._get_subtitle_path(
                    "translation",
                    self.config.get('input'),
                    self.config.get('target_language'),
                )
            return self.config.get("transcription_path")
        finally:
            self._cleanup()

    @_with_pipeline_context
    def run_analyze_emotions_only(
        self,
        save_translated_subtitles: bool = False,
    ) -> str:
        """Re-run emotion analysis over an existing translation.

        Reuses cached diarization/transcription and the cached translation
        pickle from a previous full-pipeline (or ``translate_only``) run,
        wipes the emotion cache so classification runs fresh, then persists
        the updated ``emotion`` / ``style_prompt`` fields back into the
        translation cache. The Dubbing Texts editor and ``tts_to_end`` pick
        them up on the next load without re-running TTS.

        Fails loudly if there is no cached translation to annotate — this
        step deliberately does not fall back to running the full pipeline.

        Returns:
            The translation cache path (or the translated subtitles path
            when requested), for parity with the other resume-style steps.
        """
        logger.info("Running emotion-analysis-only step")
        self.performance_tracker.start_timing("total")
        try:
            if not self.config.get("enable_emotion_analysis", False):
                logger.warning(
                    "enable_emotion_analysis is False in the current config; "
                    "the analyze_emotions step will still run because it was "
                    "requested explicitly."
                )

            audio_file, _, _ = self._prepare_audio_inputs()
            self._restore_semantic_plan_fingerprint(audio_file)

            translated_segments = self._load_required_cached_step(
                step_name="translation",
                cache_key=self._build_translation_cache_key(audio_file),
                hint="translation",
            )
            self._validate_plan_dependent_segments(translated_segments)

            try:
                self.cache_manager.clear_cache("emotions")
            except Exception as e:
                logger.warning(f"Could not clear emotions cache: {e}")

            annotated_segments = self.analyze_emotions(translated_segments, audio_file)
            self._persist_synthesis_results(annotated_segments, audio_file)

            filled = sum(
                1 for seg in annotated_segments
                if (seg.get("style_prompt") or "").strip()
            )
            logger.info(
                "Emotion analysis complete: %d/%d segments have style_prompt set.",
                filled,
                len(annotated_segments),
            )

            if save_translated_subtitles:
                self._save_requested_subtitles(
                    self._apply_speaker_filter(annotated_segments),
                    save_original_subtitles=False,
                    save_translated_subtitles=True,
                )
                return self._get_subtitle_path(
                    "translation",
                    self.config.get("input"),
                    self.config.get("target_language"),
                )

            return self.config.get("transcription_path")
        finally:
            self._cleanup()

    @_with_pipeline_context
    def run_from_scratch(
        self,
        save_original_subtitles: bool = False,
        save_translated_subtitles: bool = False,
    ) -> str:
        """Wipe cached artifacts for this input and run the full pipeline fresh.

        Fresh results are persisted to cache so subsequent resume steps
        (`tts_to_end`, `combine_video`) can reuse them.
        """
        self._reset_input_cache("run_step=from_scratch")
        return self.run_pipeline(
            save_original_subtitles=save_original_subtitles,
            save_translated_subtitles=save_translated_subtitles,
        )

    @_with_pipeline_context
    def run_from_tts(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str:
        """Resume from cached translation artifacts, rerun TTS, and finish the video."""
        logger.info("Resuming dubbing process from the TTS step")
        output_video_path = ""

        try:
            audio_file, background_audio_path, segment_reference_audio_file = self._prepare_audio_inputs()
            self._restore_semantic_plan_fingerprint(audio_file)
            translated_segments = self._load_required_cached_step(
                step_name="translation",
                cache_key=self._build_translation_cache_key(audio_file),
                hint="translation",
            )
            self._validate_plan_dependent_segments(translated_segments)

            self.debug_data["translation"] = translated_segments
            segments_for_output = self._apply_speaker_filter(translated_segments)
            self.subtitle_manager.save_debug_tsv(segments_for_output, output_dir=self.config.get("debug_dir"))

            if self.config.get('enable_emotion_analysis', True):
                # Prefer emotion/style_prompt already embedded in the translation
                # cache (populated by analyze_emotions_only or a previous full
                # run's _persist_synthesis_results). Only fall back to a
                # standalone `emotions` cache entry when the translation cache
                # is missing that data — that keeps back-compat with older
                # projects while ensuring fresh analyze_emotions_only edits are
                # respected on resume.
                translation_has_emotions = any(
                    (segment.get("emotion") is not None)
                    or (segment.get("style_prompt") or "").strip()
                    for segment in segments_for_output
                )
                if not translation_has_emotions:
                    emotion_provider = str(
                        self.config.get("emotion_provider") or "gemini"
                    ).lower()
                    emotion_model = str(
                        self.config.get("emotion_model") or "gemini-3.1-flash-lite"
                    )
                    emotions_key = self._build_emotions_cache_key(
                        audio_file,
                        segments_for_output,
                        emotion_provider,
                        emotion_model,
                    )
                    if self.cache_manager.cache_exists("emotions", emotions_key):
                        segments_for_output = self._load_required_cached_step(
                            step_name="emotions",
                            cache_key=emotions_key,
                            hint="emotion-analysis",
                        )
                        self._validate_plan_dependent_segments(segments_for_output)
                    else:
                        logger.warning(
                            "enable_emotion_analysis is True but no emotion data was "
                            "found in the translation cache or a standalone 'emotions' "
                            "cache; proceeding with Neutral for all segments. Run "
                            "--run_step analyze_emotions_only to populate emotions."
                        )
                        for segment in segments_for_output:
                            segment.setdefault("emotion", "Neutral")
            else:
                for segment in segments_for_output:
                    segment["emotion"] = "Neutral"

            speakers_rolls = self._build_speaker_rolls_from_segments(segments_for_output)

            # Ensure per-speaker reference clips exist. Cloning-based TTS
            # backends (OmniVoice, XTTS, F5, BexTTS) silently skip segments
            # when no reference audio is available, which used to produce a
            # `translated_audio.wav` full of silence when `tts_to_end` was
            # resumed after `_reset_input_cache` wiped `speakers_audio/`.
            try:
                self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
            except Exception as e:
                logger.warning(f"Could not (re)extract speaker reference audio: {e}")

            original_use_cache = getattr(self.cache_manager, "use_cache", True)
            try:
                self.cache_manager.use_cache = False
                logger.info("Clearing TTS audio caches for run_step=tts_to_end to force fresh audio generation")
                if hasattr(self.cache_manager, "clear_cache"):
                    self.cache_manager.clear_cache("synthesized_speech")
                    self.cache_manager.clear_cache("segment_synthesis")

                output_audio_path = Path(self.config.get("translated_audio_path"))
                if output_audio_path.exists():
                    try:
                        output_audio_path.unlink()
                    except Exception as e:
                        logger.warning(f"Could not delete existing translated audio file: {e}")

                audio_chunks_dir = getattr(self, "audio_chunks_dir", Path(self.config.get("audio_chunks_dir", ""))) if self.config.get("audio_chunks_dir") else None
                su_audio_chunks_dir = getattr(self, "su_audio_chunks_dir", Path(self.config.get("su_audio_chunks_dir", ""))) if self.config.get("su_audio_chunks_dir") else None

                for chunk_dir in [audio_chunks_dir, su_audio_chunks_dir]:
                    if chunk_dir and chunk_dir.exists():
                        for item in chunk_dir.glob("*.wav"):
                            try:
                                item.unlink()
                            except Exception as e:
                                logger.warning(f"Could not delete chunk file {item}: {e}")

                translated_audio_path = self.synthesize_speech(
                    segments_for_output,
                    speakers_rolls,
                    segment_reference_audio_file,
                )
            finally:
                self.cache_manager.use_cache = original_use_cache

            self._persist_synthesis_results(segments_for_output, audio_file)

            self.speaker_processor.save_translated_samples(segments_for_output, audio_file)
            output_video_path, pause_adjustments = self._combine_final_video(
                translated_audio_path=translated_audio_path,
                background_audio_path=background_audio_path,
                speakers_rolls=speakers_rolls,
            )
            self.pause_adjustments = pause_adjustments
            self._save_requested_subtitles(
                segments_for_output,
                save_original_subtitles=save_original_subtitles,
                save_translated_subtitles=save_translated_subtitles,
                pause_adjustments=pause_adjustments,
            )
        except Exception as e:
            logger.error(f"Error in TTS resume pipeline: {e}", exc_info=True)
            raise
        finally:
            self._cleanup()

        return output_video_path
    
    @_with_pipeline_context
    def run_pipeline(self, save_original_subtitles: bool = False, save_translated_subtitles: bool = False) -> str:
        """Run the full dubbing pipeline."""
        pipeline_start_time = time.perf_counter()
        self.performance_tracker.start_timing("total")
        
        logger.info(f"Starting dubbing process for {self.config.get('input')}")
        output_video_path = ""
        
        try:
            audio_file, background_audio_path, segment_reference_audio_file = self._prepare_audio_inputs()
            
            # Perform speaker diarization and transcription
            speakers_rolls, transcription = self.diarize_and_transcribe(audio_file)
            if speakers_rolls is None or len(speakers_rolls) == 0:
                raise ValueError("No speakers found in the video")
            
            # If debug_diarize_only is True, generate debug video and exit early
            if self.config.get('debug_diarize_only', False):
                return self._handle_debug_diarize_only(audio_file, speakers_rolls)
            
            # Extract audio for each speaker
            self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
            
            # Translate segments
            translated_segments = self.translate_segments(transcription, audio_file)

            # Apply optional speaker mute filter for downstream steps
            segments_for_output = self._apply_speaker_filter(translated_segments)
            
            # Save debug TSV
            self.subtitle_manager.save_debug_tsv(segments_for_output, output_dir=self.config.get("debug_dir"))
            
            # Analyze emotions (if enabled)
            if self.config.get('enable_emotion_analysis', True):
                segments_for_output = self.analyze_emotions(segments_for_output, audio_file)
            else:
                logger.debug("Emotion analysis disabled")
                for segment in segments_for_output:
                    segment["emotion"] = "Neutral"
            
            # Synthesize speech or generate silence if no segments remain after muting
            if segments_for_output and len(segments_for_output) > 0:
                translated_audio_path = self.synthesize_speech(
                    segments_for_output,
                    speakers_rolls,
                    segment_reference_audio_file,
                )
                self._persist_synthesis_results(segments_for_output, audio_file)
            else:
                logger.info("All segments filtered by mute_speakers; generating silent audio track...")
                total_duration_sec = self.audio_processor.get_total_duration() or 0
                silent_ms = int(max(0, total_duration_sec) * 1000)
                silent_audio = AudioSegment.silent(duration=silent_ms)
                self.audio_dir.mkdir(parents=True, exist_ok=True)
                translated_audio_path = self.config.get("translated_audio_path")
                silent_audio.export(translated_audio_path, format="wav")

            # Save translated samples
            self.speaker_processor.save_translated_samples(segments_for_output, audio_file)
            
            # Generate final debug video if needed
            if self.config.get('debug_info', False):
                self.debug_generator.generate_debug_video(
                    self.config.get('input'),
                    self.debug_data,
                    self.config.get("debug_dir"),
                    self.config.get('start_time'),
                    self.config.get('duration'),
                    self.audio_processor.get_total_duration()
                )
            
            output_video_path, pause_adjustments = self._combine_final_video(
                translated_audio_path=translated_audio_path,
                background_audio_path=background_audio_path,
                speakers_rolls=speakers_rolls,
            )
            
            # Store pause adjustments for potential future use
            self.pause_adjustments = pause_adjustments
            self._save_requested_subtitles(
                segments_for_output,
                save_original_subtitles=save_original_subtitles,
                save_translated_subtitles=save_translated_subtitles,
                pause_adjustments=pause_adjustments,
            )
            
            # Overall pipeline metrics
            total_elapsed = time.perf_counter() - pipeline_start_time
            logger.info("Dubbing process completed!")
            self.performance_tracker.record_metric("total", total_elapsed)
            
            # Write performance summary
            self.performance_tracker.write_performance_summary(self.audio_processor.get_total_duration())
            
        except Exception as e:
            logger.error(f"Error in dubbing pipeline: {e}", exc_info=True)
            raise
        finally:
            # Clean up
            self._cleanup()
        
        return output_video_path
    
    def _handle_debug_diarize_only(self, audio_file: str, speakers_rolls: Dict) -> str:
        """Handle debug diarize-only mode."""
        logger.info("Debug diarization only mode: Generating debug video after diarization and exiting")
        
        # Set debug_info to True to ensure debug video generation works
        original_debug_info = self.config.get('debug_info', False)
        self.config.set('debug_info', True)
        
        # Extract audio for each speaker
        self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
        
        # Generate debug video
        self.debug_generator.generate_debug_video(
            self.config.get('input'),
            self.debug_data,
            self.config.get("debug_dir"),
            self.config.get('start_time'),
            self.config.get('duration'),
            self.audio_processor.get_total_duration()
        )
        
        # Create debug TSV of original transcription
        self.subtitle_manager.save_debug_tsv(self.debug_data["transcription"], output_dir=self.config.get("debug_dir"))
        
        # Reset debug_info to original value
        self.config.set('debug_info', original_debug_info)
        
        # Return path to debug video
        debug_video_path = self.config.get("debug_video_path")
        logger.info(f"Debug video generated: {debug_video_path}")
        
        # Write partial performance summary
        self.performance_tracker.record_metric("total", time.perf_counter() - self.performance_tracker._start_times.get("total", 0))
        self.performance_tracker.write_performance_summary(self.audio_processor.get_total_duration())
        
        return debug_video_path
    
    def generate_diarization_report(self) -> Tuple[str, str]:
        """Generate a report of identified speakers and their voice samples."""
        report_start_time = time.perf_counter()
        logger.info("Starting speaker report generation...")

        # Extract audio from video
        audio_file = self.audio_processor.extract_audio(
            self.config.get('input'),
            self.config.get('start_time'),
            self.config.get('duration')
        )

        # Perform speaker diarization and transcription
        speakers_rolls, transcription = self.diarize_and_transcribe(audio_file)
        if speakers_rolls is None or len(speakers_rolls) == 0:
            raise ValueError("No speakers found in the video during diarization.")

        # Extract audio samples for each speaker
        speaker_audio_paths = self.speaker_processor.extract_speaker_audio(audio_file, speakers_rolls)
        if not speaker_audio_paths:
            raise ValueError("Could not extract audio samples for speakers.")

        # Create the speaker report
        report_file_path, samples_dir_path = self.speaker_reporter.create_speaker_report(
            speaker_audio_paths, transcription, self.config.get('input')
        )
        
        # Write performance summary for this specific operation
        self.performance_tracker.record_metric("total_report_generation", time.perf_counter() - report_start_time)
        self.performance_tracker.record_metric("video_duration", self.audio_processor.get_total_duration() or 0)
        self.performance_tracker.write_performance_summary_for_report()
        
        return report_file_path, samples_dir_path
    
    def _load_cached_diarize_and_transcribe(
        self, audio_file: str
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
        return transcription_helpers.load_cached_diarize_and_transcribe(self, audio_file)

    def diarize_and_transcribe(self, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
        return transcription_helpers.diarize_and_transcribe(self, audio_file)

    def _isolated_tracks_cache_key(
        self, audio_file: str, isolated_tracks: Dict[str, str],
    ) -> str:
        return cache_key_helpers.isolated_tracks_cache_key(self, audio_file, isolated_tracks)

    def _isolated_tracks_raw_cache_key(self, isolated_tracks: Dict[str, str]) -> str:
        return cache_key_helpers.isolated_tracks_raw_cache_key(self, isolated_tracks)

    def _semantic_classifier(self) -> Tuple[Optional[Any], str]:
        return transcription_helpers.semantic_classifier(self)

    def _semantic_classifier_identity(self) -> Dict[str, Any]:
        return cache_key_helpers.semantic_classifier_identity(self)

    @staticmethod
    def _write_semantic_boundary_diagnostics(
        path: str, records: List[Dict[str, Any]]
    ) -> None:
        return transcription_helpers.write_semantic_boundary_diagnostics(path, records)

    def _diarize_and_transcribe_isolated(
        self,
        audio_file: str,
        isolated_tracks: Dict[str, str],
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
        return transcription_helpers.diarize_and_transcribe_isolated(self, audio_file, isolated_tracks)

    def _isolated_inner_kwargs(self, inner_system: str) -> Dict[str, Any]:
        return transcription_helpers.isolated_inner_kwargs(self, inner_system)
    
    def translate_segments(self, transcription: List[Dict], audio_file: str) -> List[Dict]:
        return translation_helpers.translate_segments(
            self, transcription, audio_file
        )

    def _build_translation_prompt_prefix(self, base_prompt_prefix: Optional[str]) -> str:
        return translation_helpers.build_translation_prompt_prefix(
            self, base_prompt_prefix, SMART_DUBBING_STRESS_MARKS_REQUIREMENT
        )
    
    def analyze_emotions(self, segments: List[Dict], audio_file: str) -> List[Dict]:
        return emotion_helpers.analyze_emotions(self, segments, audio_file)

    def _analyze_emotions_gemini(self, segments: List[Dict], audio_file: str, model: str) -> None:
        return emotion_helpers.analyze_emotions_gemini(
            self, segments, audio_file, model, EMOTION_ANALYSIS_PROMPT
        )

    def _analyze_emotions_speechbrain(self, segments: List[Dict], audio_file: str) -> None:
        return emotion_helpers.analyze_emotions_speechbrain(
            self, segments, audio_file, SOFT_STYLE_BY_EMOTION
        )
    
    def synthesize_speech(self, segments: List[Dict], speakers_rolls: Dict, audio_file: str) -> str:
        return synthesis_helpers.synthesize_speech(self, segments, speakers_rolls, audio_file)

    def _tts_selection_cache_fingerprint(self, segments: List[Dict[str, Any]]) -> str:
        return cache_key_helpers.tts_selection_cache_fingerprint(self, segments)

    def _synthesize_measured_candidates(
        self,
        metadata: Dict[str, Any],
        policy,
    ) -> None:
        return synthesis_helpers.synthesize_measured_candidates(self, metadata, policy)

    def _load_or_synthesize_candidate(
        self,
        metadata: Dict[str, Any],
        *,
        variant: str,
        text: str,
        attempts: int,
    ) -> Optional[Dict[str, Any]]:
        return synthesis_helpers.load_or_synthesize_candidate(
            self, metadata, variant=variant, text=text, attempts=attempts
        )

    def resynthesize_one_segment(
        self,
        segments: List[Dict],
        segment_index: int,
        override_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        return synthesis_helpers.resynthesize_one_segment(
            self, segments, segment_index, override_text
        )

    def rebuild_translated_audio_from_chunks(self) -> Optional[str]:
        return audio_assembly_helpers.rebuild_translated_audio_from_chunks(self)

    def _save_transcription_file(self, transcription: List[Dict]) -> None:
        return artifact_helpers.save_transcription_file(self, transcription)
    
    def _cleanup(self) -> None:
        """Clean up temporary files and TTS systems."""
        logger.info("Cleaning up temporary files...")
        try:
            # Call cleanup through each TTS client in the pool
            for pool_key, tts_instance in getattr(self, "tts_clients", {}).items():
                if tts_instance:
                    try:
                        tts_instance.cleanup()
                        logger.info(f"Cleaned up TTS client pool_key={pool_key}")
                    except Exception as cleanup_e:
                        logger.warning(f"Warning: Error cleaning up TTS client {pool_key}: {cleanup_e}")
            
            # Clean up temporary directories
            for temp_dir in [self.audio_chunks_dir, self.su_audio_chunks_dir]:
                if temp_dir.exists():
                    for temp_file in os.listdir(temp_dir):
                        if temp_file.startswith("temp_") or temp_file.startswith("group_"):
                            try:
                                os.remove(os.path.join(temp_dir, temp_file))
                            except Exception:
                                pass
            
            logger.info("Cleanup completed.")
        except Exception as e:
            logger.warning(f"Warning: Error during cleanup: {e}")
    
    def _get_tts_system_for_speaker(self, speaker_id: str) -> str:
        return synthesis_helpers.get_tts_system_for_speaker(self, speaker_id)

    def _resolve_segment_reference(
        self, *, tts_segment_data_args: Dict[str, Any], segment_dict: Dict[str, Any],
        profile: VoiceProfile, provider_capability: str, speaker: str, segment_index: int,
        original_audio_segment: Optional[AudioSegment], segment_reference_min_duration: float,
        for_resynthesis: bool = False, processed_source_path: Optional[str] = None,
        decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None,
    ) -> tuple[Dict[str, Any], Optional[AudioSegment]]:
        return reference_helpers.resolve_segment_reference(
            self, tts_segment_data_args=tts_segment_data_args, segment_dict=segment_dict,
            profile=profile, provider_capability=provider_capability, speaker=speaker,
            segment_index=segment_index, original_audio_segment=original_audio_segment,
            segment_reference_min_duration=segment_reference_min_duration,
            for_resynthesis=for_resynthesis, processed_source_path=processed_source_path,
            decoded_audio_cache=decoded_audio_cache,
        )

    @staticmethod
    def _preflight_tts_pools(
        segments_by_pool: Dict[tuple, List[Any]],
        clients: Dict[tuple, Any],
        initial_issues: Optional[List[tuple[int, str]]] = None,
    ) -> None:
        return synthesis_helpers.preflight_tts_pools(
            segments_by_pool, clients, initial_issues
        )

    @staticmethod
    def _trim_trailing_silence(
        audio_path: str,
        silence_threshold_db: float = -40.0,
        keep_tail_ms: int = 100,
        window_ms: int = 10,
    ) -> None:
        return audio_assembly_helpers.trim_trailing_silence(
            audio_path, silence_threshold_db, keep_tail_ms, window_ms
        )

    def _measure_raw_tts_for_timing(self, audio_path: str, segment_index: int) -> float:
        return audio_assembly_helpers.measure_raw_tts_for_timing(
            self, audio_path, segment_index
        )

    @staticmethod
    def _segment_cache_metadata_path(cache_path: Path) -> Path:
        return cache_key_helpers.segment_cache_metadata_path(cache_path)

    @staticmethod
    def _raw_tts_segment_cache_key(
        *, base_cache_prefix: str, tts_system: str, segment: Dict[str, Any],
        speaker: str, translation: str, style_prompt: str,
        reference_audio_path: Optional[str], reference_mode: Optional[str] = None,
        reference_text: Optional[str] = None, client_pool_settings: Any = None,
        legacy_index: Any = 0, emotion: Optional[str] = "Neutral",
        tts_prompt_prefix: Optional[str] = None, voice_prompt: Any = None,
    ) -> str:
        return cache_key_helpers.raw_tts_segment_cache_key(
            base_cache_prefix=base_cache_prefix, tts_system=tts_system,
            segment=segment, speaker=speaker, translation=translation,
            style_prompt=style_prompt, reference_audio_path=reference_audio_path,
            reference_mode=reference_mode, reference_text=reference_text,
            client_pool_settings=client_pool_settings, legacy_index=legacy_index,
            emotion=emotion, tts_prompt_prefix=tts_prompt_prefix,
            voice_prompt=voice_prompt,
            cache_fingerprint=SmartDubbing._cache_fingerprint,
            file_content_identity=SmartDubbing._file_content_identity,
        )

    def _cache_raw_tts_segment(
        self,
        source_path: str,
        cache_path: Path,
        *,
        synthesized_text: str = "",
    ) -> None:
        return synthesis_helpers.cache_raw_tts_segment(
            self, source_path, cache_path, synthesized_text=synthesized_text
        )

    def _cached_segment_metadata(self, cache_path: Path) -> Dict[str, Any]:
        return synthesis_helpers.cached_segment_metadata(self, cache_path)

    def _cached_segment_contract(self, cache_path: Path) -> str:
        return synthesis_helpers.cached_segment_contract(self, cache_path)

    def _adjust_and_combine_audio_grouped_legacy(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]:
        return audio_assembly_helpers.adjust_and_combine_audio_grouped_legacy(
            self, segments
        )

    def _adjust_and_combine_audio_grouped(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]:
        return audio_assembly_helpers.adjust_and_combine_audio_grouped(self, segments)

    def _get_subtitle_path(self, subtitle_type: str, input_path: str, language: str) -> str:
        return artifact_helpers.get_subtitle_path(
            self, subtitle_type, input_path, language
        )

    def adjust_subtitle_timestamps(self, segments: List[Dict], pause_adjustments: List[Dict[str, float]]) -> List[Dict]:
        return artifact_helpers.adjust_subtitle_timestamps(
            self, segments, pause_adjustments
        )
