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
import hashlib
import csv
import json
import math
import torch
import warnings
import shutil
import subprocess
from functools import wraps
from typing import Dict, Iterable, List, Tuple, Optional, Any, Literal, Union
from pathlib import Path
from urllib.parse import quote
from dotenv import load_dotenv
from pydub import AudioSegment

# Disable all warnings for a cleaner output.
warnings.filterwarnings("ignore")

# Import our components
from .config import DubbingConfig
from .cache_manager import CacheManager
from .voice_profiles import (
    FALLBACK_SPEAKER,
    VoiceProfile,
    normalize_voices,
    resolve_profile,
)
from ..audio.audio_processor import AudioProcessor
from ..audio.speaker_processor import SpeakerProcessor
from ..video.video_processor import VideoProcessor
from ..debug.performance_tracker import PerformanceTracker
from ..debug.debug_generator import DebugGenerator
from ..debug.reporter import SpeakerReporter
from ..utils.subtitle_utils import SubtitleManager
from .log_config import get_logger
from .pipeline import cache_keys as cache_key_helpers
from .pipeline import emotions as emotion_helpers
from .pipeline import references as reference_helpers
from .pipeline import transcription as transcription_helpers
from .pipeline import translation as translation_helpers
from .pipeline.context import (
    active_context,
    commit_context,
    snapshot_context,
    validate_plan_dependent_segments,
)

# Import existing factories and interfaces
from tts.tts_factory import TTSFactory
from translation.llm_translator import DEFAULT_LLM_MODELS
from translation.translator_factory import TranslatorFactory

# Disable warnings
warnings.filterwarnings("ignore")

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
        self.config = config
        self.project_dir = Path(self.config.get("project_dir"))
        self.artifacts_root = Path(self.config.get("artifacts_dir"))
        self.audio_dir = Path(self.config.get("audio_artifacts_dir"))
        self.speakers_audio_dir = Path(self.config.get("speakers_audio_dir"))
        self.audio_chunks_dir = Path(self.config.get("audio_chunks_dir"))
        self.su_audio_chunks_dir = Path(self.config.get("su_audio_chunks_dir"))
        # Legacy per-speaker mirrors (kept for backward-compat with a few call sites and tests).
        self.tts_system_mapping = self.config.get('tts_system_mapping') or {}
        self.voice_prompt = self.config.get('voice_prompt') or {}
        self.reference_audio_mapping = self.config.get('reference_audio_mapping') or {}
        self.reference_text_mapping = self.config.get('reference_text_mapping') or {}

        # Unified per-speaker profiles. DubbingConfig.process_special_parameters folds legacy
        # fields into this dict; when a plain dict-config is used (mostly in tests) we
        # normalize on the fly so downstream code always sees VoiceProfile objects.
        voices_cfg = self.config.get('voices')
        if not isinstance(voices_cfg, dict) or not all(
            isinstance(v, VoiceProfile) for v in voices_cfg.values()
        ):
            voices_cfg = normalize_voices(self.config.to_dict() if hasattr(self.config, "to_dict") else dict(self.config))
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
        """The TTS backend used as fallback when a profile does not name one."""
        return self.config.get('tts_system', 'coqui')

    def _resolve_voice_profile(self, speaker: str) -> VoiceProfile:
        """Resolve a speaker to its effective VoiceProfile (with `"*"` fallback)."""
        return resolve_profile(
            self.voice_profiles,
            speaker,
            tts_system_default=self._default_tts_system(),
        )

    def _profile_pool_key(self, profile: VoiceProfile) -> tuple:
        """Client-pool identity for a profile, taking global TTS defaults into account."""
        return (
            (profile.tts_system or self._default_tts_system() or "").lower(),
            profile.model or (self.config.get('tts_model') or ""),
            tuple(sorted(profile.params.items())),
        )

    def _global_omnivoice_kwargs(self) -> Dict[str, Any]:
        """Backend-specific globals still sourced from top-level config (legacy)."""
        return {
            "space_id": self.config.get('omnivoice_space_id'),
            "api_name": self.config.get('omnivoice_api_name'),
            "lang": self.config.get('omnivoice_lang'),
            "instruct": self.config.get('omnivoice_instruct', ''),
            "num_steps": self.config.get('omnivoice_num_steps'),
            "guidance_scale": self.config.get('omnivoice_guidance_scale'),
            "denoise": self.config.get('omnivoice_denoise'),
            "speed": self.config.get('omnivoice_speed'),
            "duration": self.config.get('omnivoice_duration'),
            "preprocess_prompt": self.config.get('omnivoice_preprocess_prompt'),
            "postprocess_output": self.config.get('omnivoice_postprocess_output'),
        }

    def _build_tts_client(self, profile: VoiceProfile) -> Any:
        """Create a single TTS client for a normalized profile.

        Voice/style mapping is not attached here — those are per-segment and are
        threaded into TTSSegmentData at synth time. This function only handles
        provider construction (system, model, provider-specific bootstrap kwargs).
        """
        tts_system = profile.tts_system or self._default_tts_system()
        model = profile.model or self.config.get('tts_model')

        # Provider-specific bootstrap kwargs. Globals still come from the top-level
        # config; per-profile `params` override them.
        bootstrap: Dict[str, Any] = {}
        if tts_system.lower() == "omnivoice":
            bootstrap.update(self._global_omnivoice_kwargs())
        bootstrap.update(profile.params or {})

        return TTSFactory.create_tts(
            tts_system=tts_system,
            device=self.device,
            voice_config=None,  # per-segment via TTSSegmentData
            voice_prompt=None,  # per-segment via TTSSegmentData
            prompt_prefix=self.config.get('tts_prompt_prefix'),
            # A manually assigned voice does not need catalog matching. This also
            # prevents OpenAI/Gemini from generating the catalog samples solely
            # for a profile whose voice is already known.
            enable_voice_matching=(
                self.config.get('voice_auto_selection', True)
                and not bool(profile.voice_name)
            ),
            debug_tts=self.config.get('debug_tts', False),
            model=model,
            default_reference_audio=self.config.get('reference_audio'),
            **bootstrap,
        )

    def _initialize_tts_systems(self) -> None:
        """Instantiate one TTS client per unique (system, model, params) profile."""
        # Client pool keyed by pool_key. Multiple speakers with identical settings
        # share the same client.
        self.tts_clients: Dict[tuple, Any] = {}
        # Speaker -> pool_key resolution cache.
        self._speaker_pool_key: Dict[str, tuple] = {}
        self.default_tts = None
        self.tts_init_error: Optional[Exception] = None

        # Build the effective set of profiles to instantiate: every named profile
        # plus a fallback (`"*"` or synthesised from global TTS defaults).
        profiles_to_build: Dict[tuple, VoiceProfile] = {}
        for speaker, profile in self.voice_profiles.items():
            resolved = self._resolve_voice_profile(speaker)
            key = self._profile_pool_key(resolved)
            profiles_to_build.setdefault(key, resolved)
            self._speaker_pool_key[speaker] = key

        fallback_profile = self._resolve_voice_profile(FALLBACK_SPEAKER)
        fallback_key = self._profile_pool_key(fallback_profile)
        profiles_to_build.setdefault(fallback_key, fallback_profile)

        first_error: Optional[Exception] = None
        for key, profile in profiles_to_build.items():
            try:
                client = self._build_tts_client(profile)
                self.tts_clients[key] = client
                logger.debug(
                    "Initialized TTS client for pool_key=%s (system=%s, model=%s)",
                    key, profile.tts_system, profile.model,
                )
            except Exception as exc:
                if first_error is None:
                    first_error = exc
                logger.warning(
                    "Failed to initialize TTS client for pool_key=%s (system=%s, model=%s): %s",
                    key, profile.tts_system, profile.model, exc,
                )

        if first_error is not None and not self.tts_clients:
            self.tts_init_error = first_error

        # default_tts is the client for the "*" profile (or first available).
        if fallback_key in self.tts_clients:
            self.default_tts = self.tts_clients[fallback_key]
        elif self.tts_clients:
            self.default_tts = next(iter(self.tts_clients.values()))

        # Backwards-compat mirror: `self.tts_systems` used to be a dict keyed by
        # backend name. A few older call sites/tests still touch it, so mirror the
        # default client under its backend name. New code should use tts_clients.
        self.tts_systems: Dict[str, Any] = {}
        if self.default_tts is not None:
            self.tts_systems[fallback_profile.tts_system or self._default_tts_system()] = self.default_tts
    
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
        """Extract the source audio and optional background/vocals tracks."""
        audio_file = self.audio_processor.extract_audio(
            self.config.get('input'),
            self.config.get('start_time'),
            self.config.get('duration')
        )
        background_audio_path = None
        segment_reference_audio_file = audio_file
        if self.config.get('keep_background', False):
            (
                background_audio_path,
                separated_vocals_path,
            ) = self.audio_processor.separate_background_and_vocals(audio_file)
            if separated_vocals_path:
                segment_reference_audio_file = separated_vocals_path

        return audio_file, background_audio_path, segment_reference_audio_file

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
        """Load a required cached artifact or raise an actionable error."""
        if not getattr(self.cache_manager, "use_cache", True):
            raise FileNotFoundError(
                f"run_step=tts_to_end requires cached {hint} artifacts from a previous full dubbing run, "
                "but caching is currently disabled. Re-enable cache or run the full pipeline first."
            )

        if not self.cache_manager.cache_exists(step_name, cache_key):
            raise FileNotFoundError(
                f"run_step=tts_to_end requires cached {hint} artifacts from a previous full dubbing run in the same project directory, "
                f"but no cache entry was found for step '{step_name}'."
            )

        cached_value = self.cache_manager.load_from_cache(step_name, cache_key)
        if cached_value is None:
            raise FileNotFoundError(
                f"run_step=tts_to_end found step '{step_name}' but could not load cached {hint} artifacts. "
                "Re-run the full pipeline to rebuild them."
            )

        return cached_value

    def _build_speaker_rolls_from_segments(self, segments: List[Dict]) -> Dict[Tuple[float, float], str]:
        """Reconstruct a speaker timeline from translated segment data."""
        speakers_rolls: Dict[Tuple[float, float], str] = {}
        for segment in segments:
            start = segment.get("start")
            end = segment.get("end")
            speaker = segment.get("speaker")
            if start is None or end is None or speaker is None:
                continue
            speakers_rolls[(float(start), float(end))] = str(speaker)
        return speakers_rolls

    def _save_requested_subtitles(
        self,
        segments_for_output: List[Dict],
        *,
        save_original_subtitles: bool,
        save_translated_subtitles: bool,
        pause_adjustments: Optional[List[Dict[str, float]]] = None,
    ) -> None:
        """Persist subtitle files for the current output state."""
        if not (save_original_subtitles or save_translated_subtitles):
            return

        remove_pauses_enabled = self.config.get('remove_pauses', False)
        if not remove_pauses_enabled:
            if save_original_subtitles:
                self.subtitle_manager.save_subtitles(
                    segments_for_output,
                    "original",
                    self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')),
                )

            if save_translated_subtitles:
                self.subtitle_manager.save_subtitles(
                    segments_for_output,
                    "translation",
                    self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')),
                )
            return

        if pause_adjustments:
            logger.info("Adjusting subtitle timestamps based on pause modifications...")
            adjusted_segments = self.adjust_subtitle_timestamps(segments_for_output, pause_adjustments)
            if save_original_subtitles:
                self.subtitle_manager.save_subtitles(
                    adjusted_segments,
                    "original",
                    self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')),
                )
                adjusted_path = self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language'))
                logger.info(f"Saved pause-corrected original subtitles to {adjusted_path}")

            if save_translated_subtitles:
                self.subtitle_manager.save_subtitles(
                    adjusted_segments,
                    "translation",
                    self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')),
                )
                adjusted_path = self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language'))
                logger.info(f"Saved pause-corrected translated subtitles to {adjusted_path}")
            return

        logger.info("No pause adjustments needed, saving subtitles with original timestamps...")
        if save_original_subtitles:
            self.subtitle_manager.save_subtitles(
                segments_for_output,
                "original",
                self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language')),
            )
            subtitle_path = self._get_subtitle_path("original", self.config.get('input'), self.config.get('source_language'))
            logger.info(f"Saved original subtitles to {subtitle_path}")

        if save_translated_subtitles:
            self.subtitle_manager.save_subtitles(
                segments_for_output,
                "translation",
                self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language')),
            )
            subtitle_path = self._get_subtitle_path("translation", self.config.get('input'), self.config.get('target_language'))
            logger.info(f"Saved translated subtitles to {subtitle_path}")

    def _combine_final_video(
        self,
        *,
        translated_audio_path: str,
        background_audio_path: Optional[str],
        speakers_rolls: Dict[Tuple[float, float], str],
    ) -> tuple[str, List[Dict[str, float]]]:
        """Combine the current translated audio track with the source video."""
        keep_original_audio_ranges = self.config.get('keep_original_audio_ranges')
        muted_speakers = getattr(self, "muted_speakers", set())
        if keep_original_audio_ranges is None and self.config.get('include_original_audio', False) and muted_speakers:
            try:
                keep_original_audio_ranges = [
                    (start, end) for (start, end), spk in (speakers_rolls or {}).items() if spk not in muted_speakers
                ]
                if keep_original_audio_ranges:
                    logger.info(f"Computed keep_original_audio_ranges excluding muted speakers ({len(keep_original_audio_ranges)} ranges)")
            except Exception:
                keep_original_audio_ranges = self.config.get('keep_original_audio_ranges')

        return self.video_processor.combine_audio_with_video(
            video_path=self.config.get('input'),
            translated_audio_path=translated_audio_path,
            background_audio_path=background_audio_path,
            watermark_path=self.config.get('watermark_path'),
            watermark_text=self.config.get('watermark_text'),
            include_original_audio=self.config.get('include_original_audio', False),
            output_file=self.config.get('output'),
            start_time=self.config.get('start_time'),
            duration=self.config.get('duration'),
            keep_original_audio_ranges=keep_original_audio_ranges,
            source_language=self.config.get('source_language'),
            target_language=self.config.get('target_language'),
            normalize_audio=self.config.get('normalize_audio', True),
            use_two_pass_encoding=self.config.get('use_two_pass_encoding', True),
            remove_pauses=self.config.get('remove_pauses', False),
            min_pause_duration=self.config.get('min_pause_duration', 300),
            preserve_pause_duration=self.config.get('preserve_pause_duration', 1.5),
            keyframe_buffer=self.config.get('keyframe_buffer', 0.2),
            ffmpeg_batch_size=self.config.get('ffmpeg_batch_size', 50),
            dubbed_volume=self.config.get('dubbed_volume', 1.0),
            background_volume=self.config.get('background_volume', 0.562341),
            upscale_factor=self.config.get('upscale_factor', 1.0),
            upscale_sharpen=self.config.get('upscale_sharpen', True),
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
        """Delete every cached artifact tied to the current input file.

        Also drops any legacy per-step caches saved without an input hash so
        callers do not silently reuse stale results from earlier runs.
        """
        logger.info(f"Clearing cached artifacts for this input ({reason})")
        try:
            if hasattr(self.cache_manager, "clear_input_cache"):
                self.cache_manager.clear_input_cache(self.config.get('input'))
        except Exception as e:
            logger.warning(f"Could not clear per-input cache: {e}")

        # Also wipe rendered chunk wavs. Without this a from-scratch run that
        # loses a segment mid-flight (e.g. OmniVoice AcceleratorError) would
        # silently reuse the previous run's chunk with the same index — audio
        # produced by a completely different TTS backend.
        for chunk_dir in (
            getattr(self, "audio_chunks_dir", None),
            getattr(self, "su_audio_chunks_dir", None),
        ):
            if chunk_dir and Path(chunk_dir).exists():
                for item in Path(chunk_dir).glob("*.wav"):
                    try:
                        item.unlink()
                    except OSError as exc:
                        logger.debug(f"Could not remove stale chunk {item}: {exc}")

        # Legacy fallback: older transcription backends write to ./cache/<step>/
        # without the input-hash prefix, so a targeted wipe is still needed.
        legacy_root = getattr(self.cache_manager, "cache_root", None)
        if legacy_root is None:
            return
        for step_name in (
            "whisperx_diarization_transcription",
            "gemini_diarization_transcription",
            "deepgram_diarization_transcription",
            "assemblyai_diarization_transcription",
            "isolated_tracks_transcription",
            "isolated_tracks_raw_transcription",
            "isolated_tracks_semantic_plan",
            "semantic_boundary_classification",
            "chunked_processing",
            "segment_transcription",
            "diarization",
            "transcription",
            "translation",
            "emotions",
        ):
            legacy_dir = Path(legacy_root) / step_name
            if legacy_dir.exists():
                try:
                    shutil.rmtree(legacy_dir)
                except Exception as e:
                    logger.warning(f"Could not remove legacy cache {legacy_dir}: {e}")

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
        """Synthesize measured text candidates within each recognized segment."""
        if not segments:
            raise ValueError("Cannot synthesize speech with no segments.")

        from .timing import TimingPolicy, plan_anchor_windows, timing_cache_fingerprint
        from tts.models import TTSSegmentData

        try:
            _update_pipeline_context(
                self,
                "timing_source_duration",
                len(AudioSegment.from_file(audio_file)) / 1000.0,
            )
        except Exception as exc:
            raise ValueError(f"Cannot measure processed source audio for timing: {audio_file}") from exc
        _update_pipeline_context(self, "timing_source_audio_file", audio_file)

        planned = plan_anchor_windows(segments, self._timing_source_duration)
        for item in planned:
            item.segment["_timing_original_index"] = item.original_index
            item.segment["_timing_available_window"] = item.available_window
            item.segment.pop("_timing_next_anchor", None)
        segments[:] = [item.segment for item in planned]

        policy = TimingPolicy.from_config(self.config)
        _update_pipeline_context(
            self,
            "plan_dependent_cache_allowed",
            active_context(self).semantic_plan_cache_persistable,
        )
        self.performance_tracker.start_timing("speech_synthesis")

        semantic_fingerprints = {
            segment.get("semantic_plan_fingerprint")
            for segment in segments
            if segment.get("semantic_plan_fingerprint")
        }
        semantic_segments = [
            segment for segment in segments if segment.get("semantic_unit_id")
        ]
        if semantic_segments and (
            len(semantic_fingerprints) != 1
            or any(
                segment.get("semantic_plan_fingerprint") not in semantic_fingerprints
                for segment in semantic_segments
            )
        ):
            raise ValueError(
                "Semantic TTS segments have a missing or inconsistent semantic_plan_fingerprint"
            )
        semantic_suffix = (
            f"_{next(iter(semantic_fingerprints))}"
            if len(semantic_fingerprints) == 1
            else ""
        )
        tts_fingerprint = self._effective_tts_cache_fingerprint(
            segment["speaker"] for segment in segments
        )
        source_cache_key = self.cache_manager.generate_cache_key(
            audio_file,
            self.config.get("source_language"),
            self.config.get("target_language"),
            self.config.get("whisper_model", "large-v3"),
            self.config.get("start_time"),
            self.config.get("duration"),
        )
        selection_fingerprint = self._tts_selection_cache_fingerprint(segments)
        has_segment_references = any(
            self._resolve_voice_profile(segment["speaker"]).reference_mode == "segment"
            for segment in segments
        )
        cache_key = (
            f"{source_cache_key}_{self.config.get('target_language')}_"
            f"{self.config.get('tts_system')}_{timing_cache_fingerprint(policy)}_"
            f"{tts_fingerprint}_{selection_fingerprint}{semantic_suffix}"
        )
        aggregate_cache_dir = self.cache_manager.get_cache_path("synthesized_speech")
        cached_audio_path = aggregate_cache_dir / f"{cache_key}.wav"
        output_path = self.config.get("translated_audio_path")
        if (
            self.cache_manager.use_cache
            and self._plan_dependent_cache_allowed
            and not self.config.get("debug_info", False)
            and not has_segment_references
            and cached_audio_path.exists()
        ):
            shutil.copy(cached_audio_path, output_path)
            self.performance_tracker.end_timing("speech_synthesis")
            return output_path

        if not self.tts_clients:
            raise ValueError("No TTS systems are initialized properly")

        segment_cache_dir = self.cache_manager.get_cache_path("segment_synthesis")
        base_cache_prefix = f"{source_cache_key}_{tts_fingerprint}"
        self.audio_chunks_dir.mkdir(parents=True, exist_ok=True)
        self.su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)

        segment_reference_min_duration = self.config.get(
            "segment_reference_min_duration", 2.0
        )
        try:
            segment_reference_min_duration = max(
                0.0, float(segment_reference_min_duration)
            )
        except (TypeError, ValueError, OverflowError):
            segment_reference_min_duration = 2.0

        prepared: List[Dict[str, Any]] = []
        preflight_segments: Dict[tuple, List[Any]] = {}
        preflight_clients: Dict[tuple, Any] = {}
        reference_issues: List[tuple[int, str]] = []
        original_audio_segment = None
        segment_reference_audio_cache: Dict[tuple[str, float], AudioSegment] = {}

        for chronological_index, segment in enumerate(segments):
            speaker = segment["speaker"]
            profile = self._resolve_voice_profile(speaker)
            pool_key = self._profile_pool_key(profile)
            tts_instance = self.tts_clients.get(pool_key) or getattr(
                self, "default_tts", None
            )
            if tts_instance is None:
                raise ValueError(
                    f"TTS client for {profile.tts_system or self._default_tts_system()} is not available"
                )
            tts_system = profile.tts_system or self._default_tts_system()
            original_index = self._canonical_segment_index(
                segment, chronological_index
            )
            style_prompt = (
                (segment.get("style_prompt") or "").strip()
                or profile.style_prompt
            )
            voice_name = profile.voice_name
            if voice_name is None and isinstance(self.config.get("voice_name"), str):
                voice_name = self.config.get("voice_name")
            if self.config.get("debug_info", False):
                self.debug_data.setdefault("voices", {})[chronological_index] = {
                    "speaker": speaker,
                    "voice": voice_name,
                    "style_prompt": style_prompt,
                    "tts_system": tts_system,
                    "model": profile.model,
                }
            base_args: Dict[str, Any] = {
                "speaker": speaker,
                "text": segment.get("translation", ""),
                "emotion": segment.get("emotion", "Neutral"),
                "style_prompt": style_prompt,
                "reference_audio_path": profile.reference_audio,
                "reference_text": profile.reference_text,
                "reference_mode": profile.reference_mode,
                "segment_index": original_index,
                "voice": voice_name,
                "speed": 1.0,
                "target_duration": segment["_timing_available_window"],
            }
            provider_capability = getattr(
                tts_instance, "reference_capability", "unsupported"
            )
            try:
                base_args, original_audio_segment = self._resolve_segment_reference(
                    tts_segment_data_args=base_args,
                    segment_dict=segment,
                    profile=profile,
                    provider_capability=provider_capability,
                    speaker=speaker,
                    segment_index=original_index,
                    original_audio_segment=original_audio_segment,
                    segment_reference_min_duration=segment_reference_min_duration,
                    processed_source_path=audio_file,
                    decoded_audio_cache=segment_reference_audio_cache,
                )
            except Exception as exc:
                reference_issues.append(
                    (
                        original_index,
                        f"provider={tts_system} speaker={speaker} segment={original_index} "
                        f"mode={profile.reference_mode or '<missing>'}: {exc}",
                    )
                )

            final_path = str(self.audio_chunks_dir / f"{chronological_index}.wav")
            initial_data = TTSSegmentData(
                **{
                    **base_args,
                    "text": segment.get("translation", ""),
                    "output_path": final_path,
                }
            )
            preflight_segments.setdefault(pool_key, []).append(initial_data)
            preflight_clients[pool_key] = tts_instance
            prepared.append(
                {
                    "index": chronological_index,
                    "segment": segment,
                    "profile": profile,
                    "pool_key": pool_key,
                    "tts_instance": tts_instance,
                    "tts_system": tts_system,
                    "base_args": base_args,
                    "base_cache_prefix": base_cache_prefix,
                    "segment_cache_dir": segment_cache_dir,
                    "final_path": final_path,
                    "style_prompt": style_prompt,
                }
            )

        self._preflight_tts_pools(
            preflight_segments,
            preflight_clients,
            reference_issues,
        )

        for metadata in prepared:
            self._synthesize_measured_candidates(metadata, policy)

        combined_audio, real_segment_positions = self._adjust_and_combine_audio_grouped(
            segments
        )
        combined_audio.export(output_path, format="wav")
        self.real_segment_positions = real_segment_positions

        if self.cache_manager.use_cache and self._plan_dependent_cache_allowed:
            shutil.copy(output_path, cached_audio_path)

        track_usage: Dict[str, int] = {}
        for segment in segments:
            variant = segment.get("selected_variant", "missing")
            track_usage[variant] = track_usage.get(variant, 0) + 1
        logger.debug("Measured TTS candidate usage: %s", track_usage)
        self.performance_tracker.end_timing("speech_synthesis")
        return output_path

    def _tts_selection_cache_fingerprint(self, segments: List[Dict[str, Any]]) -> str:
        return cache_key_helpers.tts_selection_cache_fingerprint(self, segments)

    def _synthesize_measured_candidates(
        self,
        metadata: Dict[str, Any],
        policy,
    ) -> None:
        segment = metadata["segment"]
        target_duration = float(segment["_timing_available_window"])
        generated: List[Dict[str, Any]] = []
        tried_texts: set[str] = set()
        try:
            Path(metadata["final_path"]).unlink()
        except OSError:
            pass

        def generate(variant: str, attempts: int = 1) -> Optional[Dict[str, Any]]:
            value = segment.get(variant, "")
            text = value.strip() if isinstance(value, str) else ""
            if not text or text in tried_texts:
                return None
            tried_texts.add(text)
            candidate = self._load_or_synthesize_candidate(
                metadata,
                variant=variant,
                text=text,
                attempts=attempts,
            )
            if candidate is not None:
                generated.append(candidate)
            return candidate

        normal = generate("translation", attempts=3)
        if normal is None:
            segment["synthesized_speech_len"] = 0.0
            segment["synthesized_speech_file"] = None
            segment["selected_variant"] = "missing"
            return

        if normal["duration"] < target_duration:
            generate("long_translation")
        elif normal["duration"] > target_duration + policy.max_overflow:
            short = generate("short_translation")
            if (
                short is not None
                and short["duration"] > target_duration + policy.max_overflow
            ):
                generate("very_short_translation")

        winner = generated[0]
        best_difference = abs(winner["duration"] - target_duration)
        for candidate in generated[1:]:
            difference = abs(candidate["duration"] - target_duration)
            if difference + 1e-9 < best_difference:
                winner = candidate
                best_difference = difference
        final_path = metadata["final_path"]
        shutil.copy(winner["path"], final_path)
        segment["_tts_cache_contract"] = "anchor_raw_v2"
        segment["synthesized_speech_len"] = winner["duration"]
        segment["synthesized_speech_file"] = final_path
        segment["synthesized_text"] = winner["text"]
        segment["selected_variant"] = winner["variant"]

        for candidate in generated:
            try:
                Path(candidate["path"]).unlink()
            except OSError:
                pass

    def _load_or_synthesize_candidate(
        self,
        metadata: Dict[str, Any],
        *,
        variant: str,
        text: str,
        attempts: int,
    ) -> Optional[Dict[str, Any]]:
        from tts.models import TTSSegmentData

        segment = metadata["segment"]
        index = metadata["index"]
        candidate_path = str(
            self.audio_chunks_dir / f"candidate_{index}_{variant}.wav"
        )
        cache_key = self._raw_tts_segment_cache_key(
            base_cache_prefix=metadata["base_cache_prefix"],
            tts_system=metadata["tts_system"],
            segment=segment,
            speaker=segment["speaker"],
            translation=text,
            style_prompt=metadata["style_prompt"],
            reference_audio_path=metadata["base_args"].get("reference_audio_path"),
            reference_mode=metadata["base_args"].get("reference_mode"),
            reference_text=metadata["base_args"].get("reference_text"),
            client_pool_settings=metadata["pool_key"],
            legacy_index=segment.get("_timing_original_index", index),
            emotion=segment.get("emotion", "Neutral"),
            tts_prompt_prefix=self.config.get("tts_prompt_prefix"),
            voice_prompt=self.config.get("voice_prompt"),
        )
        cache_path = metadata["segment_cache_dir"] / f"{cache_key}.wav"

        if (
            self.cache_manager.use_cache
            and self._plan_dependent_cache_allowed
            and cache_path.exists()
        ):
            try:
                shutil.copy(cache_path, candidate_path)
                duration = self._measure_raw_tts_for_timing(candidate_path, index)
                if duration > 0:
                    return {
                        "variant": variant,
                        "text": text,
                        "path": candidate_path,
                        "duration": duration,
                    }
            except Exception as exc:
                logger.warning("Could not reuse cached TTS candidate %s: %s", variant, exc)
            try:
                cache_path.unlink()
                self._segment_cache_metadata_path(cache_path).unlink()
            except OSError:
                pass

        segment_data = TTSSegmentData(
            **{
                **metadata["base_args"],
                "text": text,
                "output_path": candidate_path,
            }
        )
        last_error = None
        for attempt in range(max(1, attempts)):
            try:
                Path(candidate_path).unlink()
            except OSError:
                pass
            try:
                metadata["tts_instance"].synthesize(
                    segments_data=[segment_data],
                    language=self.config.get("target_language"),
                )
                if not os.path.exists(candidate_path):
                    continue
                duration = self._measure_raw_tts_for_timing(candidate_path, index)
                if duration <= 0:
                    continue
                if self.cache_manager.use_cache and self._plan_dependent_cache_allowed:
                    self._cache_raw_tts_segment(
                        candidate_path,
                        cache_path,
                        synthesized_text=text,
                    )
                return {
                    "variant": variant,
                    "text": text,
                    "path": candidate_path,
                    "duration": duration,
                }
            except Exception as exc:
                last_error = exc
                logger.warning(
                    "TTS candidate %s failed for segment %d (attempt %d/%d): %s",
                    variant,
                    index,
                    attempt + 1,
                    max(1, attempts),
                    exc,
                )
        if last_error is not None:
            logger.error(
                "No usable %s candidate for segment %d: %s",
                variant,
                index,
                last_error,
            )
        try:
            Path(candidate_path).unlink()
        except OSError:
            pass
        return None

    def resynthesize_one_segment(
        self,
        segments: List[Dict],
        segment_index: int,
        override_text: Optional[str] = None,
    ) -> Dict[str, Any]:
        """Resynthesize a single segment (in-place) with the currently
        configured TTS and reference audio.

        Args:
            segments: The full list of cached translation segments.
            segment_index: Index of the segment to resynthesize.
            override_text: Optional text used instead of ``segment.translation``.

        Returns:
            The updated segment dict (also mutated in-place inside ``segments``).
        """
        from tts.models import TTSSegmentData

        if not (0 <= segment_index < len(segments)):
            raise IndexError(f"segment_index {segment_index} out of range (0..{len(segments)-1})")

        segment_dict = segments[segment_index]
        original_segment_index = self._canonical_segment_index(
            segment_dict, segment_index
        )
        speaker = segment_dict.get("speaker") or "SPEAKER_00"

        profile = self._resolve_voice_profile(speaker)
        pool_key = self._profile_pool_key(profile)
        tts_instance = self.tts_clients.get(pool_key) or self.default_tts
        if tts_instance is None:
            raise RuntimeError(
                f"TTS client for '{profile.tts_system or self._default_tts_system()}' is not initialised"
            )
        tts_system = profile.tts_system or self._default_tts_system()

        text_to_synthesize = (override_text or segment_dict.get("translation") or "").strip()
        if not text_to_synthesize:
            raise ValueError("Cannot resynthesize a segment with empty text")

        voice_name = profile.voice_name
        if voice_name is None:
            voice_cfg = self.config.get('voice_name')
            if isinstance(voice_cfg, str):
                voice_name = voice_cfg

        segment_style_override = (segment_dict.get("style_prompt") or "").strip()
        segment_style_prompt = segment_style_override or profile.style_prompt

        self.audio_chunks_dir.mkdir(parents=True, exist_ok=True)
        output_path = str(self.audio_chunks_dir / f"{segment_index}.wav")

        tts_segment_data_args: Dict[str, Any] = {
            "speaker": speaker,
            "text": text_to_synthesize,
            "emotion": segment_dict.get("emotion", "Neutral"),
            "style_prompt": segment_style_prompt,
            "reference_audio_path": profile.reference_audio,
            "reference_text": profile.reference_text,
            "reference_mode": profile.reference_mode,
            "segment_index": original_segment_index,
            "voice": voice_name,
            "speed": 1.0,
            "target_duration": segment_dict.get(
                "_timing_available_window",
                max(segment_dict.get("end", 0) - segment_dict.get("start", 0), 0.0),
            ),
        }

        segment_reference_min_duration = float(self.config.get('segment_reference_min_duration', 2.0) or 0.0)
        provider_capability = getattr(
            tts_instance, "reference_capability", "unsupported"
        )
        tts_segment_data_args, _ = self._resolve_segment_reference(
            tts_segment_data_args=tts_segment_data_args,
            segment_dict=segment_dict,
            profile=profile,
            provider_capability=provider_capability,
            speaker=speaker,
            segment_index=original_segment_index,
            original_audio_segment=None,
            segment_reference_min_duration=segment_reference_min_duration,
            for_resynthesis=True,
        )

        segment_data = TTSSegmentData(**{**tts_segment_data_args, "text": text_to_synthesize, "output_path": output_path})
        self._preflight_tts_pools(
            {pool_key: [segment_data]},
            {pool_key: tts_instance},
        )

        # Remove any stale zero-byte file so `os.path.exists` reflects reality.
        try:
            if os.path.exists(output_path):
                os.remove(output_path)
        except OSError:
            pass

        max_attempts = 3
        last_error: Optional[Exception] = None
        for attempt in range(max_attempts):
            logger.info(
                f"Resynthesizing segment {segment_index} (speaker={speaker}, tts={tts_system}) "
                f"attempt {attempt + 1}/{max_attempts}"
            )
            try:
                tts_instance.synthesize(
                    segments_data=[segment_data],
                    language=self.config.get('target_language'),
                )
            except Exception as exc:
                last_error = exc
                logger.error(f"Resynthesize attempt {attempt + 1} failed for segment {segment_index}: {exc}")
                continue

            if os.path.exists(output_path):
                try:
                    audio_info = AudioSegment.from_file(output_path)
                except Exception as exc:
                    last_error = exc
                    logger.error(f"Resynthesize produced unreadable audio for segment {segment_index}: {exc}")
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                    continue
                if len(audio_info) <= 0:
                    try:
                        os.remove(output_path)
                    except OSError:
                        pass
                    continue

                segment_dict['_tts_cache_contract'] = 'anchor_raw_v2'
                segment_dict['synthesized_speech_len'] = self._measure_raw_tts_for_timing(
                    output_path,
                    segment_index,
                )
                if segment_dict['synthesized_speech_len'] <= 0:
                    last_error = RuntimeError("TTS produced only silence")
                    continue
                segment_dict['synthesized_speech_file'] = output_path
                segment_dict['synthesized_text'] = text_to_synthesize
                if override_text:
                    segment_dict['translation'] = text_to_synthesize
                return segment_dict

        raise RuntimeError(
            f"TTS ({tts_system}) did not produce an audio file for segment {segment_index} "
            f"after {max_attempts} attempts. Last error: {last_error!r}. "
            f"Reference audio: {segment_data.reference_audio_path!r}"
        )

    def rebuild_translated_audio_from_chunks(self) -> Optional[str]:
        """Rebuild the aggregate dubbed track from the latest editor snapshot.

        ``Regenerate selected row`` deliberately updates only one raw chunk.
        The combine-video step calls this method so its mux input reflects that
        chunk without invoking TTS again for the other segments.
        """
        source_audio_path = Path(self.config.get("audio_artifacts_dir")) / "source.wav"
        if not source_audio_path.is_file():
            logger.warning(
                "Cannot rebuild translated audio from chunks because source audio is missing: %s",
                source_audio_path,
            )
            return None

        snapshot_key = self._build_dubbing_text_snapshot_key(str(source_audio_path))
        if not self.cache_manager.cache_exists("dubbing_texts", snapshot_key):
            logger.info(
                "No Dubbing Texts snapshot found; reusing the existing translated audio track."
            )
            return None

        snapshot = self.cache_manager.load_from_cache("dubbing_texts", snapshot_key)
        if isinstance(snapshot, list):
            segments = snapshot
        elif isinstance(snapshot, dict) and snapshot.get("version") == 1:
            segments = snapshot.get("segments")
        else:
            raise ValueError("Unexpected Dubbing Texts snapshot payload")

        if not isinstance(segments, list) or not segments:
            raise ValueError("Dubbing Texts snapshot has no segments to combine")

        _update_pipeline_context(
            self, "timing_source_audio_file", str(source_audio_path)
        )
        _update_pipeline_context(
            self,
            "timing_source_duration",
            len(AudioSegment.from_file(source_audio_path)) / 1000.0,
        )
        combined_audio, real_segment_positions = self._adjust_and_combine_audio_grouped(
            segments
        )

        output_path = Path(self.config.get("translated_audio_path"))
        output_path.parent.mkdir(parents=True, exist_ok=True)
        combined_audio.export(output_path, format="wav")
        self.real_segment_positions = real_segment_positions
        logger.info("Rebuilt translated audio from current chunks: %s", output_path)
        return str(output_path)

    def _save_transcription_file(self, transcription: List[Dict]) -> None:
        """Save transcription to a readable text file."""
        from src.utils.time_utils import format_seconds_to_hms
        
        transcription_output_path = self.config.get("transcription_path")
        os.makedirs(os.path.dirname(transcription_output_path), exist_ok=True)
        
        try:
            with open(transcription_output_path, 'w', encoding='utf-8') as f:
                for segment in transcription:
                    start_seconds = segment['start']
                    end_seconds = segment['end']
                    formatted_time = format_seconds_to_hms(
                        start_seconds, include_milliseconds=True
                    )
                    formatted_time_end = format_seconds_to_hms(
                        end_seconds, include_milliseconds=True
                    )
                    
                    f.write(f"[{formatted_time}-{formatted_time_end}] {segment['speaker']}: {segment['text']}\n")
            logger.info(f"Transcription saved to {transcription_output_path}")
        except Exception as e:
            logger.warning(f"Failed to save transcription to file: {e}")
    
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
        """Return the TTS backend name a speaker is routed to.

        Preserved for callers/tests that only need the backend name; new code
        should use :meth:`_resolve_voice_profile` to get the full profile.
        """
        return self._resolve_voice_profile(speaker_id).tts_system or self._default_tts_system()

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
        """Validate every active pool before the first synthesis request."""
        issues = list(initial_issues or [])
        resolved_error_indices = {index for index, _ in issues}
        for pool_key, segments_data in segments_by_pool.items():
            if not segments_data:
                continue
            client = clients.get(pool_key)
            if client is None:
                continue
            validator = getattr(client, "validate_segments", None)
            if validator is None:
                continue
            for issue in validator(segments_data):
                if issue[0] not in resolved_error_indices:
                    issues.append(issue)

        if issues:
            issues.sort(key=lambda item: item[0])
            details = "\n".join(f"- {message}" for _, message in issues)
            raise ValueError(f"TTS reference validation failed before synthesis:\n{details}")
    
    @staticmethod
    def _trim_trailing_silence(
        audio_path: str,
        silence_threshold_db: float = -40.0,
        keep_tail_ms: int = 100,
        window_ms: int = 10,
    ) -> None:
        """Trim silence at the end of a synthesized WAV in place.

        Applied to every TTS backend right after the segment file is
        written and before ``synthesized_speech_len`` is computed.
        Without this, tail silence from the TTS is counted as speech,
        which:
          - makes ``ratio = original / actual`` look closer to 1 than
            it really is, so the comfort-zone check skips alternative-
            text resynthesis;
          - leaves the group-level ``atempo`` stretching a hunk of
            silence, producing the "audio ends early, then a pause"
            artefact in the final track.
        """
        if not audio_path or not os.path.exists(audio_path):
            return
        try:
            seg = AudioSegment.from_file(audio_path)
        except Exception as exc:
            logger.debug(f"Trailing silence trim skipped for {audio_path}: {exc}")
            return
        total_ms = len(seg)
        if total_ms == 0:
            return
        last_active_end_ms = 0
        for start in range(0, total_ms, window_ms):
            window = seg[start:start + window_ms]
            level = window.dBFS
            if level == float('-inf'):
                continue
            if level > silence_threshold_db:
                last_active_end_ms = start + window_ms
        if last_active_end_ms == 0:
            return
        end_ms = min(total_ms, last_active_end_ms + keep_tail_ms)
        if total_ms - end_ms < window_ms:
            return
        try:
            seg[:end_ms].export(audio_path, format="wav")
        except Exception as exc:
            logger.debug(f"Trailing silence trim export failed for {audio_path}: {exc}")
            return
        logger.debug(
            f"Trimmed {total_ms - end_ms}ms trailing silence from {audio_path} "
            f"(kept {end_ms}ms of {total_ms}ms)"
        )

    def _measure_raw_tts_for_timing(self, audio_path: str, segment_index: int) -> float:
        """Measure audible duration without modifying the raw synthesized WAV."""
        from .timing import trim_audio_edges

        self.su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)
        measured_path = self.su_audio_chunks_dir / f"measure_{segment_index}.wav"
        result = trim_audio_edges(audio_path, measured_path)
        if result.error:
            logger.warning("Could not trim TTS edges for %s: %s", audio_path, result.error)
        return result.trimmed_duration if result.usable else 0.0

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
        """Cache raw TTS plus a version marker, independent of timing policy."""
        shutil.copy(source_path, cache_path)
        metadata_path = self._segment_cache_metadata_path(cache_path)
        metadata_path.write_text(
            json.dumps(
                {
                    "audio_contract": "anchor_raw_v2",
                    "synthesized_text": synthesized_text,
                },
                ensure_ascii=False,
                sort_keys=True,
            ),
            encoding="utf-8",
        )

    def _cached_segment_metadata(self, cache_path: Path) -> Dict[str, Any]:
        metadata_path = self._segment_cache_metadata_path(cache_path)
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            return data if isinstance(data, dict) else {}
        except (OSError, ValueError, TypeError):
            return {}

    def _cached_segment_contract(self, cache_path: Path) -> str:
        contract = self._cached_segment_metadata(cache_path).get("audio_contract")
        if contract in {"anchor_raw_v1", "anchor_raw_v2"}:
            return contract
        return "legacy"

    def _adjust_and_combine_audio_grouped_legacy(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]:
        """
        Adjusts timing and combines audio segments with optimizations for speaker continuity.
        
        This method:
        1. Groups segments by speaker
        2. Optimizes each speaker's segments in continuous groups
        3. Creates separate audio tracks for each speaker
        4. Mixes all speaker tracks together
        
        Args:
            segments: List of transcript segments with translations and speaker info
            
        Returns:
            Tuple of (Combined AudioSegment with proper timing, List of real segment positions)
        """
        logger.warning("Legacy group timing is disabled; using anchor-based timing instead.")
        return self._adjust_and_combine_audio_grouped(segments)

        # Retained unreachable for one compatibility cycle as implementation
        # history; all callers are routed to the active anchor assembler.
        if not segments:
            return AudioSegment.empty(), []
        
        logger.info("Grouping segments by speaker and optimizing timing...")
        
        SPLITTING_PAUSE_THRESHOLD_SECONDS = 3
        MAX_GROUP_DURATION_SECONDS = 60
        LIMIT_MIN_ADJUSTMENT_RATIO = 0.5
        LIMIT_MAX_ADJUSTMENT_RATIO = 1.15
        
        # Get all unique speakers
        all_speakers = set(segment["speaker"] for segment in segments)
        logger.debug(f"Found {len(all_speakers)} unique speakers")
        
        # Create empty audio tracks for each speaker (full duration)
        total_duration_ms = int(max(segment["end"] for segment in segments) * 1000) + 1000  # Add 1s padding
        speaker_tracks = {speaker: AudioSegment.silent(duration=total_duration_ms) for speaker in all_speakers}
        
        # Track real segment positions for pause calculation
        real_segment_positions = []
        
        # For debug: store all speaker groups for later use in debug video
        speaker_groups_info = {}
        
        # Process each speaker's segments separately
        for speaker in sorted(all_speakers):
            # Get segments for this speaker
            speaker_segments = [segment for segment in segments if segment["speaker"] == speaker]
            logger.debug(f"Processing {len(speaker_segments)} segments for speaker {speaker}")
            
            # Group segments by continuous speech
            speaker_groups = []
            current_group = []
            
            for i, segment in enumerate(speaker_segments):
                start_new_group = False
                
                if not current_group:
                    start_new_group = True
                elif i > 0:
                    prev_segment = speaker_segments[i-1]
                    pause_duration = segment["start"] - prev_segment["end"]
                    
                    if pause_duration > SPLITTING_PAUSE_THRESHOLD_SECONDS:
                        start_new_group = True
                
                if current_group and segment["end"] - current_group[0]["start"] > MAX_GROUP_DURATION_SECONDS:
                    start_new_group = True
                
                if start_new_group and current_group:
                    speaker_groups.append(current_group)
                    current_group = []
                
                current_group.append(segment)
            
            # Add the last group
            if current_group:
                speaker_groups.append(current_group)
            
            logger.debug(f"Divided speaker {speaker} into {len(speaker_groups)} continuous groups")
            
            # Store groups information for debug
            if self.config.get('debug_info', False):
                speaker_groups_info[speaker] = speaker_groups
            self.debug_data["speaker_groups"] = speaker_groups_info
            
            # Process each group for this speaker
            for group_idx, group in enumerate(speaker_groups):
                group_start_time_ms = group[0]["start"] * 1000
                group_end_time_ms = group[-1]["end"] * 1000
                target_duration_ms = int(group_end_time_ms - group_start_time_ms)
                
                # Combine all synthesized audio in this group
                combined_group_audio = AudioSegment.empty()
                group_segment_positions = []  # Track individual segment positions within the group
                
                for i, segment in enumerate(group):
                    # Add pause between segments if not the first segment
                    segment_start_in_group_ms = len(combined_group_audio)
                    
                    if i > 0:
                        prev_segment_end = group[i-1]["end"]
                        current_segment_start = segment["start"]
                        pause_duration_ms = int((current_segment_start - prev_segment_end) * 1000)
                        if pause_duration_ms > 0:
                            combined_group_audio += AudioSegment.silent(duration=pause_duration_ms)
                            segment_start_in_group_ms = len(combined_group_audio)
                    
                    # Load segment audio
                    segment_file = segment.get('synthesized_speech_file')
                    if not segment_file:
                        segment_file = str(self.audio_chunks_dir / f"{segments.index(segment)}.wav")
                    segment_audio = None
                    if segment_file and os.path.exists(segment_file):
                        try:
                            candidate = AudioSegment.from_file(segment_file)
                            if len(candidate) > 0:
                                segment_audio = candidate
                        except Exception as e:
                            logger.warning(
                                f"Could not read segment audio {segment_file}: {e}. Substituting silence."
                            )
                    if segment_audio is None:
                        # Fallback: create silence with original duration when no
                        # usable audio was produced. This preserves timing for the
                        # rest of the track while making the gap obvious.
                        duration_ms = int((segment["end"] - segment["start"]) * 1000)
                        segment_audio = AudioSegment.silent(duration=max(duration_ms, 1))
                        logger.warning(
                            f"Segment #{segments.index(segment)} has no synthesized audio; "
                            f"inserting {duration_ms}ms of silence."
                        )
                    
                    combined_group_audio += segment_audio
                    segment_end_in_group_ms = len(combined_group_audio)
                    
                    # Store the segment position within the group (before speed adjustment)
                    group_segment_positions.append({
                        "segment": segment,
                        "start_in_group_ms": segment_start_in_group_ms,
                        "end_in_group_ms": segment_end_in_group_ms,
                        "original_index": segments.index(segment)
                    })
                
                # Calculate required speed adjustment for the entire group
                actual_duration_ms = len(combined_group_audio)
                ratio = target_duration_ms / actual_duration_ms if actual_duration_ms > 0 else 1.0
                
                # Clamp ratio to maintain natural speech
                ratio_clamped = min(max(ratio, LIMIT_MIN_ADJUSTMENT_RATIO), LIMIT_MAX_ADJUSTMENT_RATIO)
                
                # Apply speed adjustment to the entire group if needed
                adjusted_group_audio = combined_group_audio
                if abs(ratio_clamped - 1.0) > 0.01:
                    try:
                        # Save the combined group audio to a temporary file
                        tmp_in = str(self.audio_chunks_dir / f"group_{speaker}_{group_idx}.wav")
                        tmp_out = str(self.su_audio_chunks_dir / f"group_{speaker}_{group_idx}.wav")
                        os.makedirs(os.path.dirname(tmp_out), exist_ok=True)
                        combined_group_audio.export(tmp_in, format="wav")
                        
                        # Store ratio for debugging
                        if self.config.get('debug_info', False):
                            for segment in group:
                                self.debug_data.setdefault("speed_ratios", {})[segments.index(segment)] = ratio_clamped
                        
                        # Apply tempo filter — pass argv as a list so paths
                        # with spaces (e.g. "prj/My Video/artifacts/...") do
                        # not break the shell split and silently skip the
                        # tempo adjustment, which would otherwise leave the
                        # group short and pad the tail with silence.
                        tempo = 1.0 / ratio_clamped
                        cmd = [
                            "ffmpeg", "-y",
                            "-i", tmp_in,
                            "-filter:a", f"atempo={tempo}",
                            "-vn", tmp_out,
                        ]
                        result = subprocess.run(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
                        if result.returncode == 0:
                            adjusted_group_audio = AudioSegment.from_file(tmp_out)
                        else:
                            stderr_tail = (result.stderr or b"").decode("utf-8", errors="replace").strip().splitlines()[-3:]
                            logger.warning(
                                f"Warning: Speed adjustment failed for group {speaker}_{group_idx}: "
                                + " | ".join(stderr_tail)
                            )
                    except Exception as exc:
                        logger.warning(f"Warning: Speed adjustment failed for group {speaker}_{group_idx}: {exc}")
                
                # Enforce allowed overflow beyond the group's original timeframe
                overflow_tolerance = 1.0
                overflow_tolerance = max(0.0, min(1.0, overflow_tolerance))

                original_group_span_ms = target_duration_ms
                adjusted_len_ms = len(adjusted_group_audio)
                if adjusted_len_ms > original_group_span_ms:
                    overflow_ms = adjusted_len_ms - original_group_span_ms
                    allowed_len_ms = original_group_span_ms + int(overflow_ms * overflow_tolerance)
                    if adjusted_len_ms > allowed_len_ms:
                        adjusted_group_audio = adjusted_group_audio[:allowed_len_ms]
                
                # Calculate real segment positions after speed adjustment
                final_group_duration_ms = len(adjusted_group_audio)
                position_ms = int(group_start_time_ms)
                
                for seg_pos in group_segment_positions:
                    # Apply speed ratio to get actual positions in final audio
                    real_start_ms = position_ms + (seg_pos["start_in_group_ms"] * ratio_clamped)
                    real_end_ms = position_ms + (seg_pos["end_in_group_ms"] * ratio_clamped)
                    # Clamp to the trimmed group end if overflow was restricted
                    group_final_end_ms = position_ms + final_group_duration_ms
                    if real_end_ms > group_final_end_ms:
                        real_end_ms = group_final_end_ms
                    if real_start_ms > group_final_end_ms:
                        real_start_ms = group_final_end_ms
                    
                    real_segment_positions.append({
                        "start": real_start_ms / 1000.0,
                        "end": real_end_ms / 1000.0,
                        "speaker": seg_pos["segment"]["speaker"],
                        "text": seg_pos["segment"]["text"],
                        "translation": seg_pos["segment"]["translation"],
                        "original_index": seg_pos["original_index"],
                        "original_start": seg_pos["segment"]["start"],
                        "original_end": seg_pos["segment"]["end"]
                    })
                
                # Place the adjusted group at the correct position in the speaker's track
                speaker_track = speaker_tracks[speaker]
                
                # Extend track if needed
                if position_ms + len(adjusted_group_audio) > len(speaker_track):
                    extension = position_ms + len(adjusted_group_audio) - len(speaker_track)
                    speaker_track += AudioSegment.silent(duration=extension)
                
                # Overlay the adjusted group audio at the correct position
                speaker_tracks[speaker] = speaker_track.overlay(
                    adjusted_group_audio,
                    position=position_ms
                )
                
                from src.utils.time_utils import format_seconds_to_srt
                duration_ms = len(adjusted_group_audio)
                duration_minutes = duration_ms / (1000 * 60)
                logger.debug(f"Processed group {group_idx+1}/{len(speaker_groups)} for speaker {speaker}: "
                      f"{len(group)} segments, Speed ratio: {ratio_clamped:.2f}, "
                      f"Time: {format_seconds_to_srt(group_start_time_ms / 1000)}-{format_seconds_to_srt((group_start_time_ms + duration_ms) / 1000)}, "
                      f"Duration: {duration_minutes:.2f} minutes")
        
        # Mix all speaker tracks together
        logger.info("Mixing all speaker tracks together...")
        final_audio = AudioSegment.silent(duration=total_duration_ms)
        
        for speaker, track in speaker_tracks.items():
            # Overlay each speaker track onto the final audio
            final_audio = final_audio.overlay(track)
            logger.debug(f"Added speaker {speaker}'s track to the mix")
        
        # If we processed the full video, pad to match original length if necessary
        input_path = self.config.get('input')
        if input_path and os.path.exists(str(input_path)) and self.config.get('start_time') is None and self.config.get('duration') is None:
            try:
                total_original_ms = len(AudioSegment.from_file(input_path))
                if len(final_audio) < total_original_ms:
                    final_audio += AudioSegment.silent(duration=total_original_ms - len(final_audio))
            except Exception as e:
                logger.warning(f"Warning: Could not pad audio to match original length: {e}")
        
        # Sort real segment positions by start time
        real_segment_positions.sort(key=lambda x: x["start"])
        
        return final_audio, real_segment_positions 

    def _adjust_and_combine_audio_grouped(self, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]:
        """Place speech at immutable recognized starts using per-clip tempo only."""
        if not segments:
            return AudioSegment.empty(), []

        from .timing import TimingPolicy, calculate_segment_timing, plan_anchor_windows, trim_audio_edges

        source_duration = getattr(self, "_timing_source_duration", None)
        if source_duration is None:
            source_path = getattr(self, "_timing_source_audio_file", None)
            if source_path and os.path.exists(str(source_path)):
                source_duration = len(AudioSegment.from_file(source_path)) / 1000.0
            else:
                # Compatibility for direct callers that predate the source-duration
                # contract. Normal synthesis always sets the exact processed length.
                source_duration = max(float(segment["end"]) for segment in segments) + 1.0

        policy = TimingPolicy.from_config(self.config)
        planned = plan_anchor_windows(segments, source_duration)
        for item in planned:
            item.segment["_timing_original_index"] = item.original_index
            item.segment["_timing_available_window"] = item.available_window
            item.segment.pop("_timing_next_anchor", None)
        segments[:] = [item.segment for item in planned]

        source_duration_ms = max(0, round(float(source_duration) * 1000))
        final_audio = AudioSegment.silent(duration=source_duration_ms)
        real_segment_positions: List[Dict[str, Any]] = []
        diagnostics: List[Dict[str, Any]] = []
        self.su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)

        speaker_groups = {}
        for item in planned:
            speaker_groups.setdefault(item.segment.get("speaker", "UNKNOWN"), []).append(item.segment)
        self.debug_data["speaker_groups"] = {
            speaker: [speaker_segments] for speaker, speaker_segments in speaker_groups.items()
        }

        for chronological_index, item in enumerate(planned):
            segment = item.segment
            segment_file = segment.get("synthesized_speech_file")
            if not segment_file:
                candidate_path = self.audio_chunks_dir / f"{chronological_index}.wav"
                segment_file = str(candidate_path) if candidate_path.exists() else None

            raw_duration = None
            leading_removed = None
            trailing_removed = None
            trim_error = None
            used_fallback = False
            clip = None

            if segment_file and os.path.exists(str(segment_file)):
                try:
                    raw_clip = AudioSegment.from_file(segment_file)
                    if len(raw_clip) > 0:
                        if segment.get("_tts_cache_contract") in {
                            "anchor_raw_v1",
                            "anchor_raw_v2",
                        }:
                            timing_path = self.su_audio_chunks_dir / f"timed_{item.original_index}.wav"
                            trim_result = trim_audio_edges(segment_file, timing_path)
                            raw_duration = trim_result.raw_duration
                            leading_removed = trim_result.leading_removed
                            trailing_removed = trim_result.trailing_removed
                            trim_error = trim_result.error
                            if trim_result.usable:
                                if timing_path.exists():
                                    clip = AudioSegment.from_file(timing_path)
                                else:
                                    clip = raw_clip
                                    leading_removed = 0.0
                                    trailing_removed = 0.0
                            elif trim_result.error:
                                clip = raw_clip
                                leading_removed = 0.0
                                trailing_removed = 0.0
                            else:
                                logger.warning(
                                    "Segment #%d contains no usable synthesized speech; inserting silence.",
                                    item.original_index,
                                )
                        else:
                            # Legacy cache entries may already have been tail-trimmed.
                            # Their current readable duration is authoritative, while
                            # unavailable raw/removed-edge values stay explicitly unknown.
                            if raw_clip.dBFS != float("-inf"):
                                clip = raw_clip
                except Exception as exc:
                    trim_error = str(exc)
                    logger.warning("Could not read segment audio %s: %s", segment_file, exc)

            if clip is None:
                fallback_ms = max(0, round((item.end - item.start) * 1000))
                clip = AudioSegment.silent(duration=fallback_ms)
                used_fallback = True
                logger.warning(
                    "Segment #%d has no synthesized audio; inserting %dms of silence.",
                    item.original_index,
                    fallback_ms,
                )

            trimmed_duration = len(clip) / 1000.0
            timing = calculate_segment_timing(
                start=item.start,
                end=item.end,
                next_start=None,
                source_duration=float(source_duration),
                audio_duration=trimmed_duration,
                policy=policy,
            )

            actual_tempo = timing.tempo
            adjusted_clip = clip
            tempo_error = None
            if abs(timing.tempo - 1.0) > 0.0005 and len(clip) > 0 and not used_fallback:
                input_path = self.su_audio_chunks_dir / f"tempo_in_{item.original_index}.wav"
                output_path = self.su_audio_chunks_dir / f"tempo_{item.original_index}.wav"
                try:
                    clip.export(input_path, format="wav")
                    result = subprocess.run(
                        [
                            "ffmpeg", "-y", "-i", str(input_path),
                            "-filter:a", f"atempo={timing.tempo:.8f}",
                            "-vn", str(output_path),
                        ],
                        stdout=subprocess.PIPE,
                        stderr=subprocess.PIPE,
                    )
                    if result.returncode != 0:
                        stderr = (result.stderr or b"").decode("utf-8", errors="replace")
                        raise RuntimeError(stderr.strip().splitlines()[-1] if stderr.strip() else "ffmpeg failed")
                    adjusted_clip = AudioSegment.from_file(output_path)
                except Exception as exc:
                    tempo_error = str(exc)
                    actual_tempo = 1.0
                    adjusted_clip = clip
                    logger.warning(
                        "Tempo adjustment failed for segment #%d; using natural speed: %s",
                        item.original_index,
                        exc,
                    )

            actual_duration = len(adjusted_clip) / 1000.0
            actual_overflow = max(0.0, actual_duration - item.available_window)
            within_policy = actual_overflow <= policy.max_overflow + 0.002
            if not within_policy:
                logger.warning(
                    "Segment #%d remains %.3fs beyond its anchor window after %.3fx tempo.",
                    item.original_index,
                    actual_overflow,
                    actual_tempo,
                )

            start_ms = round(item.start * 1000)
            final_audio = final_audio.overlay(adjusted_clip, position=start_ms)
            natural_end_ms = start_ms + len(adjusted_clip)
            clipped_end_ms = min(natural_end_ms, source_duration_ms)
            boundary_truncated_ms = max(0, natural_end_ms - source_duration_ms)
            if boundary_truncated_ms:
                logger.warning(
                    "Segment #%d is truncated by %dms at the processed source boundary.",
                    item.original_index,
                    boundary_truncated_ms,
                )

            position = {
                "start": round(start_ms / 1000.0, 3),
                "end": round(clipped_end_ms / 1000.0, 3),
                "speaker": segment.get("speaker", "UNKNOWN"),
                "text": segment.get("text", ""),
                "translation": segment.get("translation", ""),
                "original_index": item.original_index,
                "original_start": item.start,
                "original_end": item.end,
                "tempo": actual_tempo,
                "overflow": actual_overflow,
                "within_policy": within_policy,
                "boundary_truncated_ms": boundary_truncated_ms,
            }
            real_segment_positions.append(position)
            diagnostics.append(
                {
                    "segment_index": item.original_index,
                    "speaker": segment.get("speaker", "UNKNOWN"),
                    "recognized_start": item.start,
                    "recognized_end": item.end,
                    "available_window": item.available_window,
                    "raw_tts_duration": "" if raw_duration is None else raw_duration,
                    "trimmed_tts_duration": trimmed_duration,
                    "leading_silence_removed": "" if leading_removed is None else leading_removed,
                    "trailing_silence_removed": "" if trailing_removed is None else trailing_removed,
                    "tempo": actual_tempo,
                    "actual_final_duration": max(0.0, (clipped_end_ms - start_ms) / 1000.0),
                    "final_start": start_ms / 1000.0,
                    "final_end": clipped_end_ms / 1000.0,
                    "overflow": actual_overflow,
                    "within_policy": within_policy,
                    "boundary_truncated_ms": boundary_truncated_ms,
                    "trim_error": trim_error or "",
                    "tempo_error": tempo_error or "",
                    "cache_contract": segment.get("_tts_cache_contract", "legacy"),
                    "selected_variant": segment.get("selected_variant", ""),
                    "semantic_unit_id": segment.get("semantic_unit_id", ""),
                    "semantic_plan_fingerprint": segment.get("semantic_plan_fingerprint", ""),
                    "continuation_id": segment.get("continuation_id", ""),
                }
            )

        real_segment_positions.sort(key=lambda value: (value["start"], value["original_index"]))
        self.debug_data["timing_alignment"] = diagnostics
        self.debug_data["speed_ratios"] = {
            row["segment_index"]: row["tempo"] for row in diagnostics
        }

        if self.config.get("debug_info", False):
            debug_dir = Path(self.config.get("debug_dir") or ".")
            debug_dir.mkdir(parents=True, exist_ok=True)
            diagnostics_path = debug_dir / "timing_alignment.tsv"
            with diagnostics_path.open("w", encoding="utf-8", newline="") as diagnostics_file:
                writer = csv.DictWriter(
                    diagnostics_file,
                    fieldnames=list(diagnostics[0].keys()),
                    delimiter="\t",
                )
                writer.writeheader()
                writer.writerows(diagnostics)

        # Overlay uses a source-sized base, so this remains exactly the rounded
        # processed source duration even when the last clip crosses the boundary.
        if len(final_audio) < source_duration_ms:
            final_audio += AudioSegment.silent(duration=source_duration_ms - len(final_audio))
        elif len(final_audio) > source_duration_ms:
            final_audio = final_audio[:source_duration_ms]
        return final_audio, real_segment_positions

    def _get_subtitle_path(self, subtitle_type: str, input_path: str, language: str) -> str:
        """Generate subtitle path based on input file and language.
        
        Args:
            subtitle_type: "original" or "translation"
            input_path: Path to the input video file
            language: Language code (source for original, target for translation)
            
        Returns:
            Path for the subtitle file in current working directory
        """
        input_file = Path(input_path)
        project_dir = Path(self.config.get("project_dir"))
        # Base filenames for source and target
        source_lang = self.config.get('source_language')
        target_lang = self.config.get('target_language')
        source_name = f"{input_file.stem}_{source_lang}.srt"
        target_name = f"{input_file.stem}_{target_lang}.srt"

        # If names coincide, disambiguate with explicit prefixes
        if source_name == target_name:
            if subtitle_type == "original":
                return str(project_dir / f"source_{source_name}")
            else:
                return str(project_dir / f"target_{target_name}")

        # Default: use requested language-specific filename
        return str(project_dir / f"{input_file.stem}_{language}.srt")

    def adjust_subtitle_timestamps(self, segments: List[Dict], pause_adjustments: List[Dict[str, float]]) -> List[Dict]:
        """Adjust subtitle timestamps based on pause adjustments from video processing.
        
        Args:
            segments: List of subtitle segments with start/end times
            pause_adjustments: List of pause adjustments from video processing
            
        Returns:
            List of segments with adjusted timestamps
        """
        if not pause_adjustments:
            logger.debug("No pause adjustments to apply to subtitles")
            return segments
        
        logger.info(f"Adjusting subtitle timestamps based on {len(pause_adjustments)} pause modifications")
        
        adjusted_segments = []
        for segment in segments:
            adjusted_segment = segment.copy()
            
            # Calculate cumulative time offset for this segment's timestamps
            start_offset = 0.0
            end_offset = 0.0
            
            for adjustment in pause_adjustments:
                # If the segment starts after this pause was shortened, apply the full offset
                if segment['start'] >= adjustment['original_end']:
                    start_offset = adjustment['cumulative_offset']
                # If the segment starts during this pause, apply partial offset
                elif segment['start'] >= adjustment['original_start']:
                    # Segment starts within the pause - calculate partial offset
                    if segment['start'] <= adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed']):
                        # Segment starts in the preserved part of the pause
                        start_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                    else:
                        # Segment would have started in the removed part - move to end of preserved pause
                        start_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                        adjusted_segment['start'] = adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed'])
                
                # Apply same logic for end time
                if segment['end'] >= adjustment['original_end']:
                    end_offset = adjustment['cumulative_offset']
                elif segment['end'] >= adjustment['original_start']:
                    if segment['end'] <= adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed']):
                        end_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                    else:
                        end_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                        adjusted_segment['end'] = adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed'])
            
            # Apply the calculated offsets
            adjusted_segment['start'] = max(0, adjusted_segment['start'] - start_offset)
            adjusted_segment['end'] = max(adjusted_segment['start'], adjusted_segment['end'] - end_offset)
            
            adjusted_segments.append(adjusted_segment)
        
        logger.debug(f"Adjusted timestamps for {len(adjusted_segments)} subtitle segments")
        return adjusted_segments 
