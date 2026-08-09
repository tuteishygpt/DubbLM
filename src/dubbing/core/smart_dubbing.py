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
import torch
import warnings
import shutil
import subprocess
from typing import Dict, List, Tuple, Optional, Any, Literal, Union
from pathlib import Path
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

# Import existing factories and interfaces
from tts.tts_factory import TTSFactory
from translation.translator_factory import TranslatorFactory
from transcription.transcription_factory import TranscriptionFactory

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
        """Initialize translator based on configuration."""
        self.translator = None
        self.translator_init_error = None
        try:
            self.translator = TranslatorFactory.create_translator(
                translator_type=self.config.get('translator_type', 'llm'),
                llm_provider=self.config.get('llm_provider', 'gemini'),
                model_name=self.config.get('llm_model_name'),
                temperature=self.config.get('llm_temperature', 0.5),
                refinement_llm_provider=self.config.get('refinement_llm_provider'),
                refinement_model_name=self.config.get('refinement_model_name'),
                refinement_temperature=self.config.get('refinement_temperature', 1.0),
                refinement_max_tokens=self.config.get('refinement_max_tokens'),
                refinement_persona=self.config.get('refinement_persona', 'normal'),
                translation_prompt_prefix=self.config.get('translation_prompt_prefix'),
                glossary=self.config.get('glossary'),
                cache_manager=self.cache_manager
            )
            logger.debug(f"Using {self.config.get('translator_type', 'llm')} translator")
        except Exception as e:
            self.translator_init_error = e
            logger.warning(f"Failed to initialize translator: {e}")
    
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
            profile.fallback_model or (self.config.get('tts_fallback_model') or ""),
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
        fallback_model = profile.fallback_model or self.config.get('tts_fallback_model')

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
            fallback_model=fallback_model,
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
        """Initialize transcriber based on configuration."""
        self.transcriber = None
        self.transcriber_init_error = None
        try:
            self.transcriber = TranscriptionFactory.create_transcriber(
                transcription_system=self.config.get('transcription_system', 'whisper'),
                source_language=self.config.get('source_language'),
                device=self.device,
                transcription_model=self.config.get('transcription_model'),
                whisper_model=self.config.get('whisper_model', 'large-v3'),
                gemini_transcription_model=self.config.get('gemini_transcription_model', 'gemini-3-flash-preview'),
                deepgram_model=self.config.get('deepgram_model', 'nova-3'),
                cache_manager=self.cache_manager,
                artifacts_root=self.config.get("artifacts_dir"),
            )
            logger.debug(f"Initialized {self.transcriber.name} transcriber")
        except Exception as e:
            self.transcriber_init_error = e
            logger.warning(f"Failed to initialize transcriber: {e}")

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
        """Return the initialized transcriber or raise an actionable error."""
        if self.transcriber is not None:
            return self.transcriber

        raise RuntimeError(
            self._format_component_init_error(
                "Transcriber",
                self.config.get('transcription_system', 'whisper'),
                getattr(self, "transcriber_init_error", None),
            )
        )

    def _require_translator(self):
        """Return the initialized translator or raise an actionable error."""
        if self.translator is not None:
            return self.translator

        raise RuntimeError(
            self._format_component_init_error(
                "Translator",
                self.config.get('translator_type', 'llm'),
                getattr(self, "translator_init_error", None),
            )
        )

    def _attach_segment_reference(
        self,
        *,
        tts_segment_data_args: Dict[str, Any],
        segment_dict: Dict[str, Any],
        speaker: str,
        segment_index: int,
        original_audio_segment: Optional[AudioSegment],
        segment_reference_min_duration: float,
        segment_reference_min_duration_ms: int,
    ) -> tuple[Dict[str, Any], Optional[AudioSegment]]:
        """Attach a segment-specific reference clip and its transcription when possible."""
        segment_duration = segment_dict["end"] - segment_dict["start"]
        if (
            original_audio_segment is None
            or (
                segment_reference_min_duration > 0.0
                and segment_duration < segment_reference_min_duration
            )
        ):
            return tts_segment_data_args, original_audio_segment

        start_ms = max(int(segment_dict["start"] * 1000), 0)
        end_ms = min(int(segment_dict["end"] * 1000), len(original_audio_segment))

        if end_ms <= start_ms:
            return tts_segment_data_args, original_audio_segment

        segment_audio = original_audio_segment[start_ms:end_ms]
        if segment_reference_min_duration_ms != 0 and len(segment_audio) < segment_reference_min_duration_ms:
            return tts_segment_data_args, original_audio_segment

        segment_ref_dir = self.speakers_audio_dir / "segments"
        segment_ref_dir.mkdir(parents=True, exist_ok=True)
        segment_ref_path = segment_ref_dir / f"{speaker}_{segment_index}.wav"
        segment_audio.export(segment_ref_path, format="wav")
        tts_segment_data_args["reference_audio_path"] = str(segment_ref_path)
        tts_segment_data_args["reference_text"] = segment_dict.get("text")

        return tts_segment_data_args, original_audio_segment

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

    def _build_dubbing_text_snapshot_key(self, audio_file: str) -> str:
        """Compute the stable, plan-independent key used by the text editor."""
        effective_prompt_prefix = self._build_translation_prompt_prefix(
            self.config.get("translation_prompt_prefix")
        )
        prompt_prefix_hash = hashlib.md5(
            effective_prompt_prefix.encode("utf-8")
        ).hexdigest()[:12]
        return (
            f"{self.cache_manager.generate_cache_key(audio_file, self.config.get('source_language'), self.config.get('target_language'), self.config.get('whisper_model', 'large-v3'), self.config.get('start_time'), self.config.get('duration'))}"
            f"_{self.config.get('target_language')}_{prompt_prefix_hash}"
        )

    def _build_translation_cache_key(self, audio_file: str) -> str:
        """Compute the plan-aware translation cache key used by the pipeline."""
        base_key = self._build_dubbing_text_snapshot_key(audio_file)
        semantic_fingerprint = getattr(self, "_semantic_plan_fingerprint", None)
        semantic_suffix = f"_semantic_{semantic_fingerprint}" if semantic_fingerprint else ""
        return f"{base_key}{semantic_suffix}"

    def _build_emotions_cache_key(
        self,
        audio_file: str,
        provider: Optional[str] = None,
        model: Optional[str] = None,
    ) -> str:
        """Compute the plan-aware emotion-analysis cache identity."""
        provider = str(provider or self.config.get("emotion_provider") or "gemini").lower()
        model = str(model or self.config.get("emotion_model") or "gemini-3.1-flash-lite")
        cache_extra = f"{provider}_{model}" if provider == "gemini" else provider
        base_key = self.cache_manager.generate_cache_key(
            audio_file, "", "", cache_extra
        )
        semantic_fingerprint = getattr(self, "_semantic_plan_fingerprint", None)
        semantic_suffix = (
            f"_semantic_{semantic_fingerprint}" if semantic_fingerprint else ""
        )
        return f"{base_key}_{cache_extra}{semantic_suffix}"

    def _restore_semantic_plan_fingerprint(self, audio_file: str) -> None:
        """Restore the semantic identity needed by resume cache keys."""
        isolated_tracks = self.config.get("isolated_tracks")
        if not isolated_tracks or not self.config.get("semantic_split_enabled", True):
            return
        step_name = "isolated_tracks_semantic_plan"
        cache_key = self._isolated_tracks_cache_key(audio_file, isolated_tracks)
        if not self.cache_manager.cache_exists(step_name, cache_key):
            raise FileNotFoundError(
                "Semantic-plan cache is missing or stale; run transcribe_only or the "
                "full pipeline before resuming plan-dependent translation/TTS."
            )
        cached = self.cache_manager.load_from_cache(step_name, cache_key)
        transcription = cached.get("transcription") if isinstance(cached, dict) else None
        fingerprints = {
            segment.get("semantic_plan_fingerprint")
            for segment in (transcription or [])
            if segment.get("semantic_plan_fingerprint")
        }
        if (
            len(fingerprints) != 1
            or any(
                segment.get("semantic_plan_fingerprint") not in fingerprints
                for segment in (transcription or [])
            )
        ):
            raise ValueError("Cached semantic plan has a missing or inconsistent fingerprint")
        self._semantic_plan_fingerprint = next(iter(fingerprints))
        self._semantic_plan_cache_persistable = True

    def _validate_plan_dependent_segments(self, segments: List[Dict[str, Any]]) -> None:
        expected = getattr(self, "_semantic_plan_fingerprint", None)
        if expected is None:
            return
        if any(
            segment.get("semantic_plan_fingerprint") != expected
            for segment in segments
        ):
            raise ValueError(
                "Cached artifact semantic_plan_fingerprint is absent or does not "
                "match the active semantic plan"
            )

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

        remove_pauses_enabled = self.config.get('remove_pauses', True)
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
            remove_pauses=self.config.get('remove_pauses', True),
            min_pause_duration=self.config.get('min_pause_duration', 3),
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
        """Persist the latest real segment state for the Dubbing Texts editor."""
        translation_cache_reusable = getattr(
            self, "_semantic_plan_cache_persistable", True
        )
        try:
            snapshot_key = self._build_dubbing_text_snapshot_key(audio_file)
            snapshot_payload = {
                "version": 1,
                "segments": segments,
                "translation_cache_reusable": translation_cache_reusable,
                "translation_cache_key": (
                    self._build_translation_cache_key(audio_file)
                    if translation_cache_reusable
                    else None
                ),
            }
            self.cache_manager.save_to_cache(
                "dubbing_texts", snapshot_key, snapshot_payload
            )
        except Exception as e:
            logger.warning(f"Could not persist Dubbing Texts snapshot: {e}")

    def _persist_synthesis_results(
        self, segments: List[Dict], audio_file: str
    ) -> None:
        """Persist post-synthesis editor state and reusable pipeline state."""
        self._persist_dubbing_text_snapshot(segments, audio_file)

        translation_cache_reusable = getattr(
            self, "_semantic_plan_cache_persistable", True
        )
        if not translation_cache_reusable:
            return
        try:
            cache_key = self._build_translation_cache_key(audio_file)
            self.cache_manager.save_to_cache("translation", cache_key, segments)
        except Exception as e:
            logger.warning(f"Could not persist synthesis results to translation cache: {e}")

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
                        audio_file, emotion_provider, emotion_model
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
        """Return a cached diarize+transcribe result or raise a clear error.

        Used by ``run_translate_only`` to guarantee we never re-run the
        transcriber during a resume step. Reproduces the exact cache_key that
        ``SmartDubbing.diarize_and_transcribe`` writes — a full-pipeline or
        ``transcribe_only`` run passes that key to the transcriber, so any
        cached result lives under it. If the transcriber does not advertise
        ``cache_step_name`` (legacy pyannote_openai with its multi-step cache
        layout), fall back to a normal ``diarize_and_transcribe`` call so its
        own internal cache is still consulted.
        """
        # Isolated-tracks path publishes its own step_name outside of any
        # transcriber. Reproduce the same cache_key the write path used;
        # miss → fall back to a fresh isolated run (its own cache is a no-op
        # then, so we just recompute).
        isolated_tracks = self.config.get('isolated_tracks')
        if isolated_tracks:
            step_name = (
                "isolated_tracks_semantic_plan"
                if self.config.get("semantic_split_enabled", True)
                else "isolated_tracks_transcription"
            )
            cache_key = self._isolated_tracks_cache_key(audio_file, isolated_tracks)
            if not self.cache_manager.cache_exists(step_name, cache_key):
                raise FileNotFoundError(
                    f"run_step=translate_only requires cached isolated-tracks "
                    f"diarization+transcription from a previous transcribe_only or "
                    f"full pipeline run in the same project directory, but no cache "
                    f"entry was found for step '{step_name}'. Run "
                    f"--run_step transcribe_only (or the full pipeline) first."
                )
            return self._diarize_and_transcribe_isolated(audio_file, isolated_tracks)

        transcriber = self._require_transcriber()
        step_name = (getattr(transcriber, "cache_step_name", "") or "").strip()

        if not step_name:
            logger.warning(
                "Transcriber %s does not advertise a cache_step_name; falling back "
                "to a normal diarize_and_transcribe call (its own cache is still consulted).",
                transcriber.name,
            )
            speakers_rolls, transcription = transcriber.diarize_and_transcribe(
                audio_file=audio_file,
                cache_key=None,
                use_cache=True,
            )
            self.debug_data["diarization"] = speakers_rolls
            self.debug_data["transcription"] = transcription
            self._save_transcription_file(transcription)
            return speakers_rolls, transcription

        # Use the SAME cache_key that SmartDubbing.diarize_and_transcribe would
        # compute on a full-pipeline / transcribe_only run — that's what any
        # existing cache entry was written under.
        cache_key = self.cache_manager.generate_cache_key(
            audio_file,
            self.config.get('source_language'),
            self.config.get('target_language'),
            self.config.get('whisper_model', 'large-v3'),
            self.config.get('start_time'),
            self.config.get('duration'),
        )

        if not self.cache_manager.cache_exists(step_name, cache_key):
            raise FileNotFoundError(
                f"run_step=translate_only requires cached diarization+transcription "
                f"from a previous transcribe_only or full pipeline run in the same "
                f"project directory, but no cache entry was found for step "
                f"'{step_name}'. Run --run_step transcribe_only (or the full pipeline) "
                f"first."
            )

        speakers_rolls, transcription = transcriber.diarize_and_transcribe(
            audio_file=audio_file,
            cache_key=cache_key,
            use_cache=True,
        )
        self.debug_data["diarization"] = speakers_rolls
        self.debug_data["transcription"] = transcription
        self._save_transcription_file(transcription)
        return speakers_rolls, transcription

    def diarize_and_transcribe(self, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
        """Perform speaker diarization and transcription."""
        # Opt-in isolated-tracks path: only active when the user supplies
        # per-speaker isolated audio files. Skips the standard transcriber
        # entirely; the standard path below is left untouched.
        isolated_tracks = self.config.get('isolated_tracks')
        if isolated_tracks:
            return self._diarize_and_transcribe_isolated(audio_file, isolated_tracks)

        transcriber = self._require_transcriber()

        # Generate cache key
        cache_key = self.cache_manager.generate_cache_key(
            audio_file,
            self.config.get('source_language'),
            self.config.get('target_language'),
            self.config.get('whisper_model', 'large-v3'),
            self.config.get('start_time'),
            self.config.get('duration')
        )

        # Perform diarization and transcription
        speakers_rolls, transcription = transcriber.diarize_and_transcribe(
            audio_file=audio_file,
            cache_key=cache_key,
            use_cache=self.cache_manager.use_cache
        )

        # Store for debug
        self.debug_data["diarization"] = speakers_rolls
        self.debug_data["transcription"] = transcription

        # Save transcription to file
        self._save_transcription_file(transcription)

        return speakers_rolls, transcription

    def _isolated_tracks_cache_key(
        self,
        audio_file: str,
        isolated_tracks: Dict[str, str],
    ) -> str:
        """Cache key for the isolated-tracks path.

        Includes the main audio_file hash (to stay consistent with the
        CacheManager's input-hash directory layout) plus a fingerprint of
        every isolated track file, the inner transcription system, and the
        language pair. That way, swapping a track or the inner backend
        invalidates the cache automatically.
        """
        inner_system = self.config.get('inner_transcription_system', 'deepgram')
        base_key = self.cache_manager.generate_cache_key(
            audio_file,
            self.config.get('source_language'),
            self.config.get('target_language'),
            self.config.get('whisper_model', 'large-v3'),
            self.config.get('start_time'),
            self.config.get('duration'),
        )

        fingerprint = hashlib.md5()
        for speaker in sorted(isolated_tracks.keys()):
            path = isolated_tracks[speaker]
            fingerprint.update(speaker.encode('utf-8'))
            fingerprint.update(b'=')
            try:
                stat = os.stat(path)
                fingerprint.update(str(stat.st_size).encode('utf-8'))
                fingerprint.update(str(int(stat.st_mtime)).encode('utf-8'))
            except OSError:
                fingerprint.update(str(path).encode('utf-8'))
            fingerprint.update(b';')
        fingerprint.update(inner_system.encode('utf-8'))
        if self.config.get("semantic_split_enabled", True):
            from ..audio.semantic_planner import SEMANTIC_PLANNER_VERSION

            fingerprint.update(
                self._isolated_tracks_raw_cache_key(isolated_tracks).encode("utf-8")
            )
            semantic_payload = {
                "algorithm": SEMANTIC_PLANNER_VERSION,
                "incomplete_tail_rules": "incomplete_tail_en_v1",
                "prompt_parser": "semantic_boundary_prompt_v1",
                "semantic_split_enabled": True,
                "tts_preferred_segment_duration": self.config.get("tts_preferred_segment_duration", 15.0),
                "tts_hard_segment_duration": self.config.get("tts_hard_segment_duration", 35.0),
                "semantic_split_search_window": self.config.get("semantic_split_search_window", 10.0),
                "source_language": self.config.get("source_language"),
                "classifier": self._semantic_classifier_identity(),
                "classifier_timeout": 30.0,
                "classifier_batch_size": 50,
                "classifier_batch_characters": 12000,
            }
            fingerprint.update(
                json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8")
            )

        return f"{base_key}_isolated_{fingerprint.hexdigest()[:16]}"

    def _isolated_tracks_raw_cache_key(
        self,
        isolated_tracks: Dict[str, str],
    ) -> str:
        """Fingerprint provider-level VAD/ASR output independently of planning."""
        payload: Dict[str, Any] = {
            "algorithm": "isolated_raw_v1",
            "source_language": self.config.get("source_language"),
            "inner_transcription_system": self.config.get(
                "inner_transcription_system", "deepgram"
            ),
            "start_time": self.config.get("start_time"),
            "duration": self.config.get("duration"),
            "deepgram_model": self.config.get("deepgram_model"),
            "assemblyai_model": self.config.get("transcription_model"),
            "gemini_model": self.config.get("gemini_transcription_model"),
            "tracks": [],
        }
        for speaker in sorted(isolated_tracks):
            path = isolated_tracks[speaker]
            try:
                track_digest = hashlib.sha256()
                with open(path, "rb") as track_file:
                    for chunk in iter(lambda: track_file.read(1024 * 1024), b""):
                        track_digest.update(chunk)
                identity = [speaker, track_digest.hexdigest()]
            except OSError:
                identity = [speaker, str(path), None, None]
            payload["tracks"].append(identity)
        return f"isolated_raw_v1_{hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(',', ':')).encode('utf-8')).hexdigest()[:20]}"

    def _semantic_classifier(self) -> Tuple[Optional[Any], str]:
        if self.config.get("translator_type", "llm") != "llm":
            return None, "deterministic-only"
        translator = getattr(self, "translator", None)
        classifier = getattr(translator, "classify_semantic_boundaries", None)
        if callable(classifier):
            return classifier, "ready"
        if getattr(self, "translator_init_error", None) is not None:
            return None, "initialization_failed"
        return None, "unavailable"

    def _semantic_classifier_identity(self) -> Dict[str, Any]:
        """Return the effective classifier identity used by semantic-plan caches."""
        _classifier, status = self._semantic_classifier()
        if status == "deterministic-only":
            return {"mode": status}
        translator = getattr(self, "translator", None)
        return {
            "mode": status,
            "provider": getattr(
                translator, "llm_provider", self.config.get("llm_provider")
            ),
            "model": getattr(
                translator, "model_name", self.config.get("llm_model_name")
            ),
            "temperature": getattr(
                translator,
                "temperature",
                self.config.get("llm_temperature", 0.5),
            ),
            "max_tokens": getattr(
                translator,
                "max_tokens",
                self.config.get("llm_max_tokens", 16384),
            ),
        }

    @staticmethod
    def _write_semantic_boundary_diagnostics(
        path: str, records: List[Dict[str, Any]]
    ) -> None:
        debug_path = Path(path)
        debug_path.parent.mkdir(parents=True, exist_ok=True)
        with debug_path.open("w", encoding="utf-8", newline="\n") as handle:
            for record in records:
                handle.write(
                    json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
                )

    def _diarize_and_transcribe_isolated(
        self,
        audio_file: str,
        isolated_tracks: Dict[str, str],
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
        """Isolated-tracks branch of ``diarize_and_transcribe``.

        Delegates to ``dubbing.audio.isolated_tracks.run_isolated_tracks``
        and caches the result under ``isolated_tracks_transcription`` so
        ``translate_only`` / ``tts_to_end`` can resume without re-running
        VAD or the inner transcriber.
        """
        from dubbing.audio.isolated_tracks import (
            collect_isolated_tracks_raw,
            run_isolated_tracks,
        )

        semantic_enabled = self.config.get("semantic_split_enabled", True)
        semantic_debug_path = (
            str(Path(self.config.get("debug_dir")) / "semantic_boundaries.jsonl")
            if self.config.get("debug_info", False) and self.config.get("debug_dir")
            else None
        )
        step_name = (
            "isolated_tracks_semantic_plan"
            if semantic_enabled
            else "isolated_tracks_transcription"
        )
        cache_key = self._isolated_tracks_cache_key(audio_file, isolated_tracks)

        if (
            self.cache_manager.use_cache
            and self.cache_manager.cache_exists(step_name, cache_key)
        ):
            logger.debug("Loading isolated-tracks diarization+transcription from cache...")
            cached = self.cache_manager.load_from_cache(step_name, cache_key)
            if not isinstance(cached, dict):
                cached = None
            if cached is not None:
                if (
                    semantic_enabled
                    and semantic_debug_path
                    and not isinstance(cached.get("semantic_diagnostics"), list)
                ):
                    logger.info(
                        "Cached semantic plan predates boundary diagnostics; replanning from raw transcription."
                    )
                    cached = None
                if cached is None:
                    logger.warning("Semantic-plan cache cannot satisfy debug diagnostics.")
                else:
                    if semantic_enabled and semantic_debug_path:
                        self._write_semantic_boundary_diagnostics(
                            semantic_debug_path,
                            cached["semantic_diagnostics"],
                        )
            if cached is not None:
                speakers_rolls = cached["diarization"]
                transcription = cached["transcription"]
                if transcription and semantic_enabled:
                    fingerprints = {
                        segment.get("semantic_plan_fingerprint")
                        for segment in transcription
                        if segment.get("semantic_plan_fingerprint")
                    }
                    if (
                        len(fingerprints) != 1
                        or any(
                            segment.get("semantic_plan_fingerprint") not in fingerprints
                            for segment in transcription
                        )
                    ):
                        logger.warning(
                            "Cached semantic plan has a missing or inconsistent fingerprint; replanning."
                        )
                        cached = None
                    else:
                        self._semantic_plan_fingerprint = next(iter(fingerprints))
                if cached is not None:
                    self._semantic_plan_cache_persistable = True
                    self.debug_data["diarization"] = speakers_rolls
                    self.debug_data["transcription"] = transcription
                    self._save_transcription_file(transcription)
                    return speakers_rolls, transcription
            logger.warning("Corrupt isolated-tracks cache entry, recomputing.")

        inner_system = self.config.get('inner_transcription_system', 'deepgram')
        logger.info(
            "Isolated-tracks path: %d tracks, inner_transcription_system=%s",
            len(isolated_tracks),
            inner_system,
        )

        raw_step_name = "isolated_tracks_raw_transcription"
        raw_cache_key = self._isolated_tracks_raw_cache_key(isolated_tracks)
        raw_tracks_data = None
        if (
            self.cache_manager.use_cache
            and self.cache_manager.cache_exists(raw_step_name, raw_cache_key)
        ):
            raw_tracks_data = self.cache_manager.load_from_cache(
                raw_step_name, raw_cache_key
            )
        if raw_tracks_data is None:
            raw_tracks_data = collect_isolated_tracks_raw(
                tracks=isolated_tracks,
                inner_system=inner_system,
                source_language=self.config.get('source_language'),
                device=self.config.get('device'),
                cache_manager=self.cache_manager,
                inner_kwargs=self._isolated_inner_kwargs(inner_system),
                start_time=self.config.get('start_time'),
                duration=self.config.get('duration'),
            )
            self.cache_manager.save_to_cache(
                raw_step_name, raw_cache_key, raw_tracks_data
            )

        semantic_classifier, classifier_status = self._semantic_classifier()
        classification_step = "semantic_boundary_classification"

        def load_classification(key: str) -> Any:
            if self.cache_manager.cache_exists(classification_step, key):
                return self.cache_manager.load_from_cache(classification_step, key)
            return None

        def save_classification(key: str, value: Any) -> None:
            self.cache_manager.save_to_cache(classification_step, key, value)

        classifier_cache_context = {
            **self._semantic_classifier_identity(),
            "timeout": 30.0,
            "batch_size": 50,
            "batch_characters": 12000,
        }
        semantic_diagnostics: List[Dict[str, Any]] = []
        speakers_rolls, transcription = run_isolated_tracks(
            tracks=isolated_tracks,
            inner_system=inner_system,
            source_language=self.config.get('source_language'),
            device=self.config.get('device'),
            cache_manager=self.cache_manager,
            inner_kwargs=self._isolated_inner_kwargs(inner_system),
            start_time=self.config.get('start_time'),
            duration=self.config.get('duration'),
            semantic_split_enabled=semantic_enabled,
            tts_preferred_segment_duration=self.config.get(
                "tts_preferred_segment_duration", 15.0
            ),
            tts_hard_segment_duration=self.config.get(
                "tts_hard_segment_duration", 35.0
            ),
            semantic_split_search_window=self.config.get(
                "semantic_split_search_window", 10.0
            ),
            semantic_classifier=semantic_classifier,
            semantic_classifier_status=classifier_status,
            semantic_debug_path=semantic_debug_path,
            raw_tracks_data=raw_tracks_data,
            classification_cache_get=load_classification,
            classification_cache_set=save_classification,
            classifier_cache_context=classifier_cache_context,
            semantic_diagnostics_out=semantic_diagnostics,
        )

        plan_persistable = all(
            segment.get("_semantic_plan_cache_persistable", True)
            for segment in transcription
        )
        if not semantic_enabled or plan_persistable:
            self.cache_manager.save_to_cache(
                step_name,
                cache_key,
                {
                    "diarization": speakers_rolls,
                    "transcription": transcription,
                    "semantic_diagnostics": semantic_diagnostics,
                },
            )
        else:
            logger.warning(
                "Semantic boundary classification used a transient fallback; "
                "semantic plan and downstream plan-dependent caches will not be persisted."
            )
        if transcription and semantic_enabled:
            self._semantic_plan_fingerprint = transcription[0].get(
                "semantic_plan_fingerprint"
            )
            self._semantic_plan_cache_persistable = plan_persistable

        self.debug_data["diarization"] = speakers_rolls
        self.debug_data["transcription"] = transcription
        self._save_transcription_file(transcription)
        return speakers_rolls, transcription

    def _isolated_inner_kwargs(self, inner_system: str) -> Dict[str, Any]:
        """Extra kwargs for the inner transcriber used by isolated-tracks."""
        if inner_system == "deepgram":
            model = self.config.get("deepgram_model")
            return {"deepgram_model": model} if model else {}
        if inner_system == "gemini":
            model = self.config.get("gemini_transcription_model")
            return {"gemini_transcription_model": model} if model else {}
        if inner_system == "assemblyai":
            model = self.config.get("transcription_model")
            return {"speech_model": model} if model else {}
        return {}
    
    def translate_segments(self, transcription: List[Dict], audio_file: str) -> List[Dict]:
        """Translate segments using the translator."""
        semantic_fingerprints = {
            segment.get("semantic_plan_fingerprint")
            for segment in transcription
            if segment.get("semantic_plan_fingerprint")
        }
        semantic_segments = [
            segment for segment in transcription if segment.get("semantic_unit_id")
        ]
        if semantic_segments and (
            len(semantic_fingerprints) != 1
            or any(
                segment.get("semantic_plan_fingerprint") not in semantic_fingerprints
                for segment in semantic_segments
            )
        ):
            raise ValueError(
                "Semantic transcription has a missing or inconsistent semantic_plan_fingerprint"
            )
        if len(semantic_fingerprints) > 1:
            raise ValueError("Transcription contains multiple semantic plan fingerprints")
        if semantic_fingerprints:
            self._semantic_plan_fingerprint = next(iter(semantic_fingerprints))
        cache_key = self._build_translation_cache_key(audio_file)
        step_name = "translation"
        
        translated_segments = None
        # Check if results are cached
        if self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading translations from cache...")
            translated_segments = self.cache_manager.load_from_cache(step_name, cache_key)
            if translated_segments is not None:
                self._validate_plan_dependent_segments(translated_segments)
                self.performance_tracker.record_metric("translation", 0.0)
            else:
                logger.warning("Found corrupted translation cache, re-translating.")

        if translated_segments is None:
            # Start timing
            self.performance_tracker.start_timing("translation")
            translator = self._require_translator()
            
            if not translator.is_available():
                raise ValueError("No translator available")

            original_prompt_prefix = getattr(translator, "prompt_prefix", None)
            if hasattr(translator, "prompt_prefix"):
                translator.prompt_prefix = self._build_translation_prompt_prefix(
                    original_prompt_prefix
                )

            try:
                translated_segments = translator.translate(
                    segments=transcription,
                    source_language=self.config.get('source_language'),
                    target_language=self.config.get('target_language'),
                    refinement_persona=self.config.get('refinement_persona', 'normal'),
                    debug=self.debug_data,
                    debug_dir=self.config.get("translation_debug_dir"),
                    refinement_debug_dir=self.config.get("translation_refinement_debug_dir"),
                    timecodes_report_path=self.config.get("timecodes_report_path"),
                )
            finally:
                if hasattr(translator, "prompt_prefix"):
                    translator.prompt_prefix = original_prompt_prefix
            
            # Save results to cache
            if getattr(self, "_semantic_plan_cache_persistable", True):
                self.cache_manager.save_to_cache(step_name, cache_key, translated_segments)
            
            # End timing
            elapsed_time = self.performance_tracker.end_timing("translation")
            logger.info(f"Finished translation in {elapsed_time:.2f} seconds (≈ {elapsed_time/60:.2f} minutes)")
        
        # Store for debug
        self.debug_data["translation"] = translated_segments
        self._persist_dubbing_text_snapshot(translated_segments, audio_file)
        
        return translated_segments

    def _build_translation_prompt_prefix(self, base_prompt_prefix: Optional[str]) -> str:
        """Combine any user-provided translation prompt prefix with SmartDubbing TTS stress rules."""
        base_prompt = (base_prompt_prefix or "").strip()
        if "U+0301" in base_prompt or "каса́" in base_prompt:
            return base_prompt
        if base_prompt:
            return f"{base_prompt}\n\n{SMART_DUBBING_STRESS_MARKS_REQUIREMENT}"
        return SMART_DUBBING_STRESS_MARKS_REQUIREMENT
    
    def analyze_emotions(self, segments: List[Dict], audio_file: str) -> List[Dict]:
        """Analyze emotions in the audio for each segment."""
        if not segments:
            return []

        provider = str(self.config.get("emotion_provider") or "gemini").lower()
        model = str(self.config.get("emotion_model") or "gemini-3.1-flash-lite")

        cache_key = self._build_emotions_cache_key(audio_file, provider, model)
        step_name = "emotions"

        if self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading emotion analysis from cache...")
            cached_segments = self.cache_manager.load_from_cache(step_name, cache_key)
            if cached_segments is not None:
                try:
                    self._validate_plan_dependent_segments(cached_segments)
                except ValueError:
                    logger.warning(
                        "Emotion cache does not match the active semantic plan; "
                        "re-analyzing."
                    )
                else:
                    return cached_segments
            logger.warning("Found corrupted emotion cache, re-analyzing.")

        logger.info("Analyzing speech emotions (provider=%s, model=%s)...", provider, model)
        self.performance_tracker.start_timing("emotion_analysis")

        try:
            if provider == "gemini":
                self._analyze_emotions_gemini(segments, audio_file, model)
            elif provider == "speechbrain":
                self._analyze_emotions_speechbrain(segments, audio_file)
            else:
                logger.warning("Unknown emotion_provider '%s'; defaulting all segments to Neutral.", provider)
                for segment in segments:
                    segment["emotion"] = "Neutral"
        finally:
            self.performance_tracker.end_timing("emotion_analysis")

        self.cache_manager.save_to_cache(step_name, cache_key, segments)
        return segments

    def _analyze_emotions_gemini(self, segments: List[Dict], audio_file: str, model: str) -> None:
        """Classify each segment's emotion via a Gemini multimodal model on Vertex AI."""
        try:
            from google import genai
            from google.genai import types as genai_types
        except ImportError:
            logger.warning(
                "google-genai package unavailable; falling back to Neutral for all segments. "
                "Install google-genai or switch emotion_provider to 'speechbrain'."
            )
            for segment in segments:
                segment["emotion"] = "Neutral"
            return

        try:
            from google_vertex import get_vertex_ai_settings
            vertex_settings = get_vertex_ai_settings()
            client = genai.Client(**vertex_settings.genai_client_kwargs)
        except Exception as exc:
            logger.warning("Vertex AI unavailable for emotion analysis (%s); defaulting to Neutral.", exc)
            for segment in segments:
                segment["emotion"] = "Neutral"
            return

        import io
        import mimetypes
        from pydub import AudioSegment

        prompt = EMOTION_ANALYSIS_PROMPT
        config_obj = genai_types.GenerateContentConfig(temperature=0.2)
        allowed = {"Neutral", "Angry", "Happy", "Sad"}

        audio = AudioSegment.from_file(audio_file)
        for segment in segments:
            try:
                start = max(int(segment["start"] * 1000), 0)
                end = min(int(segment["end"] * 1000), len(audio))
                if end <= start:
                    segment["emotion"] = "Neutral"
                    segment.setdefault("style_prompt", "")
                    continue

                segment_audio = audio[start:end]
                buffer = io.BytesIO()
                segment_audio.export(buffer, format="wav")
                audio_part = genai_types.Part.from_bytes(
                    data=buffer.getvalue(),
                    mime_type="audio/wav",
                )

                response = client.models.generate_content(
                    model=model,
                    contents=[audio_part, prompt],
                    config=config_obj,
                )
                raw = (getattr(response, "text", None) or "").strip()
                style_text = ""
                emotion_label = "Neutral"
                for line in raw.splitlines():
                    stripped = line.strip()
                    if not stripped:
                        continue
                    lower = stripped.lower()
                    if lower.startswith("style:"):
                        style_text = stripped.split(":", 1)[1].strip().strip('"\'')
                    elif lower.startswith("emotion:"):
                        tag = stripped.split(":", 1)[1].strip().split()[0]
                        tag = tag.strip(".,!?\"'").capitalize()
                        if tag in allowed:
                            emotion_label = tag
                if not style_text and raw:
                    # Fallback: model ignored the format — take the first non-empty line.
                    first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
                    style_text = first_line.strip('"\'')
                segment["emotion"] = emotion_label
                segment["style_prompt"] = style_text
            except Exception as exc:
                logger.warning("Gemini emotion classification failed for segment: %s", exc)
                segment["emotion"] = "Neutral"
                segment.setdefault("style_prompt", "")

    def _analyze_emotions_speechbrain(self, segments: List[Dict], audio_file: str) -> None:
        """Legacy speechbrain-based classifier (IEMOCAP wav2vec2)."""
        try:
            from speechbrain.inference.interfaces import foreign_class
        except ModuleNotFoundError as exc:
            if exc.name != "speechbrain":
                raise
            logger.warning(
                "Skipping emotion analysis because optional dependency "
                "'speechbrain' is unavailable. Install it or switch emotion_provider to 'gemini'."
            )
            for segment in segments:
                segment["emotion"] = "Neutral"
            return

        classifier = foreign_class(
            source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            pymodule_file="custom_interface.py",
            classname="CustomEncoderWav2vec2Classifier",
            run_opts={"device": self.torch_device},
        )
        emotion_dict = {
            'neu': 'Neutral',
            'ang': 'Angry',
            'hap': 'Happy',
            'sad': 'Sad',
            'None': None,
        }

        from pydub import AudioSegment
        audio = AudioSegment.from_file(audio_file)
        for segment in segments:
            try:
                start = int(segment["start"] * 1000)
                end = int(segment["end"] * 1000)
                segment_audio = audio[start:end]
                temp_segment_path = self.config.get("temp_segment_audio_path")
                segment_audio.export(temp_segment_path, format="wav")
                out_prob, score, index, text_lab = classifier.classify_file(temp_segment_path)
                emotion = emotion_dict[text_lab[0]] or "Neutral"
                segment["emotion"] = emotion
                segment["style_prompt"] = SOFT_STYLE_BY_EMOTION.get(emotion, "")
                os.remove(temp_segment_path)
            except Exception as e:
                logger.warning(f"Error analyzing emotion: {e}")
                segment["emotion"] = "Neutral"
                segment.setdefault("style_prompt", "")
    
    def synthesize_speech(self, segments: List[Dict], speakers_rolls: Dict, audio_file: str) -> str:
        """
        Synthesize speech for translated segments with optimized batching and estimation.
        
        Args:
            segments: List of transcript segments with translations
            speakers_rolls: Dictionary mapping time ranges to speaker IDs
            audio_file: Path to the audio file for cache key
            
        Returns:
            Path to the output audio file
        """
        if not segments:
            raise ValueError("Cannot synthesize speech with no segments.")

        from .timing import TimingPolicy, plan_anchor_windows, timing_cache_fingerprint

        try:
            self._timing_source_duration = len(AudioSegment.from_file(audio_file)) / 1000.0
        except Exception as exc:
            raise ValueError(f"Cannot measure processed source audio for timing: {audio_file}") from exc
        self._timing_source_audio_file = audio_file
        anchor_plan = plan_anchor_windows(segments, self._timing_source_duration)
        for item in anchor_plan:
            item.segment["_timing_original_index"] = item.original_index
            item.segment["_timing_next_anchor"] = item.next_start
            item.segment["_timing_available_window"] = item.available_window
        segments[:] = [item.segment for item in anchor_plan]
        timing_policy = TimingPolicy.from_config(self.config)
        self._plan_dependent_cache_allowed = getattr(
            self, "_semantic_plan_cache_persistable", True
        )

        # Start timing
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
        cache_key = f"{self.cache_manager.generate_cache_key(audio_file, self.config.get('source_language'), self.config.get('target_language'), self.config.get('whisper_model', 'large-v3'), self.config.get('start_time'), self.config.get('duration'))}_{self.config.get('target_language')}_{self.config.get('tts_system')}_{timing_cache_fingerprint(timing_policy)}{semantic_suffix}"
        step_name = "synthesized_speech"
        
        # Final audio is stored as WAV rather than CacheManager's pickle
        # payload, so its existence must be checked directly.
        cached_audio_path = self.cache_manager.get_cache_path(step_name) / f"{cache_key}.wav"
        if (
            self.cache_manager.use_cache
            and self._plan_dependent_cache_allowed
            and not self.config.get("debug_info", False)
            and cached_audio_path.exists()
        ):
            logger.debug("Loading synthesized speech from cache...")
            output_path = self.config.get("translated_audio_path")
            shutil.copy(cached_audio_path, output_path)
            self.performance_tracker.end_timing("speech_synthesis")
            return output_path
        
        logger.info(f"Synthesizing translated speech using multiple TTS systems...")
        
        # Create segment cache directory if needed
        segment_cache_path = self.cache_manager.get_cache_path("segment_synthesis")
        base_cache_prefix = self.cache_manager.generate_cache_key(audio_file, self.config.get('source_language'), self.config.get('target_language'), self.config.get('whisper_model', 'large-v3'), self.config.get('start_time'), self.config.get('duration'))
        
        # Check if any TTS clients are initialized
        if not self.tts_clients:
            raise ValueError("No TTS systems are initialized properly")
        
        # Import TTSSegmentData for synthesis
        from tts.models import TTSSegmentData
        
        # Define comfort ratio constants
        COMFORT_MIN_ADJUSTMENT_RATIO = 0.75
        # Anchor timing never stretches short speech to fill its window. Silence
        # after a naturally short clip is valid slack, not a mismatch.
        COMFORT_MAX_ADJUSTMENT_RATIO = float("inf")

        # Lazily-loaded original audio for creating segment-specific reference clips
        original_audio_segment = None

        # Minimum duration threshold for exporting segment reference audio
        segment_reference_min_duration = self.config.get('segment_reference_min_duration', 2.0)
        try:
            segment_reference_min_duration = float(segment_reference_min_duration)
        except (TypeError, ValueError):
            logger.warning(
                "Invalid segment_reference_min_duration value '%s'. Falling back to 2.0 seconds.",
                segment_reference_min_duration
            )
            segment_reference_min_duration = 2.0
        if segment_reference_min_duration < 0.0:
            logger.warning("segment_reference_min_duration cannot be negative. Using 0 seconds instead.")
            segment_reference_min_duration = 0.0
        segment_reference_min_duration_ms = max(int(segment_reference_min_duration * 1000), 0)

        # Group segments by (pool_key, profile) — profiles that share a pool_key
        # share the same TTS client and are synthesised together in one batch call.
        segments_by_pool_key: Dict[tuple, List[Tuple[int, Dict[str, Any]]]] = {}
        pool_key_profile: Dict[tuple, VoiceProfile] = {}
        segment_to_pool_key: Dict[int, tuple] = {}
        pool_key_label: Dict[tuple, str] = {}

        for i, segment_dict in enumerate(segments):
            speaker = segment_dict["speaker"]
            profile = self._resolve_voice_profile(speaker)
            pool_key = self._profile_pool_key(profile)
            segments_by_pool_key.setdefault(pool_key, []).append((i, segment_dict))
            pool_key_profile.setdefault(pool_key, profile)
            segment_to_pool_key[i] = pool_key
            pool_key_label.setdefault(
                pool_key,
                f"{profile.tts_system or self._default_tts_system()}"
                + (f":{profile.model}" if profile.model else ""),
            )

        logger.debug(
            "Segments grouped by TTS pool: %s",
            [(pool_key_label[k], len(v)) for k, v in segments_by_pool_key.items()],
        )

        # First pass: Determine the best text version for each segment using estimation
        segments_to_synthesize_by_pool: Dict[tuple, List[Any]] = {k: [] for k in segments_by_pool_key.keys()}
        segments_metadata = []
        
        # Initialize progress tracking
        total_segments = len(segments)
        completed_segments = 0
        
        logger.info(f"Estimating audio durations to select optimal text versions for {total_segments} segments...")
        
        for pool_key, segment_list in segments_by_pool_key.items():
            pool_profile = pool_key_profile[pool_key]
            tts_system = pool_key_label[pool_key]
            # Get the shared TTS instance for this pool
            tts_instance = self.tts_clients.get(pool_key)
            if not tts_instance:
                logger.warning(f"Warning: TTS client for {tts_system} not available, using default")
                tts_instance = self.default_tts

            logger.debug(f"Processing {len(segment_list)} segments with {tts_system} TTS ...")

            for i, segment_dict in segment_list:
                speaker = segment_dict["speaker"]
                profile = self._resolve_voice_profile(speaker)

                segment_style_override = (segment_dict.get("style_prompt") or "").strip()
                segment_style_prompt = segment_style_override or profile.style_prompt

                voice_name = profile.voice_name
                if voice_name is None:
                    # Legacy fallback: bare string voice_name applies to every speaker.
                    voice_cfg = self.config.get('voice_name')
                    if isinstance(voice_cfg, str):
                        voice_name = voice_cfg

                # Store the voice information for debug
                if self.config.get('debug_info', False):
                    self.debug_data["voices"][i] = {
                        "speaker": speaker,
                        "voice": voice_name,
                        "style_prompt": segment_style_prompt,
                        "tts_system": profile.tts_system or self._default_tts_system(),
                        "model": profile.model,
                    }

                # Prepare base TTSSegmentData & resolve reference audio/text first
                tts_segment_data_args = {
                    "speaker": speaker,
                    "text": segment_dict["translation"],
                    "emotion": segment_dict.get("emotion", "Neutral"),
                    "style_prompt": segment_style_prompt,
                    "reference_audio_path": profile.reference_audio,
                    "reference_text": profile.reference_text,
                    "voice": voice_name,
                    "speed": 1.0,
                    "target_duration": segment_dict["_timing_available_window"],
                }
                try:
                    if original_audio_segment is None:
                        original_audio_segment = AudioSegment.from_file(audio_file)

                    tts_segment_data_args, original_audio_segment = self._apply_reference_fallbacks(
                        tts_segment_data_args=tts_segment_data_args,
                        segment_dict=segment_dict,
                        speaker=speaker,
                        segment_index=i,
                        original_audio_segment=original_audio_segment,
                        segment_reference_min_duration=segment_reference_min_duration,
                        segment_reference_min_duration_ms=segment_reference_min_duration_ms,
                    )
                except Exception as exc:
                    logger.warning(
                        f"Failed to resolve reference audio for segment {i+1} ({speaker}): {exc}"
                    )

                # Check cache (including semantic-plan and reference identities).
                segment_cache_key = self._raw_tts_segment_cache_key(
                    base_cache_prefix=base_cache_prefix,
                    tts_system=tts_system,
                    segment=segment_dict,
                    speaker=speaker,
                    translation=segment_dict["translation"],
                    style_prompt=segment_style_prompt,
                    reference_audio_path=tts_segment_data_args.get("reference_audio_path"),
                    legacy_index=i,
                )
                current_segment_output_path = str(self.audio_chunks_dir / f"{i}.wav")
                os.makedirs(os.path.dirname(current_segment_output_path), exist_ok=True)
                
                segment_cached_file_path = segment_cache_path / f"{segment_cache_key}.wav"
                
                # Try to use cached segment
                if self.cache_manager.use_cache and self._plan_dependent_cache_allowed and segment_cached_file_path.exists():
                    try:
                        cached_audio_info = AudioSegment.from_file(segment_cached_file_path)
                        if len(cached_audio_info) > 0:
                            logger.debug(f"Using valid cached segment {i+1}/{len(segments)} ({tts_system})")
                            shutil.copy(segment_cached_file_path, current_segment_output_path)
                            cache_contract = self._cached_segment_contract(segment_cached_file_path)
                            segment_dict['_tts_cache_contract'] = cache_contract
                            if cache_contract == "anchor_raw_v1":
                                measured_duration = self._measure_raw_tts_for_timing(current_segment_output_path, i)
                            else:
                                measured_duration = len(cached_audio_info) / 1000.0
                            if measured_duration > 0:
                                segment_dict['synthesized_speech_len'] = measured_duration
                                segment_dict['synthesized_speech_file'] = current_segment_output_path
                                segment_dict['synthesized_text'] = segment_dict.get('translation', '')
                                continue
                            os.remove(segment_cached_file_path)
                            try:
                                self._segment_cache_metadata_path(segment_cached_file_path).unlink()
                            except OSError:
                                pass
                        else:
                            os.remove(segment_cached_file_path)
                    except Exception as e:
                        try:
                            os.remove(segment_cached_file_path)
                        except:
                            pass
                
                # Calculate original duration and estimate current translation
                original_duration = segment_dict["_timing_available_window"]
                
                # Create TTSSegmentData for estimation
                segment_data_model = TTSSegmentData(**tts_segment_data_args)
                
                # Estimate duration for normal translation
                estimated_duration_normal = tts_instance.estimate_audio_segment_length(
                    segment_data_model,
                    language=self.config.get('target_language')
                )
                
                best_text = segment_dict["translation"]
                best_estimated_duration = estimated_duration_normal
                best_ratio = float('inf')
                best_deviation = float('inf')
                best_track_type = "translation"  # Track which translation variant was selected
                
                if estimated_duration_normal is None or estimated_duration_normal <= 0:
                    logger.warning(f"Segment {i+1}: Duration estimation failed for normal translation, using it directly.")
                else:
                    ratio_normal = original_duration / estimated_duration_normal
                    deviation_normal = self._calculate_percentage_deviation(
                        ratio_normal,
                        COMFORT_MIN_ADJUSTMENT_RATIO,
                        COMFORT_MAX_ADJUSTMENT_RATIO,
                    )
                    best_ratio = ratio_normal
                    best_deviation = deviation_normal
                    logger.debug(
                        f"Segment {i+1} ({tts_system}): Normal translation - Estimated duration: {estimated_duration_normal:.2f}s, Ratio: {ratio_normal:.2f}, Deviation: {deviation_normal:.2%}"
                    )
                    
                    # If normal is perfect, no need to check alternatives
                    if deviation_normal == 0.0:
                        logger.debug(f"  Normal translation is within comfort zone. Selecting it.")
                    else:
                        alternatives = []
                        # Decide which alternatives to consider based on whether we need to shorten or lengthen
                        if ratio_normal < COMFORT_MIN_ADJUSTMENT_RATIO:
                            # Synthesized audio longer than original – try shorter variants first
                            if "very_short_translation" in segment_dict:
                                alternatives.append(("very_short_translation", segment_dict["very_short_translation"]))
                            if "short_translation" in segment_dict:
                                alternatives.append(("short_translation", segment_dict["short_translation"]))
                        elif ratio_normal > COMFORT_MAX_ADJUSTMENT_RATIO:
                            # Synthesized audio shorter than original – try longer variant
                            if "long_translation" in segment_dict:
                                alternatives.append(("long_translation", segment_dict["long_translation"]))
                            # Fallback to original text if long not available (handled later)
                        
                        if alternatives:
                            logger.debug(f"  Normal translation is outside comfort. Estimating {len(alternatives)} alternative(s)...")
                        
                        # Preference: when remove_pauses enabled and non-zero deviation remains, prefer negative deviation (shorter) in tie
                        prefer_shorter = self.config.get('remove_pauses', True)
                        def deviation_key(dev: float) -> tuple:
                            # Primary: minimal absolute deviation; Secondary: prefer negative when enabled
                            return (abs(dev), 0 if (prefer_shorter and dev < 0) else 1)

                        for alt_key, alt_text_content in alternatives:
                            alt_segment_data_args = {**tts_segment_data_args, "text": alt_text_content}
                            alt_segment_data_model = TTSSegmentData(**alt_segment_data_args)
                            estimated_duration_alt = tts_instance.estimate_audio_segment_length(
                                alt_segment_data_model,
                                language=self.config.get('target_language')
                            )
                            
                            if estimated_duration_alt is None or estimated_duration_alt <= 0:
                                logger.warning(f"    {alt_key.replace('_', ' ').title()}: Estimation failed.")
                                continue
                            
                            ratio_alt = original_duration / estimated_duration_alt
                            deviation_alt = self._calculate_percentage_deviation(
                                ratio_alt,
                                COMFORT_MIN_ADJUSTMENT_RATIO,
                                COMFORT_MAX_ADJUSTMENT_RATIO,
                            )
                            logger.debug(f"    {alt_key.replace('_', ' ').title()} - Estimated duration: {estimated_duration_alt:.2f}s, Ratio: {ratio_alt:.2f}, Deviation: {deviation_alt:.2%}")
                            
                            # Update if this alternative is better per deviation_key
                            if deviation_key(deviation_alt) < deviation_key(best_deviation):
                                best_deviation = deviation_alt
                                best_ratio = ratio_alt
                                best_text = alt_text_content
                                best_estimated_duration = estimated_duration_alt
                                best_track_type = alt_key  # Track the selected variant
                                logger.debug(
                                    f"      New best: {alt_key.replace('_', ' ').title()} (Deviation: {best_deviation:.2%})"
                                )
                                if best_deviation == 0.0:
                                    break
                
                logger.debug(f"  Selected for synthesis: '{best_text[:50]}...' (Ratio: {best_ratio:.2f}, Deviation: {best_deviation:.2%})")
                
                # Remove any stale wav left over from a previous run so a
                # silent TTS failure (e.g. OmniVoice AcceleratorError) can't
                # be masked by an old file with the same index — otherwise
                # ``os.path.exists(output_path)`` below would happily accept
                # audio produced by a completely different TTS backend.
                try:
                    if os.path.exists(current_segment_output_path):
                        os.remove(current_segment_output_path)
                except OSError as exc:
                    logger.debug(
                        f"Could not remove stale chunk {current_segment_output_path}: {exc}"
                    )

                # Prepare segment for synthesis with chosen text
                final_segment_data = TTSSegmentData(**{**tts_segment_data_args, "text": best_text, "output_path": current_segment_output_path})
                segments_to_synthesize_by_pool[pool_key].append(final_segment_data)
                segments_metadata.append({
                    "index": i,
                    "segment_dict": segment_dict,
                    "cache_path": segment_cached_file_path,
                    "output_path": current_segment_output_path,
                    "chosen_text": best_text,
                    "estimated_ratio": best_ratio,
                    "tts_system": tts_system,
                    "pool_key": pool_key,
                    "segment_data_args": tts_segment_data_args,
                    "selected_track_type": best_track_type  # Store the selected track type
                })

        # Second pass: Batch synthesize all segments by pool_key
        for pool_key, segments_to_synthesize in segments_to_synthesize_by_pool.items():
            if not segments_to_synthesize:
                continue

            tts_system = pool_key_label[pool_key]
            tts_instance = self.tts_clients.get(pool_key)
            if not tts_instance:
                logger.warning(f"Warning: TTS client for {tts_system} not available, skipping segments")
                continue
            
            logger.info(f"Synthesizing {len(segments_to_synthesize)} segments with {tts_system} TTS ...")
            
            try:
                # Synthesize all segments for this TTS system in one call
                segment_alignments = tts_instance.synthesize(
                    segments_data=segments_to_synthesize,
                    language=self.config.get('target_language')
                )
                
                # Process results and update segment metadata
                for segment_data in segments_to_synthesize:
                    # Find corresponding metadata
                    metadata = next((m for m in segments_metadata if m["output_path"] == segment_data.output_path), None)
                    if not metadata:
                        continue
                    
                    segment_dict = metadata["segment_dict"]
                    output_path = metadata["output_path"]
                    
                    # Update progress tracking
                    completed_segments += 1
                    logger.info(f"Processing segment {completed_segments}/{total_segments} (Speaker: {segment_dict['speaker']})")
                    
                    # Check if file was created successfully
                    if os.path.exists(output_path):
                        audio_info = AudioSegment.from_file(output_path)
                        segment_dict['_tts_cache_contract'] = 'anchor_raw_v1'
                        segment_dict['synthesized_speech_len'] = self._measure_raw_tts_for_timing(
                            output_path,
                            metadata['index'],
                        )
                        segment_dict['synthesized_speech_file'] = (
                            output_path if segment_dict['synthesized_speech_len'] > 0 else None
                        )
                        segment_dict['synthesized_text'] = metadata.get('chosen_text', segment_dict.get('translation', ''))

                        # Cache the synthesized segment
                        if self.cache_manager.use_cache and self._plan_dependent_cache_allowed and segment_dict['synthesized_speech_len'] > 0:
                            try:
                                self._cache_raw_tts_segment(output_path, metadata["cache_path"])
                                logger.debug(f"Cached synthesized segment {metadata['index']+1} ({tts_system})")
                            except Exception as e:
                                logger.error(f"Error caching segment {metadata['index']+1}: {e}")

                        # Validate actual duration and resynthesize if needed
                        original_dur = segment_dict["_timing_available_window"]
                        actual_dur = segment_dict['synthesized_speech_len']
                        ratio = original_dur / actual_dur if actual_dur > 0 else 1.0
                        deviation = abs(original_dur - actual_dur) / original_dur if original_dur > 0 else 0.0
                        if actual_dur <= 0 or not (COMFORT_MIN_ADJUSTMENT_RATIO <= ratio <= COMFORT_MAX_ADJUSTMENT_RATIO):
                            logger.debug(f"Segment {metadata['index']+1}: duration mismatch after synthesis (ratio={ratio:.2f}, dev={deviation:.2%}). Trying alternatives...")
                            self._resynthesize_segment(
                                metadata,
                                tts_instance,
                                COMFORT_MIN_ADJUSTMENT_RATIO,
                                COMFORT_MAX_ADJUSTMENT_RATIO,
                                current_ratio=ratio,
                            )
                    else:
                        logger.warning(
                            f"Segment {metadata['index']+1} ({tts_system}) skipped by TTS. "
                            f"Will retry individually."
                        )
                        segment_dict['synthesized_speech_len'] = 0
                        segment_dict['synthesized_speech_file'] = None
                        # Remove any stale empty file left over from a previous run so
                        # the downstream combiner treats this slot as truly missing
                        # instead of loading a 0ms clip.
                        try:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                        except OSError:
                            pass

                # Retry any segments that the batch pass produced no audio for.
                self._retry_missing_segments(
                    segments_to_synthesize,
                    segments_metadata,
                    tts_instance,
                    tts_system,
                    COMFORT_MIN_ADJUSTMENT_RATIO,
                    COMFORT_MAX_ADJUSTMENT_RATIO,
                    pool_key=pool_key,
                )

                logger.debug(f"Batch synthesis completed for {len(segments_to_synthesize)} segments with {tts_system}")
                
            except Exception as e:
                logger.error(f"Batch synthesis failed for {tts_system}: {e}. Falling back to individual synthesis...")
                
                # Fallback: synthesize individually
                for segment_data in segments_to_synthesize:
                    metadata = next((m for m in segments_metadata if m["output_path"] == segment_data.output_path), None)
                    if not metadata:
                        continue
                    
                    segment_dict = metadata["segment_dict"]
                    output_path = metadata["output_path"]
                    
                    # Update progress tracking for fallback synthesis
                    completed_segments += 1
                    logger.info(f"Processing segment {completed_segments}/{total_segments} (Fallback - Speaker: {segment_dict['speaker']})")
                    
                    try:
                        tts_instance.synthesize(
                            segments_data=[segment_data],
                            language=self.config.get('target_language')
                        )
                        
                        if os.path.exists(output_path):
                            audio_info = AudioSegment.from_file(output_path)
                            segment_dict['_tts_cache_contract'] = 'anchor_raw_v1'
                            segment_dict['synthesized_speech_len'] = self._measure_raw_tts_for_timing(
                                output_path,
                                metadata['index'],
                            )
                            segment_dict['synthesized_speech_file'] = (
                                output_path if segment_dict['synthesized_speech_len'] > 0 else None
                            )
                            segment_dict['synthesized_text'] = metadata.get('chosen_text', segment_dict.get('translation', ''))

                            # Cache the synthesized segment
                            if self.cache_manager.use_cache and self._plan_dependent_cache_allowed and segment_dict['synthesized_speech_len'] > 0:
                                self._cache_raw_tts_segment(output_path, metadata["cache_path"])
                        
                        # After fallback individual synthesis, validate duration again
                        original_dur = segment_dict["_timing_available_window"]
                        actual_dur = segment_dict.get('synthesized_speech_len', 0)
                        ratio = original_dur / actual_dur if actual_dur > 0 else 1.0
                        deviation = abs(original_dur - actual_dur) / original_dur if original_dur > 0 else 0.0
                        if actual_dur <= 0 or not (COMFORT_MIN_ADJUSTMENT_RATIO <= ratio <= COMFORT_MAX_ADJUSTMENT_RATIO):
                            logger.info(f"Segment {metadata['index']+1}: duration mismatch after fallback synthesis (ratio={ratio:.2f}, dev={deviation:.2%}). Trying alternatives...")
                            self._resynthesize_segment(
                                metadata,
                                tts_instance,
                                COMFORT_MIN_ADJUSTMENT_RATIO,
                                COMFORT_MAX_ADJUSTMENT_RATIO,
                                current_ratio=ratio,
                            )
                        
                        logger.debug(f"Synthesized segment {metadata['index']+1}/{len(segments)} individually ({tts_system})")
                        
                    except Exception as e_synth:
                        logger.error(f"Failed to synthesize segment {metadata['index']+1} ({tts_system}): {e_synth}")
                        segment_dict['synthesized_speech_len'] = 0
                        segment_dict['synthesized_speech_file'] = None
                        try:
                            if os.path.exists(output_path):
                                os.remove(output_path)
                        except OSError:
                            pass

                # After the fallback pass, retry anything that is still missing
                # once more with the same TTS – handles transient network errors
                # from cloud TTS backends (OmniVoice, Gemini, OpenAI).
                self._retry_missing_segments(
                    segments_to_synthesize,
                    segments_metadata,
                    tts_instance,
                    tts_system,
                    COMFORT_MIN_ADJUSTMENT_RATIO,
                    COMFORT_MAX_ADJUSTMENT_RATIO,
                    pool_key=pool_key,
                )
        
        # Adjust timing and combine audio segments
        combined_audio, real_segment_positions = self._adjust_and_combine_audio_grouped(segments)
        output_path = self.config.get("translated_audio_path")
        combined_audio.export(output_path, format="wav")
        
        # Store real segment positions for later use in pause removal
        self.real_segment_positions = real_segment_positions
        
        # Log information about real vs original timing
        if real_segment_positions:
            total_real_duration = real_segment_positions[-1]["end"] - real_segment_positions[0]["start"]
            total_original_duration = max(s["original_end"] for s in real_segment_positions) - min(s["original_start"] for s in real_segment_positions)
            logger.debug(f"Real segments timing: {len(real_segment_positions)} segments, "
                        f"Real duration: {total_real_duration:.2f}s, Original duration: {total_original_duration:.2f}s")
        
        # Save to cache
        if self.cache_manager.use_cache and self._plan_dependent_cache_allowed:
            # Save the output audio
            shutil.copy(output_path, self.cache_manager.get_cache_path(step_name) / f"{cache_key}.wav")
        
        # Generate track usage report
        track_usage_stats = {}
        for metadata in segments_metadata:
            track_type = metadata.get("selected_track_type", "translation")
            track_usage_stats[track_type] = track_usage_stats.get(track_type, 0) + 1
        
        # Log the track usage report
        logger.debug("Voice sample track usage report:")
        total_samples = sum(track_usage_stats.values())
        for track_type, count in sorted(track_usage_stats.items()):
            percentage = (count / total_samples * 100) if total_samples > 0 else 0
            logger.debug(f"  {track_type}: {count} samples ({percentage:.1f}%)")
        logger.debug(f"Total voice samples processed: {total_samples}")
        
        # Log completion summary
        logger.info(f"Speech synthesis completed! Processed {total_segments} segments successfully.")
        
        # End timing
        self.performance_tracker.end_timing("speech_synthesis")
        
        return output_path
    
    def _retry_missing_segments(
        self,
        segments_to_synthesize: List[Any],
        segments_metadata: List[Dict[str, Any]],
        tts_instance,
        tts_system: str,
        min_ratio: float,
        max_ratio: float,
        max_attempts: int = 2,
        pool_key: Optional[tuple] = None,
    ) -> None:
        """Retry any segments for which the previous synthesis pass did not
        produce an audio file. Cloud TTS backends (OmniVoice, Gemini API) can
        silently drop segments on transient errors — one more attempt clears
        those up and stops empty WAVs from ending up in the final track.

        Only segments belonging to ``pool_key`` are retried. Passing ``None``
        preserves legacy behaviour (retry every metadata entry regardless of
        pool) and is only there so older callers keep working; every new call
        site sets a pool_key so segments routed to a different TTS client are
        never synthesised by the wrong backend.
        """
        from tts.models import TTSSegmentData

        # Build a mapping from output_path back to the original segment_data so
        # we can rerun exactly the same request.
        segment_data_by_path = {s.output_path: s for s in segments_to_synthesize if s.output_path}

        for attempt in range(max_attempts):
            still_missing: List[Dict[str, Any]] = []
            for metadata in segments_metadata:
                if pool_key is not None and metadata.get("pool_key") != pool_key:
                    continue
                segment_dict = metadata["segment_dict"]
                output_path = metadata["output_path"]
                if segment_dict.get("synthesized_speech_file") and os.path.exists(output_path):
                    try:
                        if len(AudioSegment.from_file(output_path)) > 0:
                            continue
                    except Exception:
                        pass
                still_missing.append(metadata)

            if not still_missing:
                return

            logger.warning(
                f"Retrying {len(still_missing)} missing segment(s) with {tts_system} (attempt {attempt + 1}/{max_attempts})"
            )

            for metadata in still_missing:
                segment_dict = metadata["segment_dict"]
                output_path = metadata["output_path"]
                segment_data = segment_data_by_path.get(output_path)
                if segment_data is None:
                    text = metadata.get("chosen_text") or segment_dict.get("translation", "")
                    segment_data = TTSSegmentData(**{**metadata.get("segment_data_args", {}), "text": text, "output_path": output_path})

                # Ensure the reference audio still resolves; fall back to the
                # per-speaker wav in speakers_audio_dir when the previous
                # attempt lost it (e.g. temp segment ref clip removed).
                if not segment_data.reference_audio_path:
                    speaker_ref = self.speakers_audio_dir / f"{segment_dict.get('speaker', '')}.wav"
                    if speaker_ref.is_file():
                        segment_data.reference_audio_path = str(speaker_ref)

                try:
                    tts_instance.synthesize(
                        segments_data=[segment_data],
                        language=self.config.get('target_language'),
                    )
                except Exception as exc:
                    logger.error(
                        f"Retry {attempt + 1}: segment {metadata['index']+1} ({tts_system}) failed again: {exc}"
                    )
                    continue

                if os.path.exists(output_path):
                    try:
                        audio_info = AudioSegment.from_file(output_path)
                    except Exception as exc:
                        logger.error(
                            f"Retry {attempt + 1}: segment {metadata['index']+1} produced unreadable audio: {exc}"
                        )
                        continue
                    if len(audio_info) <= 0:
                        try:
                            os.remove(output_path)
                        except OSError:
                            pass
                        continue

                    segment_dict['_tts_cache_contract'] = 'anchor_raw_v1'
                    segment_dict['synthesized_speech_len'] = self._measure_raw_tts_for_timing(
                        output_path,
                        metadata['index'],
                    )
                    if segment_dict['synthesized_speech_len'] <= 0:
                        segment_dict['synthesized_speech_file'] = None
                        continue
                    segment_dict['synthesized_speech_file'] = output_path
                    segment_dict['synthesized_text'] = metadata.get('chosen_text', segment_dict.get('translation', ''))
                    logger.info(
                        f"Retry {attempt + 1}: recovered segment {metadata['index']+1} ({tts_system})"
                    )
                    if self.cache_manager.use_cache and getattr(self, "_plan_dependent_cache_allowed", True) and segment_dict['synthesized_speech_len'] > 0:
                        try:
                            self._cache_raw_tts_segment(output_path, metadata["cache_path"])
                        except Exception:
                            pass

        # Log any that are still missing after all retries (scoped to this pool).
        missing_indexes = [
            metadata["index"] + 1
            for metadata in segments_metadata
            if (pool_key is None or metadata.get("pool_key") == pool_key)
            and not metadata["segment_dict"].get("synthesized_speech_file")
        ]
        if missing_indexes:
            logger.error(
                f"Segments still missing audio after {max_attempts} retries ({tts_system}): {missing_indexes}. "
                "Use the 'Regenerate selected row' button in the Dubbing Texts tab to retry individually."
            )

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
            "reference_audio_path": None,
            "reference_text": None,
            "voice": voice_name,
            "speed": 1.0,
            "target_duration": segment_dict.get(
                "_timing_available_window",
                max(segment_dict.get("end", 0) - segment_dict.get("start", 0), 0.0),
            ),
        }

        segment_reference_min_duration = float(self.config.get('segment_reference_min_duration', 2.0) or 0.0)
        segment_reference_min_duration_ms = max(int(segment_reference_min_duration * 1000), 0)

        original_audio_segment: Optional[AudioSegment] = None
        try:
            audio_source_path = None
            if os.path.exists(self.config.get("audio_artifacts_dir", "")):
                candidate = Path(self.config.get("audio_artifacts_dir")) / "source.wav"
                if candidate.is_file():
                    audio_source_path = str(candidate)
            if audio_source_path is None:
                audio_source_path = self.config.get('input')
            if audio_source_path and os.path.exists(audio_source_path):
                original_audio_segment = AudioSegment.from_file(audio_source_path)

            tts_segment_data_args, original_audio_segment = self._apply_reference_fallbacks(
                tts_segment_data_args=tts_segment_data_args,
                segment_dict=segment_dict,
                speaker=speaker,
                segment_index=segment_index,
                original_audio_segment=original_audio_segment,
                segment_reference_min_duration=segment_reference_min_duration,
                segment_reference_min_duration_ms=segment_reference_min_duration_ms,
            )
        except Exception as exc:
            logger.warning(f"Reference resolution failed for segment {segment_index}: {exc}")

        segment_data = TTSSegmentData(**{**tts_segment_data_args, "text": text_to_synthesize, "output_path": output_path})

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

                segment_dict['_tts_cache_contract'] = 'anchor_raw_v1'
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

    def _apply_configured_reference_mapping(self, tts_segment_data_args: Dict[str, Any], speaker: str) -> Dict[str, Any]:
        """Apply manual per-speaker reference settings with priority over auto-generated references.

        Prefers the unified VoiceProfile; falls back to legacy per-speaker mappings on
        the config or the SmartDubbing instance itself (older tests set those directly).
        """
        profile = None
        profiles = getattr(self, "voice_profiles", None)
        if isinstance(profiles, dict) and profiles:
            profile = resolve_profile(profiles, speaker)

        if profile and profile.reference_audio:
            tts_segment_data_args["reference_audio_path"] = profile.reference_audio
        else:
            audio_mapping = getattr(self, "reference_audio_mapping", None)
            if audio_mapping is None:
                audio_mapping = self.config.get("reference_audio_mapping") or {}
            reference_audio_path = audio_mapping.get(speaker)
            if reference_audio_path:
                tts_segment_data_args["reference_audio_path"] = reference_audio_path

        if profile and profile.reference_text:
            tts_segment_data_args["reference_text"] = profile.reference_text
        else:
            text_mapping = getattr(self, "reference_text_mapping", None)
            if text_mapping is None:
                text_mapping = self.config.get("reference_text_mapping") or {}
            reference_text = text_mapping.get(speaker)
            if reference_text:
                tts_segment_data_args["reference_text"] = reference_text

        return tts_segment_data_args

    def _apply_reference_fallbacks(
        self,
        *,
        tts_segment_data_args: Dict[str, Any],
        segment_dict: Dict[str, Any],
        speaker: str,
        segment_index: int,
        original_audio_segment: Optional[AudioSegment],
        segment_reference_min_duration: float,
        segment_reference_min_duration_ms: int,
    ) -> tuple[Dict[str, Any], Optional[AudioSegment]]:
        """Resolve reference audio/text in priority order for synthesis.

        Order:
        1. Explicit per-speaker mappings from config
        2. Segment-specific clip + matching original text
        3. Extracted per-speaker wav in speakers_audio_dir
        4. Wrapper-level global fallback (handled by the TTS wrapper)
        """
        tts_segment_data_args = self._apply_configured_reference_mapping(tts_segment_data_args, speaker)

        if not tts_segment_data_args["reference_audio_path"]:
            tts_segment_data_args, original_audio_segment = self._attach_segment_reference(
                tts_segment_data_args=tts_segment_data_args,
                segment_dict=segment_dict,
                speaker=speaker,
                segment_index=segment_index,
                original_audio_segment=original_audio_segment,
                segment_reference_min_duration=segment_reference_min_duration,
                segment_reference_min_duration_ms=segment_reference_min_duration_ms,
            )

        if not tts_segment_data_args["reference_audio_path"]:
            potential_ref_audio_for_speaker = str(self.speakers_audio_dir / f"{speaker}.wav")
            if os.path.exists(potential_ref_audio_for_speaker):
                tts_segment_data_args["reference_audio_path"] = potential_ref_audio_for_speaker

        return tts_segment_data_args, original_audio_segment
    
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
        return cache_path.with_suffix(cache_path.suffix + ".json")

    @staticmethod
    def _raw_tts_segment_cache_key(
        *,
        base_cache_prefix: str,
        tts_system: str,
        segment: Dict[str, Any],
        speaker: str,
        translation: str,
        style_prompt: str,
        reference_audio_path: Optional[str],
        legacy_index: Any = 0,
    ) -> str:
        """Build a raw-unit cache identity without timing-policy settings."""
        translation_hash = hashlib.md5(translation.encode()).hexdigest()[:8]
        voice_prompt_hash = hashlib.md5((style_prompt or "").encode()).hexdigest()[:8]
        ref_audio_hash = hashlib.md5(
            str(reference_audio_path or "").encode()
        ).hexdigest()[:8]
        semantic_unit_identity = segment.get("semantic_unit_id", legacy_index)
        semantic_plan_identity = segment.get("semantic_plan_fingerprint", "legacy")
        return (
            f"{base_cache_prefix}_{tts_system}_{semantic_plan_identity}_"
            f"{semantic_unit_identity}_{speaker}_{translation_hash}_"
            f"{voice_prompt_hash}_{ref_audio_hash}"
        )

    def _cache_raw_tts_segment(self, source_path: str, cache_path: Path) -> None:
        """Cache raw TTS plus a version marker, independent of timing policy."""
        shutil.copy(source_path, cache_path)
        metadata_path = self._segment_cache_metadata_path(cache_path)
        metadata_path.write_text(
            json.dumps({"audio_contract": "anchor_raw_v1"}, sort_keys=True),
            encoding="utf-8",
        )

    def _cached_segment_contract(self, cache_path: Path) -> str:
        metadata_path = self._segment_cache_metadata_path(cache_path)
        try:
            data = json.loads(metadata_path.read_text(encoding="utf-8"))
            if data.get("audio_contract") == "anchor_raw_v1":
                return "anchor_raw_v1"
        except (OSError, ValueError, TypeError):
            pass
        return "legacy"

    def _calculate_percentage_deviation(self, ratio: float, min_ratio_comfort: float, max_ratio_comfort: float) -> float:
        """
        Calculates the signed percentage deviation of a given ratio from the comfort zone.
        
        Args:
            ratio: The speech ratio (original_duration / synthesized_duration).
            min_ratio_comfort: The minimum acceptable ratio for comfort.
            max_ratio_comfort: The maximum acceptable ratio for comfort.
            
        Returns:
            0.0 if the ratio is within the comfort zone.
            Positive value if the synthesized segment is longer than comfortable (ratio < min).
            Negative value if the synthesized segment is shorter than comfortable (ratio > max).
        """
        if ratio >= min_ratio_comfort and ratio <= max_ratio_comfort:
            return 0.0
        elif ratio < min_ratio_comfort:
            if min_ratio_comfort == 0:
                return float('inf')  # Avoid division by zero
            # Synthesized audio is longer than original → ratio is too small → positive deviation
            return (min_ratio_comfort - ratio) / min_ratio_comfort
        else:  # ratio > max_ratio_comfort
            if max_ratio_comfort == 0:
                return float('inf')  # Avoid division by zero
            # Synthesized audio is shorter than original → ratio is too large → negative deviation
            return -((ratio - max_ratio_comfort) / max_ratio_comfort)

    def _resynthesize_segment(
        self,
        metadata: Dict[str, Any],
        tts_instance,
        min_ratio: float,
        max_ratio: float,
        current_ratio: Optional[float] = None,
    ) -> None:
        """Attempt to resynthesize a segment using alternative translations,
        focusing on minimizing deviation from the target ratio range.

        Args:
            metadata: Metadata dictionary for the segment.
            tts_instance: The TTS instance used for synthesis.
            min_ratio: Minimum acceptable ratio original/actual.
            max_ratio: Maximum acceptable ratio original/actual.
            current_ratio: Current ratio to help prioritize alternatives.
        """

        from tts.models import TTSSegmentData

        segment_dict = metadata["segment_dict"]
        original_duration = segment_dict.get(
            "_timing_available_window",
            segment_dict["end"] - segment_dict["start"],
        )
        output_path = metadata["output_path"]
        base_args = metadata["segment_data_args"]
        
        # Log resynthesis attempt
        logger.info(f"Resynthesizing segment {metadata['index']+1} (Speaker: {segment_dict['speaker']}) for better duration matching...")

        # Decide search direction based on how the current ratio deviates
        if current_ratio is None and segment_dict.get("synthesized_speech_len", 0) > 0:
            current_ratio = original_duration / max(segment_dict["synthesized_speech_len"], 1e-6)

        if current_ratio is not None:
            if current_ratio < min_ratio:
                # synthesized audio longer than original – prioritize shorter variants
                candidate_keys = ["very_short_translation", "short_translation", "translation", "long_translation"]
            elif current_ratio > max_ratio:
                # synthesized audio shorter than original – prioritize longer variants
                candidate_keys = ["long_translation", "translation", "short_translation", "very_short_translation"]
            else:
                candidate_keys = ["very_short_translation", "short_translation", "translation", "long_translation"]
        else:
            candidate_keys = ["very_short_translation", "short_translation", "translation", "long_translation"]

        # Calculate current deviation to ensure we only accept improvements
        current_deviation = float('inf')
        if current_ratio is not None:
            current_deviation = self._calculate_percentage_deviation(current_ratio, min_ratio, max_ratio)
            logger.debug(f"Current ratio: {current_ratio:.2f}, current deviation: {current_deviation:.2%}")
        
        best_alternative = None
        best_deviation_from_range = current_deviation  # Start with current deviation as baseline
        best_ratio = None
        best_key = None

        # Preference for shorter audio when pause removal is enabled
        prefer_shorter = self.config.get('remove_pauses', True)
        def deviation_key(dev: float) -> tuple:
            # Primary: minimal absolute deviation; Secondary: prefer negative when enabled
            return (abs(dev), 0 if (prefer_shorter and dev < 0) else 1)

        # Keep track of already tried texts to avoid duplicate synthesis
        tried_texts = {metadata["chosen_text"]}

        # Try all alternatives and find the one with minimum deviation from target range
        for key in candidate_keys:
            if key not in segment_dict:
                continue

            alt_text = segment_dict[key]
            # Skip already used text or duplicate text
            if alt_text in tried_texts:
                continue
            
            # Add this text to tried set
            tried_texts.add(alt_text)

            # Create temporary output path for this alternative
            temp_output_path = f"{output_path}.temp_{key}"
            new_segment_data = TTSSegmentData(**{**base_args, "text": alt_text, "output_path": temp_output_path})

            try:
                tts_instance.synthesize(
                    segments_data=[new_segment_data],
                    language=self.config.get('target_language')
                )

                if not os.path.exists(temp_output_path):
                    continue

                audio_info = AudioSegment.from_file(temp_output_path)
                actual_duration = self._measure_raw_tts_for_timing(
                    temp_output_path,
                    metadata['index'],
                )

                if actual_duration == 0:
                    os.remove(temp_output_path)
                    continue

                ratio = original_duration / actual_duration

                # Calculate deviation from target range
                deviation_from_range = self._calculate_percentage_deviation(ratio, min_ratio, max_ratio)
                
                logger.debug(f"Alternative '{key}': ratio={ratio:.2f}, deviation_from_range={deviation_from_range:.2%}")

                # Check if this is the best alternative so far (consider signed deviation preference)
                if deviation_key(deviation_from_range) < deviation_key(best_deviation_from_range):
                    # Clean up previous best alternative if exists
                    if best_alternative and os.path.exists(best_alternative):
                        os.remove(best_alternative)
                    
                    best_alternative = temp_output_path
                    best_deviation_from_range = deviation_from_range
                    best_ratio = ratio
                    best_key = key
                    
                    logger.debug(f"New best alternative: '{key}' with deviation {deviation_from_range:.2%}")
                else:
                    # Clean up this alternative since it's not the best
                    os.remove(temp_output_path)

            except Exception as e:
                logger.error(f"Alternative synthesis failed for segment {metadata['index']+1} with '{key}': {e}")
                if os.path.exists(temp_output_path):
                    os.remove(temp_output_path)

        # If deviation remains large (>15%), try LLM-based text length adjustment
        try:
            LLM_DEVIATION_THRESHOLD = 0.15
            # Compute absolute deviation key for comparison
            if self.translator and self.translator.is_available() and deviation_key(current_deviation) > deviation_key(0.0) and abs(current_deviation) > LLM_DEVIATION_THRESHOLD:
                baseline_text = metadata.get("chosen_text") or segment_dict.get("translation", "")
                if baseline_text:
                    # Aim for center of comfort zone (prefer near 1.0), compute duration factor
                    target_ratio = 1.0
                    actual_duration = segment_dict.get("synthesized_speech_len", 0) or 1e-6
                    desired_duration = original_duration / max(target_ratio, 1e-6)
                    duration_factor = max(0.2, min(2.0, desired_duration / max(actual_duration, 1e-6)))

                    # Ask LLM to adjust text length
                    adjusted_text = self.translator.adjust_segment_text_length(
                        original_text=baseline_text,
                        source_language=self.config.get('source_language'),
                        target_language=self.config.get('target_language'),
                        desired_ratio=duration_factor,
                        target_char_count=int(len(baseline_text) * duration_factor),
                        context_info=None,
                        max_attempts=2,
                    )

                    if adjusted_text and adjusted_text.strip() and adjusted_text.strip() != baseline_text.strip():
                        # Estimate duration and synthesize to temp file
                        temp_output_path = f"{output_path}.temp_llm_adjusted"
                        from tts.models import TTSSegmentData
                        new_segment_data = TTSSegmentData(**{**base_args, "text": adjusted_text, "output_path": temp_output_path})

                        try:
                            tts_instance.synthesize(
                                segments_data=[new_segment_data],
                                language=self.config.get('target_language')
                            )

                            if os.path.exists(temp_output_path):
                                audio_info = AudioSegment.from_file(temp_output_path)
                                actual_duration_llm = self._measure_raw_tts_for_timing(
                                    temp_output_path,
                                    metadata['index'],
                                )
                                if actual_duration_llm > 0:
                                    ratio_llm = original_duration / actual_duration_llm
                                    deviation_llm = self._calculate_percentage_deviation(ratio_llm, min_ratio, max_ratio)
                                    logger.info(f"LLM-adjusted alternative: ratio={ratio_llm:.2f}, deviation_from_range={deviation_llm:.2%}")

                                    if deviation_key(deviation_llm) < deviation_key(best_deviation_from_range):
                                        # Clean up previous best alternative if exists
                                        if best_alternative and os.path.exists(best_alternative):
                                            os.remove(best_alternative)

                                        best_alternative = temp_output_path
                                        best_deviation_from_range = deviation_llm
                                        best_ratio = ratio_llm
                                        best_key = "llm_adjusted"
                                        # Also update chosen text on success path later
                                        metadata["_llm_adjusted_text"] = adjusted_text
                                    else:
                                        # Not better; remove temp
                                        os.remove(temp_output_path)
                        except Exception as e:
                            logger.error(f"LLM-adjusted synthesis failed for segment {metadata['index']+1}: {e}")
                            if os.path.exists(temp_output_path):
                                try:
                                    os.remove(temp_output_path)
                                except Exception:
                                    pass
        except Exception as e:
            logger.warning(f"LLM adjustment step encountered an error: {e}")

        # Use the best alternative found only if it's actually better than current
        if best_alternative and os.path.exists(best_alternative) and deviation_key(best_deviation_from_range) < deviation_key(current_deviation):
            # Move the best alternative to the final output path
            if os.path.exists(output_path):
                os.remove(output_path)
            shutil.move(best_alternative, output_path)
            
            # Update segment data
            audio_info = AudioSegment.from_file(output_path)
            segment_dict['_tts_cache_contract'] = 'anchor_raw_v1'
            segment_dict["synthesized_speech_len"] = self._measure_raw_tts_for_timing(
                output_path,
                metadata['index'],
            )
            segment_dict["synthesized_speech_file"] = output_path
            if best_key == "llm_adjusted":
                # Persist the adjusted text
                segment_dict["translation"] = metadata.get("_llm_adjusted_text", segment_dict.get("translation"))
                metadata["chosen_text"] = segment_dict["translation"]
            else:
                metadata["chosen_text"] = segment_dict[best_key]
            metadata["selected_track_type"] = best_key  # Update the selected track type
            segment_dict["synthesized_text"] = metadata["chosen_text"]

            # Update cache if needed
            if self.cache_manager.use_cache and getattr(self, "_plan_dependent_cache_allowed", True) and len(audio_info) > 0:
                try:
                    self._cache_raw_tts_segment(output_path, metadata["cache_path"])
                except Exception:
                    pass

            logger.info(f"Resynthesis successful for segment {metadata['index']+1} using '{best_key}' "
                        f"(improved from {current_deviation:.2%} to {best_deviation_from_range:.2%} deviation)")
        else:
            # Clean up the best alternative if it exists but isn't better
            if best_alternative and os.path.exists(best_alternative):
                os.remove(best_alternative)
            
            if current_ratio is not None:
                logger.info(f"No alternative found that improves deviation for segment {metadata['index']+1} "
                           f"(current: {current_deviation:.2%}). Keeping original synthesis.")
            else:
                logger.warning(f"No suitable alternative translation could improve duration for segment {metadata['index']+1}.")


    
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
            item.segment["_timing_next_anchor"] = item.next_start
            item.segment["_timing_available_window"] = item.available_window
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
                candidate_path = self.audio_chunks_dir / f"{item.original_index}.wav"
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
                        if segment.get("_tts_cache_contract") == "anchor_raw_v1":
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
                next_start=item.next_start,
                source_duration=float(source_duration),
                audio_duration=trimmed_duration,
                policy=policy,
            )

            actual_tempo = timing.tempo
            adjusted_clip = clip
            tempo_error = None
            if timing.tempo > 1.0005 and len(clip) > 0 and not used_fallback:
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
                    "next_anchor_start": "" if item.next_start is None else item.next_start,
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
