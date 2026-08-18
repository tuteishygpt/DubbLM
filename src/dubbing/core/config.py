"""Configuration management for the Smart Dubbing system."""

import os
import sys
import json
import re
from typing import Dict, Any, Optional, List, Tuple, Union
from pathlib import Path
import yaml
from .log_config import get_logger

logger = get_logger(__name__)


DEFAULT_PROJECTS_ROOT = Path(__file__).resolve().parents[3] / "prj"

_UNSAFE_PROJECT_DIR_CHARS = re.compile(r'[<>:"/\\|?*\x00-\x1f]')

def sanitize_project_dir_name(name: str) -> str:
    """Sanitize user-provided project name to be safe for directory names across OSes."""
    if not name:
        return ""
    return _UNSAFE_PROJECT_DIR_CHARS.sub("_", str(name)).strip(" .")

def _parse_time_to_seconds(time_str: str) -> float:
    """Parse time string (HH:MM:SS, MM:SS, or SS) into seconds."""
    parts = time_str.split(':')
    if len(parts) == 1:
        return float(parts[0])
    elif len(parts) == 2:
        return float(parts[0]) * 60 + float(parts[1])
    elif len(parts) == 3:
        return float(parts[0]) * 3600 + float(parts[1]) * 60 + float(parts[2])
    else:
        raise ValueError(f"Invalid time format: {time_str}. Expected HH:MM:SS, MM:SS, or SS.")


class DubbingConfig:
    """Configuration class for the Smart Dubbing system."""
    
    def __init__(self):
        """Initialize configuration with default values."""
        self.defaults = {
            'transcription_model': None,
            'whisper_model': 'large-v3',
            'gemini_transcription_model': 'gemini-3-flash-preview',
            'keep_background': False,
            'start_time': None,
            'duration': None,
            'no_cache': False,
            'tts_system': 'coqui',
            'tts_model': None,
            'omnivoice_space_id': 'k2-fsa/OmniVoice',
            'omnivoice_api_name': '/_clone_fn',
            'omnivoice_lang': None,
            'omnivoice_instruct': '',
            'omnivoice_num_steps': 64,
            'omnivoice_guidance_scale': 2.0,
            'omnivoice_denoise': True,
            'omnivoice_speed': 1.0,
            'omnivoice_duration': 3.0,
            'omnivoice_preprocess_prompt': True,
            'omnivoice_postprocess_output': True,
            'transcription_system': 'whisper',
            'translator_type': 'llm',
            'llm_provider': 'gemini',
            'llm_model_name': None,
            'llm_temperature': 0.5,
            'refinement_llm_provider': None,
            'refinement_model_name': None,
            'refinement_temperature': 1.0,
            'refinement_max_tokens': None,
            'refinement_persona': 'normal',
            'translation_prompt_prefix': None,
            'voice_name': None,
            'debug_info': False,
            'debug_tts': False,
            'debug_diarize_only': False,
            'save_original_subtitles': False,
            'save_translated_subtitles': False,
            'reference_audio': None,
            'reference_text': None,
            'watermark_path': None,
            'watermark_text': None,
            'glossary': None,
            'voice_auto_selection': True,
            'enable_emotion_analysis': False,
            'include_original_audio': False,
            'output': None, 
            'keep_original_audio_ranges': None,
            'voices': None,
            'tts_prompt_prefix': None,
            'remove_pauses': False,
            'min_pause_duration': 300,
            'use_two_pass_encoding': True,
            'keyframe_buffer': 0.2,
            'dubbed_volume': 1.0,
            'background_volume': 0.562341,
            'group_overflow_tolerance': 1.0,
            'timing_short_segment_threshold': 1.5,
            'timing_short_segment_max_speed': 1.08,
            'timing_max_speed': 1.15,
            'timing_max_stretch': 1.15,
            'timing_max_overflow': 0.25,
            'semantic_split_enabled': True,
            'tts_preferred_segment_duration': 15.0,
            'tts_hard_segment_duration': 35.0,
            'semantic_split_search_window': 10.0,
            'segment_reference_min_duration': 2.0,
            'isolated_tracks': None,
            'inner_transcription_system': 'deepgram',
            'project_name': None,
        }
        
        # Required runtime parameters supplied by the active caller.
        self.required_params = ['input', 'source_language', 'target_language']
        
        # Configuration data
        self.config = self.defaults.copy()
    
    def load_from_yaml(self, config_path: str) -> None:
        """Load configuration from YAML file."""
        if config_path and os.path.exists(config_path):
            with open(config_path, 'r', encoding='utf-8') as config_file:
                yaml_config = yaml.safe_load(config_file)
                if yaml_config:
                    from .voice_profiles import reject_legacy_voice_config
                    reject_legacy_voice_config(yaml_config, source="YAML config")
                    # Runtime input is supplied by the caller rather than persisted.
                    if 'input' in yaml_config:
                        logger.warning("Warning: 'input' parameter found in YAML config will be ignored. Input must be provided at runtime.")
                        del yaml_config['input']
                    self.config.update(yaml_config)
            logger.info(f"Loaded configuration from {config_path}")
        elif config_path:
            logger.warning(f"Config file {config_path} not found, using defaults and runtime overrides")
    
    def load_overrides(self, overrides: Dict[str, Any]) -> None:
        """Apply non-null runtime overrides."""
        for key, value in overrides.items():
            if key == 'config':
                continue
            if value is not None:
                self.config[key] = value
    
    def validate(self) -> None:
        """Validate required configuration parameters."""
        # Check that runtime input is provided.
        if not self.config.get('input'):
            logger.error("Error: Input video file must be specified")
            sys.exit(1)
        
        # Check other required parameters  
        for param in ['source_language', 'target_language']:
            if not self.config.get(param):
                logger.error(f"Error: {param.replace('_', ' ').title()} not specified")
                logger.error(f"Please provide '{param}' in the config file or runtime overrides")
                sys.exit(1)
        
        # Check if video file exists
        input_file = self.config['input']
        if not os.path.exists(input_file):
            logger.error(f"Error: Video file '{input_file}' not found")
            logger.error(f"Current working directory: {os.getcwd()}")
            logger.error(f"Absolute path would be: {os.path.abspath(input_file)}")
            # Try to suggest similar files
            input_dir = os.path.dirname(input_file) or '.'
            if os.path.exists(input_dir):
                similar_files = [f for f in os.listdir(input_dir) if f.lower().endswith(('.mp4', '.avi', '.mkv', '.mov'))]
                if similar_files:
                    logger.error(f"Video files found in directory '{input_dir}':")
                    for f in similar_files[:5]:  # Show first 5 files
                        logger.error(f"  - {f}")
            sys.exit(1)

        input_path = Path(input_file)
        if self.config.get("project_dir"):
            project_dir = Path(self.config["project_dir"])
        elif input_path.parent != Path(".") and (input_path.parent / "artifacts").is_dir():
            project_dir = input_path.parent
        else:
            projects_root = (
                Path(os.environ["DUBBLM_PROJECTS_ROOT"])
                if os.environ.get("DUBBLM_PROJECTS_ROOT")
                else DEFAULT_PROJECTS_ROOT
            )
            raw_project_name = self.config.get("project_name")
            folder_name = (
                sanitize_project_dir_name(str(raw_project_name))
                if raw_project_name
                else ""
            )
            if not folder_name:
                folder_name = input_path.stem
            project_dir = projects_root / folder_name

        if self.config.get("artifacts_dir"):
            artifacts_dir = Path(self.config["artifacts_dir"])
        else:
            artifacts_dir = project_dir / "artifacts"
        
        project_dir.mkdir(parents=True, exist_ok=True)
        artifacts_dir.mkdir(parents=True, exist_ok=True)

        self.config["project_dir"] = str(project_dir)
        self.config["artifacts_dir"] = str(artifacts_dir)
        self.config["audio_artifacts_dir"] = str(artifacts_dir / "audio")
        self.config["speakers_audio_dir"] = str(artifacts_dir / "speakers_audio")
        self.config["audio_chunks_dir"] = str(artifacts_dir / "audio_chunks")
        self.config["su_audio_chunks_dir"] = str(artifacts_dir / "su_audio_chunks")
        self.config["debug_dir"] = str(artifacts_dir / "debug")
        self.config["translation_debug_dir"] = str(artifacts_dir / "debug" / "translation")
        self.config["translation_refinement_debug_dir"] = str(artifacts_dir / "debug" / "translation_refinement")
        self.config["speaker_report_dir"] = str(artifacts_dir / "speaker_report")
        self.config["translated_samples_dir"] = str(artifacts_dir / "translated_samples")
        self.config["transcription_path"] = str(artifacts_dir / "transcription.txt")
        self.config["timecodes_report_path"] = str(artifacts_dir / "timecodes.txt")
        self.config["translated_audio_path"] = str(artifacts_dir / "audio" / "output.wav")
        self.config["background_audio_path"] = str(artifacts_dir / "audio" / "background.wav")
        self.config["debug_video_path"] = str(artifacts_dir / "debug" / "dubbing_debug.mp4")
        self.config["temp_segment_audio_path"] = str(artifacts_dir / "audio" / "temp_segment.wav")
        self.config["temp_final_audio_path"] = str(artifacts_dir / "audio" / "temp_final_for_pause_analysis.wav")
        self.config["temp_video_with_cuts_path"] = str(artifacts_dir / "temp_video_with_cuts.mp4")
        
        # Generate output filename if not provided
        if not self.config.get('output'):
            target_lang = self.config['target_language']
            output_filename = f"{project_dir.name}_{target_lang}{input_path.suffix}"
            self.config['output'] = str(project_dir / output_filename)
            logger.info(f"Auto-generated output filename: {self.config['output']}")
        else:
            output_path = Path(self.config['output'])
            if not output_path.suffix:
                normalized_output_path = output_path.with_suffix(input_path.suffix)
                self.config['output'] = str(normalized_output_path)
                logger.info(
                    "Added missing output extension '%s': %s",
                    input_path.suffix,
                    self.config['output'],
                )
    
    def process_special_parameters(self) -> None:
        """Process special parameters that need parsing."""
        from .voice_profiles import reject_legacy_voice_config
        reject_legacy_voice_config(self.config, source="merged config")

        def _parse_mapping_parameter(name: str) -> None:
            value = self.config.get(name)
            if isinstance(value, str):
                try:
                    parsed_value = json.loads(value)
                    if isinstance(parsed_value, dict):
                        self.config[name] = parsed_value
                        logger.info(f"Parsed {name} from JSON: {parsed_value}")
                    else:
                        logger.warning(f"Warning: {name} must decode to a JSON object. Ignoring.")
                        self.config[name] = None
                except json.JSONDecodeError as e:
                    logger.warning(f"Warning: Could not parse {name} JSON '{value}': {e}. Ignoring.")
                    self.config[name] = None
            elif isinstance(value, dict):
                logger.info(f"Using {name} from config: {value}")
            else:
                self.config[name] = None

        duration = self.config.get('duration')
        if duration is not None:
            try:
                duration_value = float(duration)
                if duration_value <= 0:
                    logger.info("Ignoring non-positive duration value; processing until the end of the file.")
                    self.config['duration'] = None
                else:
                    self.config['duration'] = duration_value
            except (TypeError, ValueError):
                logger.warning("Warning: Invalid duration value. Ignoring it.")
                self.config['duration'] = None

        voice_name = self.config.get('voice_name')
        if voice_name:
            # Single voice for all speakers
            logger.info(f"Using single voice: {voice_name}")
        
        # Process keep_original_audio_ranges
        if self.config.get('keep_original_audio_ranges'):
            parsed_ranges = []
            for range_str in self.config['keep_original_audio_ranges']:
                try:
                    start_str, end_str = range_str.split('-')
                    start_seconds = _parse_time_to_seconds(start_str.strip())
                    end_seconds = _parse_time_to_seconds(end_str.strip())
                    if start_seconds >= end_seconds:
                        logger.warning(f"Warning: Invalid range {range_str} in keep_original_audio_ranges (start >= end). Skipping.")
                        continue
                    parsed_ranges.append((start_seconds, end_seconds))
                except ValueError as e:
                    logger.warning(f"Warning: Could not parse range '{range_str}' in keep_original_audio_ranges: {e}. Skipping.")
            
            self.config['keep_original_audio_ranges'] = parsed_ranges if parsed_ranges else None
        
        _parse_mapping_parameter('isolated_tracks')

        # Validate isolated_tracks: strip empties, verify files exist
        isolated_tracks = self.config.get('isolated_tracks')
        if isinstance(isolated_tracks, dict) and isolated_tracks:
            validated = {}
            for speaker, path in isolated_tracks.items():
                if not path:
                    continue
                speaker_label = str(speaker).strip()
                path_str = str(path).strip()
                if not speaker_label or not path_str:
                    continue
                if not os.path.exists(path_str):
                    logger.error(
                        "Isolated track for speaker '%s' not found: %s",
                        speaker_label,
                        path_str,
                    )
                    sys.exit(1)
                validated[speaker_label] = path_str
            self.config['isolated_tracks'] = validated or None
            if validated:
                logger.info(
                    "Isolated speaker tracks enabled for %d speakers: %s",
                    len(validated),
                    ", ".join(validated.keys()),
                )
        else:
            self.config['isolated_tracks'] = None

        # Validate inner_transcription_system
        inner_sys = self.config.get('inner_transcription_system')
        if inner_sys not in {'deepgram', 'assemblyai', 'gemini'}:
            logger.warning(
                "Warning: invalid inner_transcription_system '%s'. Falling back to 'deepgram'.",
                inner_sys,
            )
            self.config['inner_transcription_system'] = 'deepgram'

        # Normalize the modern `voices` block into VoiceProfile objects.
        raw_voices = self.config.get('voices')
        if isinstance(raw_voices, str):
            try:
                parsed_voices = json.loads(raw_voices)
                if isinstance(parsed_voices, dict):
                    self.config['voices'] = parsed_voices
                else:
                    logger.warning(
                        "Warning: 'voices' must decode to a JSON object. Ignoring."
                    )
                    self.config['voices'] = None
            except json.JSONDecodeError as e:
                logger.warning(
                    f"Warning: Could not parse voices JSON '{raw_voices}': {e}. Ignoring."
                )
                self.config['voices'] = None

        from .voice_profiles import normalize_voices
        self.config['voices'] = normalize_voices(self.config)
        self.config.pop('tts_fallback_model', None)

        from .timing import normalize_timing_config
        normalize_timing_config(self.config, warn=logger.warning)

        # Validate segment_reference_min_duration
        ref_min_duration = self.config.get('segment_reference_min_duration')
        try:
            ref_min_duration_f = float(0.0 if ref_min_duration is None else ref_min_duration)
            if ref_min_duration_f < 0.0:
                logger.warning("Warning: segment_reference_min_duration cannot be negative. Using 0 seconds.")
                ref_min_duration_f = 0.0
            self.config['segment_reference_min_duration'] = ref_min_duration_f
        except (TypeError, ValueError):
            default_ref_duration = self.defaults['segment_reference_min_duration']
            logger.warning(
                "Warning: Invalid segment_reference_min_duration value. Falling back to default of %.2f seconds.",
                default_ref_duration
            )
            self.config['segment_reference_min_duration'] = default_ref_duration

        # Dynamically map omnivoice_lang from target_language when not explicitly set
        if not self.config.get('omnivoice_lang'):
            target_lang = self.config.get('target_language')
            if target_lang:
                from tts.omnivoice_wrapper import resolve_omnivoice_language
                self.config['omnivoice_lang'] = resolve_omnivoice_language(None, target_lang)
            else:
                self.config['omnivoice_lang'] = 'Belarusian'
    
    def get(self, key: str, default: Any = None) -> Any:
        """Get configuration value."""
        return self.config.get(key, default)
    
    def set(self, key: str, value: Any) -> None:
        """Set configuration value."""
        self.config[key] = value
    
    def to_dict(self) -> Dict[str, Any]:
        """Return configuration as dictionary."""
        return self.config.copy()
