"""Internal artifact, subtitle, cache-reset, and final-video helpers."""

import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from ..log_config import get_logger

logger = get_logger(__name__)


def prepare_audio_inputs(facade) -> tuple[str, Optional[str], str]:
    """Extract the source audio and optional background/vocals tracks."""
    audio_file = facade.audio_processor.extract_audio(
        facade.config.get('input'),
        facade.config.get('start_time'),
        facade.config.get('duration')
    )
    background_audio_path = None
    segment_reference_audio_file = audio_file
    if facade.config.get('keep_background', False):
        (
            background_audio_path,
            separated_vocals_path,
        ) = facade.audio_processor.separate_background_and_vocals(audio_file)
        if separated_vocals_path:
            segment_reference_audio_file = separated_vocals_path

    return audio_file, background_audio_path, segment_reference_audio_file


def load_required_cached_step(
    facade, *, step_name: str, cache_key: str, hint: str
) -> Any:
    """Load a required cached artifact or raise an actionable error."""
    if not getattr(facade.cache_manager, "use_cache", True):
        raise FileNotFoundError(
            f"run_step=tts_to_end requires cached {hint} artifacts from a previous full dubbing run, "
            "but caching is currently disabled. Re-enable cache or run the full pipeline first."
        )

    if not facade.cache_manager.cache_exists(step_name, cache_key):
        raise FileNotFoundError(
            f"run_step=tts_to_end requires cached {hint} artifacts from a previous full dubbing run in the same project directory, "
            f"but no cache entry was found for step '{step_name}'."
        )

    cached_value = facade.cache_manager.load_from_cache(step_name, cache_key)
    if cached_value is None:
        raise FileNotFoundError(
            f"run_step=tts_to_end found step '{step_name}' but could not load cached {hint} artifacts. "
            "Re-run the full pipeline to rebuild them."
        )

    return cached_value


def build_speaker_rolls_from_segments(
    facade, segments: List[Dict]
) -> Dict[Tuple[float, float], str]:
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


def save_requested_subtitles(
    facade,
    segments_for_output: List[Dict],
    *,
    save_original_subtitles: bool,
    save_translated_subtitles: bool,
    pause_adjustments: Optional[List[Dict[str, float]]] = None,
) -> None:
    """Persist subtitle files for the current output state."""
    if not (save_original_subtitles or save_translated_subtitles):
        return

    remove_pauses_enabled = facade.config.get('remove_pauses', False)
    if not remove_pauses_enabled:
        if save_original_subtitles:
            facade.subtitle_manager.save_subtitles(
                segments_for_output,
                "original",
                facade._get_subtitle_path("original", facade.config.get('input'), facade.config.get('source_language')),
            )

        if save_translated_subtitles:
            facade.subtitle_manager.save_subtitles(
                segments_for_output,
                "translation",
                facade._get_subtitle_path("translation", facade.config.get('input'), facade.config.get('target_language')),
            )
        return

    if pause_adjustments:
        logger.info("Adjusting subtitle timestamps based on pause modifications...")
        adjusted_segments = facade.adjust_subtitle_timestamps(segments_for_output, pause_adjustments)
        if save_original_subtitles:
            facade.subtitle_manager.save_subtitles(
                adjusted_segments,
                "original",
                facade._get_subtitle_path("original", facade.config.get('input'), facade.config.get('source_language')),
            )
            adjusted_path = facade._get_subtitle_path("original", facade.config.get('input'), facade.config.get('source_language'))
            logger.info(f"Saved pause-corrected original subtitles to {adjusted_path}")

        if save_translated_subtitles:
            facade.subtitle_manager.save_subtitles(
                adjusted_segments,
                "translation",
                facade._get_subtitle_path("translation", facade.config.get('input'), facade.config.get('target_language')),
            )
            adjusted_path = facade._get_subtitle_path("translation", facade.config.get('input'), facade.config.get('target_language'))
            logger.info(f"Saved pause-corrected translated subtitles to {adjusted_path}")
        return

    logger.info("No pause adjustments needed, saving subtitles with original timestamps...")
    if save_original_subtitles:
        facade.subtitle_manager.save_subtitles(
            segments_for_output,
            "original",
            facade._get_subtitle_path("original", facade.config.get('input'), facade.config.get('source_language')),
        )
        subtitle_path = facade._get_subtitle_path("original", facade.config.get('input'), facade.config.get('source_language'))
        logger.info(f"Saved original subtitles to {subtitle_path}")

    if save_translated_subtitles:
        facade.subtitle_manager.save_subtitles(
            segments_for_output,
            "translation",
            facade._get_subtitle_path("translation", facade.config.get('input'), facade.config.get('target_language')),
        )
        subtitle_path = facade._get_subtitle_path("translation", facade.config.get('input'), facade.config.get('target_language'))
        logger.info(f"Saved translated subtitles to {subtitle_path}")


def combine_final_video(
    facade,
    *,
    translated_audio_path: str,
    background_audio_path: Optional[str],
    speakers_rolls: Dict[Tuple[float, float], str],
) -> tuple[str, List[Dict[str, float]]]:
    """Combine the current translated audio track with the source video."""
    keep_original_audio_ranges = facade.config.get('keep_original_audio_ranges')
    muted_speakers = getattr(facade, "muted_speakers", set())
    if keep_original_audio_ranges is None and facade.config.get('include_original_audio', False) and muted_speakers:
        try:
            keep_original_audio_ranges = [
                (start, end) for (start, end), spk in (speakers_rolls or {}).items() if spk not in muted_speakers
            ]
            if keep_original_audio_ranges:
                logger.info(f"Computed keep_original_audio_ranges excluding muted speakers ({len(keep_original_audio_ranges)} ranges)")
        except Exception:
            keep_original_audio_ranges = facade.config.get('keep_original_audio_ranges')

    return facade.video_processor.combine_audio_with_video(
        video_path=facade.config.get('input'),
        translated_audio_path=translated_audio_path,
        background_audio_path=background_audio_path,
        watermark_path=facade.config.get('watermark_path'),
        watermark_text=facade.config.get('watermark_text'),
        include_original_audio=facade.config.get('include_original_audio', False),
        output_file=facade.config.get('output'),
        start_time=facade.config.get('start_time'),
        duration=facade.config.get('duration'),
        keep_original_audio_ranges=keep_original_audio_ranges,
        source_language=facade.config.get('source_language'),
        target_language=facade.config.get('target_language'),
        normalize_audio=facade.config.get('normalize_audio', True),
        use_two_pass_encoding=facade.config.get('use_two_pass_encoding', True),
        remove_pauses=facade.config.get('remove_pauses', False),
        min_pause_duration=facade.config.get('min_pause_duration', 300),
        preserve_pause_duration=facade.config.get('preserve_pause_duration', 1.5),
        keyframe_buffer=facade.config.get('keyframe_buffer', 0.2),
        ffmpeg_batch_size=facade.config.get('ffmpeg_batch_size', 50),
        dubbed_volume=facade.config.get('dubbed_volume', 1.0),
        background_volume=facade.config.get('background_volume', 0.562341),
        upscale_factor=facade.config.get('upscale_factor', 1.0),
        upscale_sharpen=facade.config.get('upscale_sharpen', True),
    )


def reset_input_cache(facade, reason: str) -> None:
    """Delete every cached artifact tied to the current input file."""
    logger.info(f"Clearing cached artifacts for this input ({reason})")
    try:
        if hasattr(facade.cache_manager, "clear_input_cache"):
            facade.cache_manager.clear_input_cache(facade.config.get('input'))
    except Exception as e:
        logger.warning(f"Could not clear per-input cache: {e}")

    for chunk_dir in (
        getattr(facade, "audio_chunks_dir", None),
        getattr(facade, "su_audio_chunks_dir", None),
    ):
        if chunk_dir and Path(chunk_dir).exists():
            for item in Path(chunk_dir).glob("*.wav"):
                try:
                    item.unlink()
                except OSError as exc:
                    logger.debug(f"Could not remove stale chunk {item}: {exc}")

    legacy_root = getattr(facade.cache_manager, "cache_root", None)
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


def save_transcription_file(facade, transcription: List[Dict]) -> None:
    """Save transcription to a readable text file."""
    from src.utils.time_utils import format_seconds_to_hms

    transcription_output_path = facade.config.get("transcription_path")
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


def get_subtitle_path(
    facade, subtitle_type: str, input_path: str, language: str
) -> str:
    """Generate subtitle path based on input file and language."""
    input_file = Path(input_path)
    project_dir = Path(facade.config.get("project_dir"))
    name_base = project_dir.name if facade.config.get("project_name") else input_file.stem
    source_lang = facade.config.get('source_language')
    target_lang = facade.config.get('target_language')
    source_name = f"{name_base}_{source_lang}.srt"
    target_name = f"{name_base}_{target_lang}.srt"

    if source_name == target_name:
        if subtitle_type == "original":
            return str(project_dir / f"source_{source_name}")
        else:
            return str(project_dir / f"target_{target_name}")

    return str(project_dir / f"{name_base}_{language}.srt")


def adjust_subtitle_timestamps(
    facade,
    segments: List[Dict],
    pause_adjustments: List[Dict[str, float]],
) -> List[Dict]:
    """Adjust subtitle timestamps based on pause adjustments."""
    if not pause_adjustments:
        logger.debug("No pause adjustments to apply to subtitles")
        return segments

    logger.info(f"Adjusting subtitle timestamps based on {len(pause_adjustments)} pause modifications")

    adjusted_segments = []
    for segment in segments:
        adjusted_segment = segment.copy()

        start_offset = 0.0
        end_offset = 0.0

        for adjustment in pause_adjustments:
            if segment['start'] >= adjustment['original_end']:
                start_offset = adjustment['cumulative_offset']
            elif segment['start'] >= adjustment['original_start']:
                if segment['start'] <= adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed']):
                    start_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                else:
                    start_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                    adjusted_segment['start'] = adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed'])

            if segment['end'] >= adjustment['original_end']:
                end_offset = adjustment['cumulative_offset']
            elif segment['end'] >= adjustment['original_start']:
                if segment['end'] <= adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed']):
                    end_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                else:
                    end_offset = adjustment['cumulative_offset'] - adjustment['time_removed']
                    adjusted_segment['end'] = adjustment['original_start'] + (adjustment['original_end'] - adjustment['original_start'] - adjustment['time_removed'])

        adjusted_segment['start'] = max(0, adjusted_segment['start'] - start_offset)
        adjusted_segment['end'] = max(adjusted_segment['start'], adjusted_segment['end'] - end_offset)

        adjusted_segments.append(adjusted_segment)

    logger.debug(f"Adjusted timestamps for {len(adjusted_segments)} subtitle segments")
    return adjusted_segments
