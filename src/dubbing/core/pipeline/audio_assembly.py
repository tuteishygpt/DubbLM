"""Internal timing and final dubbed-audio assembly implementations."""

import csv
import os
import subprocess
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from pydub import AudioSegment

from ..log_config import get_logger
from .context import update_context as _update_pipeline_context

logger = get_logger(__name__)


def trim_trailing_silence(
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


def measure_raw_tts_for_timing(facade, audio_path: str, segment_index: int) -> float:
    """Measure audible duration without modifying the raw synthesized WAV."""
    from ..timing import trim_audio_edges

    facade.su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)
    measured_path = facade.su_audio_chunks_dir / f"measure_{segment_index}.wav"
    result = trim_audio_edges(audio_path, measured_path)
    if result.error:
        logger.warning("Could not trim TTS edges for %s: %s", audio_path, result.error)
    return result.trimmed_duration if result.usable else 0.0


def adjust_and_combine_audio_grouped_legacy(
    facade, segments: List[Dict]
) -> Tuple[AudioSegment, List[Dict]]:
    """Compatibility route to the active anchor-based assembler."""
    logger.warning("Legacy group timing is disabled; using anchor-based timing instead.")
    return facade._adjust_and_combine_audio_grouped(segments)


def adjust_and_combine_audio_grouped(facade, segments: List[Dict]) -> Tuple[AudioSegment, List[Dict]]:
    """Place speech at immutable recognized starts using per-clip tempo only."""
    if not segments:
        return AudioSegment.empty(), []

    from ..timing import TimingPolicy, calculate_segment_timing, plan_anchor_windows, trim_audio_edges

    source_duration = getattr(facade, "_timing_source_duration", None)
    if source_duration is None:
        source_path = getattr(facade, "_timing_source_audio_file", None)
        if source_path and os.path.exists(str(source_path)):
            source_duration = len(AudioSegment.from_file(source_path)) / 1000.0
        else:
            # Compatibility for direct callers that predate the source-duration
            # contract. Normal synthesis always sets the exact processed length.
            source_duration = max(float(segment["end"]) for segment in segments) + 1.0

    policy = TimingPolicy.from_config(facade.config)
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
    facade.su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)

    speaker_groups = {}
    for item in planned:
        speaker_groups.setdefault(item.segment.get("speaker", "UNKNOWN"), []).append(item.segment)
    facade.debug_data["speaker_groups"] = {
        speaker: [speaker_segments] for speaker, speaker_segments in speaker_groups.items()
    }

    for chronological_index, item in enumerate(planned):
        segment = item.segment
        segment_file = segment.get("synthesized_speech_file")
        if not segment_file:
            candidate_path = facade.audio_chunks_dir / f"{chronological_index}.wav"
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
                        timing_path = facade.su_audio_chunks_dir / f"timed_{item.original_index}.wav"
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
            input_path = facade.su_audio_chunks_dir / f"tempo_in_{item.original_index}.wav"
            output_path = facade.su_audio_chunks_dir / f"tempo_{item.original_index}.wav"
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
    facade.debug_data["timing_alignment"] = diagnostics
    facade.debug_data["speed_ratios"] = {
        row["segment_index"]: row["tempo"] for row in diagnostics
    }

    if facade.config.get("debug_info", False):
        debug_dir = Path(facade.config.get("debug_dir") or ".")
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


def rebuild_translated_audio_from_chunks(facade) -> Optional[str]:
    """Rebuild the aggregate dubbed track from the latest editor snapshot.

    ``Regenerate selected row`` deliberately updates only one raw chunk.
    The combine-video step calls this method so its mux input reflects that
    chunk without invoking TTS again for the other segments.
    """
    source_audio_path = Path(facade.config.get("audio_artifacts_dir")) / "source.wav"
    if not source_audio_path.is_file():
        logger.warning(
            "Cannot rebuild translated audio from chunks because source audio is missing: %s",
            source_audio_path,
        )
        return None

    snapshot_key = facade._build_dubbing_text_snapshot_key(str(source_audio_path))
    if not facade.cache_manager.cache_exists("dubbing_texts", snapshot_key):
        logger.info(
            "No Dubbing Texts snapshot found; reusing the existing translated audio track."
        )
        return None

    snapshot = facade.cache_manager.load_from_cache("dubbing_texts", snapshot_key)
    if isinstance(snapshot, list):
        segments = snapshot
    elif isinstance(snapshot, dict) and snapshot.get("version") == 1:
        segments = snapshot.get("segments")
    else:
        raise ValueError("Unexpected Dubbing Texts snapshot payload")

    if not isinstance(segments, list) or not segments:
        raise ValueError("Dubbing Texts snapshot has no segments to combine")

    _update_pipeline_context(
        facade, "timing_source_audio_file", str(source_audio_path)
    )
    _update_pipeline_context(
        facade,
        "timing_source_duration",
        len(AudioSegment.from_file(source_audio_path)) / 1000.0,
    )
    combined_audio, real_segment_positions = facade._adjust_and_combine_audio_grouped(
        segments
    )

    output_path = Path(facade.config.get("translated_audio_path"))
    output_path.parent.mkdir(parents=True, exist_ok=True)
    combined_audio.export(output_path, format="wav")
    facade.real_segment_positions = real_segment_positions
    logger.info("Rebuilt translated audio from current chunks: %s", output_path)
    return str(output_path)
