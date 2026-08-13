"""Internal segment-reference implementations for the dubbing pipeline."""

import math
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydub import AudioSegment


def attach_segment_reference(
    facade: Any, *, tts_segment_data_args: Dict[str, Any],
    segment_dict: Dict[str, Any], speaker: str, segment_index: int,
    original_audio_segment: Optional[AudioSegment],
    segment_reference_min_duration: float,
    segment_reference_min_duration_ms: int,
) -> tuple[Dict[str, Any], Optional[AudioSegment]]:
    segment_duration = segment_dict["end"] - segment_dict["start"]
    if original_audio_segment is None or (
        segment_reference_min_duration > 0.0
        and segment_duration < segment_reference_min_duration
    ):
        return tts_segment_data_args, original_audio_segment
    start_ms = max(int(segment_dict["start"] * 1000), 0)
    end_ms = min(int(segment_dict["end"] * 1000), len(original_audio_segment))
    if end_ms <= start_ms:
        return tts_segment_data_args, original_audio_segment
    segment_audio = original_audio_segment[start_ms:end_ms]
    if segment_reference_min_duration_ms != 0 and len(segment_audio) < segment_reference_min_duration_ms:
        return tts_segment_data_args, original_audio_segment
    segment_ref_dir = facade.speakers_audio_dir / "segments"
    segment_ref_dir.mkdir(parents=True, exist_ok=True)
    segment_ref_path = segment_ref_dir / f"{speaker}_{segment_index}.wav"
    segment_audio.export(segment_ref_path, format="wav")
    tts_segment_data_args["reference_audio_path"] = str(segment_ref_path)
    tts_segment_data_args["reference_text"] = segment_dict.get("text")
    return tts_segment_data_args, original_audio_segment


def canonical_segment_index(
    segment_dict: Dict[str, Any], chronological_index: int
) -> int:
    stored = segment_dict.get("_timing_original_index")
    if isinstance(stored, int) and not isinstance(stored, bool) and stored >= 0:
        return stored
    return chronological_index


def segment_reference_artifact_paths(
    *, config: Any, processed_source_path: Optional[str] = None
) -> tuple[Path, Path]:
    if processed_source_path:
        processed = Path(processed_source_path)
        if config.get("keep_background", False):
            vocals_path = processed
            source_path = processed.with_name("source.wav")
        else:
            source_path = processed
            vocals_path = processed.with_name("vocals.wav")
    elif config.get("audio_artifacts_dir"):
        audio_dir = Path(config.get("audio_artifacts_dir"))
        source_path = audio_dir / "source.wav"
        vocals_path = audio_dir / "vocals.wav"
    else:
        source_path = Path("source.wav")
        vocals_path = Path("vocals.wav")
    return vocals_path, source_path


def segment_reference_error(
    speaker: str, segment_index: int, source_path: Path, reason: str
) -> ValueError:
    return ValueError(
        f"speaker={speaker} segment={segment_index} mode=segment "
        f"source={source_path}: {reason}"
    )


def prepare_segment_reference(
    facade: Any, *, segment_dict: Dict[str, Any], speaker: str,
    chronological_index: int, reuse_existing: bool,
    processed_source_path: Optional[str] = None,
    decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None,
) -> tuple[str, Optional[str]]:
    canonical_index = facade._canonical_segment_index(segment_dict, chronological_index)
    reference_path = facade.speakers_audio_dir / "segments" / f"{speaker}_{canonical_index}.wav"
    reference_text = segment_dict.get("text")
    try:
        minimum_duration = max(0.0, float(facade.config.get("segment_reference_min_duration", 2.0) or 0.0))
    except (TypeError, ValueError, OverflowError):
        minimum_duration = 2.0
    minimum_ms = int(minimum_duration * 1000)
    if reuse_existing and reference_path.is_file():
        try:
            existing = AudioSegment.from_file(reference_path)
            if len(existing) > 0 and len(existing) >= minimum_ms:
                return str(reference_path), reference_text
        except Exception:
            pass
    isolated_tracks = facade.config.get("isolated_tracks")
    mapped_isolated = (
        isolated_tracks.get(speaker)
        if isinstance(isolated_tracks, dict) and speaker in isolated_tracks else None
    )
    vocals_path, source_path = facade._segment_reference_artifact_paths(processed_source_path)
    if isinstance(isolated_tracks, dict) and speaker in isolated_tracks:
        candidates = [(Path(str(mapped_isolated or "")), True)]
    else:
        candidates = []
        if facade.config.get("keep_background", False):
            candidates.append((vocals_path, False))
        candidates.append((source_path, False))
    authoritative_path = candidates[0][0]
    try:
        start = float(segment_dict["start"])
        end = float(segment_dict["end"])
    except (KeyError, TypeError, ValueError, OverflowError) as exc:
        raise facade._segment_reference_error(
            speaker, canonical_index, authoritative_path,
            "segment reference requires numeric start and end timestamps",
        ) from exc
    if not math.isfinite(start) or not math.isfinite(end):
        raise facade._segment_reference_error(
            speaker, canonical_index, authoritative_path,
            "segment timestamps must be finite",
        )
    if start < 0 or end < 0 or end <= start:
        raise facade._segment_reference_error(
            speaker, canonical_index, authoritative_path,
            "segment timestamps must be non-negative with end greater than start",
        )
    if end - start < minimum_duration:
        raise facade._segment_reference_error(
            speaker, canonical_index, authoritative_path,
            f"recognized segment duration {end - start:.2f}s is below the "
            f"configured minimum {minimum_duration:.2f}s",
        )
    cache = decoded_audio_cache if decoded_audio_cache is not None else {}
    prior_failures: List[str] = []
    selected_audio: Optional[AudioSegment] = None
    selected_path: Optional[Path] = None
    selected_start_ms = 0
    selected_end_ms = 0
    for candidate_path, is_isolated in candidates:
        offset = 0.0
        if is_isolated:
            try:
                offset = float(facade.config.get("start_time") or 0.0)
            except (TypeError, ValueError, OverflowError) as exc:
                raise facade._segment_reference_error(
                    speaker, canonical_index, candidate_path,
                    "start_time must be a finite non-negative number",
                ) from exc
            if not math.isfinite(offset) or offset < 0:
                raise facade._segment_reference_error(
                    speaker, canonical_index, candidate_path,
                    "start_time must be a finite non-negative number",
                )
        candidate_start = start + offset
        candidate_end = end + offset
        failure: Optional[str] = None
        audio: Optional[AudioSegment] = None
        if not candidate_path.is_file():
            failure = "reference source file is missing"
        else:
            cache_key = (str(candidate_path.resolve(strict=False)), offset)
            try:
                audio = cache.get(cache_key)
                if audio is None:
                    audio = AudioSegment.from_file(candidate_path)
                    cache[cache_key] = audio
            except Exception as exc:
                failure = f"reference source is unreadable: {exc}"
        if audio is not None:
            if len(audio) <= 0:
                failure = "reference source is empty"
            elif candidate_end * 1000 > len(audio) + 1e-6:
                failure = (
                    f"segment interval {candidate_start:.3f}..{candidate_end:.3f}s "
                    f"is outside source duration {len(audio) / 1000.0:.3f}s"
                )
        if failure is not None:
            if is_isolated:
                raise facade._segment_reference_error(
                    speaker, canonical_index, candidate_path, failure
                )
            prior_failures.append(f"{candidate_path}: {failure}")
            continue
        selected_audio = audio
        selected_path = candidate_path
        selected_start_ms = int(candidate_start * 1000)
        selected_end_ms = int(candidate_end * 1000)
        break
    if selected_audio is None or selected_path is None:
        reason = "no usable reference source"
        if prior_failures:
            reason += "; " + "; ".join(prior_failures)
        raise facade._segment_reference_error(speaker, canonical_index, source_path, reason)
    segment_audio = selected_audio[selected_start_ms:selected_end_ms]
    if len(segment_audio) <= 0 or len(segment_audio) < minimum_ms:
        raise facade._segment_reference_error(
            speaker, canonical_index, selected_path,
            f"exported duration {len(segment_audio) / 1000.0:.2f}s is below "
            f"the configured minimum {minimum_duration:.2f}s",
        )
    reference_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        segment_audio.export(reference_path, format="wav")
        exported = AudioSegment.from_file(reference_path)
    except Exception as exc:
        raise facade._segment_reference_error(
            speaker, canonical_index, selected_path,
            f"could not export or read reference WAV {reference_path}: {exc}",
        ) from exc
    if not reference_path.is_file() or len(exported) <= 0:
        raise facade._segment_reference_error(
            speaker, canonical_index, selected_path,
            f"exported reference WAV is missing or empty: {reference_path}",
        )
    if len(exported) < minimum_ms:
        raise facade._segment_reference_error(
            speaker, canonical_index, selected_path,
            f"exported duration {len(exported) / 1000.0:.2f}s is below "
            f"the configured minimum {minimum_duration:.2f}s",
        )
    return str(reference_path), reference_text


def resolve_segment_reference(
    facade: Any, *, tts_segment_data_args: Dict[str, Any],
    segment_dict: Dict[str, Any], profile: Any, provider_capability: str,
    speaker: str, segment_index: int,
    original_audio_segment: Optional[AudioSegment],
    segment_reference_min_duration: float, for_resynthesis: bool = False,
    processed_source_path: Optional[str] = None,
    decoded_audio_cache: Optional[Dict[tuple[str, float], AudioSegment]] = None,
) -> tuple[Dict[str, Any], Optional[AudioSegment]]:
    mode = profile.reference_mode
    tts_segment_data_args["reference_mode"] = mode
    tts_segment_data_args["segment_index"] = segment_index
    if provider_capability == "unsupported" or mode not in {"configured", "segment", "speaker", "none"}:
        return tts_segment_data_args, original_audio_segment
    if mode == "none":
        tts_segment_data_args["reference_audio_path"] = None
        tts_segment_data_args["reference_text"] = None
        return tts_segment_data_args, original_audio_segment
    if mode == "configured":
        configured = profile.reference_audio
        reference_path = Path(configured).expanduser().resolve(strict=False) if configured else None
        if reference_path is None or not reference_path.is_file():
            raise ValueError(f"reference file does not exist: {configured or '<missing>'}")
        tts_segment_data_args["reference_audio_path"] = str(reference_path)
        tts_segment_data_args["reference_text"] = profile.reference_text
        return tts_segment_data_args, original_audio_segment
    if mode == "speaker":
        reference_path = facade.speakers_audio_dir / f"{speaker}.wav"
        if not reference_path.is_file():
            raise ValueError(f"reference file does not exist: {reference_path}")
        tts_segment_data_args["reference_audio_path"] = str(reference_path)
        tts_segment_data_args["reference_text"] = profile.reference_text
        return tts_segment_data_args, original_audio_segment
    if processed_source_path is not None or original_audio_segment is None:
        reference_path, reference_text = facade._prepare_segment_reference(
            segment_dict=segment_dict, speaker=speaker,
            chronological_index=segment_index, reuse_existing=for_resynthesis,
            processed_source_path=processed_source_path,
            decoded_audio_cache=decoded_audio_cache,
        )
        tts_segment_data_args["reference_audio_path"] = reference_path
        tts_segment_data_args["reference_text"] = reference_text
        tts_segment_data_args["segment_index"] = facade._canonical_segment_index(segment_dict, segment_index)
        return tts_segment_data_args, original_audio_segment
    segment_ref_path = facade.speakers_audio_dir / "segments" / f"{speaker}_{segment_index}.wav"
    try:
        start = float(segment_dict["start"])
        end = float(segment_dict["end"])
    except (KeyError, TypeError, ValueError) as exc:
        raise ValueError("segment reference requires valid start and end timestamps") from exc
    duration = end - start
    if duration <= 0:
        raise ValueError("segment reference requires end to be greater than start")
    if duration < segment_reference_min_duration:
        raise ValueError(
            f"recognized segment duration {duration:.2f}s is below the configured "
            f"minimum {segment_reference_min_duration:.2f}s"
        )
    if original_audio_segment is None:
        raise ValueError("selected reference-audio source could not be loaded")
    start_ms = max(int(start * 1000), 0)
    end_ms = min(int(end * 1000), len(original_audio_segment))
    if end_ms <= start_ms:
        raise ValueError("segment interval is outside the selected reference-audio source")
    segment_audio = original_audio_segment[start_ms:end_ms]
    if len(segment_audio) < int(segment_reference_min_duration * 1000):
        raise ValueError(
            f"exported segment duration {len(segment_audio) / 1000.0:.2f}s is below "
            f"the configured minimum {segment_reference_min_duration:.2f}s"
        )
    segment_ref_path.parent.mkdir(parents=True, exist_ok=True)
    try:
        segment_audio.export(segment_ref_path, format="wav")
    except Exception as exc:
        raise ValueError(f"could not export segment reference: {exc}") from exc
    if not segment_ref_path.is_file():
        raise ValueError(f"reference file does not exist after export: {segment_ref_path}")
    tts_segment_data_args["reference_audio_path"] = str(segment_ref_path)
    tts_segment_data_args["reference_text"] = segment_dict.get("text")
    return tts_segment_data_args, original_audio_segment
