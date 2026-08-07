"""Diarize + transcribe from per-speaker isolated audio tracks.

This module is an opt-in alternative to the standard
``TranscriptionFactory``-based path in ``SmartDubbing.diarize_and_transcribe``.
It kicks in only when the user supplies isolated per-speaker audio tracks
(one clean voice per file). Because the speaker identity is known from the
input file mapping, we skip pyannote speaker-diarization entirely and use
only voice-activity-detection to segment each track. Each VAD region is then
labeled with the speaker key from the mapping, and text is filled in from a
standard inner transcriber (Deepgram / AssemblyAI / Gemini) called on the
whole track.

Overlapping speech survives naturally: two isolated tracks that speak at the
same time contribute two independent segments with distinct speaker labels.
The downstream TTS pipeline already overlays segments through
``AudioSegment.overlay`` (see ``SmartDubbing`` audio assembly), so nothing
else needs to change to render the overlap in the dubbed output.
"""
from __future__ import annotations

import os
import subprocess
import tempfile
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from src.dubbing.core.log_config import get_logger

if TYPE_CHECKING:
    from src.dubbing.core.cache_manager import CacheManager

logger = get_logger(__name__)

# Words shorter than this from a VAD region's boundary are still merged into
# the region — accounts for word timestamps that trail the VAD tail by tens
# of ms. Wider tolerance would let words from adjacent regions leak in.
_WORD_VAD_TOLERANCE = 0.15

# Minimum speech region length worth transcribing — anything shorter is
# usually a breath or click that the inner transcriber ignores anyway.
_MIN_REGION_DURATION = 0.20

# Merge same-speaker segments separated by a shorter gap than this. VAD
# splits on any pause it detects; short pauses (in the middle of a phrase)
# should be re-glued so TTS synthesizes the whole phrase as one chunk.
_MERGE_MAX_GAP = 1.2

# Segments longer than this are hard to dub without leaving tail silence.
_SPLIT_MAX_DURATION = 15.0

# When splitting, prefer word-boundary gaps at least this large. Ignore
# smaller gaps to avoid cutting inside a phrase.
_SPLIT_MIN_WORD_GAP = 0.4

# Sentence-ending punctuation that makes for a natural split point.
_SENTENCE_END_PUNCT = (".", "!", "?", "…")
_SOFT_END_PUNCT = (",", ";", "—", ":")


def _trim_track_if_needed(
    audio_path: str,
    start_time: Optional[float],
    duration: Optional[float],
) -> Tuple[str, Optional[str]]:
    """Trim the isolated track by ``[start_time, start_time+duration]``.

    Returns ``(path_to_use, cleanup_path)``. When no trimming is needed
    (both bounds unset), returns the original path and no cleanup handle.
    Otherwise ffmpeg writes a temp wav we clean up after transcription.
    Trimming intentionally re-bases timestamps: the returned clip starts
    at t=0, matching what ``AudioProcessor.extract_audio`` produces for
    the main audio track, so downstream segment timestamps line up.
    """
    if start_time is None and duration is None:
        return audio_path, None
    if duration is not None and duration <= 0:
        duration = None
    if start_time is None and duration is None:
        return audio_path, None

    fd, out_path = tempfile.mkstemp(prefix="isolated_track_", suffix=".wav")
    os.close(fd)

    parts = ["ffmpeg", "-y"]
    if start_time is not None:
        parts += ["-ss", str(start_time)]
    parts += ["-i", audio_path]
    if duration is not None:
        parts += ["-t", str(duration)]
    parts += ["-vn", "-acodec", "pcm_s16le", "-ar", "16000", "-ac", "1", out_path]

    result = subprocess.run(parts, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    if result.returncode != 0:
        try:
            os.remove(out_path)
        except OSError:
            pass
        stderr = result.stderr.decode("utf-8", errors="replace").strip()
        raise RuntimeError(
            f"ffmpeg failed to trim isolated track '{audio_path}' "
            f"(start={start_time}, duration={duration}). Stderr:\n{stderr}"
        )

    logger.info(
        "Trimmed isolated track '%s' → %s (start=%s, duration=%s)",
        audio_path,
        out_path,
        start_time,
        duration,
    )
    return out_path, out_path


_PYANNOTE_ACCESS_HELP = (
    "pyannote/voice-activity-detection could not be loaded. Fix:\n"
    "  1. Log in to https://huggingface.co and click 'Agree and access repository' at:\n"
    "     - https://huggingface.co/pyannote/voice-activity-detection\n"
    "     - https://huggingface.co/pyannote/segmentation\n"
    "  2. Make sure HF_TOKEN in your .env is a token from that same account\n"
    "     with 'Read' permission (create at https://huggingface.co/settings/tokens).\n"
    "  3. Restart the app so the new token is picked up."
)


_TORCH_LOAD_PATCHED = False


def _patch_torch_load_for_pyannote() -> None:
    """Force ``torch.load`` to use ``weights_only=False`` for the duration
    of a pyannote pipeline load.

    Torch 2.6 flipped ``torch.load`` default to ``weights_only=True``, which
    refuses to unpickle pytorch-lightning callbacks embedded in pyannote
    checkpoints. Adding globals one-by-one is a losing game (each class
    surfaces the next). Pyannote weights come from a trusted, gated HF
    repo we authenticate against with the user's own token, so unpickling
    is safe. Idempotent — safe to call repeatedly.
    """
    global _TORCH_LOAD_PATCHED
    if _TORCH_LOAD_PATCHED:
        return

    import torch

    _original_load = torch.load

    def _patched_load(*args, **kwargs):
        # lightning_fabric passes weights_only=None explicitly, so setdefault
        # doesn't help. Force False whenever caller didn't ask for True.
        if kwargs.get("weights_only") is not True:
            kwargs["weights_only"] = False
        return _original_load(*args, **kwargs)

    torch.load = _patched_load
    _TORCH_LOAD_PATCHED = True


def _load_vad_pipeline(device: Optional[str] = None):
    """Load pyannote/voice-activity-detection. Raises a clear error if
    HF_TOKEN is missing or user conditions are not accepted (in which case
    ``Pipeline.from_pretrained`` silently returns None).
    """
    from pyannote.audio import Pipeline  # local import — heavy dep
    import torch

    hf_token = os.environ.get("HF_TOKEN")
    if not hf_token:
        raise RuntimeError(
            "HF_TOKEN environment variable is not set.\n\n" + _PYANNOTE_ACCESS_HELP
        )

    _patch_torch_load_for_pyannote()

    try:
        pipeline = Pipeline.from_pretrained(
            "pyannote/voice-activity-detection",
            use_auth_token=hf_token,
        )
    except Exception as exc:
        raise RuntimeError(
            f"pyannote/voice-activity-detection failed to load: {exc}\n\n"
            + _PYANNOTE_ACCESS_HELP
        ) from exc

    if pipeline is None:
        raise RuntimeError(
            "pyannote/voice-activity-detection returned None — HF user "
            "conditions have not been accepted for this account.\n\n"
            + _PYANNOTE_ACCESS_HELP
        )

    if device:
        pipeline = pipeline.to(torch.device(device))
    return pipeline


def _run_vad(audio_path: str, device: Optional[str] = None) -> List[Tuple[float, float]]:
    """Return speech regions [(start, end), ...] in seconds, sorted."""
    pipeline = _load_vad_pipeline(device)
    logger.debug("Running VAD on %s", audio_path)
    vad_result = pipeline(audio_path)

    regions: List[Tuple[float, float]] = []
    for turn, _, _ in vad_result.itertracks(yield_label=True):
        start = float(turn.start)
        end = float(turn.end)
        if end - start >= _MIN_REGION_DURATION:
            regions.append((start, end))

    regions.sort(key=lambda r: r[0])
    return regions


def _extract_words(inner_segments: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
    """Flatten word-level timestamps from inner-transcriber segments.

    Deepgram and AssemblyAI populate ``segment['words']``; Gemini does not.
    Returns [] if no word-level data is present.
    """
    all_words: List[Dict[str, Any]] = []
    for seg in inner_segments:
        words = seg.get("words")
        if not isinstance(words, list):
            continue
        for w in words:
            if not isinstance(w, dict):
                continue
            start = w.get("start")
            end = w.get("end")
            text = w.get("word") or w.get("text")
            if start is None or end is None or not text:
                continue
            all_words.append({
                "word": str(text).strip(),
                "start": float(start),
                "end": float(end),
                "confidence": float(w.get("confidence")) if w.get("confidence") is not None else None,
            })

    all_words.sort(key=lambda w: w["start"])
    return all_words


def _merge_close_segments(
    segments: List[Dict[str, Any]],
    max_gap: float = _MERGE_MAX_GAP,
    max_duration: float = _SPLIT_MAX_DURATION,
) -> List[Dict[str, Any]]:
    """Glue same-speaker segments separated by a short VAD-pause.

    A VAD tuned for phone-level activity will happily split a single
    sentence in half at any breath. When the pause is under ``max_gap``
    and merging would not push the segment past ``max_duration`` (which
    ``_split_long_segments`` would then re-cut anyway), we merge them.
    Text is joined with a single space; words/confidence are concatenated.
    """
    if not segments:
        return segments
    ordered = sorted(segments, key=lambda s: (s["start"], s["end"]))
    merged: List[Dict[str, Any]] = [dict(ordered[0])]
    for seg in ordered[1:]:
        prev = merged[-1]
        same_speaker = prev.get("speaker") == seg.get("speaker")
        gap = seg["start"] - prev["end"]
        combined_duration = seg["end"] - prev["start"]

        # Skip merge when the previous segment ended on a sentence-final
        # punctuation mark: that's a strong signal the speaker finished a
        # thought, and gluing would blur meaning. Comma/dash/colon are
        # weaker (mid-thought) — we still merge on those.
        prev_text = str(prev.get("text") or "").strip()
        sentence_ended = prev_text.endswith(_SENTENCE_END_PUNCT)

        if (
            same_speaker
            and 0 <= gap <= max_gap
            and combined_duration <= max_duration * 1.5
            and not sentence_ended
        ):
            prev["end"] = seg["end"]
            prev["text"] = f"{prev['text']} {seg['text']}".strip()
            prev_words = list(prev.get("words") or [])
            prev_words.extend(seg.get("words") or [])
            if prev_words:
                prev["words"] = prev_words
            confs = [
                s.get("confidence") for s in (prev, seg)
                if s.get("confidence") is not None
            ]
            if confs:
                prev["confidence"] = sum(confs) / len(confs)
            continue
        merged.append(dict(seg))
    return merged


def _pick_split_index(words: List[Dict[str, Any]]) -> Optional[int]:
    """Return an index ``i`` such that ``words[:i]`` and ``words[i:]``
    form two well-balanced halves, split on the best word-boundary gap.
    Preference order: sentence-ending punctuation, then soft punctuation,
    then plain gap. Returns None if no gap is wide enough (<0.4s).
    """
    if len(words) < 4:
        return None

    # Consider only interior word boundaries.
    candidates: List[Tuple[int, float, int]] = []  # (index, gap, punct_rank)
    for i in range(1, len(words)):
        gap = words[i]["start"] - words[i - 1]["end"]
        if gap < _SPLIT_MIN_WORD_GAP:
            continue
        prev_word = str(words[i - 1].get("word") or "").strip()
        rank = 2
        if prev_word.endswith(_SENTENCE_END_PUNCT):
            rank = 0
        elif prev_word.endswith(_SOFT_END_PUNCT):
            rank = 1
        candidates.append((i, gap, rank))

    if not candidates:
        return None

    # Prefer the middle-most best-rank candidate — this keeps chunks
    # balanced. Sort by rank first, then distance from centre.
    center = len(words) / 2
    candidates.sort(key=lambda t: (t[2], abs(t[0] - center)))
    return candidates[0][0]


def _split_long_segments(
    segments: List[Dict[str, Any]],
    max_duration: float = _SPLIT_MAX_DURATION,
) -> List[Dict[str, Any]]:
    """Recursively split segments longer than ``max_duration``.

    Requires word-level timestamps — segments without ``words`` are left
    alone (Gemini-intersect path stays untouched). Splits at the best
    interior word gap; if no acceptable gap exists, keeps the segment.
    """
    result: List[Dict[str, Any]] = []
    queue: List[Dict[str, Any]] = [dict(s) for s in segments]

    while queue:
        seg = queue.pop(0)
        duration = seg["end"] - seg["start"]
        words = seg.get("words") or []
        if duration <= max_duration or not words:
            result.append(seg)
            continue

        idx = _pick_split_index(words)
        if idx is None:
            result.append(seg)
            continue

        left_words = words[:idx]
        right_words = words[idx:]
        if not left_words or not right_words:
            result.append(seg)
            continue

        left = {
            "text": " ".join(w["word"] for w in left_words).strip(),
            "start": seg["start"],
            "end": min(seg["end"], left_words[-1]["end"]),
            "speaker": seg["speaker"],
            "words": left_words,
        }
        right = {
            "text": " ".join(w["word"] for w in right_words).strip(),
            "start": max(seg["start"], right_words[0]["start"]),
            "end": seg["end"],
            "speaker": seg["speaker"],
            "words": right_words,
        }
        for half, half_words in ((left, left_words), (right, right_words)):
            confs = [w.get("confidence") for w in half_words if w.get("confidence") is not None]
            if confs:
                half["confidence"] = sum(confs) / len(confs)

        # Re-queue halves in order — they may still exceed max_duration.
        queue.insert(0, right)
        queue.insert(0, left)

    return result


def _segment_by_words(
    words: List[Dict[str, Any]],
    regions: List[Tuple[float, float]],
    speaker: str,
) -> List[Dict[str, Any]]:
    """Group word-level timestamps into segments bounded by VAD regions.

    Each output segment gets its ``start`` from the first word's start and
    ``end`` from the last word's end (both clipped by the region). Regions
    that contain no words are dropped — they are typically breaths or noise
    the transcriber decided not to spell.
    """
    segments: List[Dict[str, Any]] = []

    for region_start, region_end in regions:
        low = region_start - _WORD_VAD_TOLERANCE
        high = region_end + _WORD_VAD_TOLERANCE

        region_words = [
            w for w in words
            if w["end"] > low and w["start"] < high
        ]
        if not region_words:
            continue

        text = " ".join(w["word"] for w in region_words).strip()
        if not text:
            continue

        seg_start = max(region_start, region_words[0]["start"])
        seg_end = min(region_end, region_words[-1]["end"])
        if seg_end <= seg_start:
            # Word timestamps drifted past the VAD boundary; snap to region
            seg_start, seg_end = region_start, region_end

        confidences = [w["confidence"] for w in region_words if w["confidence"] is not None]
        segment: Dict[str, Any] = {
            "text": text,
            "start": seg_start,
            "end": seg_end,
            "speaker": speaker,
            "words": [
                {k: v for k, v in w.items() if v is not None}
                for w in region_words
            ],
        }
        if confidences:
            segment["confidence"] = sum(confidences) / len(confidences)
        segments.append(segment)

    return segments


def _segment_by_intersect(
    inner_segments: List[Dict[str, Any]],
    regions: List[Tuple[float, float]],
    speaker: str,
) -> List[Dict[str, Any]]:
    """Fallback for backends without word timestamps (Gemini).

    Takes each inner-transcriber segment and intersects its time span with
    the union of VAD regions. Segments that don't overlap any region are
    dropped (likely hallucinations on silence).
    """
    segments: List[Dict[str, Any]] = []
    for seg in inner_segments:
        try:
            seg_start = float(seg["start"])
            seg_end = float(seg["end"])
            text = str(seg.get("text") or "").strip()
        except (KeyError, TypeError, ValueError):
            continue
        if not text or seg_end <= seg_start:
            continue

        # Find any VAD region overlapping this segment
        overlapping = [
            (max(seg_start, rs), min(seg_end, re))
            for rs, re in regions
            if re > seg_start and rs < seg_end
        ]
        if not overlapping:
            continue

        clipped_start = min(s for s, _ in overlapping)
        clipped_end = max(e for _, e in overlapping)
        if clipped_end <= clipped_start:
            continue

        new_seg: Dict[str, Any] = {
            "text": text,
            "start": clipped_start,
            "end": clipped_end,
            "speaker": speaker,
        }
        confidence = seg.get("confidence")
        if confidence is not None:
            new_seg["confidence"] = float(confidence)
        segments.append(new_seg)

    return segments


def _build_inner_transcriber(
    inner_system: str,
    source_language: str,
    device: Optional[str],
    cache_manager: Optional["CacheManager"],
    inner_kwargs: Dict[str, Any],
):
    """Instantiate the per-track transcriber with diarize disabled.

    Deepgram accepts ``diarize=False`` directly. AssemblyAI's constructor
    doesn't take a diarize flag — its factory ignores unknown kwargs. Gemini
    is a monolithic prompt-based backend and ignores diarize hints; when
    used against a single-speaker clip it collapses to one speaker in the
    response, which is what we want.
    """
    from transcription.transcription_factory import TranscriptionFactory

    kwargs = dict(inner_kwargs)
    kwargs["cache_manager"] = cache_manager

    if inner_system == "deepgram":
        # Force a single-speaker transcription — we already know who's talking.
        kwargs.setdefault("diarize", False)

    return TranscriptionFactory.create_transcriber(
        transcription_system=inner_system,
        source_language=source_language,
        device=device,
        **kwargs,
    )


def run_isolated_tracks(
    tracks: Dict[str, str],
    inner_system: str,
    source_language: str,
    device: Optional[str] = None,
    cache_manager: Optional["CacheManager"] = None,
    inner_kwargs: Optional[Dict[str, Any]] = None,
    start_time: Optional[float] = None,
    duration: Optional[float] = None,
) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
    """Diarize + transcribe from isolated per-speaker tracks.

    Args:
        tracks: Mapping of speaker label → path to isolated audio file.
        inner_system: Underlying transcription backend
            (``deepgram`` | ``assemblyai`` | ``gemini``).
        source_language: Language code passed to the inner transcriber.
        device: torch device string (``cuda`` / ``cpu``) for VAD pipeline.
        cache_manager: Shared cache manager, passed through to the inner
            transcriber so its own per-file cache still works.
        inner_kwargs: Extra keyword arguments forwarded to the inner
            transcriber's constructor (e.g. ``deepgram_model``).
        start_time: If set, trim each isolated track to start at this
            second before VAD/transcription. Output segments are re-based
            to a t=0 origin, matching ``AudioProcessor.extract_audio``.
        duration: If set, keep only this many seconds after ``start_time``.

    Returns:
        (speakers_rolls, transcription) in the same schema the standard
        ``TranscriptionInterface.diarize_and_transcribe`` returns.
    """
    if not tracks:
        raise ValueError("run_isolated_tracks: 'tracks' mapping is empty")

    inner_kwargs = dict(inner_kwargs or {})

    all_segments: List[Dict[str, Any]] = []
    for speaker_label, audio_path in tracks.items():
        if not os.path.exists(audio_path):
            raise FileNotFoundError(
                f"Isolated track for speaker '{speaker_label}' not found: {audio_path}"
            )
        logger.info(
            "Processing isolated track speaker=%s path=%s",
            speaker_label,
            audio_path,
        )

        # Trim to the requested slice before VAD/transcription. The trimmed
        # clip starts at t=0, so downstream segment timestamps line up with
        # the main audio track that AudioProcessor.extract_audio produces.
        work_path, cleanup_path = _trim_track_if_needed(audio_path, start_time, duration)
        try:
            regions = _run_vad(work_path, device=device)
            if not regions:
                logger.warning(
                    "VAD found no speech regions in isolated track '%s' — skipping.",
                    audio_path,
                )
                continue

            inner_transcriber = _build_inner_transcriber(
                inner_system=inner_system,
                source_language=source_language,
                device=device,
                cache_manager=cache_manager,
                inner_kwargs=inner_kwargs,
            )
            _speakers_ignored, inner_segments = inner_transcriber.diarize_and_transcribe(
                audio_file=work_path,
                cache_key=None,
                use_cache=cache_manager.use_cache if cache_manager else True,
            )
        finally:
            if cleanup_path:
                try:
                    os.remove(cleanup_path)
                except OSError:
                    pass

        words = _extract_words(inner_segments)
        if words:
            track_segments = _segment_by_words(words, regions, speaker_label)
        else:
            track_segments = _segment_by_intersect(inner_segments, regions, speaker_label)

        raw_count = len(track_segments)
        # Merge short-pause splits, then re-cut anything still too long.
        # Merge first so same-speaker phrases the VAD chopped in half are
        # reunited; split second so the resulting long blocks are chunked
        # at natural sentence boundaries the TTS can pace comfortably.
        track_segments = _merge_close_segments(track_segments)
        merged_count = len(track_segments)
        track_segments = _split_long_segments(track_segments)

        logger.info(
            "Isolated track '%s': %d VAD regions → %d raw → %d merged → %d final (%s, %s)",
            speaker_label,
            len(regions),
            raw_count,
            merged_count,
            len(track_segments),
            inner_system,
            "word-based" if words else "segment-based",
        )
        all_segments.extend(track_segments)

    if not all_segments:
        raise RuntimeError(
            "Isolated-tracks path produced no transcription segments — check that "
            "the input files contain speech and that the inner transcriber is configured."
        )

    all_segments.sort(key=lambda s: (s["start"], s["end"], s["speaker"]))

    speakers_rolls: Dict[Tuple[float, float], str] = {
        (seg["start"], seg["end"]): seg["speaker"]
        for seg in all_segments
    }

    return speakers_rolls, all_segments
