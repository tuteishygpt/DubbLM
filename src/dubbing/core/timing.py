"""Deterministic anchor-based timing helpers for synthesized speech."""

from __future__ import annotations

import hashlib
import json
import math
from dataclasses import dataclass
from pathlib import Path
from typing import Any, Callable, Mapping, MutableMapping, Optional, Sequence

from pydub import AudioSegment


ANCHOR_TIMING_VERSION = "anchor_timing_v2"
TIMING_DEFAULTS = {
    "timing_short_segment_threshold": 1.5,
    "timing_short_segment_max_speed": 1.08,
    "timing_max_speed": 1.15,
    "timing_max_stretch": 1.15,
    "timing_max_overflow": 0.25,
}
TIMING_KEYS = tuple(TIMING_DEFAULTS)
SEMANTIC_DEFAULTS = {
    "semantic_split_enabled": True,
    "tts_preferred_segment_duration": 15.0,
    "tts_hard_segment_duration": 35.0,
    "semantic_split_search_window": 10.0,
}
SEMANTIC_KEYS = tuple(SEMANTIC_DEFAULTS)


@dataclass(frozen=True)
class TimingPolicy:
    short_segment_threshold: float = 1.5
    short_segment_max_speed: float = 1.08
    max_speed: float = 1.15
    max_stretch: float = 1.15
    max_overflow: float = 0.25

    @classmethod
    def from_config(cls, config: Mapping[str, Any]) -> "TimingPolicy":
        return cls(
            short_segment_threshold=float(config.get("timing_short_segment_threshold", 1.5)),
            short_segment_max_speed=float(config.get("timing_short_segment_max_speed", 1.08)),
            max_speed=float(config.get("timing_max_speed", 1.15)),
            max_stretch=float(config.get("timing_max_stretch", 1.15)),
            max_overflow=float(config.get("timing_max_overflow", 0.25)),
        )


@dataclass(frozen=True)
class AnchorWindow:
    segment: MutableMapping[str, Any]
    original_index: int
    start: float
    end: float
    next_start: Optional[float]
    available_window: float


@dataclass(frozen=True)
class SegmentTiming:
    available_window: float
    tempo: float
    expected_duration: float
    residual_overflow: float
    within_policy: bool


@dataclass(frozen=True)
class EdgeTrimResult:
    raw_duration: float
    trimmed_duration: float
    leading_removed: float
    trailing_removed: float
    usable: bool
    error: Optional[str] = None


def normalize_timing_config(
    config: MutableMapping[str, Any],
    *,
    warn: Optional[Callable[[str], None]] = None,
) -> None:
    """Normalize every timing entry point using one finite-value policy."""

    warning = warn or (lambda _message: None)
    for key, default in TIMING_DEFAULTS.items():
        value = config.get(key, default)
        valid = True
        try:
            normalized = float(value)
            valid = math.isfinite(normalized)
        except (TypeError, ValueError, OverflowError):
            valid = False
            normalized = default

        if valid:
            if key in {"timing_short_segment_threshold", "timing_max_overflow"}:
                valid = normalized >= 0.0
            else:
                valid = normalized >= 1.0

        if not valid:
            warning(f"Invalid {key} value {value!r}; using default {default}.")
            normalized = default
        config[key] = normalized

    legacy = config.get("group_overflow_tolerance", 1.0)
    try:
        legacy_is_default = math.isfinite(float(legacy)) and float(legacy) == 1.0
    except (TypeError, ValueError, OverflowError):
        legacy_is_default = False
    if not legacy_is_default:
        warning(
            "group_overflow_tolerance is deprecated and ignored; "
            "use timing_max_overflow instead."
        )

    enabled = config.get("semantic_split_enabled", True)
    if isinstance(enabled, bool):
        normalized_enabled = enabled
    elif isinstance(enabled, (int, float)) and enabled in (0, 1):
        normalized_enabled = bool(enabled)
    elif isinstance(enabled, str) and enabled.strip().lower() in {"true", "false", "1", "0"}:
        normalized_enabled = enabled.strip().lower() in {"true", "1"}
    else:
        warning(
            f"Invalid semantic_split_enabled value {enabled!r}; using default True."
        )
        normalized_enabled = True
    config["semantic_split_enabled"] = normalized_enabled

    normalized_semantic: dict[str, float] = {}
    for key in (
        "tts_preferred_segment_duration",
        "tts_hard_segment_duration",
        "semantic_split_search_window",
    ):
        default = SEMANTIC_DEFAULTS[key]
        value = config.get(key, default)
        try:
            normalized = float(value)
            valid = math.isfinite(normalized)
        except (TypeError, ValueError, OverflowError):
            normalized, valid = default, False
        if valid:
            valid = normalized >= 0.0 if key == "semantic_split_search_window" else normalized > 0.0
        if not valid:
            warning(f"Invalid {key} value {value!r}; using default {default}.")
            normalized = default
        normalized_semantic[key] = normalized

    if normalized_semantic["tts_hard_segment_duration"] < normalized_semantic["tts_preferred_segment_duration"]:
        warning(
            "Invalid tts_hard_segment_duration: it must be greater than or equal "
            "to tts_preferred_segment_duration; using default semantic durations."
        )
        normalized_semantic["tts_preferred_segment_duration"] = float(
            SEMANTIC_DEFAULTS["tts_preferred_segment_duration"]
        )
        normalized_semantic["tts_hard_segment_duration"] = float(
            SEMANTIC_DEFAULTS["tts_hard_segment_duration"]
        )
    config.update(normalized_semantic)


def plan_anchor_windows(
    segments: Sequence[MutableMapping[str, Any]],
    source_duration: float,
) -> list[AnchorWindow]:
    """Validate all timestamps, then return stable chronological anchor windows."""

    try:
        source_duration = float(source_duration)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Source duration must be a finite non-negative timestamp") from exc
    if not math.isfinite(source_duration) or source_duration < 0.0:
        raise ValueError("Source duration must be a finite non-negative timestamp")

    validated: list[tuple[float, float, int, MutableMapping[str, Any]]] = []
    for current_index, segment in enumerate(segments):
        stored_index = segment.get("_timing_original_index")
        original_index = (
            stored_index
            if isinstance(stored_index, int) and stored_index >= 0
            else current_index
        )
        try:
            start = float(segment["start"])
            end = float(segment["end"])
        except (KeyError, TypeError, ValueError, OverflowError) as exc:
            raise ValueError(f"Segment {original_index} has an invalid timestamp") from exc
        if not math.isfinite(start) or not math.isfinite(end) or start < 0.0 or end < start:
            raise ValueError(f"Segment {original_index} has an invalid timestamp range: {start!r}..{end!r}")
        validated.append((start, end, original_index, segment))

    ordered = sorted(validated, key=lambda item: (item[0], item[1], item[2]))
    distinct_starts = sorted({item[0] for item in ordered})
    next_by_start = {
        start: (distinct_starts[index + 1] if index + 1 < len(distinct_starts) else None)
        for index, start in enumerate(distinct_starts)
    }

    planned: list[AnchorWindow] = []
    for start, end, original_index, segment in ordered:
        next_start = next_by_start[start]
        window_end = max(end, next_start) if next_start is not None else source_duration
        window_end = min(window_end, source_duration)
        available_window = max(0.0, window_end - start)
        planned.append(
            AnchorWindow(
                segment=segment,
                original_index=original_index,
                start=start,
                end=end,
                next_start=next_start,
                available_window=available_window,
            )
        )
    return planned


def calculate_segment_timing(
    *,
    start: float,
    end: float,
    next_start: Optional[float],
    source_duration: float,
    audio_duration: float,
    policy: TimingPolicy,
    epsilon: float = 1e-9,
) -> SegmentTiming:
    """Select the least tempo change permitted by the anchor timing policy."""

    window_end = max(end, next_start) if next_start is not None else source_duration
    window_end = min(window_end, source_duration)
    available_window = max(0.0, window_end - start)
    allowed_duration = available_window + policy.max_overflow
    recognized_duration = end - start
    speed_limit = (
        policy.short_segment_max_speed
        if recognized_duration < policy.short_segment_threshold
        else policy.max_speed
    )
    if audio_duration <= epsilon or available_window <= epsilon:
        tempo = 1.0
    elif audio_duration < available_window:
        # Fill as much unused anchor time as the slowdown policy permits. The
        # caller has already trimmed edge silence, so only audible speech is
        # stretched; the immutable start anchor is preserved.
        required_tempo = audio_duration / available_window
        tempo = max(required_tempo, 1.0 / policy.max_stretch)
    else:
        required_speed = max(1.0, audio_duration / max(allowed_duration, epsilon))
        tempo = min(required_speed, speed_limit)
    expected_duration = audio_duration / tempo
    residual_overflow = max(0.0, expected_duration - available_window)
    return SegmentTiming(
        available_window=available_window,
        tempo=tempo,
        expected_duration=expected_duration,
        residual_overflow=residual_overflow,
        within_policy=residual_overflow <= policy.max_overflow + 0.002,
    )


def trim_audio_edges(
    source_path: str | Path,
    output_path: str | Path,
    *,
    silence_threshold_db: float = -40.0,
    keep_head_ms: int = 50,
    keep_tail_ms: int = 100,
    window_ms: int = 10,
) -> EdgeTrimResult:
    """Write a non-destructive edge-trimmed WAV while retaining safety margins."""

    source_path = Path(source_path)
    output_path = Path(output_path)
    if source_path.resolve() == output_path.resolve():
        return EdgeTrimResult(0.0, 0.0, 0.0, 0.0, False, "Output path must differ from raw input")
    try:
        output_path.unlink()
    except FileNotFoundError:
        pass
    except OSError as exc:
        return EdgeTrimResult(0.0, 0.0, 0.0, 0.0, False, str(exc))
    try:
        audio = AudioSegment.from_file(source_path)
    except Exception as exc:
        return EdgeTrimResult(0.0, 0.0, 0.0, 0.0, False, str(exc))

    total_ms = len(audio)
    active_windows: list[tuple[int, int]] = []
    for start_ms in range(0, total_ms, window_ms):
        end_ms = min(start_ms + window_ms, total_ms)
        level = audio[start_ms:end_ms].dBFS
        if level != float("-inf") and level > silence_threshold_db:
            active_windows.append((start_ms, end_ms))

    if not active_windows:
        return EdgeTrimResult(total_ms / 1000.0, 0.0, 0.0, 0.0, False)

    trim_start = max(0, active_windows[0][0] - keep_head_ms)
    trim_end = min(total_ms, active_windows[-1][1] + keep_tail_ms)
    leading_removed_ms = trim_start
    trailing_removed_ms = total_ms - trim_end
    trimmed = audio[trim_start:trim_end]
    try:
        output_path.parent.mkdir(parents=True, exist_ok=True)
        trimmed.export(output_path, format="wav")
    except Exception as exc:
        return EdgeTrimResult(
            total_ms / 1000.0,
            total_ms / 1000.0,
            0.0,
            0.0,
            True,
            str(exc),
        )
    return EdgeTrimResult(
        raw_duration=total_ms / 1000.0,
        trimmed_duration=len(trimmed) / 1000.0,
        leading_removed=leading_removed_ms / 1000.0,
        trailing_removed=trailing_removed_ms / 1000.0,
        usable=True,
    )


def timing_cache_fingerprint(policy: TimingPolicy) -> str:
    payload = {
        "algorithm": ANCHOR_TIMING_VERSION,
        "timing_short_segment_threshold": policy.short_segment_threshold,
        "timing_short_segment_max_speed": policy.short_segment_max_speed,
        "timing_max_speed": policy.max_speed,
        "timing_max_stretch": policy.max_stretch,
        "timing_max_overflow": policy.max_overflow,
    }
    serialized = json.dumps(payload, sort_keys=True, separators=(",", ":"))
    digest = hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:12]
    return f"{ANCHOR_TIMING_VERSION}_{digest}"
