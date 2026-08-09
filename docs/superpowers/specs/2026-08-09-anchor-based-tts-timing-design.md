# Anchor-Based TTS Timing Design

## Goal

Keep every dubbed utterance anchored to its recognized start time without
forcing short utterances through destructive speed changes. Use available
silence before the next recognized utterance as timing slack, and apply tempo
changes only to synthesized speech, never to pauses.

## Current problem

`SmartDubbing._adjust_and_combine_audio_grouped` currently concatenates all
synthesized clips for one speaker, inserts the original gaps between them, and
applies one FFmpeg `atempo` filter to the complete group. This keeps the group
boundary approximately aligned but scales the gaps as well as the speech, so
internal utterance starts move away from their recognized timestamps.

Short segments have the opposite failure mode when fitted independently: a
small absolute duration error can require an audibly large tempo change. The
pipeline therefore needs fixed start anchors plus flexible use of the silence
that follows an utterance.

## Selected approach

Replace group-level time stretching with anchor-based, per-segment placement.
Groups remain available for diagnostics, but they no longer determine audio
timing.

For every segment:

1. Validate timestamps, sort all segments chronologically, and assign a stable
   original index.
2. Build the anchor window before duration estimation, translation-variant
   selection, or synthesis. All of those stages use the anchor window instead
   of raw `end - start` as their timing target.
3. Trim leading and trailing synthesized silence while retaining a small safety
   margin around audible speech.
4. Anchor the clip at the recognized `start` timestamp.
5. If the trimmed clip fits the usable window plus allowed overflow, do not
   change its tempo.
6. If it does not fit, speed up the speech only, subject to the configured
   maximum. Never put inserted pauses through `atempo`.
7. Allow a small configured overflow after the window. Larger unresolved
   overflow is logged explicitly; the existing shorter-translation and
   resynthesis machinery remains the primary way to reduce it before assembly.
8. Overlay the resulting clip at the original `start` on the final timeline.

This design prioritizes stable starts and natural short utterances. It does not
silently truncate intelligible speech.

## Timing policy

The default timing policy is:

- A short segment has a recognized duration below `1.5` seconds.
- Short segments may be sped up by at most `1.08x`.
- Other segments may be sped up by at most `1.15x`.
- Up to `0.25` seconds of overflow beyond the usable window is accepted.
- A clip that fits its usable window is left at its natural speed, even when it
  is longer than the recognized `end - start` duration.

FFmpeg's `atempo` value is the speed multiplier. For example, `atempo=1.08`
shortens a clip to approximately `duration / 1.08`.

### Deterministic timing formula

All timing calculations use finite seconds as floats. Before any synthesis,
reject the complete segment list with `ValueError` if a segment has a non-finite
timestamp, a negative start, or `end < start`. This prevents a corrupt cache or
edited row from producing a partially valid output. Equal starts are valid and
represent intentional overlap.

Sort by `(start, end, original_index)`. For a segment at `start_i`, find the
next **distinct** chronological start strictly greater than `start_i`; segments
with equal starts share the same later anchor. Let `source_duration` be the
duration of the extracted/processed source audio.

The nominal window end is:

```text
if a later distinct start exists:
    window_end = max(end_i, next_distinct_start)
else:
    window_end = source_duration

window_end = min(window_end, source_duration)
available_window = max(0, window_end - start_i)
```

Using `max(end_i, next_distinct_start)` preserves intentional source overlap:
when another speaker starts before this segment ends, this segment retains its
recognized duration. For a normal gap, the segment may consume that gap without
moving the next anchor. The last segment may use the remaining source duration.

For a trimmed TTS duration `audio_duration`:

```text
allowed_duration = available_window + timing_max_overflow
required_speed = max(1.0, audio_duration / max(allowed_duration, epsilon))
speed_limit = timing_short_segment_max_speed
              if (end_i - start_i) < timing_short_segment_threshold
              else timing_max_speed
chosen_speed = min(required_speed, speed_limit)
expected_duration = audio_duration / chosen_speed
residual_overflow = max(0, expected_duration - available_window)
within_policy = residual_overflow <= timing_max_overflow + 0.002
```

Thus allowed overflow reduces the required tempo; a clip already inside the
window plus overflow remains at natural speed. Position and final duration
comparisons are rounded to the nearest millisecond. A two-millisecond tolerance
covers WAV/FFmpeg rounding.

The TTS `target_duration`, estimated-duration comfort check, translation variant
selection, and resynthesis decision all use `available_window`, not the raw
recognized duration. This is required so a natural 0.8-second clip inside a
1.1-second anchor window is not shortened or resynthesized merely because the
recognized speech ended after 0.5 seconds.

## Configuration and interface

Add four configuration fields with validation and defaults:

| Config key | Default | Meaning |
| --- | ---: | --- |
| `timing_short_segment_threshold` | `1.5` | Recognized duration below which the short-segment speed limit applies, in seconds. |
| `timing_short_segment_max_speed` | `1.08` | Maximum FFmpeg tempo multiplier for short segments. |
| `timing_max_speed` | `1.15` | Maximum FFmpeg tempo multiplier for other segments. |
| `timing_max_overflow` | `0.25` | Allowed audio overflow beyond the usable window, in seconds. |

Expose the same four values in the Gradio advanced timing/audio settings. They
participate in the existing settings save/load flow and can also be supplied by
YAML, CLI arguments, and programmatic overrides.

Validation rules:

- Threshold and overflow must be non-negative.
- Speed multipliers must be at least `1.0`.
- Invalid YAML/programmatic values, including NaN and infinity, fall back to
  defaults with a warning.
- Gradio uses numeric fields with appropriate precision.

One shared normalization helper applies these rules to YAML, CLI, Gradio-loaded
defaults, and programmatic overrides so entry points cannot disagree.

The legacy `group_overflow_tolerance` setting is retained for one compatibility
cycle as a deprecated, ignored alias. Loading a non-default value logs a warning
that points to `timing_max_overflow`; it is removed from the Gradio interface and
newly saved settings, and no timing code reads it.

Silence detection thresholds and safety margins remain internal implementation
constants. They are not exposed in the interface in this iteration.

## Components

### Timing configuration

`src/dubbing/core/config.py` owns defaults, validation, and CLI arguments.
`src/dubbing/ui/gradio_app.py` exposes the four settings and includes them in
the existing ordered input component list.

### Edge-silence trimming

Generalize the existing trailing-silence helper into an edge-silence helper.
It measures audible windows, retains small head and tail margins, rewrites the
WAV to a separate timing-adjusted path, and returns raw duration, trimmed
duration, and the amount removed from each edge. Raw synthesized/cache files are
not overwritten, so later timing-setting changes can reuse TTS without losing
diagnostics. The assembly path re-measures every produced file.

Legacy cached chunks may already have been tail-trimmed by the old algorithm.
They remain reusable: diagnostics mark raw duration and removed-edge values as
unknown, then treat the readable cached duration as the trimming input. A cache
metadata/version field distinguishes newly generated raw chunks. An all-silent
clip is treated as unusable synthesized audio and follows the existing
recognized-duration silence fallback. Trimming failures keep the raw clip and
record zero removed edges plus the error message.

### Per-segment timing calculation

Introduce a small, deterministic timing calculation helper. Given recognized
start/end times, the next chronological start, trimmed audio duration, and the
timing policy, it returns:

- available window duration;
- chosen tempo multiplier;
- expected adjusted duration;
- overflow duration;
- whether the overflow exceeds policy.

Keeping this calculation independent of FFmpeg makes edge cases easy to unit
test.

### Timeline assembly

Replace speech concatenation and group-level tempo adjustment with:

1. chronological planning;
2. optional per-clip `atempo`;
3. measurement of the actual adjusted WAV;
4. overlay at `round(start * 1000)`;
5. a final exact pad or trim to the processed source-audio duration.

Separate speaker tracks may remain to preserve intentional overlapping speech,
but their segments use the shared chronological timing plan.

The final duration invariant takes precedence at the physical source boundary.
If the last segment still crosses `source_duration` after maximum speedup, the
final trim necessarily truncates its tail. This is the only permitted speech
cut: it is logged as a boundary truncation with the removed milliseconds and is
written to timing diagnostics. The pipeline does not extend the video in this
iteration.

### Cache behavior

Raw per-segment synthesis caches remain reusable across timing-policy changes.
The final `synthesized_speech` cache key includes all four timing settings plus
an `anchor_timing_v1` algorithm version, so it can never bypass reassembly under
a changed policy. Changing only timing settings reuses raw chunks and rebuilds
the final track. The segment cache also records whether a file is raw under the
new trimming contract; legacy entries use the compatibility behavior described
above.

## Diagnostics

Generate `artifacts/debug/timing_alignment.tsv` when debug information is
enabled. Each row contains:

- segment index and speaker;
- recognized start and end;
- next anchor start;
- raw and trimmed TTS durations;
- leading and trailing silence removed;
- tempo multiplier;
- actual final duration;
- final start and end;
- overflow and policy status.

Warnings identify segments that remain beyond the allowed overflow after the
maximum permitted speed adjustment.

## Error handling

- If FFmpeg tempo adjustment fails, use the unmodified trimmed clip, record
  `tempo=1.0`, and calculate diagnostics from its actual duration.
- Missing or unreadable TTS files continue to produce silence for the recognized
  segment duration so later anchors remain stable.
- Zero-length recognized ranges are valid and use the next anchor as available
  slack. Invalid ranges fail validation before synthesis.
- Audio is never silently cut merely to satisfy the overflow limit.

## Testing

Add focused tests covering:

1. A 0.5-second recognized segment with a 0.8-second TTS clip and a 1.1-second
   next-anchor window: no speed change, no shorter-translation selection or
   resynthesis, and exact next start.
2. A short overflowing segment: speed is capped at `1.08x` and remaining
   overflow is reported.
3. A normal overflowing segment: speed is capped at `1.15x`.
4. Leading and trailing silence trimming with retained safety margins.
5. Multiple segments whose starts remain equal to recognized starts after
   assembly.
6. Original overlapping speakers remain overlapped.
7. FFmpeg failure uses actual unmodified duration in diagnostics.
8. Final output duration exactly matches the processed source audio.
9. Configuration validation, CLI parsing, UI defaults, and settings persistence
   for all four timing parameters.
10. Last-segment windows and explicit source-boundary truncation diagnostics.
11. Exact overflow boundaries, equal-start segments, zero-length segments, and
    rejection of negative, inverted, NaN, and infinite timestamps.
12. Final-cache keys change with every timing setting and the algorithm version,
    while raw segment caches remain reusable.
13. Raw and legacy cached chunks produce defined trimming diagnostics.

The existing full test suite remains the regression gate. A real-artifact check
compares `transcription.srt`, generated chunks, and final output duration for the
latest project and reports timing metrics without requiring language-dependent
waveform correlation.

## Non-goals

- Word- or phoneme-level forced alignment.
- Lip-shape synchronization.
- Automatically rewriting translations during final assembly.
- Exposing low-level silence detector settings in the UI.
