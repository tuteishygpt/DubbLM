# Segment-Bounded Sequential TTS Design

## Goal

Fit each synthesized utterance to the recognized bounds of that same segment.
Choose among translated-text variants using measured synthesized audio before
applying any tempo correction.

## Current behavior

The timing planner currently extends a segment's usable window to the next
distinct segment start when that start is later than the current recognized
end. The last segment may use all remaining source duration. Duration
estimation then selects an initial text variant using a broad comfort ratio.
After synthesis, best-effort resynthesis may try other variants, and final
assembly applies bounded tempo correction. A clip that still exceeds the
window is logged and overlaid without truncating intelligible speech.

This creates two problems:

1. A segment can consume a pause that belongs outside its recognized
   `start..end` interval.
2. Estimated-duration selection and the `0.75..1.15` comfort band can accept a
   candidate that final bounded tempo correction cannot fit.

Gemini TTS also validates every generated segment against a fixed one-second
minimum. Legitimate sub-second utterances are therefore retried even though
they are valid inputs to the dubbing timeline.

## Selected behavior

### Segment-local timing bounds

For every valid segment:

```text
target_duration = max(0, end - start)
```

The next segment's start never affects the target used by duration selection,
TTS, resynthesis, stretching, speeding, or final timing diagnostics. The same
rule applies to the last segment; it does not inherit the remaining source
duration.

Chronological sorting, stable original indexes, equal-start support, timestamp
validation, and final clipping at the physical source-audio boundary remain
unchanged. The active timing path neither calculates nor stores a next-segment
anchor.

### Sequential measured candidate selection

Candidate selection uses audible WAV duration measured through the existing
non-destructive edge-silence trimming. Provider estimates do not select the
winning text.

For each segment, synthesis proceeds as follows:

1. Synthesize `translation` and measure its audible duration.
2. If it is shorter than `target_duration`, synthesize
   `long_translation` when that field exists and contains distinct non-empty
   text. Do not synthesize short variants on this branch.
3. If the initial translation is longer than
   `target_duration + timing_max_overflow`, synthesize `short_translation`
   when available.
4. If the measured short candidate is still longer than
   `target_duration + timing_max_overflow`, synthesize
   `very_short_translation` when available.
5. Stop as soon as the applicable branch is exhausted. Missing, blank, or
   duplicate variants are skipped without an error.
6. Choose the generated candidate with the smallest absolute difference from
   `target_duration`. A tie preserves generation order, preferring the normal
   translation, then long or short, then very short.

Only candidates required by this decision tree are generated. For example, a
normal candidate already within `target_duration..target_duration + overflow`
does not trigger another TTS call. A short candidate that is no longer too long
does not trigger the very-short candidate.

Automatic LLM text-length rewriting is not part of this selection path. The
defined translated variants are authoritative, making generation order,
provider cost, caching, and chosen text deterministic.

### Tempo correction after selection

Tempo correction runs only after the winning measured candidate has been
selected:

- clips shorter than the segment may be stretched up to
  `timing_max_stretch`;
- clips longer than the segment plus allowed overflow may be sped up to
  `timing_short_segment_max_speed` for recognized segments shorter than
  `timing_short_segment_threshold`, otherwise `timing_max_speed`;
- `timing_max_overflow` remains allowed and defaults to `0.25` seconds;
- intelligible speech is not hard-truncated at the recognized segment end;
- unresolved overflow remains explicit in logs and diagnostics.

The existing `1.08x` short-segment and `1.15x` normal-segment default speed
limits are retained. They are independent of candidate selection.

### Gemini sub-second segments

Gemini's segment-synthesis validation no longer supplies a fixed
`expected_min_duration=1.0`. Segment validation continues to reject missing,
unreadable, extremely small, flat, or energy-free audio and continues to
measure trailing silence. It does not reject an otherwise valid segment only
because its total or audible duration is below one second.

For normal dubbing segments, excessive trailing silence is recoverable whenever
the validator found at least one non-silent analysis frame and did not classify
the file as flat or energy-free. Such a take is accepted without a provider
retry because `SmartDubbing` removes edge silence before duration selection and
assembly. This deliberately makes trailing-silence ratio diagnostic rather
than rejection-worthy for segment synthesis, while silence-only output still
fails. Separate voice-library sample generation retains its stricter duration
and trailing-silence validation.

The one-second minimum used when choosing optional translated speaker samples
for reports remains unchanged because it does not control TTS generation. Any
minimum used by separate voice-library sample generation also remains outside
this change.

## Components

### `dubbing.core.timing`

The timing planner owns segment-local target calculation. Its public result
continues to expose `available_window` for compatibility, but the value is now
the current recognized duration. `calculate_segment_timing` uses the same
segment-local duration and no longer derives a window from `next_start`.

The timing algorithm/cache version must change so final audio created under
the previous next-anchor policy is not reused.

### `dubbing.core.smart_dubbing`

`SmartDubbing.synthesize_speech` owns the sequential candidate state machine.
It synthesizes or loads raw cached candidates one at a time, measures each,
records the chosen text and file, then invokes the existing anchor assembler.
The assembler performs edge trimming and tempo correction on only the chosen
candidate.

Raw-candidate cache entries are keyed by exact candidate text plus the existing
input, segment, speaker, voice/profile/provider, and reference identities. The
aggregate `synthesized_speech` cache fingerprint adds the new timing/selection
version, every segment's `start` and `end`, and all available candidate texts.
This invalidates old assembled audio and edits without adding another cache
layer. Explicit single-row regeneration continues to overwrite its normal
`audio_chunks/<index>.wav`; the existing chunk-based rebuild consumes that file
directly and does not load the aggregate WAV cache.

Single-row resynthesis continues to synthesize the user-selected override or
translation directly. Rebuilding the full dubbed track applies segment-local
tempo correction to that regenerated chunk; it does not silently generate
additional variants during the explicit single-row operation.

### `tts.gemini_tts_wrapper`

Gemini segment attempts call audio validation without a fixed duration floor.
Voice sample generation retains its own validation contract. Failed provider
calls, empty output, unreadable files, and silence-only output keep their
existing retry/fallback behavior.

## Error handling

- Invalid, negative, inverted, NaN, or infinite timestamps continue to fail
  before partial synthesis.
- A zero-duration segment may synthesize normally. Results up to
  `timing_max_overflow` are accepted without short-variant generation; a result
  exceeding overflow follows the normal short, then conditional very-short,
  branch. The winning candidate is still the one closest to the zero target.
- A failed optional candidate does not replace a valid earlier candidate. The
  best successfully generated candidate is retained.
- If the initial translation produces no usable candidate, existing provider
  retry and missing-segment fallback behavior remains authoritative.
- FFmpeg tempo failure retains the unmodified winning candidate and reports
  actual unresolved overflow.
- Final output duration remains exactly the processed source-audio duration.

## Diagnostics

The existing timing report continues to show recognized bounds, target and
actual duration, tempo, overflow, and policy status. It adds only the selected
variant name; no new report is introduced.

## Testing

Focused tests will cover:

1. A segment followed by a long pause gets `end - start`, not the next start,
   as its target.
2. The last segment gets `end - start`, not the remainder of source audio.
3. Equal starts and overlapping speakers retain their recognized bounds.
4. A short normal candidate triggers only the long candidate and chooses the
   closer measured WAV.
5. A long normal candidate triggers short, then triggers very short only when
   short remains beyond target plus overflow.
6. Candidate ties preserve generation order.
7. Missing, blank, duplicate, cached, and failed optional variants follow the
   deterministic fallback rules.
8. No tempo correction is applied before candidate selection; the chosen WAV
   receives bounded stretch or speed afterward.
9. Allowed overflow and unresolved-overflow diagnostics remain correct.
10. A zero-duration target respects the overflow branch boundary.
11. A valid Gemini segment shorter than one second is accepted without a
    duration-driven retry, including a sub-second clip with removable trailing
    silence and at least one non-silent analysis frame.
12. Gemini still rejects empty, unreadable, flat, and silence-only output.
13. Raw-candidate reuse and aggregate fingerprint invalidation are covered.
14. Explicit single-row regeneration synthesizes exactly the selected text,
    generates no variants, and chunk rebuilding applies segment-local tempo
    without restoring a cached aggregate WAV.
15. Full anchor-timing, Gemini validation, runner, and TTS wrapper regression
    suites pass.

## Non-goals

- Hard-cutting speech at the recognized segment end.
- Removing the configured overflow allowance.
- Removing separate speed limits for short and normal recognized segments.
- Generating every translation variant unconditionally.
- Changing translation prompts or how translation variants are authored.
- Changing speaker-reference or report-sample minimum durations.
