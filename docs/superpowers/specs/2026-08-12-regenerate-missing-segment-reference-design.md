# Isolated-Track-Aware Segment Reference Design

## Goal

Make every `reference_mode: segment` path use a speaker's configured isolated
track when one exists. This applies to the normal `full_pipeline`, `tts_to_end`,
and `Regenerate selected row` paths. The button must also be able to synthesize
a row when its segment-specific reference WAV does not yet exist. Reference
audio and its recognized transcription must come from the same finalized
segment after semantic or legacy merge/split.

## Scope

This change affects segment-reference source selection in normal synthesis and
single-row resynthesis. Existing reference WAV files remain reusable and are
not overwritten by the button's missing-file fallback. Other reference modes
(`configured`, `speaker`, and `none`) remain unchanged.

## Ownership and Interface

`SmartDubbing` owns one shared segment-reference preparation helper because
this is TTS-domain behavior and must not be duplicated in the Gradio UI. Both
`synthesize_speech()` and `resynthesize_one_segment()` call it while resolving
`reference_mode: segment`. The helper selects and validates the source, applies
its timestamp offset, validates the interval and minimum duration, slices and
exports the WAV, attaches `reference_text`, and formats preparation errors. A
`reuse_existing` parameter is the only entry-point-specific policy.

Normal synthesis caches decoded audio by `(source path, offset)` during the run
so multiple segments from the same isolated track do not repeatedly decode it.

The Gradio handler continues to supply the complete cached segment list, row
index, and optional text override. It does not slice audio itself.

## Reference Source Selection

For every segment whose resolved voice profile has `reference_mode: segment`:

1. Derive the canonical index from a non-negative integer
   `_timing_original_index`; otherwise use the segment's current chronological
   list index. Resolve the output as
   `speakers_audio/segments/<speaker>_<canonical_index>.wav`.
2. `full_pipeline` and `tts_to_end` call the helper with
   `reuse_existing=False`, so the file is always rebuilt from the currently
   selected source. If `isolated_tracks` contains the speaker, use exactly
   that mapped track.
3. `Regenerate selected row` calls the helper with `reuse_existing=True`. An
   existing readable WAV whose duration satisfies
   `segment_reference_min_duration` is returned immediately without validating
   or opening any source. If it is absent, unreadable, empty, or too short,
   prepare it from the currently selected source; a mapped isolated track has
   priority.
4. When preparation is required, if an isolated track is mapped but is missing,
   unreadable, or does not
   contain the requested interval, raise an actionable error and stop before
   invoking TTS. Do not fall back to a mixed source in this case.
5. If no isolated track is mapped for the speaker, try the processed
   `vocals.wav` only when `keep_background` is enabled. It is usable only when
   it exists, decodes successfully, and covers the interval. Otherwise try
   `source.wav`; failure of the final source raises before TTS.

This ordering prevents a speaker with an explicitly supplied clean track from
silently receiving another speaker's or mixed-program audio as its reference.

## Timestamp Handling

Finalized dubbing segments use timestamps rebased to the processed input slice.
`source.wav` and `vocals.wav` are already rebased, so their slice is
`segment.start..segment.end`.

Configured isolated tracks point to the original untrimmed files. In every
path, when `start_time` is set, the helper adds that offset to both segment
bounds before slicing the isolated track. `duration` needs no additional
offset; the finalized row bounds already describe the selected processed
window. Bounds must be finite, non-negative, ordered, and inside the selected
audio source.

## Cache Identity

For speakers using `reference_mode: segment`, the effective TTS and raw-segment
cache identities include the selected source path/content identity,
`start_time` offset, finalized `start/end`, and `reference_text`. Therefore a
previous mixed-source or older isolated-track result cannot bypass current
source preparation or validation. `tts_to_end` continues to disable and clear
TTS caches as it does today.

## Audio/Text Pairing

Whenever a reference is prepared or rebuilt, the helper exports it using the
selected finalized segment's `start` and `end`, adjusted only by the selected
source's offset. It supplies `segment["text"]` as `reference_text`. Merge and
split happen upstream, so both values describe the same finalized semantic or
legacy unit.

Editing only `Translation`, `Synthesized text`, or `Style instructions` does
not affect the reference pair. The button leaves a valid existing reference
WAV unchanged; changing `Original`, `Start`, `End`, or `Speaker` after a
reference has already been created is outside its missing-file fallback and
continues to require a normal full reference rebuild.

## Error Handling

Before TTS is called, every synthesis path reports a clear error containing
the speaker, segment index, reference mode, source path, and failure reason for
these cases:

- mapped isolated-track file missing;
- mapped isolated track unreadable;
- fallback vocals missing, unreadable, or unable to cover the interval (the
  helper then tries `source.wav`);
- final `source.wav` missing, unreadable, or unable to cover the interval
  (the helper raises an actionable error);
- invalid segment timestamps;
- segment interval outside the selected source;
- exported duration below `segment_reference_min_duration`;
- failed or missing WAV after export.

No TTS retries occur for reference-preparation failures. Normal synthesis keeps
its existing preflight behavior: reference-preparation errors are collected for
all segments and raised together before any TTS pool starts.

## Testing

Add focused regression coverage proving that:

- normal `full_pipeline`/`tts_to_end` segment-reference export uses the mapped
  isolated speaker track rather than the mixed source or separated vocals;
- normal synthesis rebuilds an existing reference from the current source;
- a missing segment reference requested by the button is exported from the
  mapped isolated speaker track and paired with the segment's recognized
  `text`;
- `start_time` is applied when slicing an original isolated track;
- `start_time` is covered in both normal and single-row paths;
- a missing or unreadable explicitly mapped track raises before TTS and never
  falls back to `vocals.wav` or `source.wav`;
- without a mapped isolated track, `vocals.wav` is preferred when available;
- without an isolated track or usable vocals artifact, `source.wav` is used;
- an existing segment reference is reused without reopening or rewriting its
  source;
- an existing unreadable, empty, or too-short reference is rebuilt;
- an existing valid row reference is reused even if its mapped isolated track
  later becomes invalid;
- cached reloads use `_timing_original_index` consistently, and changing
  isolated-track content invalidates applicable TTS caches;
- the existing minimum-duration and preflight validation behavior remains in
  force.
