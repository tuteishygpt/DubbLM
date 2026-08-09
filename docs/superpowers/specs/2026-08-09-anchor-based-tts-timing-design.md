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

## Semantic segmentation for isolated speaker tracks

### Current isolated-track failure

The isolated-track path already runs VAD independently for every supplied
speaker track. It then maps ASR words onto VAD regions, merges close regions,
and finally calls `_split_long_segments`. The last step currently treats 15
seconds as a hard duration target and may choose any word gap of at least 0.4
seconds. As a result, a VAD breath inside an unfinished sentence can become a
new translation and TTS segment. This is the source of the observed splits
after phrases such as `I think that`.

VAD remains authoritative for acoustic activity and word timestamps remain
authoritative for placement, but neither is by itself a semantic boundary.
Running a second ASR provider is not part of the default solution: it may alter
punctuation or word timing, but it cannot correct a downstream duration-based
split reliably.

### Selected hybrid policy

Replace the hard 15-second recursive split with a semantic segment planner:

- `tts_preferred_segment_duration = 15.0` is a soft target.
- `tts_hard_segment_duration = 35.0` is the maximum normal synthesis unit.
- `semantic_split_search_window = 10.0` controls how far on either side of the
  preferred duration the planner searches before allowing the unit to grow.
- `semantic_split_enabled = true` enables the planner for isolated-track input.

The planner first rebuilds a continuity chain from adjacent VAD regions for the
same isolated speaker. A short acoustic pause is retained as metadata; it does
not force a boundary. The old `max_duration * 1.5` merge cap must not prevent a
30-second thought from being considered as one chain. A speaker change is
always a semantic boundary.

For each chain, candidate cuts are produced from word boundaries. Candidates
are ranked in this order:

1. a semantically and grammatically complete sentence;
2. a complete independent clause supported by soft punctuation and an
   acoustic pause;
3. an ASR segment boundary classified as a complete thought;
4. the strongest acoustic word gap, used only when the hard limit would
   otherwise be exceeded.

Sentence-final punctuation is a candidate even if its measured word gap is
below 0.4 seconds. Conversely, a long VAD pause is rejected as a normal cut
when the left context is syntactically incomplete, for example after a
subordinating conjunction, preposition, determiner, or construction such as
`I think that` or `if we've laid off`.

Obvious boundaries are classified locally from punctuation, pause, speaker,
and incomplete-tail rules. Ambiguous candidate boundaries are evaluated in a
batched structured LLM request using the source text on both sides. The
classifier must not rewrite the transcript. A second ASR provider is
deliberately not required.

### Exact boundary selection

Planning proceeds left-to-right. For a unit beginning at `unit_start`, define:

```text
target = unit_start + preferred_duration
hard_end = min(chain_end, unit_start + hard_duration)
effective_window = min(configured_search_window,
                       preferred_duration,
                       hard_duration - preferred_duration)
search_start = max(unit_start, target - effective_window)
search_end = min(hard_end, target + effective_window)
```

An oversized search-window setting is therefore deterministically clamped; it
can never move a normal cut beyond the hard limit. Candidate times are the end
of the word/segment on the left. The following local decisions apply:

- `HARD_CONTINUE`: the left text has an incomplete tail. This veto overrides
  sentence punctuation and an LLM `CUT`.
- `LOCAL_CUT_SENTENCE`: sentence-final punctuation and no incomplete tail.
- `LOCAL_CUT_CLAUSE`: soft punctuation, no incomplete tail, and a source pause
  of at least 0.2 seconds.
- `AMBIGUOUS`: every other ASR/VAD boundary near a possible cut.

Incomplete-tail detection normalizes case and trailing punctuation, then checks
the final source-language tokens against versioned language-specific rule
tables. The English table includes subordinating conjunctions, coordinating
conjunctions, prepositions, determiners, and explicit multi-token tails such as
`I think that` and `if we've laid off`. Unsupported languages use only
language-neutral punctuation and send non-obvious candidates to the LLM.

The LLM returns `CUT`, `CONTINUE`, or `UNCERTAIN` with confidence in `[0, 1]`.
`CUT >= 0.75` makes an ambiguous candidate eligible; `CONTINUE >= 0.60` vetoes
it. Low-confidence results, `UNCERTAIN`, missing results, and malformed results
fall back per candidate to the local decision. `HARD_CONTINUE` always wins.

Eligible candidates in `[search_start, search_end]` are ordered by this exact
authoritative tuple, ascending:

```text
(-semantic_rank, -decision_priority, abs(candidate_time - target),
 -source_pause, source_word_or_segment_index)
```

Sentence, clause, and plain-boundary semantic ranks are 3, 2, and 1. Within the
same semantic rank, an accepted LLM `CUT` has decision priority 2 and a local
decision has priority 1. Thus an LLM-approved ambiguous plain boundary can
never outrank a local complete sentence or clause. If the window has no
eligible candidate, scan eligible candidates in `(search_end, hard_end]` with
the same ordering. If the remaining chain ends by `hard_end`, emit it unsplit
rather than inventing a bad cut.

If the chain continues past `hard_end` and no eligible candidate exists, a cut
is mandatory. First choose only among non-vetoed word boundaries at or before
`hard_end`, ordered by
`(-punctuation_rank, -source_pause, -candidate_time, source_word_index)`. Only
when no non-vetoed boundary can keep the unit within the hard maximum may the
same tuple be applied to `HARD_CONTINUE` boundaries. If no complete word ends
by `hard_end`, raise
`SemanticSegmentationError` with the speaker, offending word/segment, and
timestamps rather than silently exceed the invariant.

The forced cut marks both sides with a shared `continuation_id` and
`boundary_type=technical_continuation`. Such a cut:

- adds no synthetic pause;
- retains the measured source pause separately;
- uses edge-silence trimming on both generated clips;
- is reported in timing diagnostics.

Normal semantic cuts use `boundary_type=semantic`. Each planned unit retains
the contributing VAD-region IDs and word range so its start and end are still
derived from recognized word timestamps.

For ASR backends without word timestamps, ASR/VAD segment boundaries are the
only candidates and the same selection algorithm applies with segment indexes.
An individual no-word-timestamp segment longer than the hard maximum is
indivisible and raises `SemanticSegmentationError`, explaining that a
word-timestamp backend is required or semantic splitting must be disabled. A
second ASR is never started implicitly.

### LLM integration contract

`LLMTranslator` exposes `classify_semantic_boundaries(request)`. It uses the
already initialized primary translation `llm`, `llm_provider`, `model_name`,
`temperature`, and `max_tokens`; it does not create another client.
`SmartDubbing` passes the callable plus an explicit classifier status to
`run_isolated_tracks`, which passes them into the planner. A deliberately
configured non-LLM translator uses stable `deterministic-only` mode. Failure to
initialize a configured LLM translator is `initialization_failed`, is treated
as a transient classifier failure, and disables persistence of plan-dependent
caches for the run; the later translation stage may still raise its existing
initialization error.

Candidates are batched in stable source order, at most 50 candidates and 12,000
input characters per request. Each request uses the source-language code and:

```json
{"candidates": [{"id": "stable-id", "left": "...", "right": "...", "pause": 0.644}]}
```

The response schema is:

```json
{"boundaries": [{"id": "stable-id", "decision": "CUT", "confidence": 0.92, "reason_code": "complete_sentence"}]}
```

Candidate IDs are unique within a request. Duplicate, unknown, missing, or
invalid response entries fall back independently; they do not discard valid
siblings. Calls use a 30-second timeout. Provider errors and timeouts fall back
for the affected batch and are marked transient. The prompt and parser are
versioned as `semantic_boundary_prompt_v1`.

### Semantic input normalization

Before IDs or candidates are produced, validate every VAD region, ASR segment,
and word timestamp as finite floats with `start >= 0` and `end >= start`.
Malformed records fail the semantic-planning step with their provider index and
speaker; they are not silently dropped. Sort VAD regions and ASR segments by
`(start, end, original_provider_index)` and words by
`(start, end, normalized_text, original_provider_index)`.

Exact duplicate words with equal normalized text and millisecond-rounded
start/end are collapsed, keeping highest confidence and then lowest original
provider index. Overlapping non-duplicate words are retained in deterministic
order. When VAD tolerance makes a word intersect multiple regions, assign it
to exactly one: greatest overlap, then nearest region center, then lowest
stable VAD-region ID. Define every candidate's pause as
`source_pause = max(0.0, right_start - left_end)` after finite validation, so
overlap is represented by zero rather than a negative or non-finite score.

### Boundary preservation and lineage

The translation and TTS stages consume the planned semantic units. Both
`LLMTranslator._optimize_segments` passes must treat `lock_boundary_before` as
a merge veto and preserve all semantic metadata. Translation chunking may put
multiple locked units in one context request, but decomposition and refinement
must return one translation per stable `semantic_unit_id`; it may not merge IDs
or produce a unit over the hard maximum. Missing or duplicate IDs fail that
translation batch through its existing retry/fallback path.

Every planned unit has this minimum schema:

```text
semantic_unit_id: sha256(planner_version, speaker, first_source_index,
                         last_source_index, start_ms, end_ms)[:16]
speaker: original isolated-track label
start/end: recognized word or ASR/VAD-segment timestamps
source_word_range: [first_index, end_exclusive] or null
source_segment_ids: ordered stable ASR segment IDs
vad_region_ids: ordered IDs such as SPEAKER_00:v000013
lock_boundary_before: true except for the first unit in a chain
boundary_before: {candidate_id, type, source_pause, decision, confidence,
                  reason_code, classifier_mode}
continuation_id: null or sha256(chain_id, forced_cut_source_index)[:16]
```

Raw VAD region IDs and ASR word/segment indexes are assigned before any merge
or split and never renumbered. Merging concatenates ordered lineage; splitting
slices it. Debug output preserves this lineage through translation and TTS.

For the motivating artifact, the expected planning decisions are:

- rows 6 and 7 remain one thought; row 8 starts a new semantic unit;
- the cut after `I think that` is prohibited;
- rows 13--15 are resegmented at complete sentence or clause boundaries, with
  no unit exceeding 35 seconds.

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

Add eight configuration fields with validation and defaults:

| Config key | Default | Meaning |
| --- | ---: | --- |
| `timing_short_segment_threshold` | `1.5` | Recognized duration below which the short-segment speed limit applies, in seconds. |
| `timing_short_segment_max_speed` | `1.08` | Maximum FFmpeg tempo multiplier for short segments. |
| `timing_max_speed` | `1.15` | Maximum FFmpeg tempo multiplier for other segments. |
| `timing_max_overflow` | `0.25` | Allowed audio overflow beyond the usable window, in seconds. |
| `semantic_split_enabled` | `true` | Use semantic rather than fixed-duration splitting for isolated speaker tracks. |
| `tts_preferred_segment_duration` | `15.0` | Soft target duration for a translation/TTS unit, in seconds. |
| `tts_hard_segment_duration` | `35.0` | Hard maximum duration for a normal semantic unit, in seconds. |
| `semantic_split_search_window` | `10.0` | Candidate search radius around the preferred duration, in seconds. |

Expose all timing and semantic-split values in the Gradio advanced timing/audio settings. They
participate in the existing settings save/load flow and can also be supplied by
YAML, CLI arguments, and programmatic overrides.

Validation rules:

- Threshold and overflow must be non-negative.
- Speed multipliers must be at least `1.0`.
- Preferred duration must be positive, hard duration must be greater than or
  equal to preferred duration, and the search window must be non-negative.
- `semantic_split_enabled` accepts booleans plus the existing normalized
  `true`/`false` and `1`/`0` forms. Other values fall back to `true` with a
  warning.
- The effective search window is clamped by the formula in Exact boundary
  selection; the stored user value is retained for display.
- Invalid YAML/programmatic values, including NaN and infinity, fall back to
  defaults with a warning.
- Gradio uses numeric fields with appropriate precision.

One shared normalization helper applies these rules to YAML, CLI, Gradio-loaded
defaults, and programmatic overrides so entry points cannot disagree.

When `semantic_split_enabled=false`, the isolated-track path uses the existing
legacy `_merge_close_segments` followed by `_split_long_segments`, with
`tts_preferred_segment_duration` as its fixed maximum. The hard duration,
search window, LLM classifier, semantic metadata, and semantic-plan cache are
not used. This compatibility mode is logged explicitly.

The legacy `group_overflow_tolerance` setting is retained for one compatibility
cycle as a deprecated, ignored alias. Loading a non-default value logs a warning
that points to `timing_max_overflow`; it is removed from the Gradio interface and
newly saved settings, and no timing code reads it.

Silence detection thresholds and safety margins remain internal implementation
constants. They are not exposed in the interface in this iteration.

## Components

### Timing configuration

`src/dubbing/core/config.py` owns defaults, validation, and CLI arguments.
`src/dubbing/ui/gradio_app.py` exposes all eight settings and includes them in
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

### Semantic segment planner

Add a focused planner component for isolated-track continuity chains. It owns
candidate extraction, deterministic ranking, optional batched LLM
classification, duration-constrained selection, lineage metadata, and forced
continuation diagnostics. VAD execution and ASR providers remain unchanged.
The existing `_split_long_segments` becomes a compatibility wrapper or is
replaced at its isolated-track call site; fixed recursive splitting must not run
after semantic planning and undo its decisions.

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

Split isolated-track caching into two layers:

1. `isolated_tracks_raw_transcription` is keyed by track fingerprints, VAD/ASR
   provider and model, language, trim range, and `isolated_raw_v1`. It stores
   stable VAD IDs plus raw words/segments before semantic planning.
2. `isolated_tracks_semantic_plan` is keyed by the raw fingerprint, all four
   semantic settings, `semantic_planner_v1`, incomplete-tail rule-table
   version, prompt/parser version, source language, and LLM provider/model (or
   `deterministic-only`). When an LLM classifier is used, the key also includes
   its effective temperature, maximum-token setting, timeout, and batching
   limits. Its value stores the plan and a
   `semantic_plan_fingerprint`, computed from canonical JSON containing the
   resulting ordered boundaries and lineage.

The `semantic_plan_fingerprint` is included in translation, translated-segment
resume data, raw per-unit TTS identity, final `synthesized_speech`, and
`tts_to_end` resume validation. A cached translation or final track whose
fingerprint is absent or different is rejected and rebuilt. The legacy combined
isolated-transcription cache is readable only when semantic splitting is
disabled; it is never promoted to a semantic plan without replanning.

Successful LLM classifications are cached per candidate-payload hash under
`semantic_boundary_classification`, including provider, model, source language,
temperature, maximum-token setting, timeout, batching limits, and prompt/parser
version. Intentional deterministic-only results are stable and may be
persisted. Configured-classifier initialization failure, timeout, provider
error, or malformed-response fallback is transient: the run continues, but
neither the semantic-plan cache nor translation/final artifacts derived from
that transient plan are persisted. Raw transcription and successful
per-candidate classifications remain reusable. This prevents a temporary LLM
failure from becoming a sticky segmentation result.

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

Also generate `artifacts/debug/semantic_boundaries.jsonl`. One record per
candidate contains its stable ID, speaker, source indexes, left/right context,
candidate time, measured source pause, local decision, LLM decision and
confidence when present, final decision, reason code, classifier mode, fallback
reason, chosen/not-chosen state, semantic unit ID, boundary type, lineage, and
continuation ID. The timing TSV includes `semantic_unit_id`,
`semantic_plan_fingerprint`, and `continuation_id` so a generated waveform can
be traced back to the exact plan.

## Error handling

- If FFmpeg tempo adjustment fails, use the unmodified trimmed clip, record
  `tempo=1.0`, and calculate diagnostics from its actual duration.
- Missing or unreadable TTS files continue to produce silence for the recognized
  segment duration so later anchors remain stable.
- Zero-length recognized ranges are valid and use the next anchor as available
  slack. Invalid ranges fail validation before synthesis.
- Audio is never silently cut merely to satisfy the overflow limit.
- An indivisible no-word-timestamp segment over the semantic hard maximum fails
  before translation with an actionable `SemanticSegmentationError`.
- A transient boundary-classifier failure uses deterministic per-candidate
  fallback and disables persistence of downstream plan-dependent caches for
  that run.

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
14. Check in `tests/fixtures/semantic_boundaries_matthew.json` containing the
    exact source text, timestamps, word timestamps, stable ASR/VAD IDs, and
    speaker labels for motivating rows 6--8 and 13--15. Its expected plan
    specifies 6→7 `CONTINUE`, 7→8 `CUT`, 13→14 `CONTINUE`, and 14→15
    `CONTINUE`, plus the expected resegmented unit text, start/end timestamps,
    source ranges, and lineage.
15. Run that exact fixture through both accepted LLM results and
    deterministic-only fallback. Assert that no chosen boundary follows
    `I think that`, every output unit is at most 35 seconds, unit text is a
    lossless ordered reconstruction of the fixture words, starts/ends come from
    fixture timestamps, and all source/VAD IDs occur exactly once in lineage.
16. Sentence-final punctuation is considered even with a word gap below 0.4
    seconds; a long VAD pause after an incomplete tail is a `HARD_CONTINUE`,
    including when the LLM returns high-confidence `CUT`.
17. Exact search-window and tie-break tests cover candidates on both interval
    edges, equal scores, clamped oversized windows, extension beyond the soft
    target, sentence-over-LLM-plain precedence, a mixed safe/vetoed forced-cut
    case that chooses the safe boundary, and a forced cut at or before 35
    seconds when every candidate is vetoed.
18. LLM `CUT`, `CONTINUE`, low confidence, `UNCERTAIN`, missing ID, duplicate
    ID, unknown ID, malformed response, timeout, unavailable classifier, and
    mixed-validity batch paths produce the defined per-candidate outcomes.
19. Wordless multi-segment input uses segment-boundary fallback; an indivisible
    segment over 35 seconds raises the specified error without starting another
    ASR provider.
20. Both translation optimization passes and refinement preserve locked
    semantic boundaries, IDs, lineage, and the 35-second invariant.
21. Raw transcription, successful classification, semantic plan, translation,
    raw TTS, final assembly, and resume cache tests prove the fingerprint rules;
    transient LLM fallback persists only raw transcription and successful
    candidate classifications.
22. Configuration, CLI, Gradio defaults, validation, disabled legacy mode, and
    persistence cover all four semantic-split settings.
23. Non-finite, negative, and inverted VAD/word/segment timestamps are rejected;
    input-order permutations, exact duplicate words, overlapping words, and
    multi-region word intersections produce deterministic IDs, assignments,
    pauses, boundaries, and plan fingerprints.
24. Intentional deterministic-only mode is persistable, while configured LLM
    initialization failure follows transient no-plan/downstream-cache policy;
    provider/model/temperature/max-token changes invalidate classifications and
    semantic plans.

The existing full test suite remains the regression gate. A real-artifact check
compares `transcription.srt`, generated chunks, and final output duration for the
latest project and reports timing metrics without requiring language-dependent
waveform correlation.

## Non-goals

- Word- or phoneme-level forced alignment.
- Lip-shape synchronization.
- Automatically rewriting translations during final assembly.
- Exposing low-level silence detector settings in the UI.
- Running multiple ASR providers by default or voting between their segment
  boundaries.
