# Turn-Aware Semantic Segmentation Design

## Goal

Prevent an isolated-speaker semantic unit from spanning a different speaker's
turn, and preserve strong complete utterance boundaries even when they occur
before the preferred 15-second TTS duration.

The motivating failure is the `videoplayback3` transcription:

- `SPEAKER_00` speaks at `63.122..65.065` and returns at `82.678..83.849`;
- `SPEAKER_01` speaks between those ranges;
- the current planner joins both `SPEAKER_00` ranges into `63.122..83.849`,
  creating false overlaps with the surrounding `SPEAKER_01` units;
- the current planner also joins the complete `Matt Manorina.` utterance to the
  following question because its sentence boundary is earlier than the normal
  preferred-duration search window.

## Considered approaches

### Selected: global word activity plus strong early utterance boundaries

Build one read-only activity index from the normalized words of every isolated
track. Foreign word ranges are sorted, exact duplicates are removed, and
overlapping or touching ranges are coalesced into activity components. While
planning one speaker, a non-zero foreign activity component marks a mandatory
speaker-turn boundary only when its complete interval lies strictly inside the
open gap between two consecutive current-speaker items. A foreign component
that touches or overlaps either adjacent current-speaker item is concurrent
speech and does not mark that gap. This prevents a word such as `1.8..3.2` from
splitting a current-speaker gap `2.0..3.0`, while still recognizing a genuinely
intervening turn that begins after the left item and ends before the right.

Independently, treat an early sentence-final boundary as mandatory when it also
crosses an ASR source-segment boundary and has at least 0.5 seconds of source
pause. This preserves standalone complete utterances such as `Matt Manorina.`
without turning every short sentence into a separate TTS unit.

This approach is selected because it uses the most precise evidence already
available, distinguishes intervening turns from concurrent overlap, and keeps
the preferred duration as a soft target.

### Rejected: split on every long VAD gap

This would fix the motivating 17.613-second hole but would regress the original
semantic-planner requirement: a long acoustic pause may occur inside an
unfinished sentence. VAD alone is not a semantic boundary.

### Rejected: trim overlapping segment endpoints after planning

Post-processing timestamps would hide the table symptom while leaving text
from non-contiguous turns in one translation/TTS request. It would also have no
principled way to decide which speaker owns the overlap.

## Architecture and data flow

`_assemble_isolated_raw_tracks` derives foreign activity from each raw track's
normalized words before it plans any track. For a given speaker it passes only
the other tracks' word ranges into `plan_semantic_segments` through a new
keyword-only parameter:

```python
foreign_activity: Sequence[Mapping[str, Any] | Sequence[float]] = ()
```

Each record supplies `start` and `end` in seconds, either as mapping keys or the
first two sequence values. The planner validates finite, non-negative,
non-inverted timestamps; invalid records raise `ValueError` naming the current
speaker and provider index. Zero-duration records are ignored. Valid records
are deterministically sorted, deduplicated, and coalesced as described above,
so caller order cannot affect the plan fingerprint. In wordless current-track
mode, the same rule applies between consecutive ASR segment items. A foreign
track without normalized words contributes no inferred activity: broad ASR or
VAD ranges are not substituted because they may contain internal silence.

The semantic planner annotates candidate boundaries with two new facts:

- `speaker_turn_boundary`: at least one qualifying fully-contained foreign
  activity component exists inside the source gap;
- `strong_early_utterance`: the left word ends a sentence, the right word comes
  from a different ASR segment, and the source pause is at least 0.5 seconds.

Boundary precedence is authoritative:

1. `speaker_turn_boundary` is mandatory and overrides every local or classifier
   decision, including `HARD_CONTINUE`;
2. `strong_early_utterance` is mandatory only when the pre-existing local
   decision is `LOCAL_CUT_SENTENCE`, so it can never override an incomplete-tail
   `HARD_CONTINUE`;
3. normal sentence, clause, classifier, and forced-hard-limit selection keeps
   the existing ordering.

If both mandatory facts occur on one candidate, speaker-turn precedence wins
and the boundary uses speaker-turn metadata.

Boundary selection operates against the next mandatory candidate index as a
temporary chain cap. Normal eligible cuts may still be selected before that
cap. If one is selected, planning resumes and retains the same cap until it is
reached. When no earlier normal cut is selected, the cap candidate itself is
marked `chosen`, emitted as the unit's terminal boundary, and planning resumes
at the following item. It can therefore never be skipped by the preferred or
hard-duration windows. A speaker-turn cut caused across an incomplete tail gets
a stable shared `continuation_id` but retains `boundary_type = "speaker_turn"`;
a strong early utterance is never a technical continuation.

Built units keep the existing schema. Speaker-turn boundaries use
`boundary_type = "speaker_turn"`; strong early utterances retain the normal
`semantic` boundary type and `sentence_final` reason. Diagnostics include the
new boolean facts for debugging.

## Cache compatibility

The algorithm identity changes from `semantic_planner_v1` to
`semantic_planner_v2`. One exported `SEMANTIC_PLANNER_VERSION` constant is the
source of truth for candidate IDs, per-track fingerprints, aggregate
fingerprints, and isolated-plan cache keys. Candidate IDs and semantic plan
fingerprints therefore change automatically. Provider-level VAD/ASR raw caches
remain reusable.

The isolated semantic-plan cache is looked up before a plan fingerprint exists.
Its key therefore contains the raw-track fingerprint, semantic settings,
classifier identity, and `SEMANTIC_PLANNER_VERSION`; a loaded payload is then
validated to contain one consistent embedded plan fingerprint.

Every downstream cache that stores or reloads segment records is
plan-dependent:

- translation;
- emotions;
- raw/final TTS and synthesis metadata.

Their keys include the embedded semantic-plan fingerprint, and loaded payloads are
validated with the existing plan-dependent segment validator. In particular,
`_build_emotions_cache_key` incorporates provider, model, and semantic
fingerprint; both `analyze_emotions` and `tts_to_end` use that same helper and
validate loaded emotion segments. A v1 emotion result therefore cannot replace
a newly segmented v2 list.

## Error handling and edge cases

- Missing foreign word timestamps contribute no foreign activity rather than
  guessing from VAD noise. Wordless current tracks still consume valid foreign
  activity between their ASR segments.
- Exact boundary contact and zero-duration foreign ranges are not treated as
  intervening activity.
- Unsorted and duplicate foreign ranges normalize to the same canonical
  activity components and fingerprint.
- Foreign activity concurrent with either adjacent current-speaker item remains
  intentional overlap and does not split at every word.
- A mandatory speaker turn overrides `HARD_CONTINUE` because one synthesized
  unit must not span another speaker's turn; a continuation ID preserves the
  semantic relationship without restoring the invalid bounding interval.
- Empty tracks and invalid timestamps retain existing validation behavior.

## Testing

Add regression tests that reproduce the full critical timeline:

1. A standalone `Matt Manorina.` ASR segment followed by a longer same-speaker
   question remains its own unit despite being earlier than the 15-second
   search window.
2. `SPEAKER_00` activity at `63.122..65.065` and `82.678..83.849`, with
   `SPEAKER_01` words between them, becomes two units and cannot create the
   former `63.122..83.849` bounding interval.
3. Concurrent foreign speech does not create repeated false turn boundaries.
4. A foreign range spanning both adjacent current-speaker items, exact contact,
   zero-duration input, unsorted/duplicate ranges, and a wordless foreign track
   do not create false turn boundaries.
5. Mandatory speaker turns override `HARD_CONTINUE` and carry a continuation
   ID; strong early boundaries never override `HARD_CONTINUE`; dual-fact
   candidates use speaker-turn metadata.
6. An earlier normal cut before a mandatory cap is emitted first, after which
   the mandatory cap is still emitted and planning resumes after it.
7. v1 isolated-plan, translation, emotion, and TTS segment caches are rejected
   by v2 identities, while the provider-level raw VAD/ASR cache remains reusable.
8. A wordless current track consumes valid foreign activity between its ASR
   segments, and malformed foreign timestamps raise the documented
   speaker/index-specific `ValueError`.
9. The existing incomplete-tail, deterministic classification, semantic
   lineage, overlap, cache-fingerprint, and full repository test suites remain
   green.

The regression test must be observed failing before production code changes,
then passing afterward.
