# Turn-Aware Semantic Segmentation Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Prevent semantic units from spanning intervening speaker turns and preserve strong early complete utterances without breaking intentional concurrent overlap.

**Architecture:** Normalize foreign-track word ranges into deterministic activity components and pass them into the semantic planner. Mandatory speaker-turn and strong early-utterance candidates cap the current selection chain; the planner version and every downstream segment cache identity advance together.

**Tech Stack:** Python 3.12, pytest, existing `dubbing.audio.semantic_planner`, isolated-track orchestration, and `SmartDubbing` cache helpers.

---

## Chunk 1: Planner behavior and integration

### Task 1: Add failing turn-aware planner regressions

**Files:**
- Modify: `tests/test_semantic_planner.py`

- [ ] **Step 1: Add a failing early-utterance regression**

Add `test_strong_early_source_utterance_is_mandatory_before_duration_window` using two ASR segments: `Matt Manorina.` at `58.683..60.213`, followed by a longer question beginning at `61.179`. Use `preferred_duration=15`, `hard_duration=35`, and `search_window=10`; assert that the first unit is exactly `Matt Manorina.` and its chosen diagnostic has `strong_early_utterance=True`.

- [ ] **Step 2: Add a failing full turn-interleaving regression**

Add `test_intervening_foreign_turn_splits_returning_speaker_chain`. Build two raw isolated tracks and call `_assemble_isolated_raw_tracks`: `SPEAKER_00` has words at `63.122..65.065` and `82.678..83.849`; `SPEAKER_01` has words fully between those ranges. Assert that `SPEAKER_00` produces two units, no unit spans `63.122..83.849`, and the split diagnostic uses `boundary_type="speaker_turn"`.

- [ ] **Step 3: Add atomic foreign-activity contract tests**

Add these named tests with concrete assertions:

- `test_concurrent_foreign_activity_does_not_split_current_speaker`: current words straddle `2.0..3.0`, foreign activity `1.8..3.2`; assert no `speaker_turn` boundary.
- `test_foreign_activity_normalization_is_order_duplicate_and_contact_stable`: permute duplicate, exact-contact, zero-duration, and valid contained ranges; assert identical fingerprints/units and only the contained range qualifies.
- `test_invalid_foreign_activity_names_speaker_and_provider_index`: parametrize negative, inverted, NaN, fewer-than-two sequence values, and mappings missing `start` or `end`; assert `ValueError` contains `SPEAKER_00` and the exact index. Include one valid mapping-form record in the normalization test.
- `test_wordless_current_track_splits_on_valid_foreign_activity`: two wordless ASR segments surround a contained foreign range; assert two units.
- `test_wordless_foreign_track_contributes_no_inferred_activity`: assemble a wordless foreign track between current-speaker words; assert it supplies no mandatory split.
- `test_speaker_turn_overrides_incomplete_tail_with_continuation`: split after `I think that` with contained foreign activity; assert `speaker_turn`, chosen CUT, and one shared non-empty continuation ID on adjacent units.
- `test_strong_early_boundary_never_overrides_incomplete_tail`: use an ASR source-segment change and pause after `I think that`; assert `HARD_CONTINUE`, `strong_early_utterance=False`, and no chosen cut there.
- `test_dual_fact_boundary_uses_speaker_turn_metadata`: complete early sentence plus contained foreign activity; assert both diagnostic facts are true but `boundary_type="speaker_turn"` and `reason_code="speaker_turn"`.
- `test_normal_cut_before_mandatory_cap_does_not_drop_cap`: place one normal sentence cut before a later contained foreign turn; assert both boundaries are chosen in order and the next unit begins after the mandatory candidate.

- [ ] **Step 4: Run tests and verify RED**

Run the complete test file because every new public-contract call must fail before implementation:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_semantic_planner.py -v
```

Expected: ten new tests fail because `plan_semantic_segments` does not accept `foreign_activity`, diagnostics lack the two new facts, and isolated-track assembly still glues the motivating units. The negative-control `test_wordless_foreign_track_contributes_no_inferred_activity` and all existing tests pass.

### Task 2: Implement deterministic foreign activity and mandatory chain caps

**Files:**
- Modify: `src/dubbing/audio/semantic_planner.py`
- Modify: `src/dubbing/audio/isolated_tracks.py`
- Modify: `src/dubbing/core/smart_dubbing.py`
- Test: `tests/test_semantic_planner.py`
- Modify: `tests/fixtures/semantic_boundaries_matthew.json`

- [ ] **Step 1: Normalize foreign activity**

In `semantic_planner.py`, add `_normalize_foreign_activity(records, speaker)` that accepts mappings or sequences with at least two values (extra values are ignored), validates finite non-negative ranges with speaker/provider-index-specific errors, ignores zero duration, sorts/deduplicates, and coalesces overlapping or touching ranges.

- [ ] **Step 2: Annotate candidates**

Extend `plan_semantic_segments(..., foreign_activity=())` and `_build_candidates` so a candidate records:

```python
speaker_turn_boundary = any(
    left["end"] < activity["start"]
    and activity["end"] < right["start"]
    for activity in foreign_activity
)
strong_early_utterance = (
    local_decision == "LOCAL_CUT_SENTENCE"
    and left.get("source_segment_id") != right.get("source_segment_id")
    and pause >= 0.5
)
```

The authoritative mandatory predicate is `speaker_turn_boundary or strong_early_utterance`. Speaker turns set a mandatory CUT with `boundary_type="speaker_turn"`; incomplete tails also receive a stable continuation ID. Strong early utterances qualify only when the pre-existing local decision is `LOCAL_CUT_SENTENCE`, remain semantic sentence cuts, and must participate in the mandatory cap even before `search_start`. Speaker-turn precedence wins when both facts are true.

- [ ] **Step 3: Cap boundary selection at mandatory candidates**

Update `_select_boundaries` to find the next mandatory candidate at or after `unit_start_index`, restrict normal selection to that candidate, preserve the cap after earlier normal cuts, and emit the mandatory candidate itself before resuming after it. Keep current normal and forced-hard-limit ordering inside the cap.

- [ ] **Step 4: Supply global foreign activity**

In `_assemble_isolated_raw_tracks`, collect normalized word ranges per track once. When planning one speaker, flatten ranges from all other tracks and pass them as `foreign_activity`; do not substitute VAD or broad wordless ASR segments.

- [ ] **Step 5: Advance the planner version from one exported constant**

Set `SEMANTIC_PLANNER_VERSION = "semantic_planner_v2"` and import it in `isolated_tracks.py` and `smart_dubbing.py` for aggregate fingerprints and isolated-plan cache keys instead of duplicating string literals.

- [ ] **Step 6: Refresh version-derived fixture IDs**

Run `tests/fixtures/semantic_boundaries_matthew.json` through the v2 planner and update only semantic unit/candidate IDs. Do not change expected text, timestamps, source ranges, lineage, or decisions.

- [ ] **Step 7: Run focused tests and verify GREEN**

Run the Task 1 command. Expected: all selected tests pass.

- [ ] **Step 8: Run the complete semantic planner suite**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_semantic_planner.py -v
```

Expected: all tests pass with zero failures.

---

## Chunk 2: Plan-dependent cache safety and verification

### Task 3: Make emotion caching semantic-plan aware

**Files:**
- Modify: `src/dubbing/core/smart_dubbing.py`
- Modify: `tests/test_semantic_planner.py`

- [ ] **Step 1: Add failing cache-transition tests**

Create `test_emotions_cache_key_includes_provider_model_and_semantic_fingerprint` with a `SmartDubbing` stub and deterministic cache manager. Assert that `_build_emotions_cache_key(audio_file, provider, model)` differs for semantic fingerprints `plan-a` and `plan-b`, providers, and models.

Add these named tests:

- `test_analyze_emotions_uses_plan_aware_key_and_rejects_mismatched_payload`: deliberately seed an emotion payload carrying `plan-old` under the currently requested `plan-new` key, then assert validation rejects it and the analyzer path receives the input segments.
- `test_run_from_tts_rejects_mismatched_emotion_payload`: deliberately return emotion segments with the wrong fingerprint from the currently requested key and assert the existing plan-dependent validator raises `ValueError` before synthesis.
- `test_v2_invalidates_plan_dependent_keys_but_reuses_raw_track_key`: first control exported `SEMANTIC_PLANNER_VERSION` as v1/v2 and assert `_isolated_tracks_cache_key` changes while `_isolated_tracks_raw_cache_key` remains identical for the same track/config; separately set embedded semantic fingerprints `plan-v1`/`plan-v2` and assert translation, emotion, and raw-TTS keys change.

- [ ] **Step 2: Run the test and verify RED**

Run:

```powershell
.\.venv\Scripts\python.exe -m pytest `
  tests/test_semantic_planner.py::test_emotions_cache_key_includes_provider_model_and_semantic_fingerprint `
  tests/test_semantic_planner.py::test_analyze_emotions_uses_plan_aware_key_and_rejects_mismatched_payload `
  tests/test_semantic_planner.py::test_run_from_tts_rejects_mismatched_emotion_payload `
  tests/test_semantic_planner.py::test_v2_invalidates_plan_dependent_keys_but_reuses_raw_track_key -v
```

Expected: the new tests fail because the existing helper omits the fingerprint/provider/model contract, `analyze_emotions` bypasses the helper/validator, and `run_from_tts` does not validate fallback emotion payloads. Existing fingerprint tests pass.

- [ ] **Step 3: Implement the cache-key helper**

Extend `_build_emotions_cache_key` to include provider, model, and `_semantic_plan_fingerprint`.

- [ ] **Step 4: Adopt and validate the helper in `analyze_emotions`**

Use the helper for reads/writes and call `_validate_plan_dependent_segments` on a loaded payload before returning it. A mismatch is treated as a stale/corrupt cache miss and triggers fresh analysis rather than replacing the current segment list.

- [ ] **Step 5: Adopt and validate the helper in `run_from_tts`**

Use the same provider/model/helper values for the fallback lookup and validate loaded segments before synthesis. A mismatched payload raises the existing actionable `ValueError` in resume mode.

- [ ] **Step 6: Run the tests and verify GREEN**

Run the Step 2 command. Expected: pass.

### Task 4: Verify the original failure and full repository

**Files:**
- No additional production files

- [ ] **Step 1: Run the complete semantic planner suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests/test_semantic_planner.py -v
```

Expected: all tests pass.

- [ ] **Step 2: Replay the cached `videoplayback3` raw transcription read-only**

Deterministically select the lexicographically first pickle under `cache/2f4dce08/isolated_tracks_raw_transcription`, load it, and call `_assemble_isolated_raw_tracks` with semantic splitting enabled. Run:

```powershell
@'
import pickle
from pathlib import Path
from dubbing.audio.isolated_tracks import _assemble_isolated_raw_tracks

path = sorted(Path("cache/2f4dce08/isolated_tracks_raw_transcription").glob("*.pkl"))[0]
raw = pickle.loads(path.read_bytes())
_, segments = _assemble_isolated_raw_tracks(
    raw,
    inner_system="assemblyai",
    source_language="en",
    semantic_split_enabled=True,
    tts_preferred_segment_duration=15.0,
    tts_hard_segment_duration=35.0,
    semantic_split_search_window=10.0,
    semantic_classifier=None,
    semantic_classifier_status="deterministic-only",
    semantic_debug_path=None,
    classification_cache_get=None,
    classification_cache_set=None,
    classifier_cache_context=None,
    semantic_diagnostics_out=None,
)
assert any(s["text"] == "Matt Manorina." for s in segments)
assert not any(s["start"] == 63.122 and s["end"] == 83.849 for s in segments)
speaker_zero = [s for s in segments if s["speaker"] == "SPEAKER_00"]
assert any(s["end"] <= 65.065 for s in speaker_zero)
assert any(s["start"] >= 82.678 for s in speaker_zero)
print("turn-aware replay: PASS")
'@ | .\.venv\Scripts\python.exe -
```

Expected output: `turn-aware replay: PASS` and exit code 0.

- [ ] **Step 3: Run the full repository suite**

```powershell
.\.venv\Scripts\python.exe -m pytest tests
```

Expected: zero failures.

- [ ] **Step 4: Review the final diff**

Run `git diff --check`, inspect `git diff --stat`, and request code review against this spec. Resolve all Critical and Important findings, then repeat the focused and full verification commands.
