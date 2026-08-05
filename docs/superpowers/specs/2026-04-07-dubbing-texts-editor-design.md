# Dubbing Texts Editor Design

## Goal

Add a new Gradio tab after `Settings` that exposes all translated text segments that will be sent to TTS, allows users to edit those translations manually, and saves the edits both:

- to a human-readable artifact file inside the project artifacts directory
- to the cached translation payload used by `run_step=tts_to_end`

This lets users fix problematic translations and then re-run only the TTS/audio generation step without repeating the full pipeline.

## User Experience

### New tab

Add a new `Dubbing Texts` tab after `Settings`.

The tab is optimized for review and editing of dubbing-ready text:

- main working column: `translation`
- reference columns: `speaker`, `time`, `original`
- column order: `speaker | time | translation | original`
- `original` stays last and visually compact so it can be consulted when needed without dominating the layout

### Actions in the tab

The tab provides:

- `Load texts` button
- editable table with all dubbing segments
- `Save texts` button
- status message for load/save results

### Editing model

Users primarily edit `translation`.

For usability, the table may remain generally editable at the Gradio level, but save logic will treat `speaker`, `time`, and `original` as reference fields and preserve canonical values from the cached segment source when writing back. Only the `translation` field is treated as authoritative user-editable content.

### Run step defaults

Update `Run step` choices to include an explicit main/full-process option and make it the default selection.

Expected choices:

- `full_pipeline`
- `combine_video`
- `tts_to_end`

Behavior:

- `full_pipeline` is the default UI value
- selecting `full_pipeline` behaves the same as the current normal run with no resume step
- `tts_to_end` remains the fast re-dub option after text edits

## Data Flow

### Loading texts

When the user clicks `Load texts`:

1. Build config from current UI values, especially `input`, `config`, `source_language`, `target_language`, and timing fields.
2. Compute the same translation cache key currently used by `SmartDubbing.translate_segments()`.
3. Load the cached translation payload from `cache/<input-hash>/translation/<cache-key>.pkl`.
4. Convert segment data into table rows:
   - `speaker`
   - `time` as a human-readable time range
   - `translation`
   - `original`
5. If a previously saved editable artifact exists, prefer it as the human-facing source only when it still matches the same segment structure; otherwise regenerate rows from cache.

### Saving texts

When the user clicks `Save texts`:

1. Reload the canonical cached translation payload.
2. Validate row count and row identity against the canonical segment list.
3. Apply only `translation` changes back into the cached segment list.
4. Save a human-readable artifact file to:
   - `prj/<input-stem>/artifacts/dubbing_texts.tsv`
5. Save the updated segment payload back to:
   - `cache/<input-hash>/translation/<cache-key>.pkl`

This keeps:

- a readable source for humans
- the exact cache artifact required by `run_step=tts_to_end`

## File Format

### Human-readable artifact

Use TSV because it is easy to inspect and diff.

Columns:

- `speaker`
- `time`
- `translation`
- `original`

Rules:

- UTF-8 encoding
- tabs inside text are normalized or escaped
- one row per dubbing segment

This file is treated as an exported working copy, not the system of record for playback. The cache payload remains the runtime source for `tts_to_end`.

## Validation and Error Handling

### Load failures

Show a clear message when:

- no input file is selected
- config cannot be validated
- required translation cache is missing
- cache payload is corrupted or has an unexpected structure

### Save failures

Reject save when:

- rows do not match the cached segment count
- a required translation value is missing
- the cache payload cannot be reloaded

Do not silently write partial updates.

## Integration Points

### Gradio UI

Modify `src/dubbing/ui/gradio_app.py` to:

- add the `Dubbing Texts` tab
- add load/save callbacks
- include new helper functions for reading and writing dubbing text rows
- extend the `Run step` dropdown with `full_pipeline` as default

### Runner

Modify `src/dubbing/core/runner.py` so that:

- `run_step=full_pipeline` is normalized to the same behavior as a normal full run
- existing `combine_video` and `tts_to_end` behavior remains unchanged

### SmartDubbing cache compatibility

Reuse existing translation cache key logic instead of inventing a parallel lookup path. The new editor should depend on the same cache contract currently used by `run_from_tts()`.

## Testing

Add tests that cover:

- new run-step choice list and default UI value
- presence of the new `Dubbing Texts` tab controls
- loading rows from cached translation segments
- saving edited rows writes `dubbing_texts.tsv`
- saving edited rows updates the `translation` cache `.pkl`
- `full_pipeline` default value maps to a normal full run
- save rejection when row counts do not match cached segments

## Recommendation

Implement the editor as a thin layer on top of the existing translation cache contract.

That keeps the user-facing workflow simple:

1. run full pipeline once
2. open `Dubbing Texts`
3. edit `translation`
4. save
5. run `tts_to_end`

It also minimizes behavioral risk because TTS continues reading from the same cached translation structure it already depends on today.
