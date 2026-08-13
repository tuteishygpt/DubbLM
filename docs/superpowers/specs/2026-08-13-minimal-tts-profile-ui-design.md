# Minimal TTS Profile UI

Date: 2026-08-13
Status: approved for implementation planning

## Goal

Replace the raw `Voices (YAML)` editor and obsolete per-speaker controls with a
small structured editor for `voices:` profiles. Make the existing speaker
reference library assign examples directly to the selected profile. Keep
read-only compatibility with old YAML configuration shapes, but write only the
current `voices:` shape.

## In scope

- A master-detail editor with a compact profile list and one profile form.
- Add, update, and delete a `SPEAKER_XX` or `"*"` profile.
- Form fields:
  - `tts_system`
  - `model`
  - `voice_name`
  - `style_prompt`
  - `reference_mode`
  - `reference_audio`
  - `reference_text`
  - `params`, edited as YAML
- Local dropdown choices for registered TTS systems, known models, known
  catalog voices, and supported reference modes. Model and voice dropdowns
  allow an exact custom value. The sources are defined under "Choice metadata"
  below.
- Direct assignment of a selected speaker-reference-library example to the
  currently selected voice profile.
- Complete removal of `fallback_model` and `tts_fallback_model` from the model,
  runtime routing, factory inputs, UI, configuration defaults, and tests.
- Removal of legacy per-speaker controls and persistence paths from the UI.
- Compatibility loading of old YAML fields through the existing normalization
  boundary.

## Out of scope

- A migration wizard or migration confirmation dialog.
- Network discovery of models or voices.
- A provider-specific form builder for every possible `params` key.
- A new retry strategy or automatic model substitution.
- Changes to the reference-library on-disk format.
- Redesign of unrelated TTS, emotion, timing, or dubbing-text controls.

## UI

The TTS section contains:

1. A profile table showing speaker ID, TTS system, model, voice name, and
   reference mode.
2. One editor form for the selected row.
3. `Add profile`, `Save profile`, and `Delete profile` actions.
4. The existing reference-library save/delete controls and read-only table.
5. A `Use in selected profile` action for the selected library row.
6. The existing global `TTS prompt prefix` control.

The raw `Voices (YAML)` textbox, `Speaker reference mappings` table, legacy
accordion, top-level TTS system/model/voice/reference controls, and fallback
model control are removed from the UI. The `"*"` profile is the only fallback
edited by the new UI.

`params` uses one YAML textbox. The UI does not interpret provider-specific
keys beyond requiring the value to decode to a mapping.

### Choice metadata

Keep one small static UI mapping; do not add discovery or a general capability
framework.

- TTS systems come from `TTSFactory.get_available_providers()`.
- Gemini voice choices come from `ALL_GEMINI_VOICES`.
- OpenAI voice choices come from `ALL_OPENAI_VOICES`.
- Gemini model suggestions are the current primary model defaults already
  declared by the Gemini wrapper; OpenAI suggestions are `tts-1` and
  `tts-1-hd`. Custom model IDs remain allowed.
- A model is required for Gemini and OpenAI. For the other registered backends,
  the profile `model` field is optional because their constructors use local or
  provider parameters instead.
- Reference-mode choices use a small static provider-to-capability mapping that
  matches the wrappers' existing `reference_capability` declarations.

## Data flow

### Load

1. Load YAML configuration.
2. Normalize legacy per-speaker fields into `VoiceProfile` objects.
3. Preserve an explicit `voices:` entry over a same-speaker legacy entry.
4. If no explicit `"*"` profile exists, expose old top-level `tts_system`,
   `tts_model`, string `voice_name`, `reference_audio`, and `reference_text`
   values as the editable `"*"` profile. When either top-level reference value
   is present, set the migrated profile's reference mode to `configured`.
5. Ignore legacy `fallback_model`/`tts_fallback_model` values and log one clear
   warning; they never affect routing.
6. Convert normalized profiles to plain dictionaries for Gradio state.

### Edit

- Selecting a profile populates the form. Any uncommitted edits in the previous
  form are discarded. Profile state changes only through `Save profile`,
  `Delete profile`, or the explicit `Use in selected profile` library action.
- `Save profile` validates the form and replaces that profile in Gradio state.
- `Add profile` clears the form and enters a new unsaved draft. The new speaker
  ID is supplied in that form and becomes part of state only after
  `Save profile` passes validation; duplicate or invalid IDs are rejected then.
- `Delete profile` removes the selected profile after the normal button event;
  no additional confirmation workflow is added.
- Unsaved form edits are not persisted by the outer `Save settings` action;
  the user must use `Save profile` first.

### Reference library

`Use in selected profile` copies the selected library row's audio path and text
into the selected profile and sets `reference_mode: configured`. The library
label never becomes a diarization speaker ID. The assignment updates profile
state only; the existing `Save settings` action persists it under
`voices.<selected-speaker>`.

### Save

The UI serializes profile state to one plain `voices:` mapping. It removes these
keys from the YAML written by the UI:

- `tts_system_mapping`
- dictionary-form `voice_name`
- `voice_prompt`
- `reference_audio_mapping`
- `reference_text_mapping`
- `tts_fallback_model`
- all per-profile `fallback_model` keys
- migrated top-level `tts_system`, `tts_model`, string `voice_name`,
  `reference_audio`, and `reference_text`

The compatibility reader remains so old YAML files can still be opened. The
removed legacy fields are not shown or written by the UI.

## Validation and errors

- Speaker IDs must be `"*"` or match `SPEAKER_\d+`.
- Duplicate speaker IDs are rejected.
- Validation uses the effective profile after applying the `"*"` profile.
  Therefore, a named profile may leave an inheritable field blank, while the
  `"*"` profile itself must resolve to a `tts_system`.
- An effective `model` is required for Gemini and OpenAI. It is optional for the
  other registered backends.
- `params` must be empty or valid YAML that decodes to a mapping.
- Reference validation continues to use the provider capability contract.
- `reference_mode: configured` requires an existing reference file before
  synthesis.
- Validation errors leave profile state and the saved YAML unchanged and are
  displayed in the existing status output.

## Runtime behavior

A resolved profile selects one concrete `(tts_system, model, params)` client.
There is no fallback model in the profile or client-pool identity. Existing
retry and regeneration paths reuse that same client and the same model; this
feature adds no model-switching behavior.

## Internal boundaries

- Profile-state helpers convert between `VoiceProfile` objects, plain UI state,
  table rows, and editor values. They do not read or write files.
- UI event handlers perform selection and state mutation, then return Gradio
  updates.
- Reference-library helpers continue to own library file operations. Assignment
  only copies a library row into profile state.
- Configuration loading owns backward compatibility. UI saving owns removal of
  obsolete persisted keys.
- Runtime profile resolution owns TTS client selection and has no UI dependency.

## Tests

Focused tests must cover:

- new-style profile load and profile table rendering;
- legacy mappings load into profile state while same-speaker `voices:` wins;
- old top-level defaults become `"*"` when needed;
- add, update, delete, duplicate-ID, invalid-ID, and invalid-`params` behavior;
- library assignment updates only the selected profile and sets `configured`;
- UI save writes `voices:` and removes obsolete keys;
- model and voice choices update for the selected backend while allowing custom
  values;
- no `fallback_model` or `tts_fallback_model` participates in config or runtime;
- retries/regeneration retain the selected concrete model;
- existing voice-profile and reference-validation tests continue to pass after
  their obsolete fallback assertions are removed.

## Acceptance criteria

- A user can configure every supported `VoiceProfile` field except the removed
  fallback model without editing YAML.
- A library example can be assigned to the selected profile and survives
  `Save settings`/reload under `voices:`.
- No legacy per-speaker control is visible in the UI or written by it.
- Old YAML per-speaker mappings still load.
- Runtime never substitutes a different model after a TTS error.
- No feature listed under Out of scope is introduced.
