# Higgs TTS and Strict Reference Modes Design

## Summary

Add `higgs` as a lazily loaded TTS provider backed by the public Hugging Face
Space `archivartaunik/higgs-audio-v3-tts`. At the same time, replace implicit
reference-audio fallback chains with an explicit `reference_mode` contract for
every TTS provider that supports voice cloning.

The pipeline must resolve and validate every reference required by an active
TTS generation pass before sending the first synthesis request. Missing or
invalid references fail the entire TTS stage with one aggregated error. No
provider may silently switch from one reference source to another.

This change is configured through `dubbing_config.yml` and the existing Gradio
settings surface. It adds no CLI flags and does not modify a user's local
`dubbing_config.yml` during implementation.

## Goals

- Register a new `higgs` backend through the existing lazy TTS factory.
- Implement the exact Gradio `/synthesize` contract exposed by
  `archivartaunik/higgs-audio-v3-tts`.
- Make reference acquisition explicit and provider-independent through
  `VoiceProfile.reference_mode`.
- Apply strict reference-mode rules to every cloning-capable backend, not only
  Higgs.
- Resolve all references and run all provider preflight checks before the first
  TTS synthesis request.
- Preserve new-style `voices:` entries as authoritative over deprecated legacy
  mappings.
- Keep provider-specific generation knobs in `VoiceProfile.params` rather than
  adding Higgs branches to `SmartDubbing`.

## Non-goals

- Adding or changing CLI arguments.
- Running the Higgs model locally.
- Calling the Space's `/transcribe` endpoint.
- Automatically choosing a reference mode.
- Falling back between configured, segment, and speaker references.
- Repairing unrelated baseline test failures.
- Editing the user's current `dubbing_config.yml`.

## Current State and Problems

### Existing provider architecture

All TTS backends implement `TTSInterface`, are lazily registered in
`tts_factory.py`, and receive per-segment data through `TTSSegmentData`.
`SmartDubbing` groups speakers by a client-pool key derived from system, model,
fallback model, and provider parameters.

The unified `voices:` model already carries `tts_system`, voice metadata,
`reference_audio`, `reference_text`, and a provider-specific `params` bag.
This is the correct extension point for Higgs.

### Implicit reference fallback

The current `_apply_reference_fallbacks` path tries several sources in order:
manual mappings, a clip from the current recognized segment, and an extracted
per-speaker WAV. This makes the effective source dependent on runtime file
availability rather than explicit configuration.

### Legacy mappings can override new intent

`normalize_voices` documents that a `voices:` entry overrides deprecated
per-speaker mappings for the same speaker. However,
`_apply_configured_reference_mapping` directly reads
`reference_audio_mapping` and `reference_text_mapping` again when a normalized
profile has no reference. Consequently, a new profile can accidentally reuse
an old reference from the same speaker.

### Provider errors occur too late

Existing wrappers validate references while iterating over segments. A batch
can therefore partially synthesize before a later segment discovers a missing
reference. In addition, `SmartDubbing` catches batch exceptions and retries
segments individually, which is inappropriate for configuration failures.

## Configuration Contract

### `VoiceProfile.reference_mode`

Add `reference_mode` as a first-class, optional `VoiceProfile` field. It is not
part of `params` because reference acquisition is pipeline behavior shared by
multiple providers.

Allowed values are:

| Mode | Source | Required behavior |
| --- | --- | --- |
| `configured` | `VoiceProfile.reference_audio` | Require an existing local file. Use `VoiceProfile.reference_text` when supplied. |
| `segment` | Current recognized segment | Export exactly that segment from the selected reference-audio source and use the segment's original recognized `text`. |
| `speaker` | `speakers_audio/<speaker>.wav` | Require the previously extracted per-speaker WAV. Use explicitly configured reference text when supplied. |
| `none` | No reference | Explicitly disable cloning. Allowed only for providers whose reference support is optional. |

There is no implicit default. A cloning-capable provider without an effective
`reference_mode` is a configuration error.

The normal `"*"` profile inheritance applies. A fallback profile may set a
mode and provider parameters for every otherwise-unlisted speaker.

### Examples

```yaml
voices:
  "*":
    tts_system: higgs
    reference_mode: segment
    params:
      space_id: archivartaunik/higgs-audio-v3-tts
      api_name: /synthesize
      temperature: 0.7
      top_p: 0.95
      top_k: 50
      max_new_tokens: 2048
      seed: -1

  SPEAKER_00:
    reference_mode: configured
    reference_audio: D:/voices/speaker_00.wav
    reference_text: Original reference transcript

  SPEAKER_01:
    reference_mode: speaker
```

For an optional-cloning provider:

```yaml
voices:
  SPEAKER_00:
    tts_system: bextts
    reference_mode: none
```

### Legacy mappings

`normalize_voices` may continue to fold deprecated fields into profiles for
unrelated legacy compatibility, but legacy reference mappings do not infer a
`reference_mode`. A cloning-capable legacy-only profile therefore fails with a
migration message until the user adds an explicit new-style `voices:` entry
and `reference_mode`.

When a speaker exists in the raw new-style `voices:` block, no deprecated
mapping for that same speaker may be read again later. Deprecated mappings for
other speakers may still be normalized, subject to the explicit-mode rule.

## Provider Reference Capabilities

Expose a small provider capability on `TTSInterface` with three values:

- `unsupported`: the provider does not consume reference audio;
- `optional`: the provider supports cloning but can synthesize without it;
- `required`: this application requires a valid reference for the provider.

Initial classifications:

| Provider | Capability |
| --- | --- |
| `openai`, `gemini` | `unsupported` |
| `bextts` | `optional` |
| `coqui`, `xtts`, `f5`, `omnivoice`, `higgs` | `required` |

The application classifies Higgs as `required` even though the underlying
Space accepts an empty reference, because the product requirement is to use
Higgs only with an explicitly sourced clone reference.

Validation rules:

- `unsupported` plus any `reference_mode` is an error;
- `optional` requires an explicit mode and accepts all four modes;
- `required` requires an explicit mode and rejects `none`;
- every non-`none` mode must resolve to an existing local file before
  synthesis;
- an unknown mode is an error naming the provider and speaker.

## Strict Reference Resolution

Reference resolution becomes a focused, provider-independent operation rather
than a fallback chain. Given a profile, speaker, source segment, and requested
mode, it returns one resolved audio path and reference text or raises a
descriptive error.

### `configured`

- Read only the effective new-style profile's `reference_audio`.
- Expand and normalize the local path without changing the configured value.
- Require `Path.is_file()`.
- Never try a segment clip or speaker WAV if the file is absent.

### `segment`

- Use the same reference-audio source selected by the pipeline (source audio or
  separated vocals when `keep_background` is enabled).
- Require valid `start` and `end` timestamps.
- Require duration greater than or equal to
  `segment_reference_min_duration`.
- Export `speakers_audio/segments/<speaker>_<segment_index>.wav`.
- Set `reference_text` to the recognized source `segment["text"]`.
- Never switch to a configured or speaker reference when export is impossible.

### `speaker`

- Use only `speakers_audio/<speaker>.wav`.
- Require the file to exist.
- Never switch to a configured file or segment clip.

### `none`

- Clear reference audio and text in `TTSSegmentData`.
- Rely on provider capability validation to allow or reject the mode.

## Preflight and Data Flow

For a generation pass that is not satisfied by a complete final-audio cache:

1. Resolve each segment's effective `VoiceProfile`, including `"*"`
   inheritance.
2. Determine the provider's reference capability.
3. Validate that the explicit mode is allowed for that capability.
4. Materialize the single configured reference source for every segment.
5. Build all final `TTSSegmentData` objects, including `reference_mode`.
6. In a separate preflight loop, call `validate_segments()` on every active TTS
   client.
7. Aggregate all reference errors across providers, speakers, and segments.
8. If any error exists, raise one `TTSReferenceValidationError` before entering
   the synthesis loop.
9. Only after all pools pass preflight may the first `synthesize()` call run.

The preflight loop must be outside the existing batch-synthesis `try` block so
configuration failures are not swallowed by individual retry behavior.

When a complete synthesized-audio cache is reused, no generation pass starts
and no reference is needed. Per-segment cache hits inside an active generation
pass do not bypass validation for segments that still require synthesis.

Single-segment resynthesis must use the same strict resolver and client
validation before its call. It may validate only the selected segment because
that operation is itself a one-segment TTS stage.

## TTS Interface Changes

Add the following shared concepts without making existing implementations
reimplement boilerplate:

- a reference capability class attribute with default `unsupported`;
- a concrete `validate_segments(segments_data)` hook on `TTSInterface`;
- generic validation for mode/capability compatibility and required local
  files;
- an explicit `reference_mode` field on `TTSSegmentData`.

Cloning wrappers set their capability. Each wrapper calls
`validate_segments()` defensively at the start of `synthesize()` so direct
wrapper use follows the same contract. The pipeline-level preflight remains
necessary to guarantee that no pool begins before all pools validate.

## Higgs Provider

### Registration

Register `higgs` in the existing lazy factory map. Importing `tts_factory`
must not import `higgs_audio_wrapper` or its optional Gradio dependencies.

### Wrapper defaults

`HiggsAudioWrapper` implements `TTSInterface` with these defaults:

| Argument | Default |
| --- | --- |
| `space_id` | `archivartaunik/higgs-audio-v3-tts` |
| `api_name` | `/synthesize` |
| `temperature` | `0.7` |
| `top_p` | `0.95` |
| `top_k` | `50` |
| `max_new_tokens` | `2048` |
| `seed` | `-1` |
| `hf_token_env` | `HF_TOKEN` |

These values are overridden through `VoiceProfile.params`. No `higgs_*`
top-level settings or Higgs-specific `SmartDubbing` branch is introduced.

### Space request

For each segment, call `Client.predict` with:

```python
client.predict(
    text=segment.text,
    reference_audio=handle_file(segment.reference_audio_path),
    reference_text=segment.reference_text or "",
    temperature=temperature,
    top_p=top_p,
    top_k=top_k,
    max_new_tokens=max_new_tokens,
    seed=seed,
    api_name=api_name,
)
```

The wrapper does not call `/transcribe`; `segment` mode already provides the
recognized source text and the other modes retain their configured/provider
semantics.

### Authentication and lifecycle

- Read a token from `hf_token_env`.
- Prefer `Client(space_id, hf_token=token)` and retain the headers fallback for
  older `gradio_client` versions.
- Permit anonymous access with a warning when no token is set.
- Create a private temporary directory for downloaded segment outputs.
- Copy each generated result to `TTSSegmentData.output_path`.
- Remove downloaded temporary files and clean the private directory in
  `cleanup()`.

### Output and alignment

The Space returns an audio filepath. The wrapper must accept the common Gradio
filepath shapes already handled by the OmniVoice integration, load the result
with `pydub`, measure actual duration, copy it to the requested output, and
return a `SegmentAlignment` per successful segment.

The wrapper uses the existing text-length heuristic for duration estimation;
Higgs has no duration or speed input in the inspected Space API.

### Error behavior

- Missing dependencies fail wrapper construction with actionable install
  messages.
- Client initialization errors name the Space.
- Reference validation raises before any per-segment prediction.
- Runtime Space failures are logged per segment and return no alignment for the
  failed segment, preserving the pipeline's existing retry behavior for
  transient generation failures.
- Configuration/reference errors are never treated as transient runtime
  failures.

## Gradio and Documentation

- Add `higgs` to the Gradio TTS dropdown.
- Update the `voices` YAML placeholder to demonstrate `reference_mode` and
  Higgs `params`.
- Document strict reference modes and the legacy migration requirement.
- Update provider lists in repository guidance where they describe supported
  TTS systems.
- Do not add CLI choices or flags.

## Error Messages

The aggregate exception must be actionable and deterministic. Example:

```text
TTS reference validation failed before synthesis:
- provider=higgs speaker=SPEAKER_01 segment=3 mode=configured:
  reference file does not exist: D:\voices\missing.wav
- provider=omnivoice speaker=SPEAKER_04 segment=8 mode=segment:
  recognized segment duration 0.72s is below the configured minimum 2.00s
```

Errors should be ordered by segment index. They must include provider, speaker,
segment, mode, and the concrete reason.

## Testing Strategy

### Voice profile tests

- Parse and inherit all four `reference_mode` values.
- Include `reference_mode` in profile merge behavior but not client-pool
  identity, because the same client may serve speakers with different
  per-segment reference sources.
- Prove a new `voices:` entry cannot reuse same-speaker legacy reference
  mappings.
- Prove deprecated mappings for other speakers remain normalized but do not
  acquire an implicit mode.

### Reference resolver tests

- `configured` accepts only its configured existing file.
- Missing configured files do not fall back.
- `segment` exports the exact interval and matching recognized text.
- Short, invalid, or unexportable segments fail without fallback.
- `speaker` accepts only the expected per-speaker path.
- `none` clears reference values.

### Capability and preflight tests

- Unsupported providers reject a reference mode.
- Optional providers require an explicit mode and accept `none`.
- Required providers reject missing mode and `none`.
- Mixed TTS pools collect every reference error and make zero synthesis calls.
- A valid mixed-provider pass validates all pools before the first synthesis
  call.
- Single-segment resynthesis applies the same strict rules.

### Higgs wrapper tests

- Factory lists Higgs while retaining lazy imports.
- Wrapper initialization supports authenticated and anonymous clients.
- Predict receives the exact Space field names and configured values.
- Segment reference audio and text are passed unchanged.
- Missing references fail before `Client.predict`.
- Gradio filepath results are copied to the requested output and produce the
  expected alignment duration.
- Cleanup removes temporary resources.

### UI tests

- Gradio exposes `higgs` in the TTS dropdown.
- The YAML example contains `reference_mode` and Higgs parameters.
- Settings round-trip through the existing `voices` YAML path.

### Verification

- Run focused new and modified test modules during TDD.
- Run the complete test suite at the end.
- Compare final full-suite results with the recorded baseline of 258 passing
  and 2 unrelated pre-existing failures; no new failures are acceptable.

## Compatibility and Migration

This is an intentionally strict configuration change for cloning-capable
providers. Existing configs that relied on implicit reference selection must
add `reference_mode` under `voices:`. The error message explains the required
migration rather than silently selecting a source.

Non-cloning providers continue to work without `reference_mode`. Provider
generation parameters and per-speaker references remain in the established
`voices:` schema. Client pooling remains based on provider construction
settings; reference mode and files remain per-segment data.

## Acceptance Criteria

- `higgs` can be selected in Gradio and instantiated lazily by `TTSFactory`.
- Higgs requests match the inspected Space API exactly.
- No CLI code is changed.
- Every cloning-capable provider requires an explicit allowed
  `reference_mode`.
- No reference source silently falls back to another source.
- Same-speaker legacy mappings cannot override or fill a new-style profile.
- All active pools validate before the first synthesis request.
- Missing references produce one aggregate, actionable exception.
- Existing final-audio cache reuse remains possible without an unnecessary
  reference check.
- Focused tests pass, and the full suite introduces no failures beyond the two
  confirmed baseline failures.
