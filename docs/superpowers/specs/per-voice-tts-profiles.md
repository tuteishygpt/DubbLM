# Per-voice TTS profiles

Status: implemented on `codex/per-voice-tts-profiles`.

## Motivation

Before this change, per-speaker TTS configuration was scattered across five
parallel dicts in `dubbing_config.yml`:

- `tts_system_mapping: {SPEAKER_XX: backend}`
- `voice_name: {SPEAKER_XX: voice}` (or a bare string)
- `voice_prompt: {SPEAKER_XX: style_prompt}`
- `reference_audio_mapping: {SPEAKER_XX: path}`
- `reference_text_mapping: {SPEAKER_XX: text}`

Two problems:

1. **Broken multi-backend routing.** `SmartDubbing._initialize_tts_systems` only
   instantiated one TTS client — the one named by the top-level `tts_system`.
   Any speaker mapped to a different backend via `tts_system_mapping` silently
   fell back to the default client at synthesis time.
2. **No per-speaker model / provider params.** Two speakers on Gemini couldn't
   use different Gemini model IDs; two speakers on OmniVoice couldn't have
   different `num_steps`.

## New shape

A single `voices:` block in `dubbing_config.yml`:

```yaml
voices:
  SPEAKER_00:
    tts_system: gemini
    model: gemini-2.5-flash-preview-tts
    voice_name: Kore
    style_prompt: "calm, friendly narrator"
    params:
      temperature: 0.9

  SPEAKER_01:
    tts_system: omnivoice
    reference_audio: D:/voices/speaker_03.wav
    reference_text: "И тут прямо какие-то Анадырь..."
    params:
      instruct: ""
      num_steps: 32
      guidance_scale: 3.0

  "*":
    tts_system: omnivoice
```

- Every key is a diarization speaker ID (`SPEAKER_00`, `SPEAKER_01`, …).
- `"*"` is the fallback profile applied to any speaker not listed.
- `params:` is a per-provider knob bag. Unknown top-level keys inside a profile
  are folded into `params` too, so
  `{tts_system: omnivoice, num_steps: 32}` == `{tts_system: omnivoice, params: {num_steps: 32}}`.
- Fields not set in a profile inherit from the `"*"` fallback, and then from
  the top-level `tts_system` / `tts_model` / `voice_name`
  defaults in the config.

## Runtime

`src/dubbing/core/voice_profiles.py`:

- `VoiceProfile` dataclass with `pool_key()` for identity comparison
  (`(system, model, sorted(params))`).
- `normalize_voices(config)` returns `dict[str, VoiceProfile]`, folding the
  legacy fields into the same shape.
- `resolve_profile(profiles, speaker, tts_system_default=...)` looks up a
  speaker with `"*"` fallback and TTS-system default.

`SmartDubbing._initialize_tts_systems` now builds a **client pool** keyed by
`pool_key`: one TTS client per unique `(system, model, params)` combination.
Multiple speakers with identical settings share a single client; speakers with
different backends or model IDs get their own.

Segment routing (`synthesize_speech`, `resynthesize_one_segment`) walks each
speaker through `_resolve_voice_profile → pool_key → tts_clients[pool_key]`.
Segments are grouped by `pool_key`, not by system name, so the batch synth
loop calls each client with its own subset.

## Legacy compatibility

The five legacy dicts still work — `normalize_voices` folds them into the
`voices` shape on load and emits a single `DeprecationWarning` per run listing
which legacy fields were used. `voices:` entries override legacy entries for
the same speaker; other speakers still pick up their legacy settings so
partial migrations are safe.

## UI

`Settings → TTS` has:

- Top defaults (`TTS system`, `TTS model`, `Fallback TTS model`,
  `Voice name`) — behave as the implicit `"*"` fallback.
- A `Voices (YAML)` textarea for the full per-speaker block. On load the
  legacy fields are folded into this text; on save the text is round-tripped
  as YAML back into `voices:` in `dubbing_config.yml`.
- A collapsed `Legacy per-speaker fields (deprecated)` accordion holding the
  old `TTS system mapping JSON` and `Voice prompt JSON` for reference.

## Tests

- `tests/test_voice_profiles.py` — normalize + resolve behavior, JSON string
  in override path, legacy → new-style precedence.
- `tests/test_smart_dubbing_voices.py` — client pool identity across
  duplicate/unique profiles, `"*"` fallback for unlisted speakers,
  legacy-only configs still route correctly.

## Files touched

- `src/dubbing/core/voice_profiles.py` (new)
- `src/dubbing/core/config.py` — `voices` default + YAML/JSON parsing hook +
  `normalize_voices` call in `process_special_parameters`.
- `src/dubbing/core/runner.py` — `voices` in `_YAML_FIELDS` for UI/CLI overrides.
- `src/dubbing/core/smart_dubbing.py` — `_initialize_tts_systems` rewritten as
  client pool; `_get_voice_profile`, `_profile_pool_key`, `_build_tts_client`;
  synth/resynth callsites route via `pool_key`.
- `src/dubbing/ui/gradio_app.py` — `voices` textarea in Settings tab.
