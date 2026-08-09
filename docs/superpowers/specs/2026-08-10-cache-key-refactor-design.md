# Focused Cache-Key Refactor Design

## Goal

Refactor the cache-key methods around `SmartDubbing._build_dubbing_text_snapshot_key()` so their inputs are explicit, their construction is deterministic, and changes to translation or emotion-analysis behavior cannot silently reuse incompatible cached results. Close the directly related inner translation-cache gap so an outer cache miss cannot fall through to a stale chunk hit. Compatibility with existing cache files is not required.

## Scope

The primary implementation remains the cache-key helper group beginning near `src/dubbing/core/smart_dubbing.py:512`. The directly coupled `LLMTranslator._generate_cache_key()` and its call sites are also in scope because they can otherwise return stale chunk translations after the outer key changes. Tests specifying both cache layers will change. The broader TTS and transcription cache-key construction remains unchanged.

## Design

Add a small private helper that produces a deterministic fingerprint from a mapping of cache dimensions. It will serialize the mapping as canonical JSON with sorted keys and compact separators, encode it as UTF-8, and use a truncated SHA-256 digest. Values will remain structured until serialization so underscores and similar characters cannot make different parameter sets produce the same preimage. A cache-schema version will be one of the dimensions so intentional algorithm or default changes have an explicit invalidation point.

Add a private helper for the shared audio/transcription identity currently passed to `CacheManager.generate_cache_key()`. It will use named local variables and preserve the existing input dimensions: audio content, source language, target language, Whisper model, start time, and duration.

The public-to-the-class key builders will compose readable namespaces with deterministic fingerprints:

- The dubbing-text snapshot key will include the shared audio/transcription identity and the effective translation prompt. It will remain independent of the semantic-plan fingerprint so the editor can locate the latest snapshot across transient or changed plans.
- The translation key will extend the snapshot identity with all configuration that can affect `LLMTranslator` output: translator type; effective primary provider/model/temperature/max tokens; effective refinement provider/model/temperature/max tokens/persona; glossary; effective prompt; and semantic-plan fingerprint. Provider-dependent defaults such as the Gemini/OpenRouter model and refinement fallbacks will be resolved to the same concrete values used by `LLMTranslator`, not represented as raw `None` values.
- The inner `LLMTranslator` chunk key will use the same canonical fingerprint approach and cover the chunk text, languages, effective primary model settings, glossary, prompt prefix, and every actual context input interpolated into the chunk prompt: preceding/following chunk context, source summary, domain, tone, ordered themes, and ordered terminology. Refinement-only settings will not be included in this initial-translation chunk key because refinement runs after the cached value is read; they remain dimensions of the outer final-translation key.
- The emotion-analysis key will accept the segments being annotated and fingerprint their full ordered input payload before analysis mutates it. This is required because the current cache stores and returns complete segment dictionaries, not only emotion fields; unchanged timings must not resurrect stale text or translations. The key will also include audio identity, normalized provider, provider-specific model identity, semantic-plan fingerprint, and algorithm identity. Gemini identity includes the effective model, exact prompt, fixed generation temperature, and accepted labels. SpeechBrain identity includes its fixed Hub model/classifier and label/style mapping. Provider/model components will occur once rather than being duplicated in both the base key and suffix.

Defaults used by the key builders must match the effective defaults used where the corresponding component is initialized or called. Resolution will be covered by tests for both a fully initialized `SmartDubbing` instance and the lightweight config-only instances used by the UI to locate cache files. The produced strings are internal cache identities; their exact old format is intentionally not preserved.

## Error Handling

Cache dimensions are expected to be JSON-compatible configuration values. The fingerprint helper will fail clearly for unsupported values rather than silently stringifying objects into unstable representations. Existing callers already catch persistence errors where cache writes are optional; translation and emotion cache lookup should surface invalid configuration early.

## Tests

Tests will be written before production changes and will first fail against the current implementation. They will verify:

1. Equivalent inputs produce identical keys regardless of mapping construction order.
2. Snapshot keys change when the effective translation prompt changes but do not change with the semantic plan.
3. Translation keys change for every translation-affecting configuration dimension listed above and for the semantic plan; raw missing values resolve identically to the translator's effective defaults.
4. Inner chunk keys change with each prompt-driving context field (neighboring context, source summary, domain, tone, themes, and terminology), glossary, and effective primary model settings, preventing a fresh outer translation from reusing an incompatible chunk.
5. Emotion keys change with any change to the ordered input segment payload, provider-specific algorithm/model identity, and semantic plan without duplicated readable components. A regression test will keep timings fixed while changing translation text to prove stale complete segments cannot be returned.
6. Existing snapshot persistence and translation/emotion cache behavior tests continue to pass after exact-key expectations are updated to behavioral assertions.

## Code Review Criteria

The final review will check for omitted behavior-affecting inputs, mismatched defaults, ambiguous serialization, accidental expansion into unrelated cache paths, and regression coverage. The unrelated existing modification to `dubbing_config.yml` will not be touched.
