# CLAUDE.md

This file provides guidance to Claude Code (claude.ai/code) when working with code in this repository.

## Environment

- Windows dev environment; the project uses `.venv/` in the repo root. Always run through it:
  - PowerShell: `.\.venv\Scripts\Activate.ps1` then `python …`
  - Or directly without activating: `.\.venv\Scripts\python.exe …`
- Python 3.10+ (README claims 3.12+); FFmpeg on `PATH`.
- Credentials come from `.env` (loaded by the Gradio entry point). Google features run through Vertex AI + ADC (`gcloud auth application-default login`). See `docs/LAUNCH.md` for the Windows-specific launch notes.

## Common commands

Run the Gradio UI (http://localhost:7860):
```powershell
python gradio_app.py
```

Tests (pytest is not declared in `pyproject.toml`; use the venv):
```powershell
.\.venv\Scripts\python.exe -m pytest tests
.\.venv\Scripts\python.exe -m pytest tests/test_runner.py::test_build_config_from_overrides_parses_structured_fields
```
`tests/conftest.py` inserts both the repo root and `src/` onto `sys.path`, so tests import as `dubbing.*`, `tts.*`, `translation.*`, `transcription.*` without an editable install.

There is no lint/format configuration checked in. Do not add one unless asked.

## Architecture

`gradio_app.py` is a thin launcher that prepends `src/` to `sys.path` and delegates to `dubbing.ui.gradio_app:main`.

The Python package layout is unusual: `src/` contains four sibling top-level packages — `dubbing`, `transcription`, `translation`, `tts` — and a top-level module `google_vertex.py`. `pyproject.toml` sets `package-dir = {"" = "src"}` and lists them under `packages.find`. Import them as top-level, not as `dubblm.tts`.

### Pipeline orchestration

`src/dubbing/core/smart_dubbing.py::SmartDubbing` is the orchestrator. Everything else in `src/dubbing/` is a component it composes:
- `core/config.py::DubbingConfig` — config merge (defaults → YAML → programmatic/UI overrides). `validate()` requires `input`, `source_language`, `target_language`; `process_special_parameters()` creates the per-video working tree.
- `core/runner.py::run_dubbing_job` — shared programmatic/Gradio entry point. It normalizes structured overrides, then dispatches by `run_step` / `generate_speaker_report` and captures logs for the UI.
- `core/cache_manager.py` — caches transcription/translation between runs.
- `audio/`, `video/`, `debug/`, `utils/subtitle_utils.py` — audio extraction/separation, ffmpeg muxing, debug reporters, SRT helpers.

### Per-run working tree

Every input video gets a **project directory** (default: `prj/<input_stem>/`, controlled by `DEFAULT_PROJECTS_ROOT` in `core/config.py`). Under it:
- `artifacts/` — all intermediates
- `artifacts/audio/`, `speakers_audio/`, `audio_chunks/`, `su_audio_chunks/`, `debug/`, `translated_samples/`
- Well-known files: `transcription_path`, `timecodes_report_path`, `translated_audio_path`, `background_audio_path`

Do not hard-code paths — everything is derived from `config.get("project_dir")` and siblings. See `docs/dubblm-runtime-reference-be.md` for the runtime map (in Belarusian).

### Pluggable backends

Each stage is a factory returning an interface implementation:
- Transcription: `src/transcription/transcription_factory.py` — `whisperx`, `assemblyai`, `deepgram`, `gemini`, `pyannote_openai`
- Translation: `src/translation/translator_factory.py` — LLM-only; providers `gemini` (Vertex) or `openrouter`. Prompt scaffolding lives in `src/translation/prompts.py`.
- TTS: `src/tts/tts_factory.py` — `omnivoice`, `gemini`, `openai`, `coqui`/`xtts`, `bextts`, `f5`. `voices:` selects per-speaker backends.

When adding a new backend, implement its `*_interface.py` contract and register it in the factory; do not branch on system names outside the factory.

### google_vertex module

`src/google_vertex.py` is a top-level module (declared in `pyproject.toml` as `py-modules`). Vertex-based transcription/translation/TTS all route through it — keep auth logic centralized there.

### Special pipeline modes

Dispatch lives in `core/runner.py::run_dubbing_job` (Gradio + programmatic path). Each mode maps to a `SmartDubbing` method:

- `full_pipeline` → `run_pipeline` — normal end-to-end.
- `from_scratch` → `run_from_scratch` — wipes cache via `SmartDubbing._reset_input_cache` and then runs the full pipeline. Use this instead of `--no_cache` when you want previously cached blobs actually deleted, not just skipped.
- `transcribe_only` → `run_transcribe_only` — same cache wipe (`_reset_input_cache`) before running, then stops after `diarize_and_transcribe`; returns `transcription_path`. Emits original subtitles if requested.
- `translate_only` → `run_translate_only` — **resume mode**. Loads cached diarization+transcription from a previous `transcribe_only` (or full-pipeline) run via `SmartDubbing._load_cached_diarize_and_transcribe` and only re-runs translation. Fails with `FileNotFoundError` if no cached transcription exists — never re-runs the transcriber (would waste API credits). Returns the translated-subtitle path when saved, else `transcription_path`. Cache lookup uses the transcriber's `cache_step_name` + `default_cache_key(audio_file)` (each backend advertises its own).

`_reset_input_cache` deletes both the per-input tree (`cache/<input-hash>/…`) via `CacheManager.clear_input_cache` **and** the legacy top-level `cache/<step_name>/` directories that some transcription backends still write without an input-hash prefix — this is what makes "transcribe again" actually re-run the transcriber instead of loading a stale pickle. The wipe intentionally leaves `cache_manager.use_cache=True` so the fresh transcription/translation is written to cache; a follow-up `tts_to_end` then finds it. `--no_cache` would break that hand-off.
- `combine_video` → `VideoProcessor.combine_audio_with_video` — requires `translated_audio_path` and (optionally) `background_audio_path` from a prior run. Fails loudly if they're missing (see `_run_combine_video_step` in `core/runner.py`). Useful when iterating on watermark/mux options.
- `tts_to_end` → `run_from_tts` — resumes after translation; requires cached `translation` (and `emotions` when emotion analysis is on) in the same project directory. Clears TTS chunk caches before regenerating audio, and re-runs `speaker_processor.extract_speaker_audio` first so cloning-based TTS backends (OmniVoice/XTTS/F5/BexTTS) find their per-speaker reference wavs — without this the previous behavior silently skipped every segment and produced a translated_audio track full of silence, i.e. video with background only.
- `generate_speaker_report=true` → `generate_diarization_report` — diarize + dump samples only; used for building the reference-audio library at `speaker_reference_library/`.

When adding a new mode, wire it in `runner.py` and add the option to the Gradio dropdown in `src/dubbing/ui/gradio_app.py`.

### Dubbing Texts editor

The Gradio "Dubbing Texts" tab loads/saves the cached translation pickle (`cache/<input-hash>/translation/*.pkl`) that `tts_to_end` consumes. Each row is one segment with columns `Speaker | Start | End | Original | Translation | Synthesized text | Audio file`. All columns except `Audio file` are editable; edits are written back to the same pickle **and** to `artifacts/dubbing_texts.tsv`.

- `synthesized_text` is populated by `synthesize_speech` (and its resynthesis path) — it reflects the variant TTS actually spoke (`translation` / `short_translation` / `very_short_translation` / `long_translation` / `llm_adjusted`). To make it visible to the UI, `SmartDubbing._persist_synthesis_results` re-serializes segments back into the translation cache after `synthesize_speech` completes in both `run_pipeline` and `run_from_tts`.
- `SmartDubbing.resynthesize_one_segment(segments, index, override_text=None)` re-runs TTS for a single segment. The Gradio "Regenerate selected row" button uses the currently-selected row: if the user changed the `Synthesized text` column, that text is used for TTS; otherwise the `Translation` value is used. Only the one chunk in `artifacts/audio/chunks/<index>.wav` is rewritten — video reassembly still requires a follow-up `tts_to_end`.

## Config conventions

- **Per-speaker TTS profiles live under `voices:`.** Each entry (`SPEAKER_00`, `"*"`, …) is a `VoiceProfile` — `tts_system`, `model`, `voice_name`, `style_prompt`, `reference_mode`, `reference_audio`, `reference_text`, and provider-specific knobs in `params:`. `SmartDubbing` builds one TTS client per unique `(system, model, params)` combination — see `src/dubbing/core/voice_profiles.py` and `docs/superpowers/specs/per-voice-tts-profiles.md`.
- Legacy per-speaker TTS keys fail fast with a migration message; migrate them to `voices:`.
- `glossary` is a per-speaker dict too but unrelated to TTS. In UI/programmatic overrides structured fields may arrive as strings; `runner._normalize_override_value` decodes them.
- `keep_background: true` runs source separation; the README and `docs/LAUNCH.md` both warn it's RAM-heavy on videos >30 min.

## Docs to consult

- `docs/dubblm-runtime-reference-be.md` — runtime behavior reference (Belarusian).
- `docs/LAUNCH.md` — Windows launch/troubleshooting (Belarusian).
- `docs/superpowers/plans/` and `docs/superpowers/specs/` — historical design docs for individual subsystems (Gradio UI, Deepgram/Gemini transcription, Vertex migration, dubbing texts editor, TTS resume, per-video project dirs). Read these before touching the corresponding subsystem.
