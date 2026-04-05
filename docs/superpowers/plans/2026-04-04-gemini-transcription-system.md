# Gemini Transcription System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `gemini` as a full transcription backend in config, CLI, UI, and runtime, with a configurable Gemini transcription model and structured diarization/transcription output.

**Architecture:** Introduce a dedicated `GeminiTranscriber` backend that uploads the full audio file to Gemini, requests structured JSON segments, validates and normalizes those segments, and converts them into the existing internal transcription format. Wire the backend through the existing transcription factory and expose both backend selection and model selection in CLI/config/UI without changing downstream pipeline contracts.

**Tech Stack:** Python, `google-genai`, pytest, Gradio, existing DubbLM transcription interfaces

---

## Chunk 1: Surface Area

### Task 1: Add failing tests for factory and parser support

**Files:**
- Modify: `tests/test_transcription_factory.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_transcription_factory_supports_gemini_backend():
    ...

def test_argument_parser_accepts_gemini_transcription_model():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcription_factory.py tests/test_runner.py -k gemini -v`
Expected: FAIL because `gemini` is not supported yet.

- [ ] **Step 3: Write minimal implementation**

Update the transcription factory and CLI/config parser to accept:
- `transcription_system="gemini"`
- `gemini_transcription_model`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcription_factory.py tests/test_runner.py -k gemini -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_transcription_factory.py tests/test_runner.py src/transcription/transcription_factory.py src/dubbing/core/config.py
git commit -m "feat: add gemini transcription config surface"
```

### Task 2: Add failing tests for UI support

**Files:**
- Modify: `tests/test_gradio_app.py`
- Modify: `src/dubbing/ui/gradio_app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_app_lists_gemini_in_transcription_system_choices():
    ...

def test_build_app_exposes_gemini_transcription_model_field():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gradio_app.py -k gemini -v`
Expected: FAIL because the UI does not expose the backend/model yet.

- [ ] **Step 3: Write minimal implementation**

Expose `gemini` in the transcription dropdown and add a `Gemini transcription model` input that round-trips through UI load/save.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gradio_app.py -k gemini -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_gradio_app.py src/dubbing/ui/gradio_app.py
git commit -m "feat: expose gemini transcription backend in gradio"
```

## Chunk 2: Backend

### Task 3: Add failing tests for Gemini transcriber parsing and validation

**Files:**
- Create: `tests/test_gemini_transcriber.py`
- Create: `src/transcription/gemini_transcriber.py`

- [ ] **Step 1: Write the failing test**

```python
def test_gemini_transcriber_parses_segments_into_internal_format():
    ...

def test_gemini_transcriber_requires_google_api_key():
    ...

def test_gemini_transcriber_rejects_invalid_model_payload():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gemini_transcriber.py -v`
Expected: FAIL because the backend does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement `GeminiTranscriber` that:
- checks `GOOGLE_API_KEY`
- uses `google-genai`
- uploads the full audio file
- requests strict JSON with `segments`
- validates `start/end/speaker/text`
- normalizes speakers to `SPEAKER_00+`
- returns `(speakers_rolls, transcription)`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gemini_transcriber.py -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_gemini_transcriber.py src/transcription/gemini_transcriber.py
git commit -m "feat: add gemini transcription backend"
```

### Task 4: Wire runtime and dependencies

**Files:**
- Modify: `src/transcription/transcription_factory.py`
- Modify: `src/dubbing/core/smart_dubbing.py`
- Modify: `pyproject.toml`

- [ ] **Step 1: Write the failing test**

```python
def test_transcription_factory_passes_gemini_model_to_backend():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcription_factory.py -k gemini -v`
Expected: FAIL until kwargs wiring is complete.

- [ ] **Step 3: Write minimal implementation**

Pass `gemini_transcription_model` through runtime initialization and add `google-genai` as a dependency.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcription_factory.py -k gemini -v`
Expected: PASS

- [ ] **Step 5: Commit**

```bash
git add tests/test_transcription_factory.py src/transcription/transcription_factory.py src/dubbing/core/smart_dubbing.py pyproject.toml
git commit -m "feat: wire gemini transcription backend"
```

## Chunk 3: Verification

### Task 5: Run focused and regression verification

**Files:**
- Test: `tests/test_transcription_factory.py`
- Test: `tests/test_gemini_transcriber.py`
- Test: `tests/test_gradio_app.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_transcription_factory.py tests/test_gemini_transcriber.py tests/test_gradio_app.py tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 2: Run broader regression check**

Run: `pytest tests/test_translation.py tests/test_tts_factory.py -v`
Expected: PASS

- [ ] **Step 3: Review changed files for accidental scope creep**

Run: `git diff -- src/transcription src/dubbing/core src/dubbing/ui tests pyproject.toml`
Expected: only Gemini transcription surface/backend changes

