# Deepgram Transcription System Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add `deepgram` as a full transcription backend in config, CLI, runtime, and Gradio UI, using Deepgram utterances as the canonical diarized segment format.

**Architecture:** Introduce a dedicated `DeepgramTranscriber` backend that reads a local audio file, calls Deepgram with diarization and utterances enabled, and normalizes the response into the existing internal transcription contract. Wire backend selection through the current transcription factory, config parser, and Gradio UI without changing downstream pipeline consumers.

**Tech Stack:** Python, `deepgram-sdk`, pytest, Gradio, existing DubbLM transcription interfaces

---

## Chunk 1: Surface Area

### Task 1: Add failing tests for factory and parser support

**Files:**
- Modify: `tests/test_transcription_factory.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write the failing test**

```python
def test_transcription_factory_supports_deepgram_backend():
    ...

def test_argument_parser_accepts_deepgram_backend():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_transcription_factory.py tests/test_runner.py -k deepgram -v`
Expected: FAIL because `deepgram` is not supported yet.

- [ ] **Step 3: Write minimal implementation**

Update the transcription factory and CLI/config parser to accept `transcription_system="deepgram"`.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_transcription_factory.py tests/test_runner.py -k deepgram -v`
Expected: PASS

### Task 2: Add failing tests for UI support

**Files:**
- Modify: `tests/test_gradio_app.py`
- Modify: `src/dubbing/ui/gradio_app.py`

- [ ] **Step 1: Write the failing test**

```python
def test_build_app_lists_deepgram_in_transcription_system_choices():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_gradio_app.py -k deepgram -v`
Expected: FAIL because the UI does not expose the backend yet.

- [ ] **Step 3: Write minimal implementation**

Expose `deepgram` in the transcription dropdown.

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_gradio_app.py -k deepgram -v`
Expected: PASS

## Chunk 2: Backend

### Task 3: Add failing tests for Deepgram normalization

**Files:**
- Create: `tests/test_deepgram_transcriber.py`
- Create: `src/transcription/deepgram_transcriber.py`

- [ ] **Step 1: Write the failing test**

```python
def test_deepgram_transcriber_normalizes_utterances():
    ...

def test_deepgram_transcriber_requires_api_key():
    ...
```

- [ ] **Step 2: Run test to verify it fails**

Run: `pytest tests/test_deepgram_transcriber.py -v`
Expected: FAIL because the backend does not exist yet.

- [ ] **Step 3: Write minimal implementation**

Implement `DeepgramTranscriber` that:
- checks `DEEPGRAM_API_KEY`
- calls Deepgram with `utterances=True` and `diarize=True`
- uses utterances as canonical segments
- normalizes speakers to `SPEAKER_00+`
- returns `(speakers_rolls, transcription)`

- [ ] **Step 4: Run test to verify it passes**

Run: `pytest tests/test_deepgram_transcriber.py -v`
Expected: PASS

## Chunk 3: Verification

### Task 4: Run focused verification

**Files:**
- Test: `tests/test_transcription_factory.py`
- Test: `tests/test_deepgram_transcriber.py`
- Test: `tests/test_gradio_app.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Run focused tests**

Run: `pytest tests/test_transcription_factory.py tests/test_deepgram_transcriber.py tests/test_gradio_app.py tests/test_runner.py -v`
Expected: PASS

- [ ] **Step 2: Review changed files for scope**

Run: `git diff -- src/transcription src/dubbing/core src/dubbing/ui tests README.md requirements.txt pyproject.toml`
Expected: only Deepgram backend and surface wiring changes
