# TTS Resume Run Step Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a `run_step` mode that resumes from cached translation data, regenerates TTS audio, replaces generated audio artifacts, and finishes by rendering a new video.

**Architecture:** Route the new workflow through `runner.py`, but keep the resume logic in `SmartDubbing` so CLI and UI share the same implementation. The resume path should reconstruct the same extracted-audio cache key, load cached translation and optional emotion artifacts, rerun synthesis, and then reuse the existing video-combine path.

**Tech Stack:** Python, Gradio, pytest, existing DubbLM pipeline/cache infrastructure

---

## Chunk 1: Surface The New Step

### Task 1: Add failing interface tests

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `tests/test_gradio_app.py`

- [ ] **Step 1: Write failing parser/runner/UI tests**
- [ ] **Step 2: Run targeted tests to verify they fail**
- [ ] **Step 3: Implement the minimal parser/UI/runner changes**
- [ ] **Step 4: Re-run targeted tests to verify they pass**

## Chunk 2: Resume From TTS

### Task 2: Add failing SmartDubbing resume tests

**Files:**
- Modify: `tests/test_runner.py`
- Modify: `src/dubbing/core/smart_dubbing.py`
- Modify: `src/dubbing/core/runner.py`

- [ ] **Step 1: Write failing tests for `tts_to_end` success and missing-cache errors**
- [ ] **Step 2: Run targeted tests to verify they fail**
- [ ] **Step 3: Implement `SmartDubbing.run_from_tts()` and wire it into `runner.py`**
- [ ] **Step 4: Re-run targeted tests to verify they pass**

## Chunk 3: Verify End-To-End Wiring

### Task 3: Confirm integration behavior

**Files:**
- Modify: `src/dubbing/core/config.py`
- Modify: `src/dubbing/ui/gradio_app.py`

- [ ] **Step 1: Verify `run_step` choices and help text cover `tts_to_end`**
- [ ] **Step 2: Run focused pytest coverage for runner/UI/config behavior**
- [ ] **Step 3: Fix any regressions and keep tests green**
