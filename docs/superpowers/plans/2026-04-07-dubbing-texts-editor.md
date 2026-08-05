# Dubbing Texts Editor Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Add a Gradio tab for loading, editing, and saving dubbing translations, while keeping the editable artifact file and translation cache `.pkl` synchronized for `tts_to_end`.

**Architecture:** Extend the existing Gradio UI with a new tab and helper callbacks that derive the current project/cache paths from the same config logic used by the runtime pipeline. Persist edits both to a TSV artifact and to the cached translated segment list so the existing `run_from_tts()` path can continue unmodified.

**Tech Stack:** Python, Gradio, pytest, pickle, YAML, existing DubbLM config/runner/UI modules

---

## Chunk 1: Tests

### Task 1: Add failing UI tests for run-step defaults and the dubbing texts tab

**Files:**
- Modify: `D:/CodexPRJ/DubbLM/tests/test_gradio_app.py`

- [ ] **Step 1: Write failing tests**
- [ ] **Step 2: Run `pytest tests/test_gradio_app.py -k "dubbing_texts or full_pipeline" -v` and verify failure**
- [ ] **Step 3: Implement the minimum UI/runtime surface**
- [ ] **Step 4: Re-run the same tests and verify pass**

### Task 2: Add failing helper tests for translation text load/save behavior

**Files:**
- Modify: `D:/CodexPRJ/DubbLM/tests/test_gradio_app.py`

- [ ] **Step 1: Write failing tests for loading rows from translation cache**
- [ ] **Step 2: Write failing tests for saving rows to TSV and `.pkl`**
- [ ] **Step 3: Run the focused tests and verify failure**
- [ ] **Step 4: Implement the minimum helpers to satisfy them**
- [ ] **Step 5: Re-run the focused tests and verify pass**

## Chunk 2: Implementation

### Task 3: Add translation editor helpers and tab wiring

**Files:**
- Modify: `D:/CodexPRJ/DubbLM/src/dubbing/ui/gradio_app.py`

- [ ] **Step 1: Add path/cache helper functions derived from existing config/cache logic**
- [ ] **Step 2: Add load/save callbacks for dubbing texts**
- [ ] **Step 3: Add the `Dubbing Texts` tab with optimized column order and controls**
- [ ] **Step 4: Add `full_pipeline` as the default run-step choice**

### Task 4: Normalize `full_pipeline` in runtime execution

**Files:**
- Modify: `D:/CodexPRJ/DubbLM/src/dubbing/core/runner.py`

- [ ] **Step 1: Treat `full_pipeline` like the normal full run**
- [ ] **Step 2: Keep `combine_video` and `tts_to_end` behavior unchanged**

## Chunk 3: Verification

### Task 5: Run focused verification

**Files:**
- Test: `D:/CodexPRJ/DubbLM/tests/test_gradio_app.py`
- Test: `D:/CodexPRJ/DubbLM/tests/test_runner.py`

- [ ] **Step 1: Run `pytest tests/test_gradio_app.py tests/test_runner.py -v`**
- [ ] **Step 2: Fix any regressions**
- [ ] **Step 3: Re-run the focused suite until green**
