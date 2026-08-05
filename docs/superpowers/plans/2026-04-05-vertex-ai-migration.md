# Vertex AI Migration Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Migrate every Google-backed runtime path in DubbLM to Vertex AI with ADC-based authentication and remove Google API key based execution for transcription, translation, refinement, and TTS.

**Architecture:** Introduce one shared Vertex AI configuration helper for `google-genai` clients and LlamaIndex Google GenAI setup. Keep the existing provider names and orchestration flow intact where possible, but make Vertex the only supported execution path for all Google-backed models.

**Tech Stack:** Python, `google-genai`, `llama-index-llms-google-genai`, pytest, Gradio, YAML/CLI config

---

## Chunk 1: Vertex Runtime Config

### Task 1: Add shared Vertex AI configuration helper

**Files:**
- Create: `src/google_vertex.py`
- Test: `tests/test_google_vertex.py`

- [ ] **Step 1: Write the failing tests**
- [ ] **Step 2: Run test to verify it fails**
- [ ] **Step 3: Write minimal implementation**
- [ ] **Step 4: Run test to verify it passes**

### Task 2: Wire google-genai transcription and TTS through Vertex

**Files:**
- Modify: `src/transcription/gemini_transcriber.py`
- Modify: `src/tts/gemini_tts_wrapper.py`
- Test: `tests/test_gemini_transcriber.py`
- Test: `tests/test_tts_factory.py`

- [ ] **Step 1: Write failing tests for Vertex-only initialization**
- [ ] **Step 2: Run targeted tests to verify failures**
- [ ] **Step 3: Update runtime code to use shared Vertex helper**
- [ ] **Step 4: Re-run targeted tests to verify passes**

## Chunk 2: Translation And Refinement LLMs

### Task 3: Replace Gemini LlamaIndex client with Vertex-compatible Google GenAI client

**Files:**
- Modify: `src/translation/llm_translator.py`
- Modify: `pyproject.toml`
- Test: `tests/test_translation.py`

- [ ] **Step 1: Write failing tests for Vertex-backed LLM creation**
- [ ] **Step 2: Run targeted tests to verify failures**
- [ ] **Step 3: Implement minimal client swap and Vertex config plumbing**
- [ ] **Step 4: Re-run targeted tests to verify passes**

## Chunk 3: Product Surface And Defaults

### Task 4: Update defaults, docs, and UI for Vertex-only setup

**Files:**
- Modify: `src/dubbing/core/config.py`
- Modify: `src/dubbing/ui/gradio_app.py`
- Modify: `dubbing_config.yml`
- Modify: `.env.example`
- Modify: `README.md`
- Test: `tests/test_gradio_app.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Write failing tests for updated dependency/docs/config expectations**
- [ ] **Step 2: Run targeted tests to verify failures**
- [ ] **Step 3: Implement minimal config and documentation updates**
- [ ] **Step 4: Re-run targeted tests to verify passes**

## Chunk 4: Final Verification

### Task 5: Run focused regression suite

**Files:**
- Test: `tests/test_google_vertex.py`
- Test: `tests/test_gemini_transcriber.py`
- Test: `tests/test_translation.py`
- Test: `tests/test_transcription_factory.py`
- Test: `tests/test_tts_factory.py`
- Test: `tests/test_gradio_app.py`
- Test: `tests/test_runner.py`

- [ ] **Step 1: Run the focused pytest commands**
- [ ] **Step 2: Fix any regressions**
- [ ] **Step 3: Re-run the full focused suite and record evidence**
