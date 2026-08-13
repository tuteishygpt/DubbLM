# FastAPI + React Replacement Implementation Plan

> **For agentic workers:** REQUIRED: Use superpowers:subagent-driven-development (if subagents available) or superpowers:executing-plans to implement this plan. Steps use checkbox (`- [ ]`) syntax for tracking.

**Goal:** Replace the complete Gradio UI with a FastAPI API and React/Vite SPA, then remove Gradio.

**Architecture:** Move reusable behavior out of `dubbing.ui.gradio_app` into framework-independent services. FastAPI exposes those services plus a persisted single-worker FIFO job queue; a typed React SPA consumes the API and SSE events. Future auth and external queue/database implementations stay behind interfaces but are not built now.

**Tech Stack:** Python 3.10+, FastAPI, Pydantic, Uvicorn, pytest, React, TypeScript, Vite, Vitest, Testing Library.

**Spec:** `docs/superpowers/specs/2026-08-13-fastapi-react-replacement-design.md`

---

## File map

- `src/dubbing/web/schema.py` — field inventory, option lists, and API DTOs.
- `src/dubbing/web/contracts.py` — replaceable repository, media, queue, and current-user protocols.
- `src/dubbing/web/settings.py` — revisioned YAML settings and voice profiles.
- `src/dubbing/web/references.py` — reference-library storage.
- `src/dubbing/web/dubbing_texts.py` — cached segment editing/regeneration.
- `src/dubbing/web/storage.py` — uploads, safe registered files, atomic writes.
- `src/dubbing/web/jobs.py` — job/event repository and pipeline-facing job service.
- `src/dubbing/web/queue.py` — one-worker FIFO queue.
- `src/dubbing/web/routes/` — thin FastAPI route modules by resource.
- `src/dubbing/web/app.py` — application wiring, lifespan, errors, and SPA fallback.
- `frontend/src/api/` — typed HTTP/SSE client.
- `frontend/src/views/` — the five user-facing views.
- `web_app.py` — thin local launcher.

## Chunk 1: Framework-independent behavior

### Task 1: Settings schema and voice profiles

**Files:**
- Create: `src/dubbing/web/__init__.py`
- Create: `src/dubbing/web/contracts.py`
- Create: `src/dubbing/web/schema.py`
- Create: `src/dubbing/web/settings.py`
- Create: `tests/web/test_settings.py`
- Modify later: `src/dubbing/ui/gradio_app.py`

- [ ] **Step 1: Write the schema RED test** for the exact legacy workflow/settings field inventory, run modes, provider options, and model/voice/reference choices.
- [ ] **Step 2: Verify schema RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_settings.py -k schema -v`; expect a missing `dubbing.web.schema` failure.
- [ ] **Step 3: Move the schema constants** from `dubbing.ui.gradio_app` into `schema.py`; have legacy code import them temporarily.
- [ ] **Step 4: Verify schema GREEN.** Run the Step 2 command; expect the schema tests to pass.
- [ ] **Step 5: Write the settings RED tests** for YAML load, unrelated-key preservation, structured/list parsing, zero-duration normalization, content-hash revision, stale-write rejection, and prior-file preservation when `os.replace` fails.
- [ ] **Step 6: Verify settings RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_settings.py -k "settings or revision or atomic" -v`; expect failures for the unimplemented settings operations.
- [ ] **Step 7: Implement settings load/save** with typed results, domain exceptions, and sibling-temp atomic replacement; do not import Gradio or FastAPI.
- [ ] **Step 8: Verify settings GREEN.** Run the Step 6 command; expect all selected tests to pass.
- [ ] **Step 9: Write the profile RED tests** for validation, list, PUT upsert, delete, and reference assignment. PUT creates a missing `speaker_id` and replaces only an existing matching profile.
- [ ] **Step 10: Verify profile RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_settings.py -k profile -v`; expect failures for the unimplemented profile operations.
- [ ] **Step 11: Implement profile operations** in `SettingsService` over the same revisioned YAML document.
- [ ] **Step 12: Verify Task GREEN.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_settings.py -v`; expect all tests to pass.
- [ ] **Step 13: Commit.** `git add src/dubbing/web tests/web/test_settings.py src/dubbing/ui/gradio_app.py && git commit -m "feat: extract web settings and voice profiles"`

### Task 2: Safe media and persisted jobs

**Files:**
- Create: `src/dubbing/web/storage.py`
- Create: `src/dubbing/web/jobs.py`
- Create: `tests/web/test_storage.py`
- Create: `tests/web/test_job_repository.py`

- [ ] **Step 1: Define protocol RED tests** proving consumers can depend on `MediaStore.save/get/delete/register` and `JobRepository.create/get/list/update/append_event` without concrete filesystem types.
- [ ] **Step 2: Write media RED tests** for streamed size/extension/FFprobe validation, partial cleanup, owner isolation, traversal rejection, registered-file lookup, referenced-upload deletion rejection, and job materialization that preserves the sanitized original basename/stem.
- [ ] **Step 3: Verify media RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_storage.py -v`; expect missing contract/storage failures.
- [ ] **Step 4: Implement `FileMediaStore`** under an injected root and against the `MediaStore` protocol; routes and services receive only opaque media IDs/references.
- [ ] **Step 5: Verify media GREEN.** Run the Step 3 command; expect all tests to pass.
- [ ] **Step 6: Write repository RED tests** for owner-scoped CRUD/list pagination, atomic job JSON, monotonic JSONL events, 10 MiB retention, and restart conversion of `queued/running` to `failed(server_restarted)`.
- [ ] **Step 7: Verify repository RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_job_repository.py -v`; expect failures for the missing repository.
- [ ] **Step 8: Implement `FileJobRepository`** against the explicit protocol; keep event ordering inside the repository.
- [ ] **Step 9: Verify repository GREEN.** Run the Step 7 command; expect all tests to pass.
- [ ] **Step 10: Commit.** `git add src/dubbing/web tests/web/test_storage.py tests/web/test_job_repository.py && git commit -m "feat: add safe media and job persistence"`

### Task 3: Reference library and dubbing texts

**Files:**
- Create: `src/dubbing/web/references.py`
- Create: `src/dubbing/web/dubbing_texts.py`
- Create: `tests/web/test_references.py`
- Create: `tests/web/test_dubbing_texts.py`

- [ ] **Step 1: Write reference RED tests** by porting copy/list/delete/assign behavior; add content-hash revisions, stale-write rejection, atomic metadata replacement failure, and registered audio references.
- [ ] **Step 2: Verify reference RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_references.py -v`; expect failures for the missing service.
- [ ] **Step 3: Implement references.** Move the library helpers and depend on `MediaStore`.
- [ ] **Step 4: Verify reference GREEN.** Run the Step 2 command; expect all tests to pass.
- [ ] **Step 5: Write dubbing-text load RED tests** for translation cache, transcription seeding, snapshot preference, millisecond preservation, stable persisted `segment_id`, and synthesized-audio media references.
- [ ] **Step 6: Verify load RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_dubbing_texts.py -k load -v`; expect failures for missing load behavior.
- [ ] **Step 7: Implement loading** by moving the context/load helpers.
- [ ] **Step 8: Verify load GREEN.** Run the Step 6 command; expect all selected tests to pass.
- [ ] **Step 9: Write save/regenerate RED tests** for row-count validation, content revision conflict, atomic cache/snapshot/TSV replacement failure, preservation of the prior files, and one-row regeneration registered through `MediaStore`.
- [ ] **Step 10: Verify save/regenerate RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_dubbing_texts.py -k "save or regenerate" -v`; expect failures for missing write behavior.
- [ ] **Step 11: Implement save/regenerate** with `SmartDubbing.resynthesize_one_segment`, typed results, and domain errors; do not return filesystem paths.
- [ ] **Step 12: Verify Task GREEN.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_references.py tests/web/test_dubbing_texts.py -v`; expect all tests to pass.
- [ ] **Step 13: Commit.** `git add src/dubbing/web tests/web/test_references.py tests/web/test_dubbing_texts.py && git commit -m "feat: extract reference and dubbing text services"`

## Chunk 2: Queue and HTTP API

### Task 4: FIFO execution and pipeline integration

**Files:**
- Create: `src/dubbing/web/queue.py`
- Create: `tests/web/test_job_queue.py`
- Modify: `src/dubbing/web/jobs.py`
- Modify: `src/dubbing/core/runner.py`
- Modify: `tests/test_runner.py`

- [ ] **Step 1: Write submission RED tests** proving the service owner-checks and materializes the input plus every isolated-track upload with sanitized original basenames, resolves all paths into the config, merges and validates `DubbingConfig` before repository creation, stores the complete normalized snapshot, rejects invalid jobs, and is unaffected by later YAML edits.
- [ ] **Step 2: Add a runner RED test** for a streaming entry point that accepts an already validated/resolved `DubbingConfig` and never reloads YAML.
- [ ] **Step 3: Verify submission/runner RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_job_queue.py -k submission -v`, then `.\.venv\Scripts\python.exe -m pytest tests/test_runner.py -k validated_config -v`; expect failures for missing frozen submission and runner behavior.
- [ ] **Step 4: Implement validated submission/execution** in `JobService` and `core.runner`; keep CLI/current callers compatible.
- [ ] **Step 5: Verify submission/runner GREEN.** Run the Step 3 command; expect all selected tests to pass.
- [ ] **Step 6: Write queue RED tests** proving FIFO order, one active worker, queued/running/terminal transitions, live log events, result/report/artifact registration, failure payloads, and clean shutdown. Inject a fake validated-config runner; do not run ML models.
- [ ] **Step 7: Verify queue RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_job_queue.py -k queue -v`; expect failures for the missing worker behavior.
- [ ] **Step 8: Implement `InProcessJobQueue`.** The worker passes the frozen snapshot to the validated-config runner, appends events, and registers terminal files. Do not add cancellation or external queue code.
- [ ] **Step 9: Verify Task GREEN.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_job_queue.py tests/test_runner.py -v`; expect all tests to pass.
- [ ] **Step 10: Commit.** `git add src/dubbing/core/runner.py src/dubbing/web/queue.py src/dubbing/web/jobs.py tests/test_runner.py tests/web/test_job_queue.py && git commit -m "feat: add validated dubbing job queue"`

### Task 5: FastAPI routes, errors, and SSE

**Files:**
- Create: `src/dubbing/web/dependencies.py`
- Create: `src/dubbing/web/routes/config.py`
- Create: `src/dubbing/web/routes/uploads.py`
- Create: `src/dubbing/web/routes/jobs.py`
- Create: `src/dubbing/web/routes/voices.py`
- Create: `src/dubbing/web/routes/references.py`
- Create: `src/dubbing/web/routes/dubbing_texts.py`
- Create: `src/dubbing/web/app.py`
- Create: `tests/web/test_api.py`
- Create: `tests/web/test_sse.py`

- [ ] **Step 1: Write failing API tests** for every route in the spec, anonymous owner injection, request/response DTOs, pre-acceptance config validation, `404` ownership masking, `409` revisions, `413` upload limits, referenced-upload deletion rejection, validation errors in `{code,message,field?,details?}`, voice-profile PUT upsert, and registered audio playback.
- [ ] **Step 2: Write failing SSE tests** for event shapes, ordered IDs, `Last-Event-ID` replay, expired-history snapshot, 15-second heartbeat via an injected shorter test interval, and close-after-terminal.
- [ ] **Step 3: Verify RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_api.py tests/web/test_sse.py -v`; expect missing app/routes failures.
- [ ] **Step 4: Implement thin routes and app wiring.** Routes validate DTOs and delegate only; map domain exceptions centrally; use FastAPI lifespan to start/stop the queue.
- [ ] **Step 5: Verify GREEN.** Run the same command, then `.\.venv\Scripts\python.exe -m pytest tests/web -v`; expect all web backend tests to pass.
- [ ] **Step 6: Commit.** `git add src/dubbing/web tests/web && git commit -m "feat: expose dubbing web API"`

## Chunk 3: React/Vite SPA

### Task 6: Frontend foundation and typed client

**Files:**
- Create: `frontend/package.json`
- Create: `frontend/package-lock.json`
- Create: `frontend/tsconfig.json`
- Create: `frontend/vite.config.ts`
- Create: `frontend/index.html`
- Create: `frontend/src/api/types.ts`
- Create: `frontend/src/api/client.ts`
- Create: `frontend/src/App.tsx`
- Create: `frontend/src/main.tsx`
- Create: `frontend/src/styles.css`
- Create: `frontend/src/test/setup.ts`
- Create: `frontend/src/App.test.tsx`

- [ ] **Step 1: Add only the Vite/Vitest configuration and dependencies** needed for React, TypeScript, Testing Library, fetch, and SSE tests; do not add a state framework or component library.
- [ ] **Step 2: Write a failing shell/client test** for the five navigation views, shared loading/error states, DTO decoding, normalized API errors, and SSE reconnect using the last event ID.
- [ ] **Step 3: Verify RED.** Run `npm --prefix frontend test -- --run`; expect missing component/client failures.
- [ ] **Step 4: Implement the typed client and minimal accessible application shell.** Keep server state in view-level hooks and the API module.
- [ ] **Step 5: Verify GREEN.** Run the same command and `npm --prefix frontend run typecheck`; expect both to pass.
- [ ] **Step 6: Commit.** `git add frontend && git commit -m "feat: scaffold React web client"`

### Task 7: Workflow and Jobs views

**Files:**
- Create: `frontend/src/views/WorkflowView.tsx`
- Create: `frontend/src/views/WorkflowView.test.tsx`
- Create: `frontend/src/views/JobsView.tsx`
- Create: `frontend/src/views/JobsView.test.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write failing component tests** for video/isolated-track upload, speaker-label mapping, source/target languages, speaker-report mode, subtitle/audio switches, every schema-designated workflow field and run mode, override submission, validation display, queued/running/terminal states, reconnecting logs, and result/report/artifact links.
- [ ] **Step 2: Verify RED.** Run `npm --prefix frontend test -- --run src/views/WorkflowView.test.tsx src/views/JobsView.test.tsx`; expect missing-view failures.
- [ ] **Step 3: Implement both views** against the typed API; use native controls and preserve all workflow fields from `schema.py`.
- [ ] **Step 4: Verify GREEN.** Run the same command; expect all tests to pass.
- [ ] **Step 5: Commit.** `git add frontend/src && git commit -m "feat: add workflow and job monitoring views"`

### Task 8: Settings, voice profiles, and references

**Files:**
- Create: `frontend/src/components/SchemaField.tsx`
- Create: `frontend/src/views/SettingsView.tsx`
- Create: `frontend/src/views/SettingsView.test.tsx`
- Create: `frontend/src/views/VoicesView.tsx`
- Create: `frontend/src/views/VoicesView.test.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write failing tests** for schema-driven rendering of every settings field, structured/list values, revision conflicts, dependent model/voice/reference choices, profile upsert/delete, reference upload/list/delete, and assign-to-profile.
- [ ] **Step 2: Verify RED.** Run `npm --prefix frontend test -- --run src/views/SettingsView.test.tsx src/views/VoicesView.test.tsx`; expect missing-view failures.
- [ ] **Step 3: Implement the two views** with one reusable schema field component and accessible forms/tables. Do not replicate option lists in TypeScript.
- [ ] **Step 4: Verify GREEN.** Run the same command; expect all tests to pass.
- [ ] **Step 5: Commit.** `git add frontend/src && git commit -m "feat: add settings and voice management views"`

### Task 9: Dubbing Texts view

**Files:**
- Create: `frontend/src/views/DubbingTextsView.tsx`
- Create: `frontend/src/views/DubbingTextsView.test.tsx`
- Modify: `frontend/src/App.tsx`

- [ ] **Step 1: Write failing tests** for job selection, seeded/cached segment loading, editing all allowed columns, stable segment IDs, dirty state, revision conflict, row regeneration with synthesized-text override, missing-audio display, and playback from the registered audio URL.
- [ ] **Step 2: Verify RED.** Run `npm --prefix frontend test -- --run src/views/DubbingTextsView.test.tsx`; expect missing-view failures.
- [ ] **Step 3: Implement the editable table** with explicit save and selected-row regeneration. Keep audio file/path read-only.
- [ ] **Step 4: Verify GREEN.** Run the same command, then `npm --prefix frontend test -- --run` and `npm --prefix frontend run typecheck`; expect all frontend checks to pass.
- [ ] **Step 5: Commit.** `git add frontend/src && git commit -m "feat: add dubbing text editor"`

## Chunk 4: Integration and Gradio removal

### Task 10: Serve the SPA and prove parity

**Files:**
- Create: `web_app.py`
- Create: `tests/web/test_spa.py`
- Create: `tests/web/test_parity.py`
- Modify: `src/dubbing/web/app.py`
- Modify: `src/dubbing/core/runner.py`
- Modify: `pyproject.toml`
- Modify: `.gitignore`

- [ ] **Step 1: Write failing smoke/parity tests.** Assert `/api/*` remains API-only, static assets are served, unknown browser paths return `index.html`, the launcher uses Uvicorn, and the committed field/run-mode inventory exactly matches the extracted legacy inventory.
- [ ] **Step 2: Verify RED.** Run `.\.venv\Scripts\python.exe -m pytest tests/web/test_spa.py tests/web/test_parity.py -v`; expect missing launcher/static behavior failures.
- [ ] **Step 3: Implement production serving and entry point.** Add `dubblm-web`, serve `frontend/dist`, return a clear startup error when the build is absent, and ignore generated `frontend/dist`/`node_modules` plus `server_data`.
- [ ] **Step 4: Build and verify.** Run `npm --prefix frontend run build`, the Step 2 pytest command, and a TestClient smoke request; expect exit code 0 and built SPA HTML.
- [ ] **Step 5: Commit.** `git add web_app.py src/dubbing/web tests/web pyproject.toml .gitignore frontend && git commit -m "feat: serve the DubbLM SPA"`

### Task 11: Remove Gradio after the gate passes

**Files:**
- Delete: `gradio_app.py`
- Delete: `src/dubbing/ui/gradio_app.py`
- Delete: `src/dubbing/ui/__init__.py` if empty
- Delete: `tests/test_gradio_app.py`
- Delete: `tests/test_tts_profile_ui.py` after its service cases are covered by `tests/web/test_settings.py`
- Modify: `tests/test_anchor_timing.py`
- Modify: `tests/test_semantic_planner.py`
- Modify: `requirements.txt`
- Modify: `pyproject.toml`
- Modify: `README.md`
- Modify: `docs/LAUNCH.md`
- Modify: `CLAUDE.md`

- [ ] **Step 1: Run the pre-removal gate.** Run `.\.venv\Scripts\python.exe -m pytest tests/web tests/test_runner.py -v`, `npm --prefix frontend test -- --run`, `npm --prefix frontend run typecheck`, and `npm --prefix frontend run build`; require all four commands to pass.
- [ ] **Step 2: Port remaining UI-helper tests.** Change the affected anchor-timing and semantic-planner tests to exercise `SettingsService`/schema directly; delete `test_tts_profile_ui.py` only after its profile cases exist in `tests/web/test_settings.py`.
- [ ] **Step 3: Remove Gradio UI runtime/test files and dependency.** Remove `gradio==...` but retain `gradio-client` for OmniVoice/Higgs/BexTTS remote provider calls. Preserve historical specs/plans; update active documentation, runner messages, and commands to `web_app.py`/`dubblm-web`.
- [ ] **Step 4: Verify no active Gradio UI references.** Run `rg -n "import gradio|dubbing\.ui\.gradio_app|dubblm-gradio|gradio_app\.py|Gradio UI" src tests pyproject.toml requirements.txt README.md docs/LAUNCH.md CLAUDE.md`; expect no matches. Separately run `rg -n "gradio_client" src/tts requirements.txt`; expect matches only for the three remote TTS providers and the retained dependency.
- [ ] **Step 5: Run complete verification.** Run `.\.venv\Scripts\python.exe -m pytest tests -v`, `npm --prefix frontend test -- --run`, `npm --prefix frontend run typecheck`, and `npm --prefix frontend run build`; require zero failures and exit code 0.
- [ ] **Step 6: Manual smoke test.** Start `.\.venv\Scripts\python.exe web_app.py`, open the SPA, submit a short cached/sample run, confirm live logs and a downloadable result, then stop the server.
- [ ] **Step 7: Commit.** `git add -A && git commit -m "feat: replace Gradio with FastAPI and React"`
