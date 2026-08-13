# FastAPI + React Replacement Design

## Goal

Replace the Gradio application completely with a FastAPI backend and a React/Vite SPA while preserving every current user workflow. Remove Gradio after parity is verified.

## Scope

The replacement must support:

- video and isolated-track uploads;
- every pipeline run mode and existing configuration field;
- queued jobs with live status and logs;
- output, report, and artifact downloads;
- settings load and save;
- voice-profile CRUD;
- reference-audio library CRUD and profile assignment;
- dubbing-text load, edit, save, row selection, audio playback, and single-row regeneration.

Authentication, Redis/Celery, PostgreSQL, and true multi-user isolation are not implemented now. The boundaries described below must allow them to be added without changing the public API or frontend job model.

## Architecture

FastAPI owns the HTTP API and serves the production SPA build. React, TypeScript, and Vite provide the browser UI. Existing pipeline orchestration remains in `dubbing.core`; UI-specific business logic currently in `dubbing.ui.gradio_app` moves into framework-independent services.

Backend units:

- `api`: request validation, routes, SSE, downloads, and consistent errors;
- `services`: settings, voice profiles, reference library, dubbing texts, uploads, and jobs;
- `jobs`: a replaceable queue interface with an in-process FIFO implementation;
- `repositories`: replaceable job metadata and current-user boundaries;
- existing `dubbing.core.runner`: pipeline execution.

The in-process queue runs one pipeline job at a time. Each job has a UUID, an immutable configuration snapshot, timestamps, logs, result files, and one of `queued`, `running`, `succeeded`, `failed`, or `cancelled`. Jobs accepted while another runs remain queued.

The initial current-user dependency returns one anonymous local user. Records and service calls still carry an owner identifier so later authentication and tenant storage do not require route-contract changes.

## API

- `GET/PUT /api/config`: load or atomically save current settings.
- `GET /api/options`: field schema and available systems, models, voices, and modes.
- `POST /api/uploads`: store validated media under server-generated names.
- `POST/GET /api/jobs`, `GET /api/jobs/{id}`: submit and inspect jobs.
- `GET /api/jobs/{id}/events`: resumable SSE log and state events.
- `GET /api/jobs/{id}/files/{name}`: download registered job results only.
- `/api/voice-profiles`: list, create, update, and delete profiles.
- `/api/reference-library`: list, create, update, delete, and serve references.
- `GET/PUT /api/dubbing-texts`: load and atomically save editable segments.
- `POST /api/dubbing-texts/{index}/regenerate`: regenerate one selected segment.

Errors use `{code, message, field?, details?}`. Existing `DubbingConfig` normalization and validation run before a job is accepted.

## Frontend

The SPA contains five views:

1. **Workflow** — inputs, languages, run mode, isolated tracks, and submission.
2. **Jobs** — queue state, reconnecting live logs, results, reports, and artifacts.
3. **Settings** — all existing transcription, translation, refinement, TTS, timing, video/audio, and debug fields.
4. **Voice Profiles** — profile editor plus the reference-audio library and assignment actions.
5. **Dubbing Texts** — editable segment table, audio playback, save, and row regeneration.

API access is centralized in one typed client. Server state is not duplicated into unrelated components. SSE reconnects with the last received event ID.

## Files and safety

Uploads use generated paths and validated extensions. Download endpoints serve only files registered to the requesting owner/job or located through a repository-controlled reference. Arbitrary client-supplied filesystem paths are never served.

Settings, translation cache updates, snapshots, and library metadata use atomic replacement. Existing per-video project and artifact paths remain unchanged.

## Migration and verification

Implementation follows test-driven development. Required checks are:

- backend service and API tests, including traversal rejection and job lifecycle;
- SSE replay/reconnect tests;
- frontend component tests for forms, job monitoring, profile/reference actions, and dubbing-text editing;
- React production build;
- FastAPI smoke test serving the built SPA;
- existing non-Gradio pipeline tests.

Only after parity checks pass, remove the Gradio launcher, `dubbing.ui.gradio_app`, Gradio-only tests, console entry point, and Gradio dependencies. Documentation and launch instructions must point to the FastAPI application.

## Acceptance criteria

- Every current Gradio workflow is available in the SPA.
- A queued job runs through `dubbing.core.runner`, streams logs, and exposes registered results.
- Existing settings, voice profiles, references, caches, and project artifacts remain compatible.
- The production server serves both `/api/*` and the SPA.
- No runtime code, tests, entry points, or direct dependencies reference Gradio.
- Queue, current-user, and repository interfaces can later be replaced without changing frontend contracts.
