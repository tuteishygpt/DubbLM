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

Required interfaces are deliberately small:

- `JobQueue.enqueue(job_id)` and lifecycle `start()/stop()`; only the queue chooses when work runs;
- `JobRepository.create/get/list/update/append_event`; only the repository owns job state and event ordering;
- `CurrentUser.id`; routes never invent or accept an owner ID;
- `MediaStore.save/get/delete`; routes never construct storage paths;
- domain services expose typed operations and do not import FastAPI or React concerns.

The in-process FIFO queue runs one pipeline job at a time. Each job has a UUID, an immutable configuration snapshot, timestamps, logs, result files, and one of `queued`, `running`, `succeeded`, or `failed`. Jobs accepted while another runs remain queued. Cancellation is out of scope because the current pipeline has no safe cooperative-cancellation contract.

Job metadata and events are atomically persisted as JSON/JSONL under `server_data/jobs/<owner>/<job-id>/`. On server start, previously queued or running jobs become failed with code `server_restarted`; terminal jobs remain readable. This is not a durable external queue, but it makes restart behavior deterministic and leaves `JobRepository` replaceable.

The initial current-user dependency returns one anonymous local user. Records and service calls still carry an owner identifier so later authentication and tenant storage do not require route-contract changes.

## Core contracts and API

- `GET /api/config` returns `{revision, values, schema}`. `PUT /api/config` accepts `{revision, values}` and returns the new revision; stale revisions return `409 conflict`.
- `GET /api/options` returns available systems, models, voices, reference modes, and run modes.
- `POST /api/uploads` accepts one multipart media file and returns `{id, name, kind, size}`. `DELETE /api/uploads/{id}` removes only an unreferenced upload owned by the current user.
- `POST /api/jobs` accepts `{input_upload_id, isolated_tracks: {speaker: upload_id}, overrides}` and returns a `Job`. `GET /api/jobs?limit=&cursor=` and `GET /api/jobs/{id}` return only current-owner jobs.
- `GET /api/jobs/{id}/events`: resumable SSE log and state events.
- `GET /api/jobs/{id}/files` lists registered `{id, name, kind, size, url}` results. `GET /api/jobs/{id}/files/{file_id}` downloads one registered file.
- `GET /api/voice-profiles`, `PUT/DELETE /api/voice-profiles/{speaker_id}` provide profile CRUD using config revision checks.
- `GET/POST /api/reference-library`, `PUT/DELETE /api/reference-library/{speaker_id}`, and `GET /api/reference-library/{speaker_id}/audio` provide reference CRUD and serving.
- `GET /api/jobs/{job_id}/dubbing-texts` returns `{revision, segments}`. `PUT` accepts the same shape and rejects stale revisions.
- `POST /api/jobs/{job_id}/dubbing-texts/{segment_id}/regenerate` accepts `{revision, synthesized_text?}` and returns the new revision and segment.

Errors use `{code, message, field?, details?}`. Existing `DubbingConfig` normalization and validation run before a job is accepted.

A `Job` response contains `{id, state, created_at, started_at?, finished_at?, status, error?, files, last_event_id}`. A dubbing segment receives a persisted `segment_id` UUID on first API load; editing timing or speaker data does not change it.

Each SSE event is `{id, job_id, type, timestamp, data}` where `id` is a monotonically increasing integer scoped to the job and `type` is `snapshot`, `state`, `log`, `file`, or `error`. The server honors `Last-Event-ID`, sends a heartbeat comment every 15 seconds, and closes after the terminal event. Event history is capped at 10 MiB per job; if a requested ID has expired, the first response is a current `snapshot` containing state and the retained log tail.

## Frontend

The SPA contains five views:

1. **Workflow** — inputs, languages, run mode, isolated tracks, and submission.
2. **Jobs** — queue state, reconnecting live logs, results, reports, and artifacts.
3. **Settings** — all existing transcription, translation, refinement, TTS, timing, video/audio, and debug fields.
4. **Voice Profiles** — profile editor plus the reference-audio library and assignment actions.
5. **Dubbing Texts** — editable segment table, audio playback, save, and row regeneration.

API access is centralized in one typed client. Server state is not duplicated into unrelated components. SSE reconnects with the last received event ID.

## Files and safety

Uploads stream to `server_data/uploads/<owner>/<upload-id>/`, use generated storage names, preserve a sanitized display name, and are checked by extension and FFprobe. Video extensions are `mp4`, `mov`, `mkv`, `webm`, and `avi`; audio extensions are `wav`, `mp3`, `m4a`, `flac`, and `ogg`. The default limit is 20 GiB and is configurable with `DUBBLM_MAX_UPLOAD_BYTES`. Failed validation removes the partial file. Referenced uploads have no automatic expiry; unreferenced uploads may be explicitly deleted.

Input stems are preserved when materializing a job so existing per-video `prj/<input-stem>` projects and caches remain addressable. Existing reference paths already present in YAML remain compatible, but new browser-supplied media enters through uploads or the reference library.

Download endpoints serve only files registered to the requesting owner/job or located through a repository-controlled reference. Arbitrary client-supplied filesystem paths are never served.

Settings, translation cache updates, snapshots, and library metadata use write-to-sibling-temp plus `os.replace`. Config, profile, reference, and dubbing-text writes use revisions and return `409` on concurrent modification. A failed replacement leaves the prior file intact. Existing per-video project and artifact paths remain unchanged.

## Parity inventory

| Existing Gradio capability | New owner | SPA view |
|---|---|---|
| Input/output, languages, all run modes, speaker report, subtitle/audio switches, isolated tracks | jobs/uploads service | Workflow |
| Live status/logs, output/report/artifacts | job repository + SSE/files routes | Jobs |
| Transcription, translation, refinement, emotion, timing, video/audio, and debug settings | settings service | Settings |
| `source_language`, `target_language`, `save_*`, `keep_*`, `remove_pauses`, `inner_transcription_system`, and every current `SETTINGS_FIELDS` entry | schema-backed config API | Workflow/Settings |
| Voice profile add/edit/delete and provider-dependent model/voice/reference choices | voice-profile service + options API | Voice Profiles |
| Reference library list/save/delete/assign | reference service | Voice Profiles |
| Dubbing-text load/seed, edit, save, style/synthesized text, missing-audio state | dubbing-text service | Dubbing Texts |
| Selected-row regeneration and resulting audio playback | dubbing-text regeneration + registered media route | Dubbing Texts |

Before deleting Gradio, an automated parity test compares the new schema field names and run-mode choices with the extracted legacy constants. The constants then move to framework-independent schema modules used by the API; the final test asserts the committed inventory above and the expected field list directly.

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
