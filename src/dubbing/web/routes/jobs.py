from __future__ import annotations

import asyncio
import json
from dataclasses import asdict
from typing import Any, AsyncIterator

from fastapi import APIRouter, Depends, Header, Query, Request, status
from fastapi.responses import FileResponse, StreamingResponse
from pydantic import BaseModel, ConfigDict, Field

from ..dependencies import current_user, get_job_repository, get_job_service, get_media_store, get_project_service
from ..jobs import TERMINAL_JOB_STATUSES, JobValidationError
from .common import job_public


router = APIRouter(prefix="/api/jobs")


class JobSubmission(BaseModel):
    model_config = ConfigDict(extra="forbid")
    input_upload_id: str = Field(min_length=1)
    isolated_tracks: dict[str, str] = Field(default_factory=dict)
    overrides: dict[str, Any] = Field(default_factory=dict)


def _file_public(job_id: str, item: dict[str, Any]) -> dict[str, Any]:
    result = {key: item[key] for key in ("id", "name", "kind", "size")}
    result["url"] = f"/api/jobs/{job_id}/files/{item['id']}"
    return result


@router.post("", status_code=status.HTTP_201_CREATED)
def submit_job(payload: JobSubmission, user=Depends(current_user), jobs=Depends(get_job_service)):
    return job_public(jobs.submit(
        user.id,
        payload.input_upload_id,
        isolated_tracks=payload.isolated_tracks,
        overrides=payload.overrides,
    ))


@router.get("")
def list_jobs(
    limit: int = Query(50, ge=1, le=100),
    cursor: str | None = None,
    user=Depends(current_user),
    repository=Depends(get_job_repository),
):
    page = repository.list(user.id, limit=limit, cursor=cursor)
    return {"jobs": [job_public(job) for job in page.items], "next_cursor": page.next_cursor}


@router.get("/{job_id}")
def get_job(job_id: str, user=Depends(current_user), repository=Depends(get_job_repository)):
    return job_public(repository.get(user.id, job_id))


@router.get("/{job_id}/files")
def list_files(job_id: str, user=Depends(current_user), repository=Depends(get_job_repository)):
    job = repository.get(user.id, job_id)
    return {"files": [_file_public(job.id, item) for item in job.files]}


@router.get("/{job_id}/files/{file_id}")
def download_file(job_id: str, file_id: str, user=Depends(current_user), repository=Depends(get_job_repository), media=Depends(get_media_store)):
    job = repository.get(user.id, job_id)
    item = next((item for item in job.files if str(item.get("id")) == file_id), None)
    if item is None:
        from ..jobs import JobNotFoundError
        raise JobNotFoundError("File was not found.")
    record = media.get(user.id, file_id)
    if not record.registered:
        from ..jobs import JobNotFoundError
        raise JobNotFoundError("File was not found.")
    return FileResponse(record.path, filename=str(item.get("name") or record.name))


def _encode_event(event: object) -> str:
    raw = asdict(event)
    return f"id: {raw['id']}\nevent: {raw['type']}\ndata: {json.dumps(raw, ensure_ascii=False, separators=(',', ':'))}\n\n"


async def job_event_stream(
    repository: object,
    owner_id: str,
    job_id: str,
    *,
    after_id: int,
    heartbeat_interval: float = 15.0,
    poll_interval: float = 0.25,
    request: Request | None = None,
) -> AsyncIterator[str]:
    job = repository.get(owner_id, job_id)
    retained = repository.events(owner_id, job_id, after_id=0)
    history_expired = (
        (bool(retained) and retained[0].id > after_id + 1)
        or (not retained and after_id < job.last_event_id)
    )
    terminal_without_event = (
        not retained and job.status in TERMINAL_JOB_STATUSES
    )
    terminal_event_recorded = any(
        event.type == "error"
        or (
            event.type == "state"
            and event.data.get("status") in TERMINAL_JOB_STATUSES
        )
        for event in retained
    )
    if history_expired or terminal_without_event:
        log_tail = [
            {"id": event.id, "message": str(event.data.get("message") or "")}
            for event in retained if event.type == "log"
        ]
        snapshot = {
            "id": job.last_event_id,
            "job_id": job.id,
            "type": "snapshot",
            "timestamp": job.updated_at,
            "data": {"state": job.state, "status": job.status, "error": job.error, "files": job.files, "log_tail": log_tail},
        }
        yield f"id: {snapshot['id']}\nevent: snapshot\ndata: {json.dumps(snapshot, ensure_ascii=False, separators=(',', ':'))}\n\n"
        if job.status in TERMINAL_JOB_STATUSES:
            return
        after_id = job.last_event_id

    last_id = after_id
    elapsed = 0.0
    sleep_interval = min(poll_interval, heartbeat_interval)
    while True:
        if request is not None and await request.is_disconnected():
            return
        events = repository.events(owner_id, job_id, after_id=last_id)
        for event in events:
            yield _encode_event(event)
            last_id = event.id
            if event.type == "error":
                terminal_event_recorded = True
            if event.type == "state" and event.data.get("status") in TERMINAL_JOB_STATUSES:
                return
        current = repository.get(owner_id, job_id)
        if (
            current.status in TERMINAL_JOB_STATUSES
            and last_id >= current.last_event_id
            and terminal_event_recorded
        ):
            return
        await asyncio.sleep(sleep_interval)
        elapsed += sleep_interval
        if elapsed >= heartbeat_interval:
            yield ": heartbeat\n\n"
            elapsed = 0.0


@router.get("/{job_id}/events")
def job_events(
    request: Request,
    job_id: str,
    last_event_id_query: int | None = Query(None, alias="last_event_id", ge=0),
    last_event_id_header: int | None = Header(None, alias="Last-Event-ID"),
    user=Depends(current_user),
    repository=Depends(get_job_repository),
):
    after_id = last_event_id_header if last_event_id_header is not None else (last_event_id_query or 0)
    repository.get(user.id, job_id)
    return StreamingResponse(
        job_event_stream(
            repository,
            user.id,
            job_id,
            after_id=after_id,
            heartbeat_interval=request.app.state.heartbeat_interval,
            poll_interval=request.app.state.sse_poll_interval,
            request=request,
        ),
        media_type="text/event-stream",
        headers={"Cache-Control": "no-cache", "X-Accel-Buffering": "no"},
    )


class JobSpeakerMapBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker_map: dict[str, str]


@router.put("/{job_id}/speaker-map", status_code=status.HTTP_200_OK)
def update_job_speaker_map(
    job_id: str,
    payload: JobSpeakerMapBody,
    user=Depends(current_user),
    repository=Depends(get_job_repository),
    projects=Depends(get_project_service),
):
    from pathlib import Path
    job = repository.get(user.id, job_id)
    config = job.config if isinstance(job.config, dict) else {}
    project_dir_str = str(config.get("project_dir") or "").strip()
    project_name = str(config.get("project_name") or "").strip()
    if not project_name and project_dir_str:
        project_name = Path(project_dir_str).name
    if not project_name:
        raise JobValidationError("Job has no associated project directory.")
    projects.update_speaker_map(project_name, user.id, payload.speaker_map)
    return {"ok": True, "project_name": project_name}


@router.get("/{job_id}/speaker-map")
def get_job_speaker_map(
    job_id: str,
    user=Depends(current_user),
    repository=Depends(get_job_repository),
    projects=Depends(get_project_service),
):
    from pathlib import Path
    job = repository.get(user.id, job_id)
    config = job.config if isinstance(job.config, dict) else {}
    project_dir_str = str(config.get("project_dir") or "").strip()
    project_name = str(config.get("project_name") or "").strip()
    if not project_name and project_dir_str:
        project_name = Path(project_dir_str).name
    if not project_name:
        return {"speaker_map": {}}
    try:
        detail = projects.get_project(project_name, user.id)
        speaker_map = detail.saved_config.get("speaker_map") if detail.saved_config else {}
        return {"speaker_map": speaker_map or {}}
    except Exception:
        return {"speaker_map": {}}
