from __future__ import annotations

from dataclasses import asdict
from typing import Any

from fastapi import APIRouter, Depends, status

from ..dependencies import current_user, get_project_service
from .common import job_public

router = APIRouter(prefix="/api/projects")


def _project_summary_public(item: object) -> dict[str, Any]:
    return asdict(item)


def _project_detail_public(item: object) -> dict[str, Any]:
    return asdict(item)


@router.get("")
def list_projects(user=Depends(current_user), projects=Depends(get_project_service)):
    items = projects.list_projects(user.id)
    return {"projects": [_project_summary_public(item) for item in items]}


@router.get("/{project_name}")
def get_project(project_name: str, user=Depends(current_user), projects=Depends(get_project_service)):
    detail = projects.get_project(project_name, user.id)
    return _project_detail_public(detail)


from pydantic import BaseModel, ConfigDict, Field


class ProjectRunRequest(BaseModel):
    model_config = ConfigDict(extra="forbid")
    run_step: str = Field(default="tts_to_end")
    overrides: dict[str, Any] = Field(default_factory=dict)


@router.post("/{project_name}/open", status_code=status.HTTP_200_OK)
def open_project(project_name: str, user=Depends(current_user), projects=Depends(get_project_service)):
    job = projects.open_project(project_name, user.id)
    return {
        "project_name": project_name,
        "job": job_public(job),
    }


@router.post("/{project_name}/run", status_code=status.HTTP_201_CREATED)
def run_project(
    project_name: str,
    payload: ProjectRunRequest = ProjectRunRequest(),
    user=Depends(current_user),
    projects=Depends(get_project_service),
):
    job = projects.run_project_step(
        project_name,
        user.id,
        run_step=payload.run_step,
        overrides=payload.overrides,
    )
    return {
        "project_name": project_name,
        "job": job_public(job),
    }


class SpeakerMapBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    speaker_map: dict[str, str]  # {"SPEAKER_00": "John Male", ...}


@router.put("/{project_name}/speaker-map", status_code=status.HTTP_200_OK)
def update_speaker_map(
    project_name: str,
    payload: SpeakerMapBody,
    user=Depends(current_user),
    projects=Depends(get_project_service),
):
    """Save a SPEAKER_XX → profile-name mapping into the project's metadata."""
    projects.update_speaker_map(project_name, user.id, payload.speaker_map)
    return {"ok": True}
