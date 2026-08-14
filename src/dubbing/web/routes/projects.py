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


@router.post("/{project_name}/open", status_code=status.HTTP_200_OK)
def open_project(project_name: str, user=Depends(current_user), projects=Depends(get_project_service)):
    job = projects.open_project(project_name, user.id)
    return {
        "project_name": project_name,
        "job": job_public(job),
    }
