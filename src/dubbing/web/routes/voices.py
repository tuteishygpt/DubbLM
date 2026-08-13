from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict, Field

from ..dependencies import get_settings
from .common import public


router = APIRouter(prefix="/api/voice-profiles")


class ProfileUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    profile: dict[str, Any]


class RevisionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str


@router.get("")
def list_profiles(settings=Depends(get_settings)):
    return public(settings.list_profiles())


@router.put("/{speaker_id}")
def put_profile(speaker_id: str, payload: ProfileUpdate, settings=Depends(get_settings)):
    return public(settings.put_profile(speaker_id, payload.profile, revision=payload.revision))


@router.delete("/{speaker_id}")
def delete_profile(speaker_id: str, payload: RevisionBody, settings=Depends(get_settings)):
    return public(settings.delete_profile(speaker_id, revision=payload.revision))
