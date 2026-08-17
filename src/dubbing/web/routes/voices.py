from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends, HTTPException
from pydantic import BaseModel, ConfigDict

from ..dependencies import current_user, get_media_store, get_reference_service, get_settings
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
def put_profile(
    speaker_id: str,
    payload: ProfileUpdate,
    user=Depends(current_user),
    settings=Depends(get_settings),
    references=Depends(get_reference_service),
    media=Depends(get_media_store),
):
    """Save a voice profile, resolving reference_audio from the library when reference_mode='configured'."""
    profile = dict(payload.profile)
    if profile.get("reference_mode") == "configured" and profile.get("reference_audio"):
        ref_name = profile["reference_audio"]
        snapshot = references.list(owner_id=user.id)
        entry = next((e for e in snapshot.entries if e.speaker_id == ref_name), None)
        if entry is None:
            raise HTTPException(status_code=422, detail=f"Reference not found in library: {ref_name!r}")
        record = media.get(owner_id=user.id, media_id=entry.audio.id)
        # Replace the symbolic library name with the real filesystem path
        profile["reference_audio"] = str(getattr(record, "path", record) or "").strip()
        if not profile["reference_audio"]:
            raise HTTPException(status_code=422, detail="Reference audio has no resolvable path.")
    return public(settings.put_profile(speaker_id, profile, revision=payload.revision))


@router.delete("/{speaker_id}")
def delete_profile(speaker_id: str, payload: RevisionBody, settings=Depends(get_settings)):
    return public(settings.delete_profile(speaker_id, revision=payload.revision))
