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
    if profile.get("reference_audio") and profile.get("reference_mode") not in {"speaker", "segment", "none"}:
        profile["reference_mode"] = "configured"

    if profile.get("reference_mode") == "configured" and profile.get("reference_audio"):
        ref_name = str(profile["reference_audio"]).strip()
        snapshot = references.list(owner_id=user.id)
        norm_ref = ref_name.replace("_", " ").lower()
        entry = next((e for e in snapshot.entries if e.speaker_id == ref_name), None)
        if entry is None:
            entry = next(
                (
                    e
                    for e in snapshot.entries
                    if e.speaker_id.replace("_", " ").lower() == norm_ref
                ),
                None,
            )
        if entry is None:
            entry = next(
                (
                    e
                    for e in snapshot.entries
                    if e.audio.id == ref_name
                    or e.audio.url == ref_name
                    or e.audio.name == ref_name
                ),
                None,
            )
        if entry is not None:
            record = media.get(owner_id=user.id, media_id=entry.audio.id)
            # Replace the symbolic library name with the real filesystem path
            profile["reference_audio"] = str(getattr(record, "path", record) or "").strip()
            if not profile["reference_audio"]:
                raise HTTPException(status_code=422, detail="Reference audio has no resolvable path.")
            if not profile.get("reference_text") and entry.reference_text:
                profile["reference_text"] = entry.reference_text
        else:
            from pathlib import Path

            if not Path(ref_name).is_file():
                raise HTTPException(status_code=422, detail=f"Reference not found in library: {ref_name!r}")
    return public(settings.put_profile(speaker_id, profile, revision=payload.revision))


@router.delete("/{speaker_id}")
def delete_profile(speaker_id: str, payload: RevisionBody, settings=Depends(get_settings)):
    return public(settings.delete_profile(speaker_id, revision=payload.revision))
