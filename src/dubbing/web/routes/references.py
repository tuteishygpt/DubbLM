from __future__ import annotations

from fastapi import APIRouter, Depends, File, Form, UploadFile
from fastapi.responses import FileResponse
from pydantic import BaseModel, ConfigDict

from ..dependencies import current_user, get_media_store, get_reference_service, get_settings
from ..references import ReferenceNotFoundError
from .common import public


router = APIRouter(prefix="/api/reference-library")


class ReferenceAssignmentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    profile_speaker_id: str
    settings_revision: str


class RevisionBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str


@router.get("")
def list_references(user=Depends(current_user), references=Depends(get_reference_service)):
    return public(references.list(owner_id=user.id))


@router.post("")
def save_reference(
    speaker_id: str = Form(...),
    reference_text: str = Form(""),
    revision: str = Form(...),
    file: UploadFile = File(...),
    user=Depends(current_user),
    references=Depends(get_reference_service),
):
    return public(references.save(
        owner_id=user.id,
        speaker_id=speaker_id,
        source_audio=file.file,
        audio_name=file.filename,
        reference_text=reference_text,
        revision=revision,
    ))


@router.put("/{speaker_id}")
def assign_reference(speaker_id: str, payload: ReferenceAssignmentBody, user=Depends(current_user), references=Depends(get_reference_service), settings=Depends(get_settings)):
    return public(references.assign(
        owner_id=user.id,
        library_speaker_id=speaker_id,
        profile_speaker_id=payload.profile_speaker_id,
        settings=settings,
        settings_revision=payload.settings_revision,
    ))


@router.delete("/{speaker_id}")
def delete_reference(speaker_id: str, payload: RevisionBody, user=Depends(current_user), references=Depends(get_reference_service)):
    return public(references.delete(owner_id=user.id, speaker_id=speaker_id, revision=payload.revision))


@router.get("/{speaker_id}/audio")
def reference_audio(speaker_id: str, user=Depends(current_user), references=Depends(get_reference_service), media=Depends(get_media_store)):
    snapshot = references.list(owner_id=user.id)
    entry = next((entry for entry in snapshot.entries if entry.speaker_id == speaker_id), None)
    if entry is None:
        raise ReferenceNotFoundError(f"Reference not found: {speaker_id}.")
    record = media.get(user.id, entry.audio.id)
    return FileResponse(record.path, filename=record.name)
