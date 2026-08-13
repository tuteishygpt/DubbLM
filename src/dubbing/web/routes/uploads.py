from __future__ import annotations

from fastapi import APIRouter, Depends, File, Response, UploadFile, status

from ..dependencies import current_user, get_media_store
from .common import public


router = APIRouter(prefix="/api/uploads")


@router.post("")
def upload(file: UploadFile = File(...), user=Depends(current_user), media=Depends(get_media_store)):
    record = media.save(user.id, file.filename, file.file)
    raw = public(record)
    return {key: raw[key] for key in ("id", "name", "kind", "size")}


@router.delete("/{media_id}", status_code=status.HTTP_204_NO_CONTENT)
def delete_upload(media_id: str, user=Depends(current_user), media=Depends(get_media_store)):
    record = media.get(user.id, media_id)
    if record.registered:
        from ..storage import MediaNotFoundError
        raise MediaNotFoundError("Upload was not found.")
    media.delete(user.id, media_id)
    return Response(status_code=status.HTTP_204_NO_CONTENT)
