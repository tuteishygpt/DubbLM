from __future__ import annotations

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from ..dependencies import current_user, get_dubbing_text_service, get_job_repository
from ..dubbing_texts import DubbingTextSegment
from .common import public


router = APIRouter(prefix="/api/jobs/{job_id}/dubbing-texts")


class SegmentBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    segment_id: str
    speaker: str
    start: float
    end: float
    text: str
    translation: str
    synthesized_text: str
    style_prompt: str = ""

    def domain(self) -> DubbingTextSegment:
        return DubbingTextSegment(**self.model_dump(), audio=None)


class TextUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    segments: list[SegmentBody]


class RegenerateBody(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    synthesized_text: str | None = None


def _job(repository, owner_id: str, job_id: str):
    return repository.get(owner_id, job_id)


@router.get("")
def load_texts(job_id: str, user=Depends(current_user), repository=Depends(get_job_repository), texts=Depends(get_dubbing_text_service)):
    job = _job(repository, user.id, job_id)
    return public(texts.load(owner_id=user.id, job_id=job.id, config=job.config))


@router.put("")
def save_texts(job_id: str, payload: TextUpdate, user=Depends(current_user), repository=Depends(get_job_repository), texts=Depends(get_dubbing_text_service)):
    job = _job(repository, user.id, job_id)
    return public(texts.save(
        owner_id=user.id, job_id=job.id, config=job.config,
        segments=[segment.domain() for segment in payload.segments], revision=payload.revision,
    ))


@router.post("/{segment_id}/regenerate")
def regenerate_text(job_id: str, segment_id: str, payload: RegenerateBody, user=Depends(current_user), repository=Depends(get_job_repository), texts=Depends(get_dubbing_text_service)):
    job = _job(repository, user.id, job_id)
    return public(texts.regenerate(
        owner_id=user.id, job_id=job.id, config=job.config, segment_id=segment_id,
        revision=payload.revision, synthesized_text=payload.synthesized_text,
    ))

