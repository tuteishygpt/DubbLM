from __future__ import annotations

from typing import Any

from fastapi import APIRouter, Depends
from pydantic import BaseModel, ConfigDict

from .. import schema
from ..dependencies import get_settings


router = APIRouter(prefix="/api")


class ConfigUpdate(BaseModel):
    model_config = ConfigDict(extra="forbid")
    revision: str
    values: dict[str, Any]


def _schema() -> dict[str, Any]:
    return {
        "fields": [schema.api_field(name) for name in schema.PUBLIC_SCHEMA_FIELDS]
    }


@router.get("/config")
def get_config(settings=Depends(get_settings)):
    snapshot = settings.load()
    return {"revision": snapshot.revision, "values": snapshot.values, "schema": _schema()}


@router.put("/config")
def put_config(payload: ConfigUpdate, settings=Depends(get_settings)):
    snapshot = settings.save(payload.values, revision=payload.revision)
    return {"revision": snapshot.revision, "values": snapshot.values, "schema": _schema()}


@router.get("/options")
def get_options():
    return {
        "run_modes": schema.RUN_MODES,
        "inner_transcription_systems": schema.INNER_TRANSCRIPTION_SYSTEM_CHOICES,
        "transcription_systems": schema.TRANSCRIPTION_SYSTEM_CHOICES,
        "transcription_models": schema.TRANSCRIPTION_MODEL_CHOICES,
        "transcription_model_defaults": schema.TRANSCRIPTION_MODEL_DEFAULTS,
        "llm_providers": schema.LLM_PROVIDER_CHOICES,
        "refinement_personas": schema.REFINEMENT_PERSONA_CHOICES,
        "emotion_providers": schema.EMOTION_PROVIDER_CHOICES,
        "emotion_models": schema.EMOTION_MODEL_CHOICES,
        "tts_providers": schema.TTS_PROVIDER_CHOICES,
        "tts_models": schema.TTS_MODEL_CHOICES,
        "tts_voices": schema.TTS_VOICE_CHOICES,
        "tts_reference_capabilities": schema.TTS_REFERENCE_CAPABILITIES,
        "reference_modes": ["none", "configured", "segment", "speaker"],
    }
