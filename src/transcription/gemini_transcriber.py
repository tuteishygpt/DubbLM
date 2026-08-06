"""
Implementation of transcription and diarization using the Gemini API.
"""
import json
import mimetypes
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from pydantic import BaseModel

from google_vertex import get_vertex_ai_settings
from transcription.transcription_interface import BaseTranscriber
from src.dubbing.core.log_config import get_logger

try:
    from google import genai
    from google.genai import types as genai_types

    GEMINI_GENAI_AVAILABLE = True
except ImportError:
    genai = None
    genai_types = None
    GEMINI_GENAI_AVAILABLE = False

if TYPE_CHECKING:
    from src.dubbing.core.cache_manager import CacheManager

logger = get_logger(__name__)
TIMESTAMP_PRECISION_CACHE_VERSION = "ts_v2"


class GeminiSegment(BaseModel):
    start: str
    end: str
    speaker: str
    text: str


class GeminiTranscriptPayload(BaseModel):
    segments: List[GeminiSegment]


class GeminiTranscriber(BaseTranscriber):
    """Transcription and diarization service using the Gemini API."""

    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        gemini_transcription_model: str = "gemini-3-flash-preview",
        cache_manager: Optional["CacheManager"] = None,
        **kwargs,
    ):
        super().__init__(source_language, device, **kwargs)
        self.gemini_transcription_model = gemini_transcription_model
        self.cache_manager = cache_manager
        if not GEMINI_GENAI_AVAILABLE:
            raise ImportError(
                "google-genai package not found. Install it with: pip install google-genai"
            )
        self.vertex_ai_settings = get_vertex_ai_settings()
        self.client = genai.Client(**self.vertex_ai_settings.genai_client_kwargs)

    @property
    def name(self) -> str:
        return "Gemini"

    @property
    def cache_step_name(self) -> str:
        return "gemini_diarization_transcription"

    def default_cache_key(self, audio_file: str) -> str:
        return self._generate_cache_key(
            audio_file,
            f"_{self.gemini_transcription_model}_{TIMESTAMP_PRECISION_CACHE_VERSION}",
        )

    def diarize_and_transcribe(
        self,
        audio_file: str,
        cache_key: Optional[str] = None,
        use_cache: bool = True,
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        if cache_key is None:
            cache_key = self._generate_cache_key(
                audio_file,
                f"_{self.gemini_transcription_model}_{TIMESTAMP_PRECISION_CACHE_VERSION}",
            )

        step_name = "gemini_diarization_transcription"
        if use_cache and self.cache_manager and self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading Gemini diarization and transcription from cache...")
            cached_results = self.cache_manager.load_from_cache(step_name, cache_key)
            self.debug_data["diarization"] = cached_results["diarization"]
            self.debug_data["transcription"] = cached_results["transcription"]
            return cached_results["diarization"], cached_results["transcription"]

        logger.info("Running Gemini transcription and diarization on %s...", audio_file)
        try:
            upload_mime_type = self._guess_mime_type(audio_file)
            with open(audio_file, "rb") as audio_stream:
                audio_part = genai_types.Part.from_bytes(
                    data=audio_stream.read(),
                    mime_type=upload_mime_type,
                )

            response = self.client.models.generate_content(
                model=self.gemini_transcription_model,
                contents=[audio_part, self._build_prompt()],
                config=genai_types.GenerateContentConfig(
                    temperature=0,
                    responseMimeType="application/json",
                    responseSchema=GeminiTranscriptPayload,
                ),
            )
            payload = response.parsed if getattr(response, "parsed", None) is not None else self._parse_response_text(response)
            speakers_rolls, transcription = self._normalize_segments(payload)
            self.debug_data["diarization"] = speakers_rolls
            self.debug_data["transcription"] = transcription

            if use_cache and self.cache_manager:
                self.cache_manager.save_to_cache(
                    step_name,
                    cache_key,
                    {
                        "diarization": speakers_rolls,
                        "transcription": transcription,
                    },
                )

            return speakers_rolls, transcription
        except Exception as e:
            logger.error("Gemini transcription failed: %s", e)
            raise RuntimeError(f"Gemini transcription failed: {e}") from e

    @staticmethod
    def _build_prompt() -> str:
        return (
            "Transcribe the entire audio and perform speaker diarization. "
            "Return JSON only with a top-level 'segments' array. "
            "Each segment must contain exactly these string fields: "
            "'start', 'end', 'speaker', 'text'. "
            "Use timestamps in HH:MM:SS.mmm format with millisecond precision. "
            "Use stable speaker labels such as SPEAKER_A, SPEAKER_B, SPEAKER_C. "
            "Keep the segments chronological, do not overlap them, and do not omit spoken words. "
            "Example:\n"
            "{\n"
            '  "segments": [\n'
            '    {"start": "00:00:00.000", "end": "00:00:07.250", "speaker": "SPEAKER_A", "text": "What have you done?"}\n'
            "  ]\n"
            "}"
        )

    @staticmethod
    def _parse_response_text(response: Any) -> Dict[str, Any]:
        response_text = getattr(response, "text", None)
        if not response_text:
            raise ValueError("Gemini response did not include parsed JSON or response text.")
        return json.loads(response_text)

    @staticmethod
    def _guess_mime_type(audio_file: str) -> str:
        mime_type, _ = mimetypes.guess_type(audio_file)
        return mime_type or "audio/wav"

    @classmethod
    def _normalize_segments(
        cls, payload: Any
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        if isinstance(payload, BaseModel):
            payload = payload.model_dump()
        elif hasattr(payload, "model_dump"):
            payload = payload.model_dump()

        segments = payload.get("segments") if isinstance(payload, dict) else None
        if not isinstance(segments, list) or not segments:
            raise ValueError("Invalid Gemini response payload: segments must be a non-empty list.")

        speaker_mapping: Dict[str, str] = {}
        normalized_transcription: List[Dict[str, Any]] = []
        normalized_diarization: Dict[Tuple[float, float], str] = {}

        for segment in segments:
            if hasattr(segment, "model_dump"):
                segment = segment.model_dump()
            if not isinstance(segment, dict):
                raise ValueError("Invalid Gemini response payload: segments entries must be objects.")

            start_raw = segment.get("start")
            end_raw = segment.get("end")
            speaker_raw = str(segment.get("speaker") or "").strip()
            text_raw = str(segment.get("text") or "").strip()

            if start_raw is None or end_raw is None or not speaker_raw or not text_raw:
                raise ValueError(
                    "Invalid Gemini response payload: segments entries must include start, end, speaker, and text."
                )

            start = cls._parse_timestamp_to_seconds(start_raw)
            end = cls._parse_timestamp_to_seconds(end_raw)
            if end <= start:
                raise ValueError("Invalid Gemini response payload: segment end must be greater than start.")

            speaker_label = speaker_mapping.setdefault(
                speaker_raw, f"SPEAKER_{len(speaker_mapping):02d}"
            )
            normalized_segment = {
                "text": text_raw,
                "start": start,
                "end": end,
                "speaker": speaker_label,
            }
            normalized_transcription.append(normalized_segment)
            normalized_diarization[(start, end)] = speaker_label

        normalized_transcription.sort(key=lambda item: item["start"])
        normalized_diarization = {
            (segment["start"], segment["end"]): segment["speaker"]
            for segment in normalized_transcription
        }
        return normalized_diarization, normalized_transcription

    @staticmethod
    def _parse_timestamp_to_seconds(raw_value: Any) -> float:
        if isinstance(raw_value, (int, float)):
            return float(raw_value)

        value = str(raw_value).strip()
        if not value:
            raise ValueError("Timestamp value cannot be empty.")

        if ":" in value:
            parts = value.split(":")
            if len(parts) != 3:
                raise ValueError(f"Unsupported timestamp format: {value}")
            hours_raw, minutes_raw, seconds_raw = parts
        elif value.count(".") >= 2:
            hours_raw, minutes_raw, seconds_raw = value.split(".", 2)
        else:
            raise ValueError(f"Unsupported timestamp format: {value}")

        hours = int(hours_raw)
        minutes = int(minutes_raw)
        seconds = float(str(seconds_raw).replace(",", "."))
        return hours * 3600 + minutes * 60 + seconds
