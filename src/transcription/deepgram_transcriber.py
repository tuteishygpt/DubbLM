"""
Implementation of transcription and diarization using the Deepgram API.
"""
import os
from typing import Any, Dict, List, Optional, Tuple, TYPE_CHECKING

from transcription.transcription_interface import BaseTranscriber
from src.dubbing.core.log_config import get_logger

try:
    from deepgram import DeepgramClient
except ImportError:
    DeepgramClient = None

if TYPE_CHECKING:
    from src.dubbing.core.cache_manager import CacheManager

logger = get_logger(__name__)
DEEPGRAM_CACHE_VERSION = "dg_v1"


class DeepgramTranscriber(BaseTranscriber):
    """Transcription and diarization service using Deepgram."""

    def __init__(
        self,
        source_language: str,
        device: Optional[str] = None,
        model: str = "nova-3",
        cache_manager: Optional["CacheManager"] = None,
        smart_format: bool = True,
        diarize: bool = True,
        paragraphs: bool = True,
        utterances: bool = True,
        utterance_split: float = 0.4,
        **kwargs,
    ):
        super().__init__(source_language, device, **kwargs)
        self.model = model
        self.cache_manager = cache_manager
        self.smart_format = smart_format
        self.diarize = diarize
        self.paragraphs = paragraphs
        self.utterances = utterances
        self.utterance_split = utterance_split

        self.api_key = os.environ.get("DEEPGRAM_API_KEY")
        if not self.api_key:
            raise ValueError("DEEPGRAM_API_KEY environment variable is required")

        if DeepgramClient is None:
            raise ImportError(
                "deepgram-sdk package not found. Install it with: pip install deepgram-sdk"
            )

        self.client = DeepgramClient(api_key=self.api_key)

    @property
    def name(self) -> str:
        return "Deepgram"

    @property
    def cache_step_name(self) -> str:
        return "deepgram_diarization_transcription"

    def default_cache_key(self, audio_file: str) -> str:
        return self._generate_cache_key(
            audio_file,
            f"_{self.model}_{self.utterance_split}_{DEEPGRAM_CACHE_VERSION}",
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
                f"_{self.model}_{self.utterance_split}_{DEEPGRAM_CACHE_VERSION}",
            )

        step_name = "deepgram_diarization_transcription"
        if use_cache and self.cache_manager and self.cache_manager.cache_exists(step_name, cache_key):
            logger.debug("Loading Deepgram diarization and transcription from cache...")
            cached_results = self.cache_manager.load_from_cache(step_name, cache_key)
            self.debug_data["diarization"] = cached_results["diarization"]
            self.debug_data["transcription"] = cached_results["transcription"]
            return cached_results["diarization"], cached_results["transcription"]

        logger.info("Running Deepgram transcription and diarization on %s...", audio_file)
        try:
            with open(audio_file, "rb") as audio_stream:
                audio_bytes = audio_stream.read()

            response = self._transcribe_bytes(audio_bytes)
            speakers_rolls, transcription = self._normalize_response(response)
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
            error_message = self._build_error_message(e)
            logger.error(error_message)
            raise RuntimeError(error_message) from e

    def _transcribe_bytes(self, audio_bytes: bytes) -> Any:
        transcribe_kwargs = {
            "model": self.model,
            "language": self.source_language,
            "smart_format": self.smart_format,
            "diarize": self.diarize,
            "paragraphs": self.paragraphs,
            "utterances": self.utterances,
            "utt_split": self.utterance_split,
        }
        media_client = self.client.listen.v1.media

        try:
            return media_client.transcribe_file(
                request=audio_bytes,
                **transcribe_kwargs,
            )
        except TypeError as exc:
            if "request" not in str(exc):
                raise

            return media_client.transcribe_file(
                payload={"buffer": audio_bytes},
                **transcribe_kwargs,
            )

    @staticmethod
    def _build_error_message(error: Exception) -> str:
        status_code = getattr(error, "status_code", None)
        body = getattr(error, "body", None)
        err_code = body.get("err_code") if isinstance(body, dict) else None

        if status_code == 401 or err_code == "INVALID_AUTH":
            return (
                "Deepgram authentication failed. Check DEEPGRAM_API_KEY in your "
                "environment or .env file and make sure the key is active."
            )

        return f"Deepgram transcription failed: {error}"

    @classmethod
    def _normalize_response(
        cls, payload: Any
    ) -> Tuple[Dict[Tuple[float, float], str], List[Dict[str, Any]]]:
        payload_dict = cls._coerce_to_dict(payload)
        results = payload_dict.get("results")
        utterances = results.get("utterances") if isinstance(results, dict) else None
        if not isinstance(utterances, list) or not utterances:
            raise ValueError(
                "Invalid Deepgram response payload: utterances must be a non-empty list."
            )

        speaker_mapping: Dict[str, str] = {}
        normalized_transcription: List[Dict[str, Any]] = []

        for utterance in utterances:
            utterance_dict = cls._coerce_to_dict(utterance)

            start_raw = utterance_dict.get("start")
            end_raw = utterance_dict.get("end")
            if start_raw is None or end_raw is None:
                raise ValueError(
                    "Invalid Deepgram response payload: utterances entries must include start and end."
                )

            start = float(start_raw)
            end = float(end_raw)
            if end <= start:
                raise ValueError(
                    "Invalid Deepgram response payload: utterance end must be greater than start."
                )

            speaker_raw = str(utterance_dict.get("speaker", "")).strip()
            text = str(
                utterance_dict.get("transcript") or utterance_dict.get("text") or ""
            ).strip()
            if not speaker_raw or not text:
                raise ValueError(
                    "Invalid Deepgram response payload: utterances entries must include speaker and transcript."
                )

            speaker = speaker_mapping.setdefault(
                speaker_raw, f"SPEAKER_{len(speaker_mapping):02d}"
            )
            normalized_segment: Dict[str, Any] = {
                "text": text,
                "start": start,
                "end": end,
                "speaker": speaker,
            }

            confidence = utterance_dict.get("confidence")
            if confidence is not None:
                normalized_segment["confidence"] = float(confidence)

            words = utterance_dict.get("words")
            if isinstance(words, list) and words:
                normalized_segment["words"] = cls._normalize_words(words)

            normalized_transcription.append(normalized_segment)

        normalized_transcription.sort(key=lambda item: item["start"])
        normalized_diarization = {
            (segment["start"], segment["end"]): segment["speaker"]
            for segment in normalized_transcription
        }
        return normalized_diarization, normalized_transcription

    @staticmethod
    def _coerce_to_dict(value: Any) -> Dict[str, Any]:
        if isinstance(value, dict):
            return value
        if hasattr(value, "to_dict"):
            coerced = value.to_dict()
            if isinstance(coerced, dict):
                return coerced
        if hasattr(value, "model_dump"):
            coerced = value.model_dump()
            if isinstance(coerced, dict):
                return coerced
        raise ValueError("Invalid Deepgram response payload: expected dictionary-like data.")

    @classmethod
    def _normalize_words(cls, words: List[Any]) -> List[Dict[str, Any]]:
        normalized_words: List[Dict[str, Any]] = []
        for word in words:
            word_dict = cls._coerce_to_dict(word)
            start_raw = word_dict.get("start")
            end_raw = word_dict.get("end")
            if start_raw is None or end_raw is None:
                continue

            text = str(
                word_dict.get("punctuated_word")
                or word_dict.get("word")
                or word_dict.get("text")
                or ""
            ).strip()
            if not text:
                continue

            normalized_word: Dict[str, Any] = {
                "word": text,
                "start": float(start_raw),
                "end": float(end_raw),
            }
            confidence = word_dict.get("confidence")
            if confidence is not None:
                normalized_word["confidence"] = float(confidence)
            normalized_words.append(normalized_word)

        return normalized_words
