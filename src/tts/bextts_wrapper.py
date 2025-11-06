"""Wrapper for the BexTTS Hugging Face Space."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from .models import DiarizationSegment, SegmentAlignment, TTSSegmentData
from .tts_interface import TTSInterface
from src.dubbing.core.log_config import get_logger

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency missing at runtime
    GRADIO_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency missing at runtime
    PYDUB_AVAILABLE = False

logger = get_logger(__name__)


class BexTTSWrapper(TTSInterface):
    """Text-to-speech wrapper around the public BexTTS Space.

    The wrapper forwards synthesis requests to the Hugging Face Space
    ``archivartaunik/Bextts`` (or a user supplied alternative).  Voice cloning
    is supported by providing reference audio via ``TTSSegmentData`` or by
    configuring a ``voice_mapping`` that maps speaker identifiers to reference
    audio paths.
    """

    def __init__(
        self,
        space_id: str = "archivartaunik/Bextts",
        api_name: str = "/predict",
        default_reference_audio: Optional[str] = None,
        hf_token_env: str = "HF_TOKEN",
        text_field: str = "belarusian_story",
        audio_field: str = "speaker_audio_file",
        **_: Any,
    ) -> None:
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "gradio-client package is required for BexTTS integration. "
                "Install it with 'pip install gradio-client'."
            )
        if not PYDUB_AVAILABLE:
            raise ImportError(
                "pydub package is required to post-process BexTTS audio. "
                "Install it with 'pip install pydub'."
            )

        self.space_id = space_id
        self.api_name = api_name
        self.default_reference_audio = default_reference_audio
        self.hf_token_env = hf_token_env
        self.text_field = text_field
        self.audio_field = audio_field

        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}

        self.client: Optional[Client] = None
        self._temp_dir: Optional[str] = None

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        """Set mapping between speaker IDs and reference audio paths."""
        self.voice_mapping = mapping or {}

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        """Store optional style prompts per speaker (prefixed to the text)."""
        self.voice_prompt_mapping = mapping or {}

    def initialize(self) -> None:
        token = os.getenv(self.hf_token_env)
        try:
            if token:
                logger.info(
                    "Initializing BexTTS client with authenticated access to %s", self.space_id
                )
                self.client = Client(self.space_id, hf_token=token)
            else:
                logger.warning(
                    "%s environment variable not set – using anonymous Hugging Face access.",
                    self.hf_token_env,
                )
                self.client = Client(self.space_id)
        except Exception as exc:  # pragma: no cover - network failures at runtime
            raise RuntimeError(
                f"Failed to initialize BexTTS client for space '{self.space_id}': {exc}"
            ) from exc

        self._temp_dir = tempfile.mkdtemp(prefix="bextts_segments_")
        logger.info("BexTTS wrapper initialized for space %s", self.space_id)

    def _prepare_text(self, segment: TTSSegmentData) -> str:
        text = segment.text
        style_prompt = (
            segment.style_prompt
            or (segment.speaker and self.voice_prompt_mapping.get(segment.speaker))
        )
        if style_prompt:
            text = f"{style_prompt.strip()}\n{text}" if style_prompt.strip() else text
        return text

    def _resolve_reference_audio(self, segment: TTSSegmentData) -> Optional[str]:
        if segment.reference_audio_path:
            return segment.reference_audio_path
        if segment.speaker and segment.speaker in self.voice_mapping:
            return self.voice_mapping[segment.speaker]
        return self.default_reference_audio

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "be",
        **_: Any,
    ) -> List[SegmentAlignment]:
        if not self.client:
            raise RuntimeError("BexTTS client not initialized. Call initialize() first.")
        if not segments_data:
            logger.warning("No segments provided to BexTTS synthesis.")
            return []

        alignments: List[SegmentAlignment] = []

        for index, segment in enumerate(segments_data):
            prepared_text = self._prepare_text(segment)
            reference_audio = self._resolve_reference_audio(segment)

            if reference_audio and not Path(reference_audio).exists():
                logger.warning(
                    "Reference audio '%s' for speaker '%s' does not exist. Falling back to default voice.",
                    reference_audio,
                    segment.speaker,
                )
                reference_audio = None

            logger.info(
                "BexTTS: Synthesizing segment %d/%d for speaker '%s' (ref audio: %s)",
                index + 1,
                len(segments_data),
                segment.speaker,
                "yes" if reference_audio else "no",
            )

            try:
                result_path = self.client.predict(
                    **{
                        self.text_field: prepared_text,
                        self.audio_field: handle_file(reference_audio) if reference_audio else None,
                    },
                    api_name=self.api_name,
                )
            except Exception as exc:  # pragma: no cover - network/runtime errors
                logger.error("BexTTS: Failed to synthesize segment %s: %s", segment.speaker, exc)
                continue

            if not result_path:
                logger.error("BexTTS returned an empty result for segment %s", segment.speaker)
                continue

            result_path = Path(result_path)
            if not result_path.exists():
                logger.error(
                    "BexTTS output file %s not found for speaker '%s'", result_path, segment.speaker
                )
                continue

            temp_output = Path(self._temp_dir) / f"segment_{index}.wav" if self._temp_dir else result_path
            try:
                if temp_output != result_path:
                    shutil.copy(result_path, temp_output)
                audio = AudioSegment.from_file(temp_output)
                duration = len(audio) / 1000.0
            except Exception as exc:  # pragma: no cover - audio parsing failures
                logger.error("BexTTS: Unable to load synthesized audio: %s", exc)
                duration = 0.0

            if segment.output_path:
                try:
                    os.makedirs(os.path.dirname(segment.output_path), exist_ok=True)
                    shutil.copy(temp_output, segment.output_path)
                    logger.debug(
                        "BexTTS: Saved synthesized audio for speaker '%s' to %s",
                        segment.speaker,
                        segment.output_path,
                    )
                except Exception as exc:  # pragma: no cover - filesystem errors
                    logger.error(
                        "BexTTS: Failed to save audio for speaker '%s' to %s: %s",
                        segment.speaker,
                        segment.output_path,
                        exc,
                    )

            diarized = DiarizationSegment(
                start_time=0.0,
                end_time=duration,
                speaker=segment.speaker,
                text=segment.text,
                confidence=1.0,
            )
            alignments.append(
                SegmentAlignment(
                    original_segment=segment,
                    diarized_segment=diarized,
                    alignment_confidence=1.0,
                )
            )

            # Remove temporary file provided by the API once copied
            if result_path.exists() and (not self._temp_dir or result_path.parent != Path(self._temp_dir)):
                try:
                    result_path.unlink()
                except OSError:
                    logger.debug("BexTTS: Could not delete temporary file %s", result_path)

        return alignments

    def estimate_audio_segment_length(self, segment_data: TTSSegmentData, language: str = "be") -> Optional[float]:
        # Simple heuristic based on number of words (approx. 150 wpm -> 0.4s per word)
        if not segment_data.text:
            return 0.0
        words = segment_data.text.strip().split()
        if not words:
            return 0.0
        return max(len(words) * 0.4, 1.0)

    def is_available(self) -> bool:
        return self.client is not None

    def cleanup(self) -> None:
        if self._temp_dir and os.path.exists(self._temp_dir):
            shutil.rmtree(self._temp_dir, ignore_errors=True)
            self._temp_dir = None
        self.client = None
