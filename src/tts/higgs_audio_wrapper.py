"""Higgs Audio v3 TTS wrapper for the public Hugging Face Space."""
from __future__ import annotations

import os
import shutil
import tempfile
from pathlib import Path
from typing import Any, Dict, List, Optional

from dubbing.core.log_config import get_logger

from .models import DiarizationSegment, SegmentAlignment, TTSSegmentData
from .tts_interface import TTSInterface

try:
    from gradio_client import Client, handle_file
    GRADIO_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime dependency check
    GRADIO_AVAILABLE = False

try:
    from pydub import AudioSegment
    PYDUB_AVAILABLE = True
except ImportError:  # pragma: no cover - runtime dependency check
    PYDUB_AVAILABLE = False


logger = get_logger(__name__)


class HiggsAudioWrapper(TTSInterface):
    provider_name = "higgs"
    reference_capability = "required"

    def __init__(
        self,
        space_id: str = "archivartaunik/higgs-audio-v3-tts",
        api_name: str = "/synthesize",
        temperature: float = 0.7,
        top_p: float = 0.95,
        top_k: int = 50,
        max_new_tokens: int = 2048,
        seed: int = -1,
        hf_token_env: str = "HF_TOKEN",
        **_: Any,
    ) -> None:
        if not GRADIO_AVAILABLE:
            raise ImportError(
                "gradio-client is required for Higgs TTS. "
                "Install it with 'pip install gradio-client'."
            )
        if not PYDUB_AVAILABLE:
            raise ImportError(
                "pydub is required for Higgs TTS audio handling. "
                "Install it with 'pip install pydub'."
            )

        self.space_id = space_id
        self.api_name = api_name
        self.temperature = temperature
        self.top_p = top_p
        self.top_k = top_k
        self.max_new_tokens = max_new_tokens
        self.seed = seed
        self.hf_token_env = hf_token_env
        self.client: Optional[Client] = None
        self._temp_dir: Optional[str] = None
        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}

    def initialize(self) -> None:
        import io
        import contextlib
        token = os.getenv(self.hf_token_env)
        buf = io.StringIO()
        try:
            with contextlib.redirect_stdout(buf), contextlib.redirect_stderr(buf):
                if token:
                    try:
                        self.client = Client(self.space_id, hf_token=token)
                    except TypeError:
                        self.client = Client(
                            self.space_id,
                            headers={"Authorization": f"Bearer {token}"},
                        )
                else:
                    logger.warning(
                        "%s environment variable not set - using anonymous Hugging Face access.",
                        self.hf_token_env,
                    )
                    self.client = Client(self.space_id)
        except Exception as exc:  # pragma: no cover - network failures at runtime
            raise RuntimeError(
                f"Failed to initialize Higgs TTS client for space '{self.space_id}': {exc}"
            ) from exc

        self._temp_dir = tempfile.mkdtemp(prefix="higgs_tts_segments_")

    @staticmethod
    def _extract_result_path(prediction: Any) -> Optional[Path]:
        if isinstance(prediction, str):
            return Path(prediction)
        if isinstance(prediction, dict):
            candidate = prediction.get("path") or prediction.get("name")
            if isinstance(candidate, str):
                return Path(candidate)
        if isinstance(prediction, (list, tuple)):
            for item in prediction:
                candidate = HiggsAudioWrapper._extract_result_path(item)
                if candidate is not None:
                    return candidate
        return None

    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "be",
        **_: Any,
    ) -> List[SegmentAlignment]:
        if not segments_data:
            return []
        self.require_valid_segments(segments_data)
        if not self.client:
            raise RuntimeError("Higgs TTS client not initialized. Call initialize() first.")

        if not self._temp_dir:
            self._temp_dir = tempfile.mkdtemp(prefix="higgs_tts_segments_")
        private_dir = Path(self._temp_dir)
        alignments: List[SegmentAlignment] = []

        for batch_index, segment in enumerate(segments_data):
            downloaded_path: Optional[Path] = None
            try:
                prediction = self.client.predict(
                    text=segment.text,
                    reference_audio=handle_file(segment.reference_audio_path),
                    reference_text=segment.reference_text or "",
                    temperature=self.temperature,
                    top_p=self.top_p,
                    top_k=self.top_k,
                    max_new_tokens=self.max_new_tokens,
                    seed=self.seed,
                    api_name=self.api_name,
                )
                downloaded_path = self._extract_result_path(prediction)
                if downloaded_path is None or not downloaded_path.is_file():
                    raise FileNotFoundError(f"Space returned no readable audio file: {prediction!r}")

                private_output = private_dir / f"segment_{batch_index}{downloaded_path.suffix or '.wav'}"
                if downloaded_path != private_output:
                    shutil.copy(downloaded_path, private_output)

                audio = AudioSegment.from_file(private_output)
                duration = len(audio) / 1000.0

                if segment.output_path:
                    output_path = Path(segment.output_path)
                    output_path.parent.mkdir(parents=True, exist_ok=True)
                    shutil.copy(private_output, output_path)

                alignments.append(
                    SegmentAlignment(
                        original_segment=segment,
                        diarized_segment=DiarizationSegment(
                            start_time=0.0,
                            end_time=duration,
                            speaker=segment.speaker,
                            text=segment.text,
                            confidence=1.0,
                        ),
                        alignment_confidence=1.0,
                    )
                )

            except Exception as exc:  # transient Space/output failures stay per-segment
                logger.error(
                    "Higgs TTS failed for speaker '%s' segment %s: %s",
                    segment.speaker,
                    segment.segment_index,
                    exc,
                )
            finally:
                if (
                    downloaded_path is not None
                    and downloaded_path.exists()
                    and private_dir not in downloaded_path.parents
                ):
                    try:
                        downloaded_path.unlink()
                    except OSError:
                        logger.debug("Could not remove Higgs downloaded file %s", downloaded_path)

        return alignments

    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: str = "be",
    ) -> Optional[float]:
        words = segment_data.text.strip().split() if segment_data.text else []
        if not words:
            return 0.0
        return max(len(words) * 0.45, 0.8)

    def is_available(self) -> bool:
        return self.client is not None

    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_mapping = mapping or {}

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_prompt_mapping = mapping or {}

    def cleanup(self) -> None:
        if self._temp_dir and Path(self._temp_dir).exists():
            shutil.rmtree(self._temp_dir, ignore_errors=True)
        self._temp_dir = None
        self.client = None
