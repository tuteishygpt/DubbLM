"""Local XTTS wrapper for speaker-conditioned text-to-speech."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.dubbing.core.log_config import get_logger
from .models import DiarizationSegment, SegmentAlignment, TTSSegmentData
from .tts_interface import TTSInterface

try:  # Optional dependency used for writing WAV files
    from scipy.io.wavfile import write as write_wav
    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover - dependency may be missing at runtime
    SCIPY_AVAILABLE = False

try:  # Torch is optional but enables GPU execution and context managers
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover - torch might not be installed in minimal setups
    TORCH_AVAILABLE = False

try:  # Coqui TTS provides the XTTS model implementation
    from TTS.api import TTS as CoquiTTS

    COQUI_TTS_AVAILABLE = True
except ImportError:  # pragma: no cover - handled gracefully during initialization
    COQUI_TTS_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class _ConditioningLatents:
    """Container for cached XTTS conditioning latents."""

    gpt_cond_latent: Any
    speaker_embedding: Any


class XTTSLocalWrapper(TTSInterface):
    """Text-to-speech wrapper that runs the XTTS model locally.

    The wrapper supports voice cloning by providing reference audio either per
    segment or via the global voice mapping. The implementation closely mirrors
    the reference notebook supplied with the task by caching conditioning
    latents and trimming leading/trailing silence from the generated waveform.
    """

    def __init__(
        self,
        model: str = "tts_models/multilingual/multi-dataset/xtts_v2",
        default_voice: Optional[str] = None,
        device: str = "auto",
        silence_threshold: float = 0.005,
        margin_start_sec: float = 0.05,
        margin_end_sec: float = 0.10,
        temperature: float = 0.6,
        length_penalty: float = 1.0,
        repetition_penalty: float = 10.0,
        top_k: int = 10,
        top_p: float = 0.3,
        prompt_prefix: Optional[str] = None,
        **_: Any,
    ) -> None:
        if not COQUI_TTS_AVAILABLE:
            raise ImportError(
                "coqui-tts package is required for local XTTS synthesis. "
                "Install it with 'pip install TTS==0.22.0 coqui-tts'."
            )

        self.model_name = model
        self.default_voice_path = default_voice
        self.requested_device = device
        self.prompt_prefix = prompt_prefix

        # XTTS sampling configuration parameters (mirrors reference snippet)
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p

        # Silence trimming configuration
        self.silence_threshold = silence_threshold
        self.margin_start_sec = margin_start_sec
        self.margin_end_sec = margin_end_sec

        # Runtime attributes initialised during ``initialize``
        self._tts: Optional[CoquiTTS] = None
        self._synthesizer: Any = None
        self._xtts_model: Any = None
        self._sample_rate: int = 24000
        self._conditioning_cache: Dict[str, _ConditioningLatents] = {}

        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Interface plumbing
    # ------------------------------------------------------------------
    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        """Store mapping between speaker identifiers and reference audio paths."""

        self.voice_mapping = mapping or {}

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        """Store optional style prompts that are prefixed to the synthesized text."""

        self.voice_prompt_mapping = mapping or {}

    # ------------------------------------------------------------------
    # Initialisation / cleanup
    # ------------------------------------------------------------------
    def initialize(self) -> None:
        """Load the XTTS model and prepare runtime helpers."""

        use_gpu = False
        if TORCH_AVAILABLE and self.requested_device:
            if self.requested_device.lower() == "cuda" and torch.cuda.is_available():
                use_gpu = True
            elif self.requested_device.lower() in {"auto", "gpu"}:
                use_gpu = torch.cuda.is_available()

        logger.info(
            "Loading XTTS model '%s' on %s...",
            self.model_name,
            "GPU" if use_gpu else "CPU",
        )

        try:
            self._tts = CoquiTTS(model_name=self.model_name, progress_bar=False, gpu=use_gpu)
        except Exception as exc:  # pragma: no cover - model loading issues are environment specific
            raise RuntimeError(f"Failed to load XTTS model '{self.model_name}': {exc}") from exc

        # Extract synthesizer internals to access conditioning helpers and sample rate
        self._synthesizer = getattr(self._tts, "synthesizer", None)
        self._xtts_model = None
        if self._synthesizer is not None:
            self._xtts_model = getattr(self._synthesizer, "tts_model", None)
            if self._xtts_model is None:
                # Older releases expose the model under ``model``
                self._xtts_model = getattr(self._synthesizer, "model", None)

            if hasattr(self._synthesizer, "output_sample_rate"):
                self._sample_rate = int(self._synthesizer.output_sample_rate)
            elif hasattr(self._synthesizer, "tts_config") and hasattr(
                self._synthesizer.tts_config, "audio"
            ):
                self._sample_rate = int(getattr(self._synthesizer.tts_config.audio, "sample_rate", 24000))

        logger.info("XTTS ready (sample rate: %d Hz)", self._sample_rate)

    def cleanup(self) -> None:
        """Release references to the model to free memory."""

        self._conditioning_cache.clear()
        self._xtts_model = None
        self._synthesizer = None
        self._tts = None

    # ------------------------------------------------------------------
    # Capability reporting
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self._tts is not None

    # ------------------------------------------------------------------
    # Core synthesis helpers
    # ------------------------------------------------------------------
    def _apply_prompt_prefix(self, text: str, speaker_id: Optional[str]) -> str:
        prefix = self.prompt_prefix or ""
        speaker_prompt = (
            self.voice_prompt_mapping.get(speaker_id, "") if speaker_id and self.voice_prompt_mapping else ""
        )

        final_prefix_parts = [p.strip() for p in (prefix, speaker_prompt) if p and p.strip()]
        if final_prefix_parts:
            combined_prefix = "\n".join(final_prefix_parts)
            return f"{combined_prefix}\n{text}" if text else combined_prefix
        return text

    def _resolve_reference_audio(self, segment: TTSSegmentData) -> Optional[str]:
        candidates = [
            segment.reference_audio_path,
            self.voice_mapping.get(segment.speaker) if segment.speaker else None,
            self.default_voice_path,
        ]

        for path in candidates:
            if path and os.path.exists(path):
                return path
            if path:
                logger.debug(
                    "XTTS: reference audio '%s' for speaker '%s' was not found.",
                    path,
                    segment.speaker,
                )
        return None

    def _get_conditioning_latents(self, reference_audio: str) -> Optional[_ConditioningLatents]:
        if not reference_audio or not self._xtts_model:
            return None

        if reference_audio in self._conditioning_cache:
            return self._conditioning_cache[reference_audio]

        if not hasattr(self._xtts_model, "get_conditioning_latents"):
            return None

        model_config = getattr(self._xtts_model, "config", None)
        kwargs: Dict[str, Any] = {"audio_path": reference_audio}

        # Mirror the reference code when the configuration exposes these fields
        for attr in ("gpt_cond_len", "max_ref_len", "sound_norm_refs"):
            if model_config is not None and hasattr(model_config, attr):
                kwargs[attr if attr != "gpt_cond_len" else "gpt_cond_len"] = getattr(model_config, attr)

        try:
            latents = self._xtts_model.get_conditioning_latents(**kwargs)
        except Exception as exc:  # pragma: no cover - runtime audio failures depend on environment
            logger.warning("XTTS: failed to compute conditioning latents: %s", exc)
            return None

        if not isinstance(latents, (tuple, list)) or len(latents) != 2:
            return None

        cached = _ConditioningLatents(gpt_cond_latent=latents[0], speaker_embedding=latents[1])
        self._conditioning_cache[reference_audio] = cached
        return cached

    def _run_model_inference(
        self,
        text: str,
        language: str,
        reference_audio: Optional[str],
    ) -> np.ndarray:
        if not self._tts:
            raise RuntimeError("XTTS model not initialised. Call initialize() first.")

        text_to_speak = text or ""

        conditioning = self._get_conditioning_latents(reference_audio) if reference_audio else None

        if conditioning and self._xtts_model and hasattr(self._xtts_model, "inference"):
            inference_kwargs = {
                "text": text_to_speak,
                "language": language,
                "gpt_cond_latent": conditioning.gpt_cond_latent,
                "speaker_embedding": conditioning.speaker_embedding,
                "temperature": self.temperature,
                "length_penalty": self.length_penalty,
                "repetition_penalty": self.repetition_penalty,
                "top_k": self.top_k,
                "top_p": self.top_p,
            }

            no_grad_ctx = torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext()
            with no_grad_ctx:  # type: ignore[attr-defined]
                output = self._xtts_model.inference(**inference_kwargs)

            wav = output["wav"] if isinstance(output, dict) and "wav" in output else output
        else:
            synthesis_kwargs = {"text": text_to_speak, "language": language}
            if reference_audio:
                synthesis_kwargs["speaker_wav"] = reference_audio
            wav = self._tts.tts(**synthesis_kwargs)

        if isinstance(wav, np.ndarray):
            return wav.astype(np.float32)

        if TORCH_AVAILABLE and isinstance(wav, torch.Tensor):  # pragma: no cover - depends on runtime return type
            return wav.detach().cpu().numpy().astype(np.float32)

        return np.asarray(wav, dtype=np.float32)

    def _trim_silence(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size == 0:
            return waveform

        amplitude = np.abs(waveform)
        non_silence_indices = np.where(amplitude > self.silence_threshold)[0]

        if non_silence_indices.size == 0:
            return waveform

        start_idx = max(
            non_silence_indices[0] - int(self.margin_start_sec * self._sample_rate),
            0,
        )
        end_idx = min(
            non_silence_indices[-1] + int(self.margin_end_sec * self._sample_rate),
            len(waveform) - 1,
        )

        return waveform[start_idx : end_idx + 1]

    def _save_waveform(self, waveform: np.ndarray, output_path: Optional[str]) -> None:
        if not output_path:
            return

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if not SCIPY_AVAILABLE:  # pragma: no cover - scipy is a declared dependency
            raise RuntimeError("scipy is required to save XTTS audio output but is not available.")

        pcm16 = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
        write_wav(output_path, self._sample_rate, pcm16)

    # ------------------------------------------------------------------
    # Public synthesis interface
    # ------------------------------------------------------------------
    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "be",
        **_: Any,
    ) -> List[SegmentAlignment]:
        if not segments_data:
            logger.warning("XTTS: no segments provided for synthesis.")
            return []

        alignments: List[SegmentAlignment] = []

        for index, segment in enumerate(segments_data):
            reference_audio = self._resolve_reference_audio(segment)
            text_to_speak = self._apply_prompt_prefix(segment.text, segment.speaker)

            try:
                waveform = self._run_model_inference(text_to_speak, language, reference_audio)
            except Exception as exc:  # pragma: no cover - synthesis failures depend on runtime environment
                logger.error(
                    "XTTS: failed to synthesize segment %d for speaker '%s': %s",
                    index + 1,
                    segment.speaker,
                    exc,
                )
                continue

            trimmed_waveform = self._trim_silence(waveform)
            duration = len(trimmed_waveform) / float(self._sample_rate)

            logger.debug(
                "XTTS: segment %d duration before trim %.3fs, after trim %.3fs",
                index + 1,
                len(waveform) / float(self._sample_rate),
                duration,
            )

            self._save_waveform(trimmed_waveform, segment.output_path)

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

        return alignments

    # ------------------------------------------------------------------
    # Estimation helpers
    # ------------------------------------------------------------------
    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: str = "be",
    ) -> Optional[float]:  # noqa: D401 - the interface already documents the method
        _ = language  # The heuristic is language agnostic for now

        text = (segment_data.text or "").strip()
        if not text:
            return 0.0

        # Basic heuristic: assume ~12 characters per second, with a minimum duration of 0.8 seconds
        estimated_duration = max(len(text) / 12.0, 0.8)
        return estimated_duration

