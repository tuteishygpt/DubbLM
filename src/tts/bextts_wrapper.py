"""Local XTTS wrapper for speaker-conditioned text-to-speech (HF BE_XTTS_V2)."""

from __future__ import annotations

import contextlib
import os
from dataclasses import dataclass
from typing import Any, Dict, List, Optional

import numpy as np

from src.dubbing.core.log_config import get_logger
from .models import DiarizationSegment, SegmentAlignment, TTSSegmentData
from .tts_interface import TTSInterface

# ---------------------------------------------------------------------
# Optional deps
# ---------------------------------------------------------------------
try:  # Optional dependency used for writing WAV files
    from scipy.io.wavfile import write as write_wav

    SCIPY_AVAILABLE = True
except ImportError:  # pragma: no cover
    SCIPY_AVAILABLE = False

try:  # Torch is optional but enables GPU execution and context managers
    import torch

    TORCH_AVAILABLE = True
except ImportError:  # pragma: no cover
    TORCH_AVAILABLE = False

try:
    from huggingface_hub import hf_hub_download

    HF_AVAILABLE = True
except ImportError:  # pragma: no cover
    HF_AVAILABLE = False

try:
    from TTS.tts.configs.xtts_config import XttsConfig
    from TTS.tts.models.xtts import Xtts

    XTTS_DEPS_AVAILABLE = True
except ImportError:  # pragma: no cover
    XTTS_DEPS_AVAILABLE = False

logger = get_logger(__name__)


@dataclass
class _ConditioningLatents:
    """Container for cached XTTS conditioning latents."""

    gpt_cond_latent: Any
    speaker_embedding: Any


class XTTSLocalWrapper(TTSInterface):
    """Text-to-speech wrapper that runs the BE_XTTS_V2 model locally.

    Реалізацыя цалкам перапісана пад нізкаўзроўневы прыклад:
    - XttsConfig + Xtts з TTS.tts.models.xtts
    - загрузка ваг з Hugging Face праз hf_hub_download
    - get_conditioning_latents(...) + inference(...)
    """

    def __init__(
        self,
        repo_id: str = "archivartaunik/BE_XTTS_V2_10ep250k",
        model_dir: str = "./be_xtts_model",
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
        if not XTTS_DEPS_AVAILABLE:
            raise ImportError(
                "TTS (XTTS) пакет абавязковы для лакальнага XTTS. "
                "Усталяваць: 'pip install TTS==0.22.0 coqui-tts'."
            )
        if not HF_AVAILABLE:
            raise ImportError(
                "huggingface_hub неабходны для спампоўкі ваг. "
                "Усталяваць: 'pip install huggingface_hub'."
            )

        # Прымаем ліцэнзію Coqui (аналаг вашага прыкладу)
        os.environ["COQUI_TOS_AGREED"] = "1"

        self.repo_id = repo_id
        self.model_dir = model_dir

        self.default_voice_path = default_voice  # калі None — возьмем voice.wav з мадэлі
        self.requested_device = device
        self.prompt_prefix = prompt_prefix

        # XTTS sampling configuration (як у прыкладзе)
        self.temperature = temperature
        self.length_penalty = length_penalty
        self.repetition_penalty = repetition_penalty
        self.top_k = top_k
        self.top_p = top_p

        # Silence trimming configuration
        self.silence_threshold = silence_threshold
        self.margin_start_sec = margin_start_sec
        self.margin_end_sec = margin_end_sec

        # Runtime attributes
        self._xtts_model: Optional[Xtts] = None
        self._config: Optional[XttsConfig] = None
        self._sample_rate: int = 24000
        self._conditioning_cache: Dict[str, _ConditioningLatents] = {}

        self.voice_mapping: Dict[str, str] = {}
        self.voice_prompt_mapping: Dict[str, str] = {}

    # ------------------------------------------------------------------
    # Interface plumbing
    # ------------------------------------------------------------------
    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_mapping = mapping or {}

    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        self.voice_prompt_mapping = mapping or {}

    # ------------------------------------------------------------------
    # Initialisation / cleanup
    # ------------------------------------------------------------------
    def _ensure_model_files(self) -> None:
        os.makedirs(self.model_dir, exist_ok=True)

        checkpoint_file = os.path.join(self.model_dir, "model.pth")
        config_file = os.path.join(self.model_dir, "config.json")
        vocab_file = os.path.join(self.model_dir, "vocab.json")
        default_voice_file = os.path.join(self.model_dir, "voice.wav")

        if not os.path.exists(checkpoint_file):
            hf_hub_download(self.repo_id, filename="model.pth", local_dir=self.model_dir)

        if not os.path.exists(config_file):
            hf_hub_download(self.repo_id, filename="config.json", local_dir=self.model_dir)

        if not os.path.exists(vocab_file):
            hf_hub_download(self.repo_id, filename="vocab.json", local_dir=self.model_dir)

        if not os.path.exists(default_voice_file):
            hf_hub_download(self.repo_id, filename="voice.wav", local_dir=self.model_dir)

        # Калі default_voice не зададзены звонку — выкарыстоўваем voice.wav мадэлі
        if self.default_voice_path is None:
            self.default_voice_path = default_voice_file

        self._checkpoint_file = checkpoint_file
        self._config_file = config_file
        self._vocab_file = vocab_file

    def initialize(self) -> None:
        """Load the BE_XTTS_V2 model and prepare runtime helpers."""

        self._ensure_model_files()

        # 1) канфіг
        config = XttsConfig()
        config.load_json(self._config_file)

        # 2) ініцыялізацыя мадэлі
        model = Xtts.init_from_config(config)

        # 3) загрузка ваг і vocab
        model.load_checkpoint(
            config,
            checkpoint_path=self._checkpoint_file,
            vocab_path=self._vocab_file,
            use_deepspeed=False,
        )

        # 4) перадача на GPU / CPU
        use_gpu = False
        device_str = "cpu"
        if TORCH_AVAILABLE and self.requested_device:
            if self.requested_device.lower() == "cuda" and torch.cuda.is_available():
                use_gpu = True
                device_str = "cuda:0"
            elif self.requested_device.lower() in {"auto", "gpu"} and torch.cuda.is_available():
                use_gpu = True
                device_str = "cuda:0"

        if TORCH_AVAILABLE:
            model.to(device_str)

        self._xtts_model = model
        self._config = config

        # sample_rate бярэм з config, калі ёсць
        try:
            self._sample_rate = int(config.audio.sample_rate)
        except Exception:  # pragma: no cover
            self._sample_rate = 24000

        logger.info(
            "✅ BE_XTTS_V2 мадэль загружана і гатовая. Прылада: %s, sample_rate=%d, файлы ў '%s'",
            device_str,
            self._sample_rate,
            self.model_dir,
        )

    def cleanup(self) -> None:
        self._conditioning_cache.clear()
        self._xtts_model = None
        self._config = None

    # ------------------------------------------------------------------
    # Capability reporting
    # ------------------------------------------------------------------
    def is_available(self) -> bool:
        return self._xtts_model is not None

    # ------------------------------------------------------------------
    # Core synthesis helpers
    # ------------------------------------------------------------------
    def _apply_prompt_prefix(self, text: str, speaker_id: Optional[str]) -> str:
        prefix = self.prompt_prefix or ""
        speaker_prompt = (
            self.voice_prompt_mapping.get(speaker_id, "")
            if speaker_id and self.voice_prompt_mapping
            else ""
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
        """Поўны аналаг прыкладу get_conditioning_latents(...)."""
        if not reference_audio or not self._xtts_model or not self._config:
            return None

        if reference_audio in self._conditioning_cache:
            return self._conditioning_cache[reference_audio]

        if not hasattr(self._xtts_model, "get_conditioning_latents"):
            return None

        kwargs: Dict[str, Any] = {
            "audio_path": reference_audio,
            "gpt_cond_len": self._config.gpt_cond_len,
            "max_ref_length": self._config.max_ref_len,
            "sound_norm_refs": self._config.sound_norm_refs,
        }

        no_grad_ctx = torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext()
        try:
            with no_grad_ctx:  # type: ignore[attr-defined]
                gpt_cond_latent, speaker_embedding = self._xtts_model.get_conditioning_latents(**kwargs)
        except Exception as exc:  # pragma: no cover
            logger.warning("XTTS: failed to compute conditioning latents: %s", exc)
            return None

        cached = _ConditioningLatents(
            gpt_cond_latent=gpt_cond_latent,
            speaker_embedding=speaker_embedding,
        )
        self._conditioning_cache[reference_audio] = cached
        return cached

    def _run_model_inference(
        self,
        text: str,
        language: str,
        reference_audio: Optional[str],
    ) -> np.ndarray:
        if not self._xtts_model:
            raise RuntimeError("XTTS model not initialised. Call initialize() first.")

        # Выкарыстоўваем мову як ёсць, без падмены "be" -> "en"
        model_language = language

        text_to_speak = text or ""

        conditioning = self._get_conditioning_latents(reference_audio) if reference_audio else None

        if conditioning is None:
            logger.warning(
                "XTTS: no conditioning latents for '%s', голас можа быць не тым, што чакалася.",
                reference_audio,
            )

        inference_kwargs = {
            "text": text_to_speak,
            "language": model_language,
            "gpt_cond_latent": conditioning.gpt_cond_latent if conditioning else None,
            "speaker_embedding": conditioning.speaker_embedding if conditioning else None,
            "temperature": self.temperature,
            "length_penalty": self.length_penalty,
            "repetition_penalty": self.repetition_penalty,
            "top_k": self.top_k,
            "top_p": self.top_p,
        }

        # Прыбіраем None, каб не ламаць подпіс функцыі
        inference_kwargs = {k: v for k, v in inference_kwargs.items() if v is not None}

        no_grad_ctx = torch.no_grad() if TORCH_AVAILABLE else contextlib.nullcontext()
        with no_grad_ctx:  # type: ignore[attr-defined]
            out = self._xtts_model.inference(**inference_kwargs)

        wav = out["wav"] if isinstance(out, dict) and "wav" in out else out

        if TORCH_AVAILABLE and isinstance(wav, torch.Tensor):  # pragma: no cover
            wav = wav.detach().cpu().numpy()

        if isinstance(wav, np.ndarray):
            return wav.astype(np.float32)

        return np.asarray(wav, dtype=np.float32)

    def _trim_silence(self, waveform: np.ndarray) -> np.ndarray:
        if waveform.size == 0:
            return waveform

        amp = np.abs(waveform)
        non_silence_idx = np.where(amp > self.silence_threshold)[0]

        if non_silence_idx.size == 0:
            # усё цішыня — не кранаем
            return waveform

        start_idx = max(
            non_silence_idx[0] - int(self.margin_start_sec * self._sample_rate),
            0,
        )
        end_idx = min(
            non_silence_idx[-1] + int(self.margin_end_sec * self._sample_rate),
            len(waveform) - 1,
        )

        return waveform[start_idx : end_idx + 1]

    def _save_waveform(self, waveform: np.ndarray, output_path: Optional[str]) -> None:
        if not output_path:
            return

        output_dir = os.path.dirname(output_path)
        if output_dir:
            os.makedirs(output_dir, exist_ok=True)

        if not SCIPY_AVAILABLE:  # pragma: no cover
            raise RuntimeError("scipy is required to save XTTS audio output but is not available.")

        wav_int16 = np.clip(waveform * 32767.0, -32768, 32767).astype(np.int16)
        write_wav(output_path, self._sample_rate, wav_int16)

    # ------------------------------------------------------------------
    # Public synthesis interface
    # ------------------------------------------------------------------
    def synthesize(
        self,
        segments_data: List[TTSSegmentData],
        language: str = "en",
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
            except Exception as exc:  # pragma: no cover
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
        language: str = "en",
    ) -> Optional[float]:
        _ = language  # пакуль што без моўнай карэкцыі

        text = (segment_data.text or "").strip()
        if not text:
            return 0.0

        # Простая эмпірыка: ~12 сімвалаў у секунду, мінімум 0.8 с
        return max(len(text) / 12.0, 0.8)
