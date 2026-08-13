"""Per-speaker voice profile model.

A :class:`VoiceProfile` groups together every TTS-related knob that used to live
in a handful of parallel per-speaker mappings (``tts_system_mapping``,
``voice_name`` as dict, ``voice_prompt``, ``reference_audio_mapping``,
``reference_text_mapping``). Configs can now express all of that under a single
``voices:`` block:

.. code-block:: yaml

    voices:
      SPEAKER_00:
        tts_system: gemini
        model: gemini-2.5-flash-preview-tts
        voice_name: Kore
        style_prompt: "calm, friendly narrator"
      SPEAKER_01:
        tts_system: omnivoice
        reference_audio: D:/.../SPEAKER_03/reference.mp3
        reference_text: "И тут прямо какие-то Анадырь..."
        params:
          instruct: ""
          num_steps: 32
      "*":
        tts_system: omnivoice

:func:`normalize_voices` accepts either the new-style ``voices`` block or the
legacy mappings (still supported, with a one-time :class:`DeprecationWarning`)
and always returns ``dict[str, VoiceProfile]``. Downstream code can then look
up ``profiles[speaker]`` (falling back to ``profiles.get("*")``) without caring
which format the user wrote.
"""

from __future__ import annotations

import warnings
from dataclasses import dataclass, field
from typing import Any, Dict, Iterable, Mapping, Optional

from .log_config import get_logger

logger = get_logger(__name__)


FALLBACK_SPEAKER = "*"

_LEGACY_FIELDS = (
    "tts_system_mapping",
    "voice_name",
    "voice_prompt",
    "reference_audio_mapping",
    "reference_text_mapping",
)


@dataclass
class VoiceProfile:
    """All per-speaker TTS knobs collapsed into one object."""

    tts_system: Optional[str] = None
    model: Optional[str] = None
    voice_name: Optional[str] = None
    style_prompt: Optional[str] = None
    reference_audio: Optional[str] = None
    reference_text: Optional[str] = None
    reference_mode: Optional[str] = None
    params: Dict[str, Any] = field(default_factory=dict)

    def pool_key(self) -> tuple:
        """Client-pool identity. Profiles that share this key share a client."""
        return (
            (self.tts_system or "").lower(),
            self.model or "",
            tuple(sorted(self.params.items())),
        )

    def merged_with_fallback(self, fallback: "VoiceProfile") -> "VoiceProfile":
        """Fill missing fields from *fallback*. Params are merged shallowly."""
        merged_params = dict(fallback.params)
        merged_params.update(self.params)
        inherited_reference_mode = fallback.reference_mode
        if (
            self.reference_mode is None
            and self.tts_system
            and fallback.tts_system
            and self.tts_system.lower() != fallback.tts_system.lower()
        ):
            inherited_reference_mode = None
        return VoiceProfile(
            tts_system=self.tts_system or fallback.tts_system,
            model=self.model or fallback.model,
            voice_name=self.voice_name or fallback.voice_name,
            style_prompt=self.style_prompt or fallback.style_prompt,
            reference_audio=self.reference_audio or fallback.reference_audio,
            reference_text=self.reference_text or fallback.reference_text,
            reference_mode=(
                self.reference_mode
                if self.reference_mode is not None
                else inherited_reference_mode
            ),
            params=merged_params,
        )


def _profile_from_dict(data: Mapping[str, Any]) -> VoiceProfile:
    known = {
        "tts_system",
        "model",
        "voice_name",
        "style_prompt",
        "reference_audio",
        "reference_text",
        "reference_mode",
    }
    kwargs = {k: data.get(k) for k in known if data.get(k) is not None}
    params = dict(data.get("params") or {})
    for key, value in data.items():
        if key in known or key in {"params", "fallback_model"}:
            continue
        params[key] = value
    return VoiceProfile(params=params, **kwargs)


def _emit_legacy_warning(used_fields: Iterable[str]) -> None:
    fields_str = ", ".join(sorted(used_fields))
    message = (
        f"Legacy per-speaker fields ({fields_str}) are deprecated; migrate to "
        f"the unified 'voices:' block. See docs/superpowers/specs/"
        f"per-voice-tts-profiles.md."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning(message)


def _extract_legacy(config: Mapping[str, Any]) -> Dict[str, VoiceProfile]:
    """Fold legacy per-speaker fields into VoiceProfile objects."""
    tts_mapping = config.get("tts_system_mapping") or {}
    voice_prompt = config.get("voice_prompt") or {}
    reference_audio_mapping = config.get("reference_audio_mapping") or {}
    reference_text_mapping = config.get("reference_text_mapping") or {}

    voice_name_field = config.get("voice_name")
    voice_name_mapping: Dict[str, str] = {}
    global_voice_name: Optional[str] = None
    if isinstance(voice_name_field, dict):
        voice_name_mapping = {str(k): str(v) for k, v in voice_name_field.items()}
    elif isinstance(voice_name_field, str) and voice_name_field.strip():
        global_voice_name = voice_name_field.strip()

    used: set[str] = set()
    if tts_mapping:
        used.add("tts_system_mapping")
    if voice_name_mapping:
        used.add("voice_name")
    if voice_prompt:
        used.add("voice_prompt")
    if reference_audio_mapping:
        used.add("reference_audio_mapping")
    if reference_text_mapping:
        used.add("reference_text_mapping")

    if used:
        _emit_legacy_warning(used)

    speakers: set[str] = set()
    for source in (tts_mapping, voice_name_mapping, voice_prompt,
                   reference_audio_mapping, reference_text_mapping):
        speakers.update(str(k) for k in source.keys())

    profiles: Dict[str, VoiceProfile] = {}
    for speaker in sorted(speakers):
        profiles[speaker] = VoiceProfile(
            tts_system=tts_mapping.get(speaker) if isinstance(tts_mapping, dict) else None,
            voice_name=voice_name_mapping.get(speaker) or global_voice_name,
            style_prompt=voice_prompt.get(speaker) if isinstance(voice_prompt, dict) else None,
            reference_audio=(
                reference_audio_mapping.get(speaker)
                if isinstance(reference_audio_mapping, dict) else None
            ),
            reference_text=(
                reference_text_mapping.get(speaker)
                if isinstance(reference_text_mapping, dict) else None
            ),
        )

    top_level = VoiceProfile(
        tts_system=(str(config.get("tts_system")).strip() if config.get("tts_system") else None),
        model=(str(config.get("tts_model")).strip() if config.get("tts_model") else None),
        voice_name=global_voice_name,
        reference_audio=(
            str(config.get("reference_audio")).strip()
            if config.get("reference_audio") else None
        ),
        reference_text=(
            str(config.get("reference_text")).strip()
            if config.get("reference_text") else None
        ),
    )
    if top_level.reference_audio or top_level.reference_text:
        top_level.reference_mode = "configured"
    if any(
        value is not None
        for value in (
            top_level.tts_system,
            top_level.model,
            top_level.voice_name,
            top_level.reference_audio,
            top_level.reference_text,
        )
    ):
        profiles.setdefault(FALLBACK_SPEAKER, top_level)

    return profiles


def _warn_removed_fallback_models(config: Mapping[str, Any]) -> None:
    raw_voices = config.get("voices")
    profile_has_fallback = isinstance(raw_voices, Mapping) and any(
        isinstance(entry, Mapping) and "fallback_model" in entry
        for entry in raw_voices.values()
    )
    if "tts_fallback_model" not in config and not profile_has_fallback:
        return
    message = (
        "fallback_model and tts_fallback_model are obsolete and ignored; "
        "TTS retries keep the selected model."
    )
    warnings.warn(message, DeprecationWarning, stacklevel=3)
    logger.warning(message)


def normalize_voices(config: Mapping[str, Any]) -> Dict[str, VoiceProfile]:
    """Return the effective per-speaker profiles for *config*.

    New-style ``voices`` takes precedence over legacy fields, but any speaker
    missing from ``voices`` is still filled from the legacy mappings so partial
    migrations work.
    """
    _warn_removed_fallback_models(config)
    legacy_profiles = _extract_legacy(config)

    raw_voices = config.get("voices")
    if raw_voices is None:
        return legacy_profiles

    if not isinstance(raw_voices, Mapping):
        logger.warning(
            "Ignoring 'voices' config value; expected a mapping, got %s",
            type(raw_voices).__name__,
        )
        return legacy_profiles

    voices: Dict[str, VoiceProfile] = {}
    for speaker, entry in raw_voices.items():
        speaker_key = str(speaker)
        if entry is None:
            voices[speaker_key] = VoiceProfile()
            continue
        if not isinstance(entry, Mapping):
            logger.warning(
                "Ignoring 'voices[%s]'; expected a mapping, got %s",
                speaker_key, type(entry).__name__,
            )
            continue
        voices[speaker_key] = _profile_from_dict(entry)

    for speaker, profile in legacy_profiles.items():
        voices.setdefault(speaker, profile)

    return voices


def resolve_profile(
    profiles: Mapping[str, VoiceProfile],
    speaker: str,
    *,
    tts_system_default: Optional[str] = None,
) -> VoiceProfile:
    """Look up a profile with ``"*"`` fallback and TTS-system default.

    The default is only used when neither the speaker's profile nor the ``"*"``
    fallback specify a TTS system, so downstream code can rely on
    ``profile.tts_system`` being non-empty.
    """
    fallback = profiles.get(FALLBACK_SPEAKER, VoiceProfile())
    profile = profiles.get(speaker, VoiceProfile())
    merged = profile.merged_with_fallback(fallback)
    fallback_system = fallback.tts_system or tts_system_default
    if (
        profile.reference_mode is None
        and profile.tts_system
        and fallback.reference_mode is not None
        and fallback_system
        and profile.tts_system.lower() != fallback_system.lower()
    ):
        merged.reference_mode = None
    if not merged.tts_system and tts_system_default:
        merged.tts_system = tts_system_default
    return merged
