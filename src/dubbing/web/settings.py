"""Revisioned YAML settings and voice-profile operations.

This module deliberately has no framework dependency, keeping the settings
contract reusable by HTTP routes and other callers.
"""

from __future__ import annotations

import hashlib
import json
import os
import threading
from contextlib import contextmanager
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Mapping

import yaml

try:  # Windows locking backend
    import msvcrt as _msvcrt
except ImportError:  # pragma: no cover - exercised on POSIX
    _msvcrt = None

try:  # POSIX locking backend
    import fcntl as _fcntl
except ImportError:  # pragma: no cover - exercised on Windows
    _fcntl = None

from .schema import JSON_TEXT_FIELDS, LIST_TEXT_FIELDS
from .schema import TTS_PROVIDER_CHOICES, TTS_REFERENCE_CAPABILITIES
from ..core.voice_profiles import VoiceProfile, resolve_profile


class SettingsError(Exception):
    """Base class for settings-domain failures."""


class SettingsConflictError(SettingsError):
    """The caller attempted to write an obsolete document revision."""


class SettingsValidationError(SettingsError):
    """A supplied setting cannot be represented by the compatible YAML form."""


class SettingsWriteError(SettingsError):
    """An atomic replacement could not be completed."""


@dataclass(frozen=True)
class SettingsSnapshot:
    """A configuration document together with its content-hash revision."""

    revision: str
    values: dict[str, Any]


@dataclass(frozen=True)
class VoiceProfilesSnapshot:
    """The persisted voice-profile mapping and its document revision."""

    revision: str
    profiles: dict[str, dict[str, Any]]


class SettingsService:
    """Load and atomically update one compatible DubbLM YAML document."""

    _path_locks: dict[str, threading.RLock] = {}
    _path_locks_guard = threading.Lock()

    def __init__(self, config_path: str | Path) -> None:
        self._config_path = Path(config_path)
        key = str(self._config_path.resolve())
        with self._path_locks_guard:
            self._write_lock = self._path_locks.setdefault(key, threading.RLock())

    def load(self) -> SettingsSnapshot:
        raw = self._read_bytes()
        return SettingsSnapshot(revision=self._revision(raw), values=self._decode(raw))

    def save(self, values: Mapping[str, Any], *, revision: str) -> SettingsSnapshot:
        with self._write_lock:
            with self._interprocess_lock():
                raw = self._read_bytes()
                actual_revision = self._revision(raw)
                if revision != actual_revision:
                    raise SettingsConflictError("Settings were changed by another writer.")

                updated = self._decode(raw)
                for field, value in values.items():
                    normalized = self._normalize_value(str(field), value)
                    if normalized is None:
                        updated.pop(str(field), None)
                    else:
                        updated[str(field)] = normalized

                written = self._encode(updated)
                self._atomic_replace(written)
                return SettingsSnapshot(revision=self._revision(written), values=updated)

    def list_profiles(self) -> VoiceProfilesSnapshot:
        """Return only explicit ``voices`` entries without legacy expansion."""
        snapshot = self.load()
        return VoiceProfilesSnapshot(
            revision=snapshot.revision,
            profiles=self._profiles_from_values(snapshot.values),
        )

    def put_profile(
        self,
        speaker_id: str,
        profile: Mapping[str, Any],
        *,
        revision: str,
    ) -> VoiceProfilesSnapshot:
        speaker = self._validate_speaker_id(speaker_id)
        candidate = self._clean_profile(profile)
        snapshot = self.load()
        self._ensure_revision(revision, snapshot.revision)
        profiles = self._profiles_from_values(snapshot.values)
        profiles[speaker] = candidate
        self._validate_profiles(profiles)
        updated = self.save({"voices": profiles}, revision=revision)
        return VoiceProfilesSnapshot(revision=updated.revision, profiles=profiles)

    def delete_profile(self, speaker_id: str, *, revision: str) -> VoiceProfilesSnapshot:
        speaker = self._validate_speaker_id(speaker_id)
        snapshot = self.load()
        self._ensure_revision(revision, snapshot.revision)
        profiles = self._profiles_from_values(snapshot.values)
        profiles.pop(speaker, None)
        self._validate_profiles(profiles)
        updated = self.save({"voices": profiles}, revision=revision)
        return VoiceProfilesSnapshot(revision=updated.revision, profiles=profiles)

    def assign_reference(
        self,
        speaker_id: str,
        *,
        reference_audio: str,
        reference_text: str | None,
        revision: str,
    ) -> VoiceProfilesSnapshot:
        speaker = self._validate_speaker_id(speaker_id)
        snapshot = self.load()
        self._ensure_revision(revision, snapshot.revision)
        profiles = self._profiles_from_values(snapshot.values)
        if speaker not in profiles:
            raise SettingsValidationError(f"Voice profile not found: {speaker}.")
        audio = str(reference_audio or "").strip()
        if not audio:
            raise SettingsValidationError("Reference audio is required.")
        profile = dict(profiles[speaker])
        profile.update(
            reference_audio=audio,
            reference_text=str(reference_text).strip() if reference_text else None,
            reference_mode="configured",
        )
        profiles[speaker] = self._clean_profile(profile)
        self._validate_profiles(profiles)
        updated = self.save({"voices": profiles}, revision=revision)
        return VoiceProfilesSnapshot(revision=updated.revision, profiles=profiles)

    def _read_bytes(self) -> bytes:
        try:
            return self._config_path.read_bytes()
        except FileNotFoundError:
            return b""

    def _decode(self, raw: bytes) -> dict[str, Any]:
        if not raw.strip():
            return {}
        try:
            decoded = yaml.safe_load(raw.decode("utf-8"))
        except (UnicodeDecodeError, yaml.YAMLError) as exc:
            raise SettingsValidationError(f"Invalid settings YAML: {exc}") from exc
        if decoded is None:
            return {}
        if not isinstance(decoded, dict):
            raise SettingsValidationError("Settings YAML must contain a mapping.")
        return dict(decoded)

    @staticmethod
    def _encode(values: Mapping[str, Any]) -> bytes:
        return yaml.safe_dump(dict(values), sort_keys=False, allow_unicode=True).encode("utf-8")

    @staticmethod
    def _revision(raw: bytes) -> str:
        return hashlib.sha256(raw).hexdigest()

    @staticmethod
    def _normalize_value(field: str, value: Any) -> Any:
        if isinstance(value, str):
            value = value.strip()
            if not value:
                return None
        if field == "duration":
            try:
                if float(value) <= 0:
                    return None
            except (TypeError, ValueError):
                raise SettingsValidationError("duration must be numeric.")
        if field in JSON_TEXT_FIELDS and isinstance(value, str):
            try:
                return json.loads(value)
            except json.JSONDecodeError as exc:
                raise SettingsValidationError(f"Invalid JSON for {field}: {exc}") from exc
        if field in LIST_TEXT_FIELDS and isinstance(value, str):
            return [line.strip() for line in value.splitlines() if line.strip()]
        return value

    @staticmethod
    def _ensure_revision(expected: str, actual: str) -> None:
        if expected != actual:
            raise SettingsConflictError("Settings were changed by another writer.")

    @classmethod
    def _profiles_from_values(cls, values: Mapping[str, Any]) -> dict[str, dict[str, Any]]:
        raw_profiles = values.get("voices") or {}
        if not isinstance(raw_profiles, Mapping):
            raise SettingsValidationError("voices must be a mapping.")
        profiles: dict[str, dict[str, Any]] = {}
        for speaker, profile in raw_profiles.items():
            if not isinstance(profile, Mapping):
                raise SettingsValidationError(f"Voice profile {speaker} must be a mapping.")
            profiles[str(speaker)] = cls._clean_profile(profile)
        return profiles

    @staticmethod
    def _validate_speaker_id(speaker_id: str) -> str:
        speaker = str(speaker_id or "").strip()
        if not speaker:
            raise SettingsValidationError("Profile name must not be empty.")
        if len(speaker) > 128:
            raise SettingsValidationError("Profile name must be 128 characters or fewer.")
        return speaker

    @staticmethod
    def _clean_profile(profile: Mapping[str, Any]) -> dict[str, Any]:
        cleaned: dict[str, Any] = {}
        for key, value in profile.items():
            if value is None:
                continue
            if isinstance(value, str):
                value = value.strip()
                if not value:
                    continue
            cleaned[str(key)] = value
        provider = str(cleaned.get("tts_system") or "").lower()
        cap = TTS_REFERENCE_CAPABILITIES.get(provider, "unsupported")
        if cap in {"required", "optional"} and "reference_mode" not in cleaned:
            if cleaned.get("reference_audio"):
                cleaned["reference_mode"] = "configured"
        return cleaned

    @classmethod
    def _validate_profiles(cls, profiles: Mapping[str, Mapping[str, Any]]) -> None:
        profile_objects: dict[str, VoiceProfile] = {}
        for speaker, profile in profiles.items():
            raw_params = profile.get("params")
            if raw_params is not None and not isinstance(raw_params, Mapping):
                raise SettingsValidationError(
                    f"Voice profile {speaker} params must be a mapping."
                )
            profile_objects[speaker] = VoiceProfile(
                tts_system=profile.get("tts_system"),
                model=profile.get("model"),
                voice_name=profile.get("voice_name"),
                style_prompt=profile.get("style_prompt"),
                reference_audio=profile.get("reference_audio"),
                reference_text=profile.get("reference_text"),
                reference_mode=profile.get("reference_mode"),
                params=dict(raw_params or {}),
            )
        for speaker in profile_objects:
            cls._validate_profile(speaker, resolve_profile(profile_objects, speaker))

    @staticmethod
    def _validate_profile(speaker: str, profile: VoiceProfile) -> None:
        provider = str(profile.tts_system or "").lower()
        if not provider:
            raise SettingsValidationError("A voice profile must define tts_system.")
        supported = set(TTS_PROVIDER_CHOICES) | {"f5_tts"}
        if provider not in supported:
            raise SettingsValidationError(f"Unknown TTS system: {provider}.")
        if provider in {"gemini", "openai"} and not profile.model:
            raise SettingsValidationError(f"A model is required for {provider}.")
        capability = TTS_REFERENCE_CAPABILITIES.get(provider, "unsupported")
        mode = profile.reference_mode
        if capability == "unsupported" and mode:
            raise SettingsValidationError(f"{provider} does not support reference_mode.")
        if capability in {"required", "optional"} and not mode:
            raise SettingsValidationError(f"An explicit reference_mode is required for {provider}.")
        if capability == "required" and mode == "none":
            raise SettingsValidationError(f"reference_mode 'none' is not allowed for {provider}.")
        if mode == "configured":
            if not str(profile.reference_audio or "").strip():
                raise SettingsValidationError("reference_audio is required for mode 'configured'.")
            # Path existence is validated at route level when resolving from reference library.

    def _atomic_replace(self, content: bytes) -> None:
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        temp_name: str | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=self._config_path.parent,
                prefix=f".{self._config_path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temp_name = temporary.name
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temp_name, self._config_path)
        except OSError as exc:
            raise SettingsWriteError(f"Could not write settings: {exc}") from exc
        finally:
            if temp_name:
                try:
                    Path(temp_name).unlink(missing_ok=True)
                except OSError:
                    pass

    @contextmanager
    def _interprocess_lock(self):
        """Hold an OS-released Windows byte lock across one full write transaction."""
        lock_path = self._config_path.with_name(f".{self._config_path.name}.lock")
        self._config_path.parent.mkdir(parents=True, exist_ok=True)
        lock_path.touch(exist_ok=True)
        with lock_path.open("r+b") as lock_file:
            try:
                if _msvcrt is not None:
                    lock_file.seek(0)
                    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_LOCK, 1)
                elif _fcntl is not None:
                    _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_EX)
                else:  # pragma: no cover - supported Python platforms provide one
                    raise OSError("No supported file-locking backend is available.")
            except OSError as exc:
                raise SettingsWriteError(f"Could not lock settings for writing: {exc}") from exc
            try:
                yield
            finally:
                if _msvcrt is not None:
                    lock_file.seek(0)
                    _msvcrt.locking(lock_file.fileno(), _msvcrt.LK_UNLCK, 1)
                else:
                    _fcntl.flock(lock_file.fileno(), _fcntl.LOCK_UN)
