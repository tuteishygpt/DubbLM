"""Revisioned, owner-scoped speaker reference library operations.

The service keeps filesystem details behind the injected media store.  Legacy
``reference_audio_path`` metadata remains readable, but callers receive only
opaque media identifiers and URLs.
"""

from __future__ import annotations

import hashlib
import os
import re
import shutil
import threading
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any
from urllib.parse import quote

import yaml

from .contracts import MediaStore


class ReferenceError(Exception):
    """Base class for reference-library domain failures."""


class ReferenceConflictError(ReferenceError):
    """A mutation used an obsolete library revision."""


class ReferenceValidationError(ReferenceError):
    """A reference request is incomplete or invalid."""


class ReferenceNotFoundError(ReferenceError):
    """The requested reference entry does not exist for this owner."""


class ReferenceWriteError(ReferenceError):
    """Reference metadata could not be replaced atomically."""


@dataclass(frozen=True)
class ReferenceAudio:
    """Public, storage-independent audio description."""

    id: str
    name: str
    url: str


@dataclass(frozen=True)
class ReferenceEntry:
    speaker_id: str
    reference_text: str
    audio: ReferenceAudio


@dataclass(frozen=True)
class ReferenceLibrarySnapshot:
    revision: str
    entries: tuple[ReferenceEntry, ...]


@dataclass(frozen=True)
class ReferenceAssignment:
    profile_speaker_id: str
    library_speaker_id: str
    audio_id: str
    reference_text: str
    settings_revision: str


class ReferenceLibraryService:
    """Copy/list/delete/assign references through an opaque ``MediaStore``."""

    _locks: dict[str, threading.RLock] = {}
    _locks_guard = threading.Lock()

    def __init__(self, library_path: str | Path, media_store: MediaStore) -> None:
        self._root = Path(library_path)
        self._media_store = media_store
        self._legacy_media: dict[tuple[str, str], object] = {}
        key = str(self._root.resolve())
        with self._locks_guard:
            self._lock = self._locks.setdefault(key, threading.RLock())

    def list(self, *, owner_id: str) -> ReferenceLibrarySnapshot:
        owner = self._validate_owner(owner_id)
        with self._lock:
            revision, metadata = self._read_library(owner)
            entries = tuple(
                self._entry_from_metadata(owner, item)
                for item in metadata
            )
            return ReferenceLibrarySnapshot(revision=revision, entries=entries)

    def save(
        self,
        *,
        owner_id: str,
        speaker_id: str,
        source_audio: object,
        reference_text: str,
        revision: str,
        audio_name: str | None = None,
    ) -> ReferenceLibrarySnapshot:
        owner = self._validate_owner(owner_id)
        speaker = self._validate_speaker(speaker_id)
        if source_audio is None or source_audio == "":
            raise ReferenceValidationError("Reference audio is required.")
        name = audio_name or Path(str(source_audio)).name or "reference.wav"

        with self._lock:
            actual, metadata = self._read_library(owner)
            self._ensure_revision(revision, actual)
            old = next(
                (item for item in metadata if item[1].get("speaker_id") == speaker),
                None,
            )
            saved = self._media_store.save(
                owner_id=owner,
                source=source_audio,
                name=name,
                kind="reference",
            )
            media_id = self._record_id(saved)
            target = self._owner_dir(owner) / self._safe_component(speaker) / "meta.yml"
            payload = {
                "speaker_id": speaker,
                "reference_audio_ref": media_id,
                "reference_text": str(reference_text or "").strip(),
            }
            try:
                self._atomic_metadata_replace(target, payload)
            except ReferenceWriteError:
                self._delete_media(owner, media_id)
                raise

            if old is not None:
                old_id = str(old[1].get("reference_audio_ref") or "").strip()
                if old_id and old_id != media_id:
                    self._delete_media(owner, old_id)
            return self.list(owner_id=owner)

    def delete(
        self, *, owner_id: str, speaker_id: str, revision: str
    ) -> ReferenceLibrarySnapshot:
        owner = self._validate_owner(owner_id)
        speaker = self._validate_speaker(speaker_id)
        with self._lock:
            actual, metadata = self._read_library(owner)
            self._ensure_revision(revision, actual)
            matches = [item for item in metadata if item[1].get("speaker_id") == speaker]
            for metadata_path, item in matches:
                media_id = str(item.get("reference_audio_ref") or "").strip()
                shutil.rmtree(metadata_path.parent)
                if media_id:
                    self._delete_media(owner, media_id)
            return self.list(owner_id=owner)

    def assign(
        self,
        *,
        owner_id: str,
        library_speaker_id: str,
        profile_speaker_id: str,
        settings: Any,
        settings_revision: str,
    ) -> ReferenceAssignment:
        owner = self._validate_owner(owner_id)
        library_speaker = self._validate_speaker(library_speaker_id)
        with self._lock:
            _revision, metadata = self._read_library(owner)
            selected = next(
                (item for _path, item in metadata if item.get("speaker_id") == library_speaker),
                None,
            )
            if selected is None:
                raise ReferenceNotFoundError(f"Reference not found: {library_speaker}.")
            record = self._resolve_record(owner, selected)
            internal_path = self._record_path(record)
            snapshot = settings.assign_reference(
                profile_speaker_id,
                reference_audio=internal_path,
                reference_text=str(selected.get("reference_text") or "").strip(),
                revision=settings_revision,
            )
            return ReferenceAssignment(
                profile_speaker_id=profile_speaker_id,
                library_speaker_id=library_speaker,
                audio_id=self._record_id(record),
                reference_text=str(selected.get("reference_text") or "").strip(),
                settings_revision=str(snapshot.revision),
            )

    def _read_library(self, owner: str) -> tuple[str, list[tuple[Path, dict[str, Any]]]]:
        owner_dir = self._owner_dir(owner)
        if not owner_dir.exists():
            return hashlib.sha256(b"").hexdigest(), []
        metadata: list[tuple[Path, dict[str, Any]]] = []
        digest = hashlib.sha256()
        for path in sorted(owner_dir.glob("*/meta.yml"), key=lambda value: str(value)):
            try:
                raw = path.read_bytes()
                decoded = yaml.safe_load(raw.decode("utf-8")) or {}
            except (OSError, UnicodeDecodeError, yaml.YAMLError):
                continue
            if not isinstance(decoded, dict):
                continue
            speaker = str(decoded.get("speaker_id") or path.parent.name).strip()
            if not speaker:
                continue
            audio_ref = str(decoded.get("reference_audio_ref") or "").strip()
            legacy_path = str(decoded.get("reference_audio_path") or "").strip()
            if not audio_ref and not legacy_path:
                continue
            item = dict(decoded)
            item["speaker_id"] = speaker
            metadata.append((path, item))
            digest.update(path.parent.name.encode("utf-8"))
            digest.update(b"\0")
            digest.update(raw)
            digest.update(b"\0")
        return digest.hexdigest(), metadata

    def _entry_from_metadata(
        self, owner: str, item: tuple[Path, dict[str, Any]]
    ) -> ReferenceEntry:
        _path, metadata = item
        record = self._resolve_record(owner, metadata)
        return ReferenceEntry(
            speaker_id=str(metadata["speaker_id"]),
            reference_text=str(metadata.get("reference_text") or "").strip(),
            audio=ReferenceAudio(
                id=self._record_id(record),
                name=self._record_name(record),
                url=(
                    self._record_url(record)
                    or f"/api/reference-library/{quote(str(metadata['speaker_id']), safe='')}/audio"
                ),
            ),
        )

    def _resolve_record(self, owner: str, metadata: dict[str, Any]) -> object:
        media_id = str(metadata.get("reference_audio_ref") or "").strip()
        if media_id:
            return self._media_store.get(owner_id=owner, media_id=media_id)
        legacy_path = str(metadata.get("reference_audio_path") or "").strip()
        cache_key = (owner, legacy_path)
        if cache_key not in self._legacy_media:
            self._legacy_media[cache_key] = self._media_store.register(
                owner_id=owner,
                path=legacy_path,
                name=Path(legacy_path).name,
                kind="reference",
            )
        return self._legacy_media[cache_key]

    def _owner_dir(self, owner: str) -> Path:
        return self._root / self._safe_component(owner)

    @staticmethod
    def _safe_component(value: str) -> str:
        return "".join(char if char.isalnum() or char in {"-", "_"} else "_" for char in value)

    @classmethod
    def _validate_owner(cls, owner_id: str) -> str:
        owner = str(owner_id or "").strip()
        if not owner:
            raise ReferenceValidationError("Owner ID is required.")
        return owner

    @staticmethod
    def _validate_speaker(speaker_id: str) -> str:
        speaker = str(speaker_id or "").strip()
        if not speaker:
            raise ReferenceValidationError("Speaker ID is required.")
        return speaker

    @staticmethod
    def _ensure_revision(expected: str, actual: str) -> None:
        if expected != actual:
            raise ReferenceConflictError("Reference library was changed by another writer.")

    @staticmethod
    def _record_value(record: object, *names: str) -> object:
        for name in names:
            if isinstance(record, dict) and name in record:
                return record[name]
            if hasattr(record, name):
                return getattr(record, name)
        return ""

    @classmethod
    def _record_id(cls, record: object) -> str:
        value = str(cls._record_value(record, "id", "media_id") or "").strip()
        if not value:
            raise ReferenceValidationError("MediaStore returned no media identifier.")
        return value

    @classmethod
    def _record_name(cls, record: object) -> str:
        return str(cls._record_value(record, "name") or "reference audio")

    @classmethod
    def _record_url(cls, record: object) -> str:
        return str(cls._record_value(record, "url") or "")

    @classmethod
    def _record_path(cls, record: object) -> str:
        value = str(cls._record_value(record, "path") or "").strip()
        if not value:
            raise ReferenceValidationError("Stored reference has no internal pipeline path.")
        return value

    def _delete_media(self, owner: str, media_id: str) -> None:
        try:
            self._media_store.delete(owner_id=owner, media_id=media_id)
        except Exception:
            pass

    @staticmethod
    def _atomic_metadata_replace(path: Path, payload: dict[str, Any]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        content = yaml.safe_dump(payload, sort_keys=False, allow_unicode=True).encode("utf-8")
        temp_name: str | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temp_name = temporary.name
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temp_name, path)
        except OSError as exc:
            raise ReferenceWriteError(f"Could not write reference metadata: {exc}") from exc
        finally:
            if temp_name:
                try:
                    Path(temp_name).unlink(missing_ok=True)
                except OSError:
                    pass
