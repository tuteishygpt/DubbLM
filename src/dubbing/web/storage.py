"""Safe owner-scoped media storage with opaque references."""

from __future__ import annotations

import json
import os
import re
import shutil
import subprocess
from dataclasses import dataclass
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import BinaryIO, Callable, Iterable
from uuid import UUID, uuid4


DEFAULT_MAX_UPLOAD_BYTES = 20 * 1024**3
VIDEO_EXTENSIONS = frozenset({".mp4", ".mov", ".mkv", ".webm", ".avi"})
AUDIO_EXTENSIONS = frozenset({".wav", ".mp3", ".m4a", ".flac", ".ogg"})
_SAFE_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")
_UNSAFE_FILENAME = re.compile(r"[^\w.-]+", re.UNICODE)


class MediaStoreError(Exception):
    """Base class for media-domain failures."""


class MediaValidationError(MediaStoreError):
    """A media request is unsafe or unsupported."""


class MediaNotFoundError(MediaStoreError):
    """The opaque media reference is not visible to this owner."""


class MediaConflictError(MediaStoreError):
    """The media cannot be changed while another record references it."""


class MediaWriteError(MediaStoreError):
    """Managed media could not be persisted."""


@dataclass(frozen=True)
class MediaRecord:
    """Metadata returned after resolving an owner-scoped opaque reference."""

    id: str
    name: str
    kind: str
    size: int
    registered: bool
    path: Path


@dataclass(frozen=True)
class MaterializedMedia:
    """A safe job-local copy for the legacy path-based pipeline."""

    id: str
    basename: str
    stem: str
    path: Path


Probe = Callable[[Path, str], bool]


class FileMediaStore:
    """Persist uploads and explicitly registered results below one root."""

    def __init__(
        self,
        root: str | Path,
        *,
        max_upload_bytes: int = DEFAULT_MAX_UPLOAD_BYTES,
        probe: Probe | None = None,
    ) -> None:
        if max_upload_bytes <= 0:
            raise ValueError("max_upload_bytes must be positive")
        self._root = Path(root).resolve()
        self._max_upload_bytes = int(max_upload_bytes)
        self._probe = probe or self._ffprobe

    def save(
        self,
        owner_id: str,
        filename: str | None = None,
        stream: BinaryIO | None = None,
        *,
        source: object | None = None,
        name: str | None = None,
        kind: str | None = None,
        chunk_size: int = 1024 * 1024,
    ) -> MediaRecord:
        owner = self._validate_owner(owner_id)
        if chunk_size <= 0:
            raise MediaValidationError("Upload chunk size must be positive.")
        safe_name = self._sanitize_filename(filename or name or "")
        suffix = Path(safe_name).suffix
        media_kind = self._upload_kind(suffix)
        record_kind = str(kind or media_kind)
        upload_stream = stream if stream is not None else source
        opened_stream: BinaryIO | None = None
        if isinstance(upload_stream, (str, os.PathLike)):
            try:
                opened_stream = Path(upload_stream).open("rb")
            except OSError as exc:
                raise MediaNotFoundError("Source media was not found.") from exc
            upload_stream = opened_stream
        if upload_stream is None or not hasattr(upload_stream, "read"):
            if opened_stream is not None:
                opened_stream.close()
            raise MediaValidationError("A readable upload stream is required.")
        media_id = str(uuid4())
        media_dir = self._root / "media" / owner / media_id
        media_path = media_dir / safe_name
        metadata_path = media_dir / "metadata.json"
        size = 0
        try:
            media_dir.mkdir(parents=True, exist_ok=False)
            with media_path.open("xb") as target:
                while True:
                    chunk = upload_stream.read(chunk_size)
                    if not chunk:
                        break
                    if not isinstance(chunk, (bytes, bytearray, memoryview)):
                        raise MediaValidationError("Upload stream must yield bytes.")
                    size += len(chunk)
                    if size > self._max_upload_bytes:
                        raise MediaValidationError("Upload exceeds the configured maximum size.")
                    target.write(chunk)
            if not self._probe(media_path, media_kind):
                raise MediaValidationError("Uploaded file is not valid media.")
            metadata = self._metadata(
                media_id=media_id,
                name=safe_name,
                kind=record_kind,
                size=size,
                stored_name=media_path.name,
                registered=False,
            )
            self._atomic_json(metadata_path, metadata)
            return self._record(metadata, media_path)
        except MediaStoreError:
            self._remove_tree(media_dir)
            raise
        except (OSError, ValueError) as exc:
            self._remove_tree(media_dir)
            raise MediaWriteError(f"Could not save media: {exc}") from exc
        finally:
            if opened_stream is not None:
                opened_stream.close()

    def register(
        self,
        owner_id: str,
        source_path: str | os.PathLike[str] | None = None,
        *,
        path: str | os.PathLike[str] | None = None,
        name: str | None = None,
        kind: str = "result",
    ) -> MediaRecord:
        owner = self._validate_owner(owner_id)
        source_value = source_path if source_path is not None else path
        if source_value is None:
            raise MediaValidationError("A file path is required for registration.")
        source = Path(source_value).resolve()
        if not source.is_file():
            raise MediaNotFoundError("File to register was not found.")
        safe_name = self._sanitize_filename(name or source.name)
        media_id = str(uuid4())
        media_dir = self._root / "registered" / owner / media_id
        destination = media_dir / safe_name
        try:
            media_dir.mkdir(parents=True, exist_ok=False)
            with source.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            size = destination.stat().st_size
            metadata = self._metadata(
                media_id=media_id,
                name=safe_name,
                kind=str(kind or "result"),
                size=size,
                stored_name=safe_name,
                registered=True,
            )
            self._atomic_json(media_dir / "metadata.json", metadata)
            return self._record(metadata, destination)
        except MediaStoreError:
            self._remove_tree(media_dir)
            raise
        except OSError as exc:
            self._remove_tree(media_dir)
            raise MediaWriteError(f"Could not register file: {exc}") from exc

    def get(self, owner_id: str, media_id: str) -> MediaRecord:
        owner = self._validate_owner(owner_id)
        opaque_id = self._validate_uuid(media_id, "media")
        for category in ("media", "registered"):
            media_dir = self._root / category / owner / opaque_id
            metadata_path = media_dir / "metadata.json"
            if not metadata_path.is_file():
                continue
            try:
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                if metadata.get("id") != opaque_id:
                    break
                stored_name = str(metadata["stored_name"])
                path = self._contained(media_dir / stored_name, media_dir)
                if not path.is_file():
                    break
                return self._record(metadata, path)
            except (OSError, ValueError, KeyError, TypeError, json.JSONDecodeError):
                break
        raise MediaNotFoundError("Media was not found.")

    def delete(self, owner_id: str, media_id: str) -> None:
        record = self.get(owner_id, media_id)
        metadata_path = record.path.parent / "metadata.json"
        try:
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaWriteError(f"Could not read media metadata: {exc}") from exc
        if metadata.get("references"):
            raise MediaConflictError("Media is referenced by a job and cannot be deleted.")
        try:
            shutil.rmtree(record.path.parent)
        except OSError as exc:
            raise MediaWriteError(f"Could not delete media: {exc}") from exc

    def materialize_for_job(
        self,
        owner_id: str,
        media_id: str,
        job_id: str,
    ) -> MaterializedMedia:
        owner = self._validate_owner(owner_id)
        opaque_job_id = self._validate_uuid(job_id, "job")
        record = self.get(owner, media_id)
        destination_dir = self._root / "jobs" / owner / opaque_job_id / "media" / record.id
        destination = destination_dir / record.name
        metadata_path = record.path.parent / "metadata.json"
        try:
            destination_dir.mkdir(parents=True, exist_ok=True)
            destination.unlink(missing_ok=True)
            with record.path.open("rb") as reader, destination.open("xb") as writer:
                shutil.copyfileobj(reader, writer, length=1024 * 1024)
            metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
            references = list(metadata.get("references") or [])
            if opaque_job_id not in references:
                references.append(opaque_job_id)
            metadata["references"] = references
            self._atomic_json(metadata_path, metadata)
        except MediaStoreError:
            destination.unlink(missing_ok=True)
            raise
        except (OSError, json.JSONDecodeError) as exc:
            destination.unlink(missing_ok=True)
            raise MediaWriteError(f"Could not materialize media: {exc}") from exc
        return MaterializedMedia(
            id=record.id,
            basename=record.name,
            stem=Path(record.name).stem,
            path=destination,
        )

    def release_job_materialization(
        self,
        owner_id: str,
        job_id: str,
        media_ids: Iterable[str],
    ) -> None:
        """Remove job-local copies and release their upload references."""
        owner = self._validate_owner(owner_id)
        opaque_job_id = self._validate_uuid(job_id, "job")
        records = [self.get(owner, media_id) for media_id in media_ids]
        try:
            for record in records:
                metadata_path = record.path.parent / "metadata.json"
                metadata = json.loads(metadata_path.read_text(encoding="utf-8"))
                references = [
                    reference
                    for reference in metadata.get("references") or []
                    if reference != opaque_job_id
                ]
                if references != list(metadata.get("references") or []):
                    metadata["references"] = references
                    self._atomic_json(metadata_path, metadata)
            self._remove_tree(self._root / "jobs" / owner / opaque_job_id / "media")
        except MediaStoreError:
            raise
        except (OSError, json.JSONDecodeError) as exc:
            raise MediaWriteError(f"Could not release job media: {exc}") from exc

    @staticmethod
    def _metadata(
        *,
        media_id: str,
        name: str,
        kind: str,
        size: int,
        stored_name: str,
        registered: bool,
    ) -> dict[str, object]:
        return {
            "id": media_id,
            "name": name,
            "kind": kind,
            "size": size,
            "stored_name": stored_name,
            "registered": registered,
            "references": [],
        }

    @staticmethod
    def _record(metadata: dict[str, object], path: Path) -> MediaRecord:
        return MediaRecord(
            id=str(metadata["id"]),
            name=str(metadata["name"]),
            kind=str(metadata["kind"]),
            size=int(metadata["size"]),
            registered=bool(metadata["registered"]),
            path=path,
        )

    @staticmethod
    def _upload_kind(suffix: str) -> str:
        normalized = suffix.lower()
        if normalized in VIDEO_EXTENSIONS:
            return "video"
        if normalized in AUDIO_EXTENSIONS:
            return "audio"
        raise MediaValidationError(f"Unsupported media extension: {suffix or '<missing>'}.")

    @staticmethod
    def _sanitize_filename(filename: str) -> str:
        raw_name = str(filename or "").replace("\\", "/").rsplit("/", 1)[-1].strip()
        if not raw_name or raw_name in {".", ".."}:
            raise MediaValidationError("A valid filename is required.")
        path = Path(raw_name)
        stem = _UNSAFE_FILENAME.sub("_", path.stem).strip("._-") or "media"
        suffix = _UNSAFE_FILENAME.sub("", path.suffix)
        return f"{stem}{suffix}"

    @staticmethod
    def _validate_owner(owner_id: str) -> str:
        owner = str(owner_id or "")
        if not _SAFE_OWNER.fullmatch(owner) or ".." in owner:
            raise MediaValidationError("Invalid owner reference.")
        return owner

    @staticmethod
    def _validate_uuid(value: str, label: str) -> str:
        candidate = str(value or "")
        try:
            parsed = UUID(candidate)
        except (ValueError, AttributeError) as exc:
            raise MediaValidationError(f"Invalid {label} reference.") from exc
        if str(parsed) != candidate.lower():
            raise MediaValidationError(f"Invalid {label} reference.")
        return str(parsed)

    @staticmethod
    def _contained(path: Path, parent: Path) -> Path:
        resolved = path.resolve()
        try:
            resolved.relative_to(parent.resolve())
        except ValueError as exc:
            raise MediaValidationError("Unsafe managed path.") from exc
        return resolved

    @staticmethod
    def _atomic_json(path: Path, value: dict[str, object]) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with NamedTemporaryFile(
                mode="w",
                encoding="utf-8",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                json.dump(value, temporary, ensure_ascii=False, separators=(",", ":"))
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        except OSError as exc:
            raise MediaWriteError(f"Could not write media metadata: {exc}") from exc
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)

    @staticmethod
    def _remove_tree(path: Path) -> None:
        try:
            shutil.rmtree(path, ignore_errors=True)
        except OSError:
            pass

    @staticmethod
    def _ffprobe(path: Path, _kind: str) -> bool:
        try:
            completed = subprocess.run(
                ["ffprobe", "-v", "error", "-show_entries", "format=format_name", str(path)],
                stdout=subprocess.DEVNULL,
                stderr=subprocess.DEVNULL,
                check=False,
            )
        except OSError:
            return False
        return completed.returncode == 0
