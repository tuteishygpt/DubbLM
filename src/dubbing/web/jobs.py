"""Atomic owner-scoped job metadata and retained JSONL events."""

from __future__ import annotations

import json
import os
import re
import threading
from contextlib import contextmanager
from dataclasses import asdict, dataclass, is_dataclass
from datetime import datetime, timezone
from pathlib import Path
from tempfile import NamedTemporaryFile
from typing import Any, Callable, Mapping
from uuid import UUID, uuid4

from dubbing.core.runner import build_config_from_overrides


DEFAULT_EVENT_RETENTION_BYTES = 10 * 1024**2
JOB_STATUSES = frozenset({"queued", "running", "succeeded", "failed"})
TERMINAL_JOB_STATUSES = frozenset({"succeeded", "failed"})
EVENT_TYPES = frozenset({"snapshot", "state", "log", "file", "error"})
_SAFE_OWNER = re.compile(r"[A-Za-z0-9][A-Za-z0-9_.-]{0,127}")


class JobRepositoryError(Exception):
    """Base class for job-domain failures."""


class JobValidationError(JobRepositoryError):
    """A job operation contains invalid or unsafe data."""


class JobNotFoundError(JobRepositoryError):
    """The job is absent or not visible to the supplied owner."""


class JobWriteError(JobRepositoryError):
    """A persisted job or event log could not be written safely."""


@dataclass(frozen=True)
class Job:
    """One immutable configuration snapshot plus mutable execution metadata."""

    id: str
    owner_id: str
    config: dict[str, Any]
    created_at: str
    updated_at: str
    started_at: str | None
    finished_at: str | None
    status: str
    state: dict[str, Any]
    error: dict[str, Any] | None
    files: list[dict[str, Any]]
    last_event_id: int


@dataclass(frozen=True)
class JobPage:
    """A bounded cursor page of jobs."""

    items: list[Job]
    next_cursor: str | None


@dataclass(frozen=True)
class JobEvent:
    """A monotonic event stored as one JSONL record."""

    id: int
    job_id: str
    type: str
    data: dict[str, Any]
    timestamp: str


Clock = Callable[[], datetime]


class JobService:
    """Validate and freeze owner-scoped uploads before accepting a job."""

    def __init__(self, repository: object, media_store: object, queue: object | None = None) -> None:
        self._repository = repository
        self._media_store = media_store
        self._queue = queue

    def submit(
        self,
        owner_id: str,
        input_upload_id: str,
        *,
        isolated_tracks: Mapping[str, str] | None = None,
        overrides: Mapping[str, Any] | None = None,
    ) -> Job:
        job_id = str(uuid4())
        values = dict(overrides or {})
        # Resolve every reference first so a cross-owner track cannot leave a
        # partially materialized submission behind.
        self._media_store.get(owner_id, input_upload_id)
        for upload_id in (isolated_tracks or {}).values():
            self._media_store.get(owner_id, str(upload_id))
        input_media = self._media_store.materialize_for_job(
            owner_id, input_upload_id, job_id
        )
        values["input"] = str(input_media.path)

        resolved_tracks: dict[str, str] = {}
        for speaker, upload_id in (isolated_tracks or {}).items():
            label = str(speaker).strip()
            if not label:
                raise JobValidationError("Isolated-track speaker labels cannot be empty.")
            materialized = self._media_store.materialize_for_job(
                owner_id, str(upload_id), job_id
            )
            resolved_tracks[label] = str(materialized.path)
        if resolved_tracks:
            values["isolated_tracks"] = resolved_tracks

        try:
            config = build_config_from_overrides(values)
        except SystemExit as exc:
            raise JobValidationError("Dubbing configuration is invalid.") from exc
        except (TypeError, ValueError) as exc:
            raise JobValidationError(f"Dubbing configuration is invalid: {exc}") from exc

        config_snapshot = json.loads(
            json.dumps(
                config.to_dict(),
                ensure_ascii=False,
                default=self._config_json_default,
            )
        )
        job = self._repository.create(
            owner_id,
            config_snapshot,
            job_id=job_id,
            state={"status": "queued"},
        )
        if self._queue is not None:
            self._queue.enqueue(job.id)
        return job

    @staticmethod
    def _config_json_default(value: object) -> object:
        if is_dataclass(value) and not isinstance(value, type):
            return asdict(value)
        if isinstance(value, Path):
            return str(value)
        raise TypeError(f"Unsupported configuration value: {type(value).__name__}")


class FileJobRepository:
    """Store job state beneath ``jobs/<owner>/<job-id>``."""

    _root_locks: dict[str, threading.RLock] = {}
    _root_locks_guard = threading.Lock()

    def __init__(
        self,
        root: str | Path,
        *,
        event_retention_bytes: int = DEFAULT_EVENT_RETENTION_BYTES,
        clock: Clock | None = None,
    ) -> None:
        if event_retention_bytes <= 0:
            raise ValueError("event_retention_bytes must be positive")
        self._root = Path(root).resolve()
        self._jobs_root = self._root / "jobs"
        self._event_retention_bytes = int(event_retention_bytes)
        self._clock = clock or (lambda: datetime.now(timezone.utc))
        key = str(self._root)
        with self._root_locks_guard:
            self._lock = self._root_locks.setdefault(key, threading.RLock())
        self._recover_interrupted_jobs()

    def create(
        self,
        owner_id: str,
        config: Mapping[str, Any],
        *,
        job_id: str | None = None,
        state: Mapping[str, Any] | None = None,
        files: list[Mapping[str, Any]] | None = None,
    ) -> Job:
        owner = self._validate_owner(owner_id)
        opaque_id = self._validate_uuid(job_id, "job") if job_id else str(uuid4())
        config_snapshot = self._mapping_copy(config, "config")
        state_snapshot = self._mapping_copy(state or {}, "state")
        file_snapshot = self._files_copy(files or [])
        now = self._now()
        raw: dict[str, Any] = {
            "id": opaque_id,
            "owner_id": owner,
            "config": config_snapshot,
            "created_at": now,
            "updated_at": now,
            "started_at": None,
            "finished_at": None,
            "status": "queued",
            "state": state_snapshot,
            "error": None,
            "files": file_snapshot,
            "last_event_id": 0,
        }
        job_path = self._job_path(owner, opaque_id)
        with self._lock:
            with self._job_file_lock(owner, opaque_id):
                if job_path.exists():
                    raise JobValidationError("Job ID already exists.")
                self._atomic_json(job_path, raw)
        return self._decode_job(raw)

    def get(self, owner_id: str, job_id: str) -> Job:
        owner = self._validate_owner(owner_id)
        opaque_id = self._validate_uuid(job_id, "job")
        with self._lock:
            return self._read_job(owner, opaque_id)

    def list(
        self,
        owner_id: str,
        *,
        limit: int = 50,
        cursor: str | None = None,
    ) -> JobPage:
        owner = self._validate_owner(owner_id)
        if not isinstance(limit, int) or isinstance(limit, bool) or not 1 <= limit <= 100:
            raise JobValidationError("Job page limit must be between 1 and 100.")
        opaque_cursor = self._validate_uuid(cursor, "cursor") if cursor is not None else None
        owner_root = self._jobs_root / owner
        with self._lock:
            jobs: list[Job] = []
            if owner_root.is_dir():
                for child in owner_root.iterdir():
                    if not child.is_dir():
                        continue
                    try:
                        job_id = self._validate_uuid(child.name, "job")
                        jobs.append(self._read_job(owner, job_id))
                    except (JobValidationError, JobNotFoundError):
                        continue
            jobs.sort(key=lambda job: (job.created_at, job.id), reverse=True)
            start = 0
            if opaque_cursor is not None:
                positions = [index for index, job in enumerate(jobs) if job.id == opaque_cursor]
                if not positions:
                    raise JobValidationError("Unknown pagination cursor.")
                start = positions[0] + 1
            selected = jobs[start : start + limit]
            has_more = start + len(selected) < len(jobs)
            next_cursor = selected[-1].id if selected and has_more else None
            return JobPage(items=selected, next_cursor=next_cursor)

    def update(self, owner_id: str, job_id: str, **changes: object) -> Job:
        owner = self._validate_owner(owner_id)
        opaque_id = self._validate_uuid(job_id, "job")
        allowed = {"status", "state", "error", "files"}
        unknown = set(changes) - allowed
        if unknown:
            raise JobValidationError(
                f"Cannot update immutable or unknown job fields: {', '.join(sorted(unknown))}."
            )
        with self._lock:
            with self._job_file_lock(owner, opaque_id):
                current = self._read_raw_job(owner, opaque_id)
                now = self._now()
                if "status" in changes:
                    status = str(changes["status"])
                    if status not in JOB_STATUSES:
                        raise JobValidationError(f"Invalid job status: {status}.")
                    current["status"] = status
                    if status == "running" and current.get("started_at") is None:
                        current["started_at"] = now
                    if status in TERMINAL_JOB_STATUSES and current.get("finished_at") is None:
                        current["finished_at"] = now
                if "state" in changes:
                    current["state"] = self._mapping_copy(changes["state"], "state")
                if "error" in changes:
                    error = changes["error"]
                    current["error"] = None if error is None else self._mapping_copy(error, "error")
                if "files" in changes:
                    current["files"] = self._files_copy(changes["files"])
                current["updated_at"] = now
                self._atomic_json(self._job_path(owner, opaque_id), current)
                return self._decode_job(current)

    def append_event(
        self,
        owner_id: str,
        job_id: str,
        event_type: str,
        data: Mapping[str, Any],
    ) -> JobEvent:
        owner = self._validate_owner(owner_id)
        opaque_id = self._validate_uuid(job_id, "job")
        normalized_type = str(event_type or "").strip()
        if normalized_type not in EVENT_TYPES:
            raise JobValidationError("Invalid SSE event type.")
        event_data = self._mapping_copy(data, "event data")
        with self._lock:
            with self._job_file_lock(owner, opaque_id):
                current = self._read_raw_job(owner, opaque_id)
                event_path = self._event_path(owner, opaque_id)
                try:
                    existing = event_path.read_bytes() if event_path.is_file() else b""
                    retained_ids = [
                        int(json.loads(line)["id"])
                        for line in existing.splitlines()
                        if line
                    ]
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                    raise JobWriteError(f"Could not read job events: {exc}") from exc
                event = JobEvent(
                    id=max([int(current["last_event_id"]), *retained_ids]) + 1,
                    job_id=opaque_id,
                    type=normalized_type,
                    data=event_data,
                    timestamp=self._now(),
                )
                encoded = self._event_bytes(event)
                if len(encoded) > self._event_retention_bytes:
                    raise JobValidationError("Event exceeds the configured retention limit.")
                lines = [line + b"\n" for line in existing.splitlines() if line]
                lines.append(encoded)
                total = sum(map(len, lines))
                while lines and total > self._event_retention_bytes:
                    total -= len(lines.pop(0))
                self._atomic_bytes(event_path, b"".join(lines))
                current["last_event_id"] = event.id
                current["updated_at"] = event.timestamp
                self._atomic_json(self._job_path(owner, opaque_id), current)
                return event

    def events(self, owner_id: str, job_id: str, *, after_id: int = 0) -> list[JobEvent]:
        owner = self._validate_owner(owner_id)
        opaque_id = self._validate_uuid(job_id, "job")
        if not isinstance(after_id, int) or isinstance(after_id, bool) or after_id < 0:
            raise JobValidationError("after_id must be a non-negative integer.")
        with self._lock:
            self._read_job(owner, opaque_id)
            path = self._event_path(owner, opaque_id)
            if not path.is_file():
                return []
            try:
                events = [
                    self._decode_event(json.loads(line))
                    for line in path.read_text(encoding="utf-8").splitlines()
                    if line
                ]
            except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
                raise JobWriteError(f"Could not read job events: {exc}") from exc
            return [event for event in events if event.id > after_id]

    def _recover_interrupted_jobs(self) -> None:
        if not self._jobs_root.is_dir():
            return
        with self._lock:
            for job_path in self._jobs_root.glob("*/*/job.json"):
                try:
                    owner, job_id = job_path.parent.parent.name, job_path.parent.name
                    with self._job_file_lock(owner, job_id):
                        raw = json.loads(job_path.read_text(encoding="utf-8"))
                        if raw.get("status") not in {"queued", "running"}:
                            continue
                        raw["status"] = "failed"
                        raw["error"] = {
                            "code": "server_restarted",
                            "message": "The server restarted before the job completed.",
                        }
                        raw["finished_at"] = self._now()
                        raw["updated_at"] = raw["finished_at"]
                        self._atomic_json(job_path, raw)
                except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError):
                    continue

    def _read_job(self, owner: str, job_id: str) -> Job:
        return self._decode_job(self._read_raw_job(owner, job_id))

    def _read_raw_job(self, owner: str, job_id: str) -> dict[str, Any]:
        path = self._job_path(owner, job_id)
        try:
            raw = json.loads(path.read_text(encoding="utf-8"))
            if raw.get("owner_id") != owner or raw.get("id") != job_id:
                raise JobNotFoundError("Job was not found.")
            self._decode_job(raw)
            return raw
        except FileNotFoundError as exc:
            raise JobNotFoundError("Job was not found.") from exc
        except JobNotFoundError:
            raise
        except (OSError, ValueError, TypeError, KeyError, json.JSONDecodeError) as exc:
            raise JobNotFoundError("Job was not found.") from exc

    @staticmethod
    def _decode_job(raw: Mapping[str, Any]) -> Job:
        status = str(raw["status"])
        if status not in JOB_STATUSES:
            raise ValueError("invalid persisted job status")
        return Job(
            id=str(raw["id"]),
            owner_id=str(raw["owner_id"]),
            config=FileJobRepository._mapping_copy(raw["config"], "config"),
            created_at=str(raw["created_at"]),
            updated_at=str(raw["updated_at"]),
            started_at=None if raw.get("started_at") is None else str(raw["started_at"]),
            finished_at=None if raw.get("finished_at") is None else str(raw["finished_at"]),
            status=status,
            state=FileJobRepository._mapping_copy(raw["state"], "state"),
            error=None if raw["error"] is None else FileJobRepository._mapping_copy(raw["error"], "error"),
            files=FileJobRepository._files_copy(raw["files"]),
            last_event_id=int(raw["last_event_id"]),
        )

    @staticmethod
    def _decode_event(raw: Mapping[str, Any]) -> JobEvent:
        return JobEvent(
            id=int(raw["id"]),
            job_id=str(raw["job_id"]),
            type=str(raw["type"]),
            data=FileJobRepository._mapping_copy(raw["data"], "event data"),
            timestamp=str(raw["timestamp"]),
        )

    @staticmethod
    def _event_bytes(event: JobEvent) -> bytes:
        return (
            json.dumps(
                {
                    "id": event.id,
                    "job_id": event.job_id,
                    "type": event.type,
                    "data": event.data,
                    "timestamp": event.timestamp,
                },
                ensure_ascii=False,
                separators=(",", ":"),
            )
            + "\n"
        ).encode("utf-8")

    @staticmethod
    def _mapping_copy(value: object, label: str) -> dict[str, Any]:
        if not isinstance(value, Mapping):
            raise JobValidationError(f"Job {label} must be a mapping.")
        try:
            copied = json.loads(json.dumps(dict(value), ensure_ascii=False))
        except (TypeError, ValueError) as exc:
            raise JobValidationError(f"Job {label} must be JSON serializable.") from exc
        if not isinstance(copied, dict):
            raise JobValidationError(f"Job {label} must be a mapping.")
        return copied

    @staticmethod
    def _files_copy(value: object) -> list[dict[str, Any]]:
        if not isinstance(value, list):
            raise JobValidationError("Job files must be a list.")
        return [FileJobRepository._mapping_copy(item, "file") for item in value]

    @staticmethod
    def _validate_owner(owner_id: str) -> str:
        owner = str(owner_id or "")
        if not _SAFE_OWNER.fullmatch(owner) or ".." in owner:
            raise JobValidationError("Invalid owner reference.")
        return owner

    @staticmethod
    def _validate_uuid(value: str | None, label: str) -> str:
        candidate = str(value or "")
        try:
            parsed = UUID(candidate)
        except (ValueError, AttributeError) as exc:
            raise JobValidationError(f"Invalid {label} reference.") from exc
        if str(parsed) != candidate.lower():
            raise JobValidationError(f"Invalid {label} reference.")
        return str(parsed)

    def _job_path(self, owner: str, job_id: str) -> Path:
        return self._jobs_root / owner / job_id / "job.json"

    def _event_path(self, owner: str, job_id: str) -> Path:
        return self._jobs_root / owner / job_id / "events.jsonl"

    @contextmanager
    def _job_file_lock(self, owner: str, job_id: str):
        """Hold a portable exclusive lock for one persisted job."""
        lock_path = self._jobs_root / owner / job_id / ".lock"
        lock_path.parent.mkdir(parents=True, exist_ok=True)
        with lock_path.open("a+b") as lock_file:
            lock_file.seek(0, os.SEEK_END)
            if lock_file.tell() == 0:
                lock_file.write(b"\0")
                lock_file.flush()
            lock_file.seek(0)
            if os.name == "nt":
                import msvcrt

                msvcrt.locking(lock_file.fileno(), msvcrt.LK_LOCK, 1)
                try:
                    yield
                finally:
                    lock_file.seek(0)
                    msvcrt.locking(lock_file.fileno(), msvcrt.LK_UNLCK, 1)
            else:
                import fcntl

                fcntl.flock(lock_file.fileno(), fcntl.LOCK_EX)
                try:
                    yield
                finally:
                    fcntl.flock(lock_file.fileno(), fcntl.LOCK_UN)

    def _now(self) -> str:
        value = self._clock()
        if value.tzinfo is None:
            value = value.replace(tzinfo=timezone.utc)
        return value.astimezone(timezone.utc).isoformat()

    @staticmethod
    def _atomic_json(path: Path, value: Mapping[str, Any]) -> None:
        content = json.dumps(value, ensure_ascii=False, separators=(",", ":")).encode("utf-8")
        FileJobRepository._atomic_bytes(path, content)

    @staticmethod
    def _atomic_bytes(path: Path, content: bytes) -> None:
        path.parent.mkdir(parents=True, exist_ok=True)
        temporary_name: str | None = None
        try:
            with NamedTemporaryFile(
                mode="wb",
                dir=path.parent,
                prefix=f".{path.name}.",
                suffix=".tmp",
                delete=False,
            ) as temporary:
                temporary_name = temporary.name
                temporary.write(content)
                temporary.flush()
                os.fsync(temporary.fileno())
            os.replace(temporary_name, path)
        except OSError as exc:
            raise JobWriteError(f"Could not persist job data: {exc}") from exc
        finally:
            if temporary_name:
                Path(temporary_name).unlink(missing_ok=True)
