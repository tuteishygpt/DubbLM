"""Single-worker FIFO execution for validated dubbing jobs."""

from __future__ import annotations

import queue
import threading
from pathlib import Path
from typing import Any, Callable, Iterable

from dubbing.core.config import DubbingConfig
from dubbing.core.runner import (
    DubbingJobResult,
    config_from_snapshot,
    run_validated_dubbing_job_streaming,
)


ValidatedRunner = Callable[
    [DubbingConfig], Iterable[tuple[str, str, DubbingJobResult | None]]
]
_STOP = object()


class InProcessJobQueue:
    """Run persisted jobs in FIFO order on exactly one worker thread."""

    def __init__(
        self,
        repository: object,
        media_store: object,
        *,
        owner_id: str,
        runner: ValidatedRunner = run_validated_dubbing_job_streaming,
    ) -> None:
        self._repository = repository
        self._media_store = media_store
        self._owner_id = owner_id
        self._runner = runner
        self._items: queue.Queue[object] = queue.Queue()
        self._lifecycle_lock = threading.Lock()
        self._thread: threading.Thread | None = None
        self._accepting = False

    def start(self) -> None:
        with self._lifecycle_lock:
            if self._thread is not None and self._thread.is_alive():
                return
            self._accepting = True
            self._thread = threading.Thread(
                target=self._work,
                name="dubblm-job-worker",
                daemon=True,
            )
            self._thread.start()

    def stop(self) -> None:
        with self._lifecycle_lock:
            thread = self._thread
            if thread is None:
                return
            self._accepting = False
            self._items.put(_STOP)
        if thread is not threading.current_thread():
            thread.join()
        with self._lifecycle_lock:
            if self._thread is thread:
                self._thread = None

    def enqueue(self, job_id: str) -> None:
        with self._lifecycle_lock:
            if not self._accepting or self._thread is None or not self._thread.is_alive():
                raise RuntimeError("Job queue is not running.")
            self._items.put(str(job_id))

    def _work(self) -> None:
        while True:
            item = self._items.get()
            try:
                if item is _STOP:
                    return
                self._run_job(str(item))
            finally:
                self._items.task_done()

    def _run_job(self, job_id: str) -> None:
        emitted_log_length = 0
        pending_log = ""
        try:
            job = self._repository.get(self._owner_id, job_id)
            self._repository.update(
                self._owner_id,
                job_id,
                status="running",
                state={"status": "running"},
                error=None,
            )
            self._repository.append_event(
                self._owner_id, job_id, "state", {"status": "running"}
            )
            config = config_from_snapshot(job.config)
            final_result: DubbingJobResult | None = None
            for _status, logs, result in self._runner(config):
                emitted_log_length, pending_log = self._append_new_logs(
                    job_id, str(logs or ""), emitted_log_length, pending_log
                )
                if result is not None:
                    final_result = result
            if pending_log:
                self._repository.append_event(
                    self._owner_id, job_id, "log", {"message": pending_log}
                )
            if final_result is None:
                raise RuntimeError("Dubbing runner ended without a result.")
            if final_result.status.startswith("Failed"):
                message = final_result.status.partition(":")[2].strip() or final_result.status
                raise RuntimeError(message)

            files = self._register_files(job.config, final_result)
            for file_data in files:
                self._repository.append_event(
                    self._owner_id, job_id, "file", file_data
                )
            self._repository.update(
                self._owner_id,
                job_id,
                status="succeeded",
                state={"status": "succeeded", "message": final_result.status},
                files=files,
            )
            self._repository.append_event(
                self._owner_id,
                job_id,
                "state",
                {"status": "succeeded", "message": final_result.status},
            )
        except Exception as exc:
            error = {"code": "pipeline_failed", "message": str(exc)}
            try:
                self._repository.append_event(self._owner_id, job_id, "error", error)
                self._repository.update(
                    self._owner_id,
                    job_id,
                    status="failed",
                    state={"status": "failed"},
                    error=error,
                )
                self._repository.append_event(
                    self._owner_id, job_id, "state", {"status": "failed"}
                )
            except Exception:
                # A repository failure cannot be recovered by this in-process
                # worker; keep it alive so later queued jobs still run.
                pass

    def _append_new_logs(
        self, job_id: str, logs: str, seen: int, pending: str
    ) -> tuple[int, str]:
        delta = logs[seen:] if len(logs) >= seen else logs
        complete = pending + delta
        lines = complete.splitlines(keepends=True)
        pending = ""
        if lines and not lines[-1].endswith(("\n", "\r")):
            pending = lines.pop()
        for line in lines:
            self._repository.append_event(
                self._owner_id, job_id, "log", {"message": line.rstrip("\r\n")}
            )
        return len(logs), pending

    def _register_files(
        self, config_snapshot: dict[str, Any], result: DubbingJobResult
    ) -> list[dict[str, Any]]:
        candidates: list[tuple[Path, str]] = []
        if result.output_file:
            candidates.append((Path(result.output_file), "result"))
        if result.report_file:
            candidates.append((Path(result.report_file), "report"))

        explicit = {path.resolve() for path, _kind in candidates if path.is_file()}
        artifacts_dir = config_snapshot.get("artifacts_dir")
        if artifacts_dir:
            root = Path(str(artifacts_dir))
            if root.is_dir():
                for artifact in sorted(path for path in root.rglob("*") if path.is_file()):
                    if artifact.resolve() not in explicit:
                        candidates.append((artifact, "artifact"))

        files: list[dict[str, Any]] = []
        registered_paths: set[Path] = set()
        for path, kind in candidates:
            if not path.is_file() or path.resolve() in registered_paths:
                continue
            record = self._media_store.register(
                self._owner_id, path=path, name=path.name, kind=kind
            )
            registered_paths.add(path.resolve())
            files.append(
                {
                    "id": str(record.id),
                    "name": str(record.name),
                    "kind": str(record.kind),
                    "size": int(record.size),
                }
            )
        return files
