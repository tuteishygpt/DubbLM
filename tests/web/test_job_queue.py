"""Validated job submission and in-process FIFO execution."""

from __future__ import annotations

import threading
import time
from pathlib import Path

import pytest

import dubbing.core.config as config_module
from dubbing.core.runner import DubbingJobResult
from dubbing.web.jobs import FileJobRepository, JobService, JobValidationError
from dubbing.web.queue import InProcessJobQueue
from dubbing.web.storage import FileMediaStore, MediaNotFoundError


def _store(tmp_path: Path) -> FileMediaStore:
    return FileMediaStore(tmp_path, probe=lambda _path, _kind: True)


def _upload(store: FileMediaStore, owner: str, name: str, content: bytes = b"media"):
    return store.save(owner, name=name, source=__import__("io").BytesIO(content))


def _wait_for(repository, owner: str, job_id: str, status: str, timeout: float = 3.0):
    deadline = time.monotonic() + timeout
    while time.monotonic() < deadline:
        job = repository.get(owner, job_id)
        if job.status == status:
            return job
        time.sleep(0.01)
    raise AssertionError(f"job {job_id} did not reach {status}")


def test_submission_materializes_owner_uploads_and_freezes_normalized_config(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config_module, "DEFAULT_PROJECTS_ROOT", tmp_path / "prj")
    store = _store(tmp_path / "data")
    repository = FileJobRepository(tmp_path / "data")
    video = _upload(store, "alice", "My unsafe clip (final).mp4")
    track = _upload(store, "alice", "Speaker one!.wav")
    config_path = tmp_path / "settings.yml"
    config_path.write_text("source_language: en\ntarget_language: fr\n", encoding="utf-8")

    job = JobService(repository, store).submit(
        owner_id="alice",
        input_upload_id=video.id,
        isolated_tracks={"SPEAKER_00": track.id},
        overrides={"config": str(config_path), "target_language": "es"},
    )

    assert Path(job.config["input"]).name == "My_unsafe_clip_final.mp4"
    assert Path(job.config["input"]).stem == "My_unsafe_clip_final"
    assert Path(job.config["input"]).is_file()
    assert Path(job.config["isolated_tracks"]["SPEAKER_00"]).name == "Speaker_one.wav"
    assert Path(job.config["isolated_tracks"]["SPEAKER_00"]).is_file()
    assert job.config["source_language"] == "en"
    assert job.config["target_language"] == "es"
    assert "timing_max_speed" in job.config
    config_path.write_text("source_language: de\ntarget_language: de\n", encoding="utf-8")
    assert repository.get("alice", job.id).config["source_language"] == "en"


def test_submission_masks_cross_owner_upload_and_invalid_jobs_are_not_created(
    tmp_path, monkeypatch
):
    monkeypatch.setattr(config_module, "DEFAULT_PROJECTS_ROOT", tmp_path / "prj")
    store = _store(tmp_path / "data")
    repository = FileJobRepository(tmp_path / "data")
    video = _upload(store, "bob", "private.mp4")
    service = JobService(repository, store)

    with pytest.raises(MediaNotFoundError):
        service.submit(
            owner_id="alice",
            input_upload_id=video.id,
            overrides={"config": "", "source_language": "en", "target_language": "es"},
        )
    assert repository.list("alice").items == []

    alice_video = _upload(store, "alice", "clip.mp4")
    with pytest.raises(JobValidationError):
        service.submit(
            owner_id="alice",
            input_upload_id=alice_video.id,
            overrides={"config": "", "source_language": "en"},
        )
    assert repository.list("alice").items == []


def test_queue_runs_fifo_with_one_worker_and_streams_ordered_logs(tmp_path):
    owner = "alice"
    store = _store(tmp_path)
    repository = FileJobRepository(tmp_path)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    calls: list[str] = []
    active = 0
    max_active = 0
    guard = threading.Lock()

    def fake_runner(config):
        nonlocal active, max_active
        marker = config.get("marker")
        with guard:
            active += 1
            max_active = max(max_active, active)
        calls.append(marker)
        yield "Running", f"start {marker}\n", None
        time.sleep(0.03)
        output = tmp_path / f"{marker}.mp4"
        output.write_bytes(marker.encode())
        with guard:
            active -= 1
        result = DubbingJobResult("Completed", f"start {marker}\ndone {marker}\n", str(output))
        yield result.status, result.logs, result

    jobs = [
        repository.create(owner, {"input": str(input_path), "marker": marker})
        for marker in ("one", "two", "three")
    ]
    queue = InProcessJobQueue(repository, store, owner_id=owner, runner=fake_runner)
    queue.start()
    for job in jobs:
        queue.enqueue(job.id)
    queue.stop()

    assert calls == ["one", "two", "three"]
    assert max_active == 1
    for job in jobs:
        finished = repository.get(owner, job.id)
        assert finished.status == "succeeded"
        assert finished.started_at is not None
        assert finished.finished_at is not None
        assert finished.files[0]["kind"] == "result"
        events = repository.events(owner, job.id)
        assert [event.id for event in events] == sorted(event.id for event in events)
        assert [event.type for event in events] == ["state", "log", "log", "file", "state"]


def test_queue_registers_report_and_artifacts_and_persists_failures(tmp_path):
    owner = "alice"
    store = _store(tmp_path)
    repository = FileJobRepository(tmp_path)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    artifacts = tmp_path / "artifacts"
    artifacts.mkdir()
    report = artifacts / "report.txt"
    report.write_text("report", encoding="utf-8")
    extra = artifacts / "debug.json"
    extra.write_text("{}", encoding="utf-8")
    successful = repository.create(
        owner, {"input": str(input_path), "artifacts_dir": str(artifacts), "fail": False}
    )
    failed = repository.create(owner, {"input": str(input_path), "fail": True})

    def fake_runner(config):
        if config.get("fail"):
            raise RuntimeError("kaboom")
        result = DubbingJobResult("Speaker report generated", "report ready\n", report_file=str(report))
        yield result.status, result.logs, result

    queue = InProcessJobQueue(repository, store, owner_id=owner, runner=fake_runner)
    queue.start()
    queue.enqueue(successful.id)
    queue.enqueue(failed.id)
    queue.stop()

    completed = repository.get(owner, successful.id)
    assert completed.status == "succeeded"
    assert {item["kind"] for item in completed.files} == {"report", "artifact"}
    broken = repository.get(owner, failed.id)
    assert broken.status == "failed"
    assert broken.error == {"code": "pipeline_failed", "message": "kaboom"}
    assert repository.events(owner, failed.id)[-2].type == "error"
    assert repository.events(owner, failed.id)[-1].type == "state"


def test_queue_stop_is_idempotent_and_enqueue_requires_running_queue(tmp_path):
    repository = FileJobRepository(tmp_path)
    queue = InProcessJobQueue(repository, _store(tmp_path), owner_id="alice", runner=lambda _c: ())

    with pytest.raises(RuntimeError, match="not running"):
        queue.enqueue("job-id")
    queue.start()
    queue.stop()
    queue.stop()


def test_queue_buffers_partial_streaming_log_lines(tmp_path):
    owner = "alice"
    repository = FileJobRepository(tmp_path)
    input_path = tmp_path / "input.mp4"
    input_path.write_bytes(b"video")
    job = repository.create(owner, {"input": str(input_path)})

    def fake_runner(_config):
        yield "Running", "par", None
        result = DubbingJobResult("Completed", "partial line\n")
        yield result.status, result.logs, result

    queue = InProcessJobQueue(
        repository, _store(tmp_path), owner_id=owner, runner=fake_runner
    )
    queue.start()
    queue.enqueue(job.id)
    queue.stop()

    messages = [
        event.data["message"]
        for event in repository.events(owner, job.id)
        if event.type == "log"
    ]
    assert messages == ["partial line"]
