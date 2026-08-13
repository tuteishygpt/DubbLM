"""Owner-scoped persisted job metadata and event logs."""

from datetime import datetime, timedelta, timezone
import json
from pathlib import Path
from uuid import UUID

import pytest

from dubbing.web.contracts import JobRepository
from dubbing.web.jobs import (
    DEFAULT_EVENT_RETENTION_BYTES,
    FileJobRepository,
    JobNotFoundError,
    JobValidationError,
    JobWriteError,
)


class _MemoryJobRepository:
    """A non-filesystem adapter used to prove the consumer boundary."""

    def create(self, *args, **kwargs):
        return "created"

    def get(self, *args, **kwargs):
        return "found"

    def list(self, *args, **kwargs):
        return "listed"

    def update(self, *args, **kwargs):
        return "updated"

    def append_event(self, *args, **kwargs):
        return "event"


def _consumer(repository: JobRepository) -> tuple[object, ...]:
    return (
        repository.create("owner", {"target_language": "es"}),
        repository.get("owner", "job-id"),
        repository.list("owner", limit=10),
        repository.update("owner", "job-id", status="running"),
        repository.append_event("owner", "job-id", "log", {"message": "hello"}),
    )


def test_job_consumer_depends_only_on_replaceable_protocol():
    assert _consumer(_MemoryJobRepository()) == (
        "created", "found", "listed", "updated", "event"
    )


def test_default_event_retention_is_ten_mibibytes():
    assert DEFAULT_EVENT_RETENTION_BYTES == 10 * 1024**2


def test_create_persists_complete_job_shape_and_defensive_config_snapshot(tmp_path):
    config = {"target_language": "es", "nested": {"temperature": 0.3}}
    repository = FileJobRepository(tmp_path)

    created = repository.create("alice", config)
    config["nested"]["temperature"] = 9
    loaded = repository.get("alice", created.id)

    assert UUID(created.id).version == 4
    assert loaded.owner_id == "alice"
    assert loaded.config == {"target_language": "es", "nested": {"temperature": 0.3}}
    assert loaded.status == "queued"
    assert loaded.state == {}
    assert loaded.error is None
    assert loaded.files == []
    assert loaded.last_event_id == 0
    assert datetime.fromisoformat(loaded.created_at).tzinfo is not None
    assert loaded.updated_at == loaded.created_at
    assert loaded.started_at is None
    assert loaded.finished_at is None
    persisted = json.loads(
        (tmp_path / "jobs" / "alice" / created.id / "job.json").read_text(encoding="utf-8")
    )
    assert set(persisted) == {
        "id", "owner_id", "config", "created_at", "updated_at", "started_at",
        "finished_at", "status", "state", "error", "files", "last_event_id",
    }


def test_update_changes_mutable_state_but_never_configuration(tmp_path):
    repository = FileJobRepository(tmp_path)
    created = repository.create("alice", {"target_language": "es"})

    running = repository.update(
        "alice", created.id, status="running", state={"stage": "transcription"}
    )
    failed = repository.update(
        "alice",
        created.id,
        status="failed",
        error={"code": "pipeline_failed", "message": "boom"},
        files=[{"id": "output-id", "name": "dubbed.mp4"}],
    )

    assert running.status == "running"
    assert running.state == {"stage": "transcription"}
    assert running.started_at is not None
    assert running.finished_at is None
    assert failed.error == {"code": "pipeline_failed", "message": "boom"}
    assert failed.files == [{"id": "output-id", "name": "dubbed.mp4"}]
    assert failed.config == {"target_language": "es"}
    assert failed.started_at == running.started_at
    assert failed.finished_at is not None
    with pytest.raises(JobValidationError, match="config"):
        repository.update("alice", created.id, config={"target_language": "fr"})


def test_get_and_update_mask_other_owners_and_reject_traversal(tmp_path):
    repository = FileJobRepository(tmp_path)
    created = repository.create("alice", {})

    with pytest.raises(JobNotFoundError):
        repository.get("bob", created.id)
    with pytest.raises(JobNotFoundError):
        repository.update("bob", created.id, status="running")
    with pytest.raises(JobValidationError):
        repository.get("../alice", created.id)
    with pytest.raises(JobValidationError):
        repository.get("alice", "../job.json")


def test_list_is_owner_scoped_and_cursor_paginated(tmp_path):
    current = datetime(2026, 8, 13, 12, 0, tzinfo=timezone.utc)

    def clock():
        nonlocal current
        result = current
        current += timedelta(seconds=1)
        return result

    repository = FileJobRepository(tmp_path, clock=clock)
    alice_ids = [repository.create("alice", {"index": index}).id for index in range(5)]
    repository.create("bob", {"private": True})

    first = repository.list("alice", limit=2)
    second = repository.list("alice", limit=2, cursor=first.next_cursor)
    third = repository.list("alice", limit=2, cursor=second.next_cursor)

    assert [job.id for job in first.items] == list(reversed(alice_ids))[0:2]
    assert [job.id for job in second.items] == list(reversed(alice_ids))[2:4]
    assert [job.id for job in third.items] == list(reversed(alice_ids))[4:5]
    assert first.next_cursor == first.items[-1].id
    assert second.next_cursor == second.items[-1].id
    assert third.next_cursor is None
    assert all(job.owner_id == "alice" for page in (first, second, third) for job in page.items)


def test_list_rejects_invalid_page_controls(tmp_path):
    repository = FileJobRepository(tmp_path)

    with pytest.raises(JobValidationError):
        repository.list("alice", limit=0)
    with pytest.raises(JobValidationError):
        repository.list("alice", limit=101)
    with pytest.raises(JobValidationError):
        repository.list("alice", cursor="../job.json")


def test_job_json_replacement_is_atomic_and_cleans_sibling_temp(tmp_path, monkeypatch):
    repository = FileJobRepository(tmp_path)
    created = repository.create("alice", {})
    job_path = tmp_path / "jobs" / "alice" / created.id / "job.json"
    original = job_path.read_bytes()

    monkeypatch.setattr(
        "dubbing.web.jobs.os.replace",
        lambda *_args: (_ for _ in ()).throw(OSError("disk error")),
    )

    with pytest.raises(JobWriteError):
        repository.update("alice", created.id, status="running")

    assert job_path.read_bytes() == original
    assert list(job_path.parent.glob(".job.json.*.tmp")) == []


def test_append_event_assigns_monotonic_ids_and_persists_jsonl_order(tmp_path):
    repository = FileJobRepository(tmp_path)
    created = repository.create("alice", {})

    first = repository.append_event("alice", created.id, "state", {"status": "running"})
    second = repository.append_event("alice", created.id, "log", {"message": "hello"})
    third = repository.append_event("alice", created.id, "log", {"message": "world"})

    assert [first.id, second.id, third.id] == [1, 2, 3]
    assert [first.job_id, second.job_id, third.job_id] == [created.id] * 3
    assert [event.id for event in repository.events("alice", created.id)] == [1, 2, 3]
    lines = (
        tmp_path / "jobs" / "alice" / created.id / "events.jsonl"
    ).read_text(encoding="utf-8").splitlines()
    assert [json.loads(line)["id"] for line in lines] == [1, 2, 3]
    assert repository.get("alice", created.id).last_event_id == 3


def test_events_support_resume_after_event_id_and_mask_owner(tmp_path):
    repository = FileJobRepository(tmp_path)
    created = repository.create("alice", {})
    for index in range(4):
        repository.append_event("alice", created.id, "log", {"index": index})

    assert [event.id for event in repository.events("alice", created.id, after_id=2)] == [3, 4]
    with pytest.raises(JobNotFoundError):
        repository.events("bob", created.id)


def test_event_log_retains_newest_complete_json_lines_within_limit(tmp_path):
    repository = FileJobRepository(tmp_path, event_retention_bytes=280)
    created = repository.create("alice", {})
    for index in range(12):
        repository.append_event("alice", created.id, "log", {"message": f"event-{index}"})

    event_path = tmp_path / "jobs" / "alice" / created.id / "events.jsonl"
    retained = repository.events("alice", created.id)

    assert event_path.stat().st_size <= 280
    assert retained[-1].id == 12
    assert retained[0].id > 1
    assert [event.id for event in retained] == sorted(event.id for event in retained)
    assert repository.get("alice", created.id).last_event_id == 12


def test_event_larger_than_retention_limit_is_rejected_without_advancing_id(tmp_path):
    repository = FileJobRepository(tmp_path, event_retention_bytes=100)
    created = repository.create("alice", {})

    with pytest.raises(JobValidationError, match="retention"):
        repository.append_event("alice", created.id, "log", {"message": "x" * 500})

    assert repository.get("alice", created.id).last_event_id == 0
    assert repository.events("alice", created.id) == []


def test_append_event_rejects_unknown_sse_event_type(tmp_path):
    repository = FileJobRepository(tmp_path)
    created = repository.create("alice", {})

    with pytest.raises(JobValidationError, match="type"):
        repository.append_event("alice", created.id, "arbitrary", {})


def test_append_event_recovers_monotonic_id_after_metadata_replace_failure(tmp_path, monkeypatch):
    repository = FileJobRepository(tmp_path)
    created = repository.create("alice", {})
    original = repository._atomic_json
    failed = False

    def fail_job_metadata(path, value):
        nonlocal failed
        if path.name == "job.json" and value.get("last_event_id") == 1 and not failed:
            failed = True
            raise JobWriteError("disk error")
        original(path, value)

    monkeypatch.setattr(repository, "_atomic_json", fail_job_metadata)

    with pytest.raises(JobWriteError):
        repository.append_event("alice", created.id, "log", {"message": "first"})
    second = repository.append_event("alice", created.id, "log", {"message": "second"})

    assert second.id == 2
    assert [event.id for event in repository.events("alice", created.id)] == [1, 2]
    assert repository.get("alice", created.id).last_event_id == 2


def test_restart_fails_queued_and_running_jobs_but_preserves_terminal_jobs(tmp_path):
    repository = FileJobRepository(tmp_path)
    queued = repository.create("alice", {"index": 1})
    running = repository.create("alice", {"index": 2})
    succeeded = repository.create("alice", {"index": 3})
    failed = repository.create("alice", {"index": 4})
    repository.update("alice", running.id, status="running")
    repository.update("alice", succeeded.id, status="succeeded")
    repository.update(
        "alice", failed.id, status="failed", error={"code": "pipeline_failed", "message": "boom"}
    )

    restarted = FileJobRepository(tmp_path)

    for job_id in (queued.id, running.id):
        recovered = restarted.get("alice", job_id)
        assert recovered.status == "failed"
        assert recovered.error == {
            "code": "server_restarted",
            "message": "The server restarted before the job completed.",
        }
        assert recovered.finished_at is not None
    assert restarted.get("alice", succeeded.id).status == "succeeded"
    assert restarted.get("alice", succeeded.id).error is None
    assert restarted.get("alice", failed.id).error == {
        "code": "pipeline_failed", "message": "boom"
    }
