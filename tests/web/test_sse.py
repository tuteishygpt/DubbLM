from __future__ import annotations

import asyncio
import json
from dataclasses import replace

from dubbing.web.jobs import Job, JobEvent
from dubbing.web.routes.jobs import job_event_stream


OWNER = "local-owner"
JOB_ID = "10000000-0000-4000-8000-000000000001"


def make_job() -> Job:
    return Job(
        id=JOB_ID, owner_id=OWNER, config={},
        created_at="2026-08-13T10:00:00+00:00",
        updated_at="2026-08-13T10:00:00+00:00",
        started_at=None, finished_at=None, status="queued",
        state={"status": "queued"}, error=None, files=[], last_event_id=0,
    )


def decode(chunk: str) -> dict:
    data = next(line[6:] for line in chunk.splitlines() if line.startswith("data: "))
    return json.loads(data)


class StreamingRepository:
    def __init__(self, *, expired=False, terminal=True):
        self.expired = expired
        self.terminal = terminal
        self.calls = 0
        self.events_list = [
            JobEvent(4 if expired else 1, JOB_ID, "log", {"message": "one"}, "2026-08-13T10:00:01+00:00"),
            JobEvent(5 if expired else 2, JOB_ID, "state", {"status": "succeeded"}, "2026-08-13T10:00:02+00:00"),
        ]

    def get(self, owner_id, job_id):
        assert owner_id == OWNER and job_id == JOB_ID
        status = "succeeded" if self.terminal else "running"
        return replace(make_job(), status=status, state={"status": status}, last_event_id=self.events_list[-1].id if self.events_list else 0)

    def events(self, owner_id, job_id, *, after_id=0):
        self.get(owner_id, job_id)
        return [event for event in self.events_list if event.id > after_id]


def collect(stream):
    async def run():
        return [chunk async for chunk in stream]
    return asyncio.run(run())


def test_sse_emits_ordered_shaped_events_and_closes_after_terminal():
    chunks = collect(job_event_stream(StreamingRepository(), OWNER, JOB_ID, after_id=0, heartbeat_interval=0.01, poll_interval=0.001))
    events = [decode(chunk) for chunk in chunks if chunk.startswith("id:")]
    assert [event["id"] for event in events] == [1, 2]
    assert events[0] == {
        "id": 1, "job_id": JOB_ID, "type": "log",
        "timestamp": "2026-08-13T10:00:01+00:00", "data": {"message": "one"},
    }


def test_sse_honors_last_event_id_replay():
    chunks = collect(job_event_stream(StreamingRepository(), OWNER, JOB_ID, after_id=1, heartbeat_interval=0.01, poll_interval=0.001))
    assert [decode(chunk)["id"] for chunk in chunks if chunk.startswith("id:")] == [2]


def test_expired_history_starts_with_current_snapshot_and_retained_log_tail():
    chunks = collect(job_event_stream(StreamingRepository(expired=True), OWNER, JOB_ID, after_id=1, heartbeat_interval=0.01, poll_interval=0.001))
    snapshot = decode(chunks[0])
    assert snapshot["type"] == "snapshot"
    assert snapshot["id"] == 5
    assert snapshot["data"]["state"] == {"status": "succeeded"}
    assert snapshot["data"]["log_tail"] == [{"id": 4, "message": "one"}]


def test_fully_expired_history_still_emits_terminal_snapshot_and_closes():
    repository = StreamingRepository()
    repository.events_list = []
    repository.get = lambda owner_id, job_id: replace(
        make_job(), status="failed", state={"status": "failed"},
        error={"code": "server_restarted", "message": "Server restarted."},
        last_event_id=9,
    )
    chunks = collect(job_event_stream(repository, OWNER, JOB_ID, after_id=3, heartbeat_interval=0.01))
    assert len(chunks) == 1
    snapshot = decode(chunks[0])
    assert snapshot["id"] == 9
    assert snapshot["type"] == "snapshot"
    assert snapshot["data"]["error"]["code"] == "server_restarted"


def test_eventless_terminal_job_emits_current_snapshot_before_close():
    repository = StreamingRepository()
    repository.events_list = []
    repository.get = lambda owner_id, job_id: replace(
        make_job(), status="failed", state={"status": "failed"},
        error={"code": "server_restarted", "message": "Server restarted."},
    )
    chunks = collect(job_event_stream(repository, OWNER, JOB_ID, after_id=0, heartbeat_interval=0.01))
    assert len(chunks) == 1
    assert decode(chunks[0])["type"] == "snapshot"


def test_sse_heartbeat_comment_uses_injected_interval_without_busy_loop():
    repository = StreamingRepository(terminal=False)
    repository.events_list = []
    stream = job_event_stream(repository, OWNER, JOB_ID, after_id=0, heartbeat_interval=0.01)
    async def first():
        chunk = await asyncio.wait_for(anext(stream), timeout=0.1)
        await stream.aclose()
        return chunk
    chunk = asyncio.run(first())
    assert chunk == ": heartbeat\n\n"
