from __future__ import annotations

from dataclasses import dataclass, replace
from io import BytesIO
from pathlib import Path
from types import SimpleNamespace

from fastapi.testclient import TestClient

from dubbing.web.app import create_app
from dubbing.web.dubbing_texts import (
    DubbingTextConflictError,
    DubbingTextRegeneration,
    DubbingTextSegment,
    DubbingTextSnapshot,
    SegmentAudio,
)
from dubbing.web.jobs import Job, JobEvent, JobNotFoundError, JobPage
from dubbing.web.references import ReferenceAudio, ReferenceEntry, ReferenceLibrarySnapshot
from dubbing.web.settings import SettingsConflictError, SettingsSnapshot, VoiceProfilesSnapshot
from dubbing.web.storage import MediaConflictError, MediaRecord, MediaValidationError


OWNER = "local-owner"
JOB_ID = "10000000-0000-4000-8000-000000000001"
UPLOAD_ID = "20000000-0000-4000-8000-000000000001"
FILE_ID = "30000000-0000-4000-8000-000000000001"
REFERENCED_UPLOAD_ID = "40000000-0000-4000-8000-000000000001"


def make_job(**changes: object) -> Job:
    job = Job(
        id=JOB_ID,
        owner_id=OWNER,
        config={"input": "trusted"},
        created_at="2026-08-13T10:00:00+00:00",
        updated_at="2026-08-13T10:00:00+00:00",
        started_at=None,
        finished_at=None,
        status="queued",
        state={"status": "queued"},
        error=None,
        files=[],
        last_event_id=0,
    )
    return replace(job, **changes)


class FakeSettings:
    def __init__(self) -> None:
        self.snapshot = SettingsSnapshot("settings-1", {"source_language": "en"})

    def load(self) -> SettingsSnapshot:
        return self.snapshot

    def save(self, values, *, revision):
        if revision != self.snapshot.revision:
            raise SettingsConflictError("stale settings")
        self.snapshot = SettingsSnapshot("settings-2", dict(values))
        return self.snapshot

    def list_profiles(self):
        return VoiceProfilesSnapshot(self.snapshot.revision, {"SPEAKER_00": {"tts_system": "gemini", "model": "m"}})

    def put_profile(self, speaker_id, profile, *, revision):
        if revision != self.snapshot.revision:
            raise SettingsConflictError("stale settings")
        return VoiceProfilesSnapshot("settings-2", {speaker_id: dict(profile)})

    def delete_profile(self, speaker_id, *, revision):
        if revision != self.snapshot.revision:
            raise SettingsConflictError("stale settings")
        return VoiceProfilesSnapshot("settings-2", {})


class FakeMedia:
    def __init__(self, root: Path) -> None:
        self.path = root / "trusted.wav"
        self.path.write_bytes(b"audio")
        self.deleted: list[tuple[str, str]] = []
        self.too_large = False

    def save(self, owner_id, filename=None, stream=None, **kwargs):
        assert owner_id == OWNER
        if self.too_large:
            raise MediaValidationError("Upload exceeds the configured maximum size.")
        data = stream.read()
        return MediaRecord(UPLOAD_ID, filename or "clip.wav", "audio", len(data), False, self.path)

    def delete(self, owner_id, media_id):
        assert owner_id == OWNER
        if media_id == REFERENCED_UPLOAD_ID:
            raise MediaConflictError("Media is referenced by a job and cannot be deleted.")
        self.deleted.append((owner_id, media_id))

    def get(self, owner_id=None, media_id=None, *args):
        owner_id = owner_id or args[0]
        media_id = media_id or args[1]
        assert owner_id == OWNER
        return MediaRecord(media_id, "trusted.wav", "result", 5, media_id == FILE_ID, self.path)


class FakeRepository:
    def __init__(self) -> None:
        self.job = make_job(files=[{"id": FILE_ID, "name": "trusted.wav", "kind": "result", "size": 5}])
        self.event_items: list[JobEvent] = []
        self.after_ids: list[int] = []

    def get(self, owner_id, job_id):
        assert owner_id == OWNER
        if job_id != JOB_ID:
            raise JobNotFoundError("missing")
        return self.job

    def list(self, owner_id, *, limit=50, cursor=None):
        assert owner_id == OWNER
        return JobPage([self.job], cursor)

    def events(self, owner_id, job_id, *, after_id=0):
        self.get(owner_id, job_id)
        self.after_ids.append(after_id)
        return [event for event in self.event_items if event.id > after_id]


class FakeJobs:
    def __init__(self, repository: FakeRepository) -> None:
        self.repository = repository
        self.received = None

    def submit(self, owner_id, input_upload_id, *, isolated_tracks=None, overrides=None):
        assert owner_id == OWNER
        self.received = (input_upload_id, isolated_tracks, overrides)
        return self.repository.job


class FakeQueue:
    def __init__(self) -> None:
        self.started = self.stopped = False

    def start(self):
        self.started = True

    def stop(self):
        self.stopped = True


class FakeReferences:
    def __init__(self) -> None:
        self.snapshot = ReferenceLibrarySnapshot(
            "refs-1",
            (ReferenceEntry("Narrator", "hello", ReferenceAudio(FILE_ID, "trusted.wav", "/api/reference-library/Narrator/audio")),),
        )

    def list(self, *, owner_id):
        assert owner_id == OWNER
        return self.snapshot

    def save(self, **kwargs):
        assert kwargs["owner_id"] == OWNER
        return ReferenceLibrarySnapshot("refs-2", self.snapshot.entries)

    def assign(self, **kwargs):
        assert kwargs["owner_id"] == OWNER
        return SimpleNamespace(
            profile_speaker_id=kwargs["profile_speaker_id"],
            library_speaker_id=kwargs["library_speaker_id"],
            audio_id=FILE_ID,
            reference_text="hello",
            settings_revision="settings-2",
        )

    def delete(self, **kwargs):
        assert kwargs["owner_id"] == OWNER
        return ReferenceLibrarySnapshot("refs-2", ())


class FakeTexts:
    segment = DubbingTextSegment(
        "segment-1", "SPEAKER_00", 0, 1, "Hello", "Bonjour", "Bonjour", "", SegmentAudio(FILE_ID, "trusted.wav", f"/api/jobs/{JOB_ID}/files/{FILE_ID}")
    )

    def load(self, **kwargs):
        assert kwargs["owner_id"] == OWNER
        return DubbingTextSnapshot("text-1", (self.segment,), "snapshot")

    def save(self, **kwargs):
        assert kwargs["owner_id"] == OWNER
        if kwargs["revision"] != "text-1":
            raise DubbingTextConflictError("stale text")
        return DubbingTextSnapshot("text-2", tuple(kwargs["segments"]), "snapshot")

    def regenerate(self, **kwargs):
        assert kwargs["owner_id"] == OWNER
        return DubbingTextRegeneration("text-2", self.segment)


@dataclass(frozen=True)
class User:
    id: str = OWNER


def client_bundle(tmp_path: Path):
    settings = FakeSettings()
    media = FakeMedia(tmp_path)
    repository = FakeRepository()
    jobs = FakeJobs(repository)
    queue = FakeQueue()
    references = FakeReferences()
    texts = FakeTexts()
    app = create_app(
        settings_service=settings,
        media_store=media,
        job_repository=repository,
        job_service=jobs,
        job_queue=queue,
        reference_service=references,
        dubbing_text_service=texts,
        current_user=User(),
        heartbeat_interval=0.01,
    )
    return TestClient(app), SimpleNamespace(
        settings=settings, media=media, repository=repository, jobs=jobs,
        queue=queue, references=references, texts=texts,
    )


def test_config_options_and_revision_error_contract(tmp_path):
    client, services = client_bundle(tmp_path)
    with client:
        response = client.get("/api/config")
        assert response.status_code == 200
        assert set(response.json()) == {"revision", "values", "schema"}
        fields = {
            field["name"]: field for field in response.json()["schema"]["fields"]
        }
        assert len(fields) > 20
        assert {"input", "output", "config"}.isdisjoint(fields)
        for name in (
            "generate_speaker_report",
            "save_original_subtitles",
            "save_translated_subtitles",
            "keep_background",
            "include_original_audio",
            "remove_pauses",
        ):
            assert fields[name]["type"] == "boolean"
        assert fields["run_step"] == {
            "name": "run_step",
            "type": "select",
            "workflow": True,
            "options_key": "run_modes",
        }
        assert fields["inner_transcription_system"]["type"] == "select"
        assert fields["inner_transcription_system"]["options_key"] == "inner_transcription_systems"
        options = client.get("/api/options").json()
        assert options["run_modes"] == [
            "full_pipeline", "from_scratch", "transcribe_only", "translate_only",
            "analyze_emotions_only", "combine_video", "tts_to_end",
        ]
        assert options["inner_transcription_systems"] == ["deepgram", "assemblyai", "gemini"]
        assert client.put("/api/config", json={"revision": "settings-1", "values": {"target_language": "fr"}}).json()["revision"] == "settings-2"
        stale = client.put("/api/config", json={"revision": "stale", "values": {}})
    assert stale.status_code == 409
    assert stale.json() == {"code": "conflict", "message": "stale settings"}
    assert services.queue.started and services.queue.stopped


def test_upload_delete_maximum_and_referenced_rejection(tmp_path):
    client, services = client_bundle(tmp_path)
    with client:
        uploaded = client.post("/api/uploads", files={"file": ("clip.wav", b"abc", "audio/wav")})
        assert uploaded.json() == {"id": UPLOAD_ID, "name": "clip.wav", "kind": "audio", "size": 3}
        assert client.delete(f"/api/uploads/{UPLOAD_ID}").status_code == 204
        services.media.too_large = True
        large = client.post("/api/uploads", files={"file": ("large.wav", b"abcdef", "audio/wav")})
        conflict = client.delete(f"/api/uploads/{REFERENCED_UPLOAD_ID}")
        registered = client.delete(f"/api/uploads/{FILE_ID}")
    assert large.status_code == 413 and large.json()["code"] == "upload_too_large"
    assert conflict.status_code == 409 and conflict.json()["code"] == "conflict"
    assert registered.status_code == 404 and registered.json()["code"] == "not_found"


def test_jobs_list_detail_files_download_and_owner_masking(tmp_path):
    client, services = client_bundle(tmp_path)
    with client:
        created = client.post("/api/jobs", json={"input_upload_id": UPLOAD_ID, "isolated_tracks": {"SPEAKER_00": UPLOAD_ID}, "overrides": {"target_language": "fr"}})
        listed = client.get("/api/jobs?limit=10")
        detail = client.get(f"/api/jobs/{JOB_ID}")
        files = client.get(f"/api/jobs/{JOB_ID}/files")
        download = client.get(f"/api/jobs/{JOB_ID}/files/{FILE_ID}")
        missing_job = client.get("/api/jobs/90000000-0000-4000-8000-000000000009")
        missing_file = client.get(f"/api/jobs/{JOB_ID}/files/not-registered")
    assert created.status_code == 201 and created.json()["id"] == JOB_ID
    assert services.jobs.received == (UPLOAD_ID, {"SPEAKER_00": UPLOAD_ID}, {"target_language": "fr"})
    assert listed.json()["jobs"][0]["id"] == detail.json()["id"] == JOB_ID
    assert files.json()["files"][0]["url"].endswith(f"/{FILE_ID}")
    assert download.content == b"audio"
    assert missing_job.status_code == missing_file.status_code == 404
    assert missing_job.json()["code"] == "not_found"


def test_voice_profile_routes_are_revisioned_upserts(tmp_path):
    client, _ = client_bundle(tmp_path)
    with client:
        listed = client.get("/api/voice-profiles")
        upserted = client.put("/api/voice-profiles/SPEAKER_02", json={"revision": "settings-1", "profile": {"tts_system": "gemini", "model": "m"}})
        deleted = client.request("DELETE", "/api/voice-profiles/SPEAKER_00", json={"revision": "settings-1"})
    assert listed.json()["profiles"]["SPEAKER_00"]["model"] == "m"
    assert "SPEAKER_02" in upserted.json()["profiles"]
    assert deleted.json()["profiles"] == {}


def test_reference_library_crud_assignment_and_registered_audio(tmp_path):
    client, _ = client_bundle(tmp_path)
    with client:
        listed = client.get("/api/reference-library")
        saved = client.post(
            "/api/reference-library",
            data={"speaker_id": "Narrator", "reference_text": "hello", "revision": "refs-1"},
            files={"file": ("voice.wav", b"voice", "audio/wav")},
        )
        assigned = client.put("/api/reference-library/Narrator", json={"profile_speaker_id": "SPEAKER_00", "settings_revision": "settings-1"})
        audio = client.get("/api/reference-library/Narrator/audio")
        deleted = client.request("DELETE", "/api/reference-library/Narrator", json={"revision": "refs-1"})
    assert listed.json()["entries"][0]["speaker_id"] == "Narrator"
    assert saved.json()["revision"] == "refs-2"
    assert assigned.json()["settings_revision"] == "settings-2"
    assert audio.content == b"audio"
    assert deleted.json()["entries"] == []


def test_dubbing_text_routes_and_conflict(tmp_path):
    client, _ = client_bundle(tmp_path)
    segment = {
        "segment_id": "segment-1", "speaker": "SPEAKER_00", "start": 0, "end": 1,
        "text": "Hello", "translation": "Salut", "synthesized_text": "Salut", "style_prompt": "",
    }
    with client:
        loaded = client.get(f"/api/jobs/{JOB_ID}/dubbing-texts")
        saved = client.put(f"/api/jobs/{JOB_ID}/dubbing-texts", json={"revision": "text-1", "segments": [segment]})
        regenerated = client.post(f"/api/jobs/{JOB_ID}/dubbing-texts/segment-1/regenerate", json={"revision": "text-1", "synthesized_text": "override"})
        stale = client.put(f"/api/jobs/{JOB_ID}/dubbing-texts", json={"revision": "stale", "segments": [segment]})
    assert loaded.json()["segments"][0]["audio"]["id"] == FILE_ID
    assert saved.json()["revision"] == regenerated.json()["revision"] == "text-2"
    assert stale.status_code == 409


def test_request_validation_uses_domain_error_shape_and_never_accepts_owner(tmp_path):
    client, services = client_bundle(tmp_path)
    with client:
        invalid = client.post("/api/jobs", json={"input_upload_id": 3, "owner_id": "attacker"})
        valid = client.post("/api/jobs", json={"input_upload_id": UPLOAD_ID, "owner_id": "attacker"})
    assert invalid.status_code == 422
    assert invalid.json()["code"] == "validation_error"
    assert "message" in invalid.json() and "details" in invalid.json()
    assert valid.status_code == 422
    assert services.jobs.received is None


def test_sse_http_route_honors_header_and_sets_streaming_headers(tmp_path):
    client, services = client_bundle(tmp_path)
    services.repository.event_items = [
        JobEvent(1, JOB_ID, "log", {"message": "old"}, "2026-08-13T10:00:01+00:00"),
        JobEvent(2, JOB_ID, "state", {"status": "succeeded"}, "2026-08-13T10:00:02+00:00"),
    ]
    services.repository.job = replace(
        services.repository.job,
        status="succeeded",
        state={"status": "succeeded"},
        last_event_id=2,
    )
    with client:
        response = client.get(
            f"/api/jobs/{JOB_ID}/events?last_event_id=0",
            headers={"Last-Event-ID": "1"},
        )
    assert response.status_code == 200
    assert response.headers["content-type"].startswith("text/event-stream")
    assert response.headers["x-accel-buffering"] == "no"
    assert '"id":2' in response.text and '"id":1' not in response.text
    assert 1 in services.repository.after_ids
