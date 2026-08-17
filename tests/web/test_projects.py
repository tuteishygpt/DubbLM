"""Tests for discovering, inspecting, and opening projects from the prj directory."""

import json
from pathlib import Path
import pytest
from fastapi.testclient import TestClient

from dubbing.web.app import create_app
from dubbing.web.projects import ProjectNotFoundError, ProjectService, ProjectValidationError
from dubbing.web.storage import FileMediaStore
from dubbing.web.jobs import FileJobRepository


@pytest.fixture
def prj_fixture(tmp_path: Path):
    prj_dir = tmp_path / "prj"
    prj_dir.mkdir()

    # Project 1: complete with video, srt, artifacts
    p1 = prj_dir / "sample_project_1"
    p1.mkdir()
    (p1 / "sample_project_1_ru.mp4").write_bytes(b"\x00" * 1024)
    (p1 / "sample_project_1_ru.srt").write_text("1\n00:00:01,000 --> 00:00:05,000\nПривет мир\n", encoding="utf-8")
    
    art1 = p1 / "artifacts"
    art1.mkdir()
    audio1 = art1 / "audio"
    audio1.mkdir()
    (audio1 / "source.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    (audio1 / "background.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    (audio1 / "output.wav").write_bytes(b"RIFF" + b"\x00" * 100)

    su_chunks1 = art1 / "su_audio_chunks"
    su_chunks1.mkdir()
    (su_chunks1 / "timed_0.wav").write_bytes(b"RIFF" + b"\x00" * 50)

    (art1 / "transcription.txt").write_text("[00.00.01.000-00.00.05.000] SPEAKER_00: Hello world\n", encoding="utf-8")
    (art1 / "timecodes.txt").write_text("00:01 - Intro\n", encoding="utf-8")
    
    debug1 = art1 / "debug"
    debug1.mkdir()
    (debug1 / "translations.tsv").write_text("time\toriginal\ttranslation\n00:00:01,000\tHello world\tПривет мир\n", encoding="utf-8")

    # Project 2: has dubbing_texts.tsv
    p2 = prj_dir / "sample_project_2"
    p2.mkdir()
    (p2 / "video.mp4").write_bytes(b"\x00" * 512)
    art2 = p2 / "artifacts"
    art2.mkdir()
    audio2 = art2 / "audio"
    audio2.mkdir()
    (audio2 / "source.wav").write_bytes(b"RIFF" + b"\x00" * 100)
    chunks2 = art2 / "audio_chunks"
    chunks2.mkdir()
    (chunks2 / "0.wav").write_bytes(b"RIFF" + b"\x00" * 50)
    (art2 / "dubbing_texts.tsv").write_text(
        "speaker\tstart\tend\toriginal\ttranslation\tsynthesized_text\tstyle_prompt\taudio_file\n"
        "SPEAKER_01\t0.5\t3.5\tGood morning\tДоброе утро\tДоброе утро\t\t0.wav\n",
        encoding="utf-8",
    )

    return prj_dir


def test_list_projects(prj_fixture: Path, tmp_path: Path):
    media = FileMediaStore(tmp_path / "data")
    jobs = FileJobRepository(tmp_path / "data")
    service = ProjectService(prj_fixture, job_repository=jobs, media_store=media)

    projects = service.list_projects("local")
    names = [p.name for p in projects]
    assert "sample_project_1" in names
    assert "sample_project_2" in names

    p1 = next(p for p in projects if p.name == "sample_project_1")
    assert p1.has_video is True
    assert p1.has_subtitles is True
    assert p1.has_artifacts is True
    assert p1.segment_count == 1
    assert "sample_project_1_ru.mp4" in p1.video_files


def test_get_project_detail(prj_fixture: Path, tmp_path: Path):
    media = FileMediaStore(tmp_path / "data")
    jobs = FileJobRepository(tmp_path / "data")
    service = ProjectService(prj_fixture, job_repository=jobs, media_store=media)

    detail = service.get_project("sample_project_1", "local")
    assert detail.name == "sample_project_1"
    assert detail.target_language == "ru"
    assert len(detail.video_files) == 1
    assert len(detail.subtitle_files) == 1


def test_open_project_creates_job_and_registers_files(prj_fixture: Path, tmp_path: Path):
    media = FileMediaStore(tmp_path / "data")
    jobs = FileJobRepository(tmp_path / "data")
    service = ProjectService(prj_fixture, job_repository=jobs, media_store=media)

    job = service.open_project("sample_project_1", "local")
    assert job.status == "succeeded"
    assert len(job.files) >= 3

    kinds = {f["kind"] for f in job.files}
    assert "output_video" in kinds
    assert "background_audio" in kinds
    assert "output_audio" in kinds

    # Opening again returns the existing job
    job2 = service.open_project("sample_project_1", "local")
    assert job2.id == job.id


def test_open_nonexistent_project_raises(prj_fixture: Path, tmp_path: Path):
    media = FileMediaStore(tmp_path / "data")
    jobs = FileJobRepository(tmp_path / "data")
    service = ProjectService(prj_fixture, job_repository=jobs, media_store=media)

    with pytest.raises(ProjectNotFoundError):
        service.open_project("does_not_exist", "local")


def test_projects_api_routes(prj_fixture: Path, tmp_path: Path):
    app = create_app(root=tmp_path / "data", projects_root=prj_fixture)
    client = TestClient(app)

    # GET /api/projects
    res = client.get("/api/projects")
    assert res.status_code == 200
    data = res.json()
    assert "projects" in data
    assert len(data["projects"]) >= 2

    # GET /api/projects/sample_project_1
    res = client.get("/api/projects/sample_project_1")
    assert res.status_code == 200
    detail = res.json()
    assert detail["name"] == "sample_project_1"

    # POST /api/projects/sample_project_1/open
    res = client.post("/api/projects/sample_project_1/open")
    assert res.status_code == 200
    opened = res.json()
    assert opened["project_name"] == "sample_project_1"
    job_id = opened["job"]["id"]
    assert job_id

    # Dubbing texts for the opened job can now be loaded
    texts_res = client.get(f"/api/jobs/{job_id}/dubbing-texts")
    print("TEXTS_RES:", texts_res.status_code, texts_res.text)
    assert texts_res.status_code == 200
    texts = texts_res.json()
    print("TEXTS DATA:", texts)
    assert len(texts["segments"]) == 1
    assert texts["segments"][0]["text"] == "Hello world"
    assert texts["segments"][0]["translation"] == "Привет мир"
    assert texts["segments"][0]["audio"] is not None


def test_projects_api_dubbing_texts_tsv_project(prj_fixture: Path, tmp_path: Path):
    app = create_app(root=tmp_path / "data", projects_root=prj_fixture)
    client = TestClient(app)

    # Open project 2 with dubbing_texts.tsv
    res = client.post("/api/projects/sample_project_2/open")
    assert res.status_code == 200
    job_id = res.json()["job"]["id"]

    texts_res = client.get(f"/api/jobs/{job_id}/dubbing-texts")
    assert texts_res.status_code == 200
    texts = texts_res.json()
    assert len(texts["segments"]) == 1
    assert texts["segments"][0]["text"] == "Good morning"
    assert texts["segments"][0]["translation"] == "Доброе утро"
    assert texts["segments"][0]["audio"] is not None


class DummyQueue:
    def __init__(self):
        self.enqueued = []
    def start(self): pass
    def stop(self): pass
    def enqueue(self, job_id):
        self.enqueued.append(job_id)


def test_projects_api_run_step(prj_fixture: Path, tmp_path: Path):
    dummy_queue = DummyQueue()
    app = create_app(root=tmp_path / "data", projects_root=prj_fixture, job_queue=dummy_queue)
    with TestClient(app) as client:
        # POST /api/projects/sample_project_1/run with tts_to_end
        res = client.post(
            "/api/projects/sample_project_1/run",
            json={"run_step": "tts_to_end", "overrides": {}},
        )
        assert res.status_code == 201
        data = res.json()
        assert data["project_name"] == "sample_project_1"
        job = data["job"]
        assert job["id"]
        assert job["status"] in ("queued", "running", "succeeded")
        assert job["id"] in dummy_queue.enqueued


def test_update_speaker_map_and_config_resolution(prj_fixture: Path, tmp_path: Path):
    media = FileMediaStore(tmp_path / "data")
    jobs = FileJobRepository(tmp_path / "data")
    config_file = tmp_path / "dubbing_config.yml"
    config_file.write_text(
        "voices:\n"
        "  'John Male':\n"
        "    tts_system: gemini\n"
        "    model: gemini-2.5-pro-preview-tts\n"
        "    voice_name: Fenrir\n",
        encoding="utf-8",
    )
    service = ProjectService(
        prj_fixture, job_repository=jobs, media_store=media, config_path=config_file
    )

    # Update speaker map for sample_project_1
    service.update_speaker_map(
        "sample_project_1", "local", {"SPEAKER_00": "John Male"}
    )

    # Verify project detail has saved speaker_map
    detail = service.get_project("sample_project_1", "local")
    assert detail.saved_config is not None
    assert detail.saved_config.get("speaker_map") == {"SPEAKER_00": "John Male"}

    # Verify _build_project_config resolves SPEAKER_00 to John Male's profile
    cfg = service._build_project_config(prj_fixture / "sample_project_1")
    assert "voices" in cfg
    assert "SPEAKER_00" in cfg["voices"]
    assert cfg["voices"]["SPEAKER_00"]["voice_name"] == "Fenrir"
    assert cfg["voices"]["SPEAKER_00"]["tts_system"] == "gemini"


def test_projects_api_update_speaker_map_endpoint(prj_fixture: Path, tmp_path: Path):
    app = create_app(root=tmp_path / "data", projects_root=prj_fixture)
    client = TestClient(app)

    res = client.put(
        "/api/projects/sample_project_1/speaker-map",
        json={"speaker_map": {"SPEAKER_00": "John Male"}},
    )
    assert res.status_code == 200
    assert res.json() == {"ok": True}

    # Verify through GET /api/projects/sample_project_1
    detail_res = client.get("/api/projects/sample_project_1")
    assert detail_res.status_code == 200
    assert detail_res.json()["saved_config"]["speaker_map"] == {"SPEAKER_00": "John Male"}
