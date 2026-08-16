"""Discovery, inspection, and opening of ready projects from the `prj` directory."""

from __future__ import annotations

import json
import os
import re
import uuid
from dataclasses import dataclass
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from .contracts import MediaStore
from .jobs import FileJobRepository, Job
from .storage import AUDIO_EXTENSIONS, VIDEO_EXTENSIONS

NAMESPACE_DUBBLM_PRJ = uuid.UUID("a7b2c3d4-e5f6-4a1b-8c2d-3e4f5a6b7c8d")
SUBTITLE_EXTENSIONS = frozenset({".srt", ".vtt", ".ass"})


class ProjectError(Exception):
    """Base class for project-domain errors."""


class ProjectNotFoundError(ProjectError):
    """The requested project does not exist."""


class ProjectValidationError(ProjectError):
    """The project is invalid or unsafe."""


@dataclass(frozen=True)
class ProjectSummary:
    """Summary metadata for an existing project folder in `prj/`."""

    name: str
    display_name: str
    relative_path: str
    created_at: str
    updated_at: str
    has_video: bool
    has_subtitles: bool
    has_artifacts: bool
    has_transcription: bool
    segment_count: int
    video_files: list[str]
    audio_files: list[str]
    job_id: str | None = None


@dataclass(frozen=True)
class ProjectDetail:
    """Detailed metadata for a project folder in `prj/`."""

    name: str
    display_name: str
    relative_path: str
    created_at: str
    updated_at: str
    video_files: list[dict[str, Any]]
    audio_files: list[dict[str, Any]]
    subtitle_files: list[dict[str, Any]]
    artifacts: dict[str, Any]
    target_language: str
    source_language: str
    saved_config: dict[str, Any] | None = None
    job_id: str | None = None


class ProjectService:
    """Service for discovering, inspecting, and opening projects from `prj/`."""

    def __init__(
        self,
        projects_root: str | Path | None = None,
        *,
        job_repository: FileJobRepository | None = None,
        media_store: MediaStore | None = None,
        queue: object | None = None,
        config_path: str | Path = "dubbing_config.yml",
    ) -> None:
        if projects_root is not None:
            self._projects_root = Path(projects_root)
        elif os.environ.get("DUBBLM_PROJECTS_ROOT"):
            self._projects_root = Path(os.environ["DUBBLM_PROJECTS_ROOT"])
        else:
            default_candidate = Path(__file__).resolve().parents[3] / "prj"
            self._projects_root = default_candidate if default_candidate.is_dir() else Path("prj")
        self._projects_root = self._projects_root.resolve()
        self._job_repository = job_repository
        self._media_store = media_store
        self._queue = queue
        self._config_path = str(config_path)

    @property
    def root(self) -> Path:
        return self._projects_root

    def list_projects(self, owner_id: str) -> list[ProjectSummary]:
        """List all valid project directories beneath `projects_root`."""
        if not self._projects_root.is_dir():
            return []

        results: list[ProjectSummary] = []
        for entry in sorted(self._projects_root.iterdir()):
            if not entry.is_dir() or entry.name.startswith((".", "_")):
                continue
            summary = self._summarize_project(entry, owner_id)
            if summary is not None:
                results.append(summary)

        # Sort by latest update first
        results.sort(key=lambda item: item.updated_at, reverse=True)
        return results

    def get_project(self, project_name: str, owner_id: str) -> ProjectDetail:
        """Get full details for one project in `prj/`."""
        project_dir = self._validate_project_dir(project_name)
        return self._detail_project(project_dir, owner_id)

    def open_project(self, project_name: str, owner_id: str) -> Job:
        """Open/import a ready project from `prj/` as an active Job."""
        if self._job_repository is None or self._media_store is None:
            raise ProjectValidationError("Job repository and media store are required to open projects.")

        project_dir = self._validate_project_dir(project_name)
        job_id = self.job_id_for_project(project_name)

        # Check if already opened and valid
        try:
            existing_job = self._job_repository.get(owner_id, job_id)
            if existing_job and existing_job.status == "succeeded" and existing_job.files:
                return existing_job
        except Exception:
            pass

        # Build configuration and register files
        config = self._build_project_config(project_dir)
        registered_files = self._register_project_files(owner_id, job_id, project_dir, config)

        # Create or update Job in repository
        self._job_repository.create(
            owner_id,
            config,
            job_id=job_id,
            state={"status": "succeeded", "message": f"Loaded ready project '{project_name}'"},
            files=registered_files,
        )
        self._job_repository.update(
            owner_id,
            job_id,
            status="succeeded",
            state={"status": "succeeded", "message": f"Loaded ready project '{project_name}'"},
            files=registered_files,
        )
        return self._job_repository.get(owner_id, job_id)

    def run_project_step(
        self,
        project_name: str,
        owner_id: str,
        *,
        run_step: str = "tts_to_end",
        overrides: dict[str, Any] | None = None,
    ) -> Job:
        """Run or resume a dubbing pipeline step on an existing project folder."""
        if self._job_repository is None or self._media_store is None:
            raise ProjectValidationError("Job repository and media store are required to run projects.")

        project_dir = self._validate_project_dir(project_name)
        config = self._build_project_config(project_dir)
        config["run_step"] = run_step
        if overrides:
            config.update(overrides)

        job_id = str(uuid.uuid4())
        registered_files = self._register_project_files(owner_id, job_id, project_dir, config)

        self._job_repository.create(
            owner_id,
            config,
            job_id=job_id,
            state={"status": "queued", "message": f"Queued {run_step} for project '{project_name}'"},
            files=registered_files,
        )
        if self._queue is not None:
            try:
                self._queue.enqueue(job_id)
            except RuntimeError:
                pass
        return self._job_repository.get(owner_id, job_id)

    @staticmethod
    def job_id_for_project(project_name: str) -> str:
        """Deterministic UUID based on project name for persistent job linkage."""
        return str(uuid.uuid5(NAMESPACE_DUBBLM_PRJ, project_name.strip()))

    def _validate_project_dir(self, project_name: str) -> Path:
        clean_name = str(project_name or "").strip()
        if not clean_name or ".." in clean_name or "/" in clean_name or "\\" in clean_name:
            raise ProjectValidationError(f"Invalid project name: {project_name!r}.")
        project_dir = (self._projects_root / clean_name).resolve()
        if not str(project_dir).startswith(str(self._projects_root)):
            raise ProjectValidationError("Project directory must stay within projects root.")
        if not project_dir.is_dir():
            raise ProjectNotFoundError(f"Project '{clean_name}' not found at {project_dir}.")
        return project_dir

    def _summarize_project(self, project_dir: Path, owner_id: str) -> ProjectSummary | None:
        video_files: list[str] = []
        audio_files: list[str] = []
        subtitle_files: list[str] = []

        try:
            for item in project_dir.iterdir():
                if item.is_file():
                    ext = item.suffix.lower()
                    if ext in VIDEO_EXTENSIONS:
                        video_files.append(item.name)
                    elif ext in AUDIO_EXTENSIONS:
                        audio_files.append(item.name)
                    elif ext in SUBTITLE_EXTENSIONS:
                        subtitle_files.append(item.name)
            
            artifacts_dir_tmp = project_dir / "artifacts"
            if artifacts_dir_tmp.is_dir():
                for item in artifacts_dir_tmp.iterdir():
                    if item.is_file() and item.suffix.lower() in VIDEO_EXTENSIONS:
                        video_files.append(f"artifacts/{item.name}")
        except OSError:
            return None

        artifacts_dir = project_dir / "artifacts"
        metadata = self._read_project_metadata(artifacts_dir)
        has_artifacts = artifacts_dir.is_dir()
        transcription_file = artifacts_dir / "transcription.txt"
        dubbing_texts_file = artifacts_dir / "dubbing_texts.tsv"
        translations_file = artifacts_dir / "debug" / "translations.tsv"

        has_transcription = (
            transcription_file.is_file()
            or dubbing_texts_file.is_file()
            or translations_file.is_file()
        )

        # Count segments
        segment_count = 0
        if dubbing_texts_file.is_file():
            try:
                lines = [line for line in dubbing_texts_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
                segment_count = max(0, len(lines) - 1)
            except Exception:
                pass
        elif translations_file.is_file():
            try:
                lines = [line for line in translations_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
                segment_count = max(0, len(lines) - 1)
            except Exception:
                pass
        elif transcription_file.is_file():
            try:
                lines = [line for line in transcription_file.read_text(encoding="utf-8", errors="ignore").splitlines() if line.strip()]
                segment_count = len(lines)
            except Exception:
                pass
        elif subtitle_files:
            srt_path = project_dir / subtitle_files[0]
            try:
                content = srt_path.read_text(encoding="utf-8", errors="ignore")
                blocks = re.findall(r"\d+\s+\d{2}:\d{2}:\d{2}", content)
                segment_count = len(blocks)
            except Exception:
                pass

        stat = project_dir.stat()
        created_at = datetime.fromtimestamp(stat.st_ctime, tz=timezone.utc).isoformat()
        updated_at = datetime.fromtimestamp(stat.st_mtime, tz=timezone.utc).isoformat()

        job_id = self.job_id_for_project(project_dir.name)
        existing_job_id: str | None = None
        if self._job_repository is not None:
            try:
                job = self._job_repository.get(owner_id, job_id)
                if job:
                    existing_job_id = job.id
            except Exception:
                pass

        # Calculate relative path
        try:
            rel_path = str(project_dir.relative_to(Path.cwd()))
        except ValueError:
            rel_path = str(project_dir)

        return ProjectSummary(
            name=project_dir.name,
            display_name=str(metadata.get("project_name") or project_dir.name),
            relative_path=rel_path,
            created_at=created_at,
            updated_at=updated_at,
            has_video=bool(video_files),
            has_subtitles=bool(subtitle_files),
            has_artifacts=has_artifacts,
            has_transcription=has_transcription,
            segment_count=segment_count,
            video_files=video_files,
            audio_files=audio_files,
            job_id=existing_job_id,
        )

    def _detail_project(self, project_dir: Path, owner_id: str) -> ProjectDetail:
        summary = self._summarize_project(project_dir, owner_id)
        if summary is None:
            raise ProjectNotFoundError(f"Project '{project_dir.name}' could not be inspected.")

        video_files_meta: list[dict[str, Any]] = []
        audio_files_meta: list[dict[str, Any]] = []
        subtitle_files_meta: list[dict[str, Any]] = []

        for item in sorted(project_dir.iterdir()):
            if not item.is_file():
                continue
            ext = item.suffix.lower()
            stat = item.stat()
            meta = {"name": item.name, "size": stat.st_size, "path": str(item)}
            if ext in VIDEO_EXTENSIONS:
                video_files_meta.append(meta)
            elif ext in AUDIO_EXTENSIONS:
                audio_files_meta.append(meta)
            elif ext in SUBTITLE_EXTENSIONS:
                subtitle_files_meta.append(meta)

        artifacts_dir = project_dir / "artifacts"
        metadata = self._read_project_metadata(artifacts_dir)
        artifacts_info: dict[str, Any] = {}
        if artifacts_dir.is_dir():
            for art_item in sorted(artifacts_dir.rglob("*")):
                if art_item.is_file():
                    try:
                        rel = str(art_item.relative_to(artifacts_dir))
                    except ValueError:
                        rel = art_item.name
                    artifacts_info[rel] = {
                        "size": art_item.stat().st_size,
                        "name": art_item.name,
                    }

        target_lang = "ru"
        # Detect target language from video filename e.g. videoplayback6_ru.mp4
        for vf in summary.video_files:
            match = re.search(r"_([a-z]{2})\.(mp4|webm|mov|mkv)$", vf, re.IGNORECASE)
            if match:
                target_lang = match.group(1).lower()
                break

        return ProjectDetail(
            name=project_dir.name,
            display_name=summary.display_name,
            relative_path=summary.relative_path,
            created_at=summary.created_at,
            updated_at=summary.updated_at,
            video_files=video_files_meta,
            audio_files=audio_files_meta,
            subtitle_files=subtitle_files_meta,
            artifacts=artifacts_info,
            target_language=target_lang,
            source_language="en",
            saved_config=metadata.get("config") if isinstance(metadata.get("config"), dict) else None,
            job_id=summary.job_id,
        )

    @staticmethod
    def _read_project_metadata(artifacts_dir: Path) -> dict[str, Any]:
        try:
            metadata = json.loads((artifacts_dir / "project_metadata.json").read_text(encoding="utf-8"))
            return metadata if isinstance(metadata, dict) else {}
        except (OSError, ValueError, TypeError, json.JSONDecodeError):
            return {}

    def _build_project_config(self, project_dir: Path) -> dict[str, Any]:
        artifacts_dir = project_dir / "artifacts"
        audio_dir = artifacts_dir / "audio"

        video_files = [
            f for f in project_dir.iterdir() if f.is_file() and f.suffix.lower() in VIDEO_EXTENSIONS
        ]
        # Prefer output video like `_ru.mp4` or first video
        output_video = next((f for f in video_files if re.search(r"_[a-z]{2}\.", f.name, re.IGNORECASE)), None)
        if output_video is None and video_files:
            output_video = video_files[0]

        input_video = next((f for f in video_files if f != output_video), output_video)
        input_path = str(input_video) if input_video is not None else str(audio_dir / "source.wav")

        target_lang = "ru"
        if output_video:
            match = re.search(r"_([a-z]{2})\.", output_video.name, re.IGNORECASE)
            if match:
                target_lang = match.group(1).lower()

        config: dict[str, Any] = {
            "input": input_path,
            "project_dir": str(project_dir),
            "artifacts_dir": str(artifacts_dir),
            "audio_artifacts_dir": str(audio_dir),
            "speakers_audio_dir": str(artifacts_dir / "speakers_audio"),
            "audio_chunks_dir": str(artifacts_dir / "audio_chunks"),
            "su_audio_chunks_dir": str(artifacts_dir / "su_audio_chunks"),
            "debug_dir": str(artifacts_dir / "debug"),
            "translation_debug_dir": str(artifacts_dir / "debug" / "translation"),
            "translation_refinement_debug_dir": str(artifacts_dir / "debug" / "translation_refinement"),
            "transcription_path": str(artifacts_dir / "transcription.txt"),
            "timecodes_report_path": str(artifacts_dir / "timecodes.txt"),
            "translated_audio_path": str(audio_dir / "output.wav"),
            "background_audio_path": str(audio_dir / "background.wav"),
            "source_language": "en",
            "target_language": target_lang,
            "config": self._config_path,
        }
        if output_video is not None:
            config["output"] = str(output_video)

        metadata = self._read_project_metadata(artifacts_dir)
        saved_cfg = metadata.get("config")
        if isinstance(saved_cfg, dict):
            for k, v in saved_cfg.items():
                if k not in ("run_step", "output", "project_dir", "artifacts_dir") and v is not None:
                    config[k] = v
        return config

    def _register_project_files(
        self, owner_id: str, job_id: str, project_dir: Path, config: dict[str, Any]
    ) -> list[dict[str, Any]]:
        candidates: list[tuple[Path, str]] = []

        # Videos in project root
        for item in project_dir.iterdir():
            if not item.is_file():
                continue
            ext = item.suffix.lower()
            if ext in VIDEO_EXTENSIONS:
                if re.search(r"_[a-z]{2}\.", item.name, re.IGNORECASE):
                    candidates.append((item, "output_video"))
                else:
                    candidates.append((item, "video"))
            elif ext in SUBTITLE_EXTENSIONS:
                candidates.append((item, "subtitles"))

        # Videos in artifacts
        artifacts_dir = project_dir / "artifacts"
        if artifacts_dir.is_dir():
            for item in artifacts_dir.iterdir():
                if not item.is_file():
                    continue
                ext = item.suffix.lower()
                if ext in VIDEO_EXTENSIONS:
                    if re.search(r"_[a-z]{2}\.", item.name, re.IGNORECASE) or item.name == "output_video.mp4":
                        candidates.append((item, "output_video"))
                    else:
                        candidates.append((item, "video"))

        # Audio artifacts
        artifacts_dir = project_dir / "artifacts"
        audio_dir = artifacts_dir / "audio"
        if audio_dir.is_dir():
            for audio_file in audio_dir.iterdir():
                if not audio_file.is_file() or audio_file.suffix.lower() not in AUDIO_EXTENSIONS:
                    continue
                name_lower = audio_file.stem.lower()
                if "background" in name_lower or "music" in name_lower:
                    candidates.append((audio_file, "background_audio"))
                elif "output" in name_lower:
                    candidates.append((audio_file, "output_audio"))
                elif "vocal" in name_lower:
                    candidates.append((audio_file, "audio"))
                elif "source" in name_lower:
                    candidates.append((audio_file, "audio"))

        # Other artifacts (timecodes, transcription, etc.)
        if artifacts_dir.is_dir():
            timecodes = artifacts_dir / "timecodes.txt"
            if timecodes.is_file():
                candidates.append((timecodes, "report"))
            transcription = artifacts_dir / "transcription.txt"
            if transcription.is_file():
                candidates.append((transcription, "artifact"))
            dubbing_texts = artifacts_dir / "dubbing_texts.tsv"
            if dubbing_texts.is_file():
                candidates.append((dubbing_texts, "artifact"))

        registered: list[dict[str, Any]] = []
        registered_paths: set[Path] = set()

        for path, kind in candidates:
            resolved = path.resolve()
            if not path.is_file() or resolved in registered_paths:
                continue
            record = self._media_store.register(owner_id, path=resolved, name=path.name, kind=kind)
            registered_paths.add(resolved)
            registered.append(
                {
                    "id": str(record.id),
                    "name": str(record.name),
                    "kind": str(record.kind),
                    "size": int(record.size),
                }
            )

        return registered
