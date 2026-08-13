"""FastAPI application factory and centralized domain error translation."""

from __future__ import annotations

import os
from contextlib import asynccontextmanager
from pathlib import Path

import uvicorn
from fastapi import FastAPI, Request
from fastapi.exceptions import RequestValidationError
from fastapi.responses import FileResponse, JSONResponse
from fastapi.staticfiles import StaticFiles

from .dependencies import AnonymousCurrentUser
from .dubbing_texts import DubbingTextConflictError, DubbingTextError, DubbingTextNotFoundError, DubbingTextValidationError, DubbingTextWriteError, DubbingTextService
from .jobs import FileJobRepository, JobNotFoundError, JobRepositoryError, JobService, JobValidationError, JobWriteError
from .queue import InProcessJobQueue
from .references import ReferenceConflictError, ReferenceError, ReferenceLibraryService, ReferenceNotFoundError, ReferenceValidationError, ReferenceWriteError
from .settings import SettingsConflictError, SettingsError, SettingsService, SettingsValidationError, SettingsWriteError
from .storage import DEFAULT_MAX_UPLOAD_BYTES, FileMediaStore, MediaConflictError, MediaNotFoundError, MediaStoreError, MediaValidationError, MediaWriteError
from .routes import config, dubbing_texts, jobs, references, uploads, voices


NOT_FOUND = (JobNotFoundError, MediaNotFoundError, ReferenceNotFoundError, DubbingTextNotFoundError)
CONFLICT = (SettingsConflictError, ReferenceConflictError, DubbingTextConflictError, MediaConflictError)
VALIDATION = (SettingsValidationError, ReferenceValidationError, DubbingTextValidationError, JobValidationError, MediaValidationError)
WRITE = (SettingsWriteError, ReferenceWriteError, DubbingTextWriteError, JobWriteError, MediaWriteError)


def _error(code: str, message: str, status_code: int, **extra):
    optional = {key: value for key, value in extra.items() if value is not None}
    return JSONResponse(status_code=status_code, content={"code": code, "message": message, **optional})


def create_app(
    *,
    root: str | Path | None = None,
    config_path: str | Path | None = None,
    settings_service=None,
    media_store=None,
    job_repository=None,
    job_service=None,
    job_queue=None,
    reference_service=None,
    dubbing_text_service=None,
    current_user=None,
    heartbeat_interval: float = 15.0,
    sse_poll_interval: float = 0.25,
    static_dir: str | Path | None = None,
) -> FastAPI:
    if heartbeat_interval <= 0 or sse_poll_interval <= 0:
        raise ValueError("SSE intervals must be positive.")
    data_root = Path(root or "server_data")
    owner = current_user or AnonymousCurrentUser()
    settings_service = settings_service or SettingsService(config_path or "dubbing_config.yml")
    media_store = media_store or FileMediaStore(
        data_root,
        max_upload_bytes=int(os.environ.get("DUBBLM_MAX_UPLOAD_BYTES", DEFAULT_MAX_UPLOAD_BYTES)),
    )
    job_repository = job_repository or FileJobRepository(data_root)
    job_queue = job_queue or InProcessJobQueue(job_repository, media_store, owner_id=owner.id)
    job_service = job_service or JobService(
        job_repository,
        media_store,
        job_queue,
        config_path=config_path or "dubbing_config.yml",
    )
    reference_service = reference_service or ReferenceLibraryService(data_root / "references", media_store)
    dubbing_text_service = dubbing_text_service or DubbingTextService(media_store, job_repository=job_repository)

    @asynccontextmanager
    async def lifespan(app: FastAPI):
        job_queue.start()
        try:
            yield
        finally:
            job_queue.stop()

    app = FastAPI(lifespan=lifespan)
    app.state.current_user = owner
    app.state.settings_service = settings_service
    app.state.media_store = media_store
    app.state.job_repository = job_repository
    app.state.job_service = job_service
    app.state.job_queue = job_queue
    app.state.reference_service = reference_service
    app.state.dubbing_text_service = dubbing_text_service
    app.state.heartbeat_interval = heartbeat_interval
    app.state.sse_poll_interval = sse_poll_interval

    @app.exception_handler(RequestValidationError)
    async def request_validation(_request: Request, exc: RequestValidationError):
        errors = exc.errors()
        field = ".".join(str(part) for part in errors[0].get("loc", ())[1:]) or None
        return _error("validation_error", "Request validation failed.", 422, field=field, details=errors)

    async def domain_error(_request: Request, exc: Exception):
        if isinstance(exc, NOT_FOUND):
            return _error("not_found", str(exc), 404)
        if isinstance(exc, CONFLICT):
            return _error("conflict", str(exc), 409)
        if isinstance(exc, MediaValidationError) and "maximum size" in str(exc).lower():
            return _error("upload_too_large", str(exc), 413)
        if isinstance(exc, VALIDATION):
            return _error("validation_error", str(exc), 400)
        if isinstance(exc, WRITE):
            return _error("write_error", str(exc), 500)
        return _error("internal_error", "Internal server error.", 500)

    for error_type in (SettingsError, ReferenceError, DubbingTextError, JobRepositoryError, MediaStoreError):
        app.add_exception_handler(error_type, domain_error)

    for router in (config.router, uploads.router, jobs.router, voices.router, references.router, dubbing_texts.router):
        app.include_router(router)

    if static_dir is not None:
        spa_root = Path(static_dir)
        index_path = spa_root / "index.html"
        if not index_path.is_file():
            raise FileNotFoundError(
                f"DubbLM SPA build not found at {spa_root}. "
                "Run `npm --prefix frontend run build` before starting the web app."
            )

        assets_path = spa_root / "assets"
        if assets_path.is_dir():
            app.mount("/assets", StaticFiles(directory=assets_path), name="spa-assets")

        @app.api_route(
            "/api",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
            include_in_schema=False,
        )
        @app.api_route(
            "/api/{path:path}",
            methods=["GET", "POST", "PUT", "PATCH", "DELETE", "OPTIONS", "HEAD"],
            include_in_schema=False,
        )
        async def unknown_api(path: str = ""):
            return _error("not_found", "API endpoint not found.", 404)

        @app.get("/{path:path}", include_in_schema=False)
        async def spa_fallback(path: str):
            return FileResponse(index_path)

    return app


def create_production_app() -> FastAPI:
    """Create the local production server with the compiled React SPA."""
    project_root = Path(__file__).resolve().parents[3]
    return create_app(static_dir=project_root / "frontend" / "dist")


def main() -> None:
    """Launch the local DubbLM web application."""
    uvicorn.run(create_production_app(), host="127.0.0.1", port=8000)
