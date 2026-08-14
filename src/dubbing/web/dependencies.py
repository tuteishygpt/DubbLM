"""FastAPI dependency accessors for application-scoped web services."""

from __future__ import annotations

from dataclasses import dataclass

from fastapi import Request


@dataclass(frozen=True)
class AnonymousCurrentUser:
    """The single local owner used until authentication is introduced."""

    id: str = "local"


def current_user(request: Request) -> AnonymousCurrentUser:
    return request.app.state.current_user


def service(name: str):
    def dependency(request: Request):
        return getattr(request.app.state, name)

    return dependency


get_settings = service("settings_service")
get_media_store = service("media_store")
get_job_repository = service("job_repository")
get_job_service = service("job_service")
get_reference_service = service("reference_service")
get_dubbing_text_service = service("dubbing_text_service")
get_project_service = service("project_service")
