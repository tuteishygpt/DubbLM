"""Replaceable service boundaries used by later web adapters."""

from __future__ import annotations

from typing import Protocol


class CurrentUser(Protocol):
    """The currently authenticated owner boundary."""

    @property
    def id(self) -> str: ...


class MediaStore(Protocol):
    """Opaque media storage boundary."""

    def save(self, *args: object, **kwargs: object) -> object: ...

    def get(self, *args: object, **kwargs: object) -> object: ...

    def delete(self, *args: object, **kwargs: object) -> object: ...


class JobRepository(Protocol):
    """Persisted jobs boundary."""

    def create(self, *args: object, **kwargs: object) -> object: ...

    def get(self, *args: object, **kwargs: object) -> object: ...

    def list(self, *args: object, **kwargs: object) -> object: ...

    def update(self, *args: object, **kwargs: object) -> object: ...

    def append_event(self, *args: object, **kwargs: object) -> object: ...


class JobQueue(Protocol):
    """Queue lifecycle boundary."""

    def enqueue(self, job_id: str) -> None: ...

    def start(self) -> None: ...

    def stop(self) -> None: ...
