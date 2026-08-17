from __future__ import annotations

from dataclasses import asdict, is_dataclass
from typing import Any


def public(value: Any) -> Any:
    """Convert service dataclasses without leaking private attributes."""
    if is_dataclass(value) and not isinstance(value, type):
        return public(asdict(value))
    if isinstance(value, dict):
        return {str(key): public(item) for key, item in value.items()}
    if isinstance(value, (list, tuple)):
        return [public(item) for item in value]
    return value


def job_public(job: object) -> dict[str, Any]:
    raw = public(job)
    config = raw.get("config") if isinstance(raw.get("config"), dict) else {}
    project_name = str(config.get("project_name") or "").strip()
    if not project_name and config.get("project_dir"):
        import os
        project_name = os.path.basename(str(config["project_dir"]))
    return {
        "id": raw["id"],
        "project_name": project_name or None,
        "state": raw["state"],
        "created_at": raw["created_at"],
        "started_at": raw["started_at"],
        "finished_at": raw["finished_at"],
        "status": raw["status"],
        "error": raw["error"],
        "files": raw["files"],
        "last_event_id": raw["last_event_id"],
    }
