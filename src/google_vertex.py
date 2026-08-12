from dataclasses import dataclass
import os
from pathlib import Path


def _is_truthy(value: str | None) -> bool:
    return str(value or "").strip().lower() in {"1", "true", "yes", "on"}


def _resolve_application_credentials() -> None:
    """Make a relative ADC path work when the app runs from a git worktree."""
    configured_path = os.environ.get("GOOGLE_APPLICATION_CREDENTIALS")
    if not configured_path:
        return

    credential_path = Path(configured_path).expanduser()
    if credential_path.is_absolute():
        return

    search_roots = [Path.cwd(), *Path(__file__).resolve().parents]
    for root in search_roots:
        candidate = (root / credential_path).resolve()
        if candidate.is_file():
            os.environ["GOOGLE_APPLICATION_CREDENTIALS"] = str(candidate)
            return


@dataclass(frozen=True)
class VertexAISettings:
    project: str
    location: str

    @property
    def genai_client_kwargs(self) -> dict[str, object]:
        return {
            "vertexai": True,
            "project": self.project,
            "location": self.location,
        }

    @property
    def llamaindex_vertexai_config(self) -> dict[str, str]:
        return {
            "project": self.project,
            "location": self.location,
        }


def get_vertex_ai_settings() -> VertexAISettings:
    _resolve_application_credentials()

    if not _is_truthy(os.environ.get("GOOGLE_GENAI_USE_VERTEXAI")):
        raise ValueError(
            "GOOGLE_GENAI_USE_VERTEXAI=true is required for Vertex AI Google integrations."
        )

    project = (os.environ.get("GOOGLE_CLOUD_PROJECT") or "").strip()
    if not project:
        raise ValueError("GOOGLE_CLOUD_PROJECT environment variable is required for Vertex AI.")

    location = (os.environ.get("GOOGLE_CLOUD_LOCATION") or "").strip()
    if not location:
        raise ValueError("GOOGLE_CLOUD_LOCATION environment variable is required for Vertex AI.")

    return VertexAISettings(project=project, location=location)
