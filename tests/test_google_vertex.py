import os
import pytest
from pathlib import Path


def test_vertex_ai_settings_require_explicit_vertex_flag(monkeypatch):
    from google_vertex import get_vertex_ai_settings

    monkeypatch.delenv("GOOGLE_GENAI_USE_VERTEXAI", raising=False)
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    with pytest.raises(ValueError, match="GOOGLE_GENAI_USE_VERTEXAI"):
        get_vertex_ai_settings()


def test_vertex_ai_settings_build_genai_and_llamaindex_configs(monkeypatch):
    from google_vertex import get_vertex_ai_settings

    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")

    settings = get_vertex_ai_settings()

    assert settings.project == "demo-project"
    assert settings.location == "global"
    assert settings.genai_client_kwargs == {
        "vertexai": True,
        "project": "demo-project",
        "location": "global",
    }
    assert settings.llamaindex_vertexai_config == {
        "project": "demo-project",
        "location": "global",
    }


def test_vertex_ai_settings_resolves_relative_credentials_from_parent_workspace(monkeypatch, tmp_path):
    import google_vertex
    from google_vertex import get_vertex_ai_settings

    workspace = tmp_path / "workspace"
    worktree = workspace / ".worktrees" / "feature"
    source_file = worktree / "src" / "google_vertex.py"
    credential_file = workspace / "cred" / "service-account.json"
    source_file.parent.mkdir(parents=True)
    credential_file.parent.mkdir(parents=True)
    source_file.touch()
    credential_file.write_text("{}", encoding="utf-8")

    monkeypatch.chdir(worktree)
    monkeypatch.setattr(google_vertex, "__file__", str(source_file))
    monkeypatch.setenv("GOOGLE_GENAI_USE_VERTEXAI", "true")
    monkeypatch.setenv("GOOGLE_CLOUD_PROJECT", "demo-project")
    monkeypatch.setenv("GOOGLE_CLOUD_LOCATION", "global")
    monkeypatch.setenv("GOOGLE_APPLICATION_CREDENTIALS", "cred/service-account.json")

    get_vertex_ai_settings()

    assert Path(os.environ["GOOGLE_APPLICATION_CREDENTIALS"]) == credential_file


def test_env_example_lists_vertex_ai_variables():
    env_example = (Path(__file__).resolve().parents[1] / ".env.example").read_text(encoding="utf-8")

    assert "GOOGLE_GENAI_USE_VERTEXAI=" in env_example
    assert "GOOGLE_CLOUD_PROJECT=" in env_example
    assert "GOOGLE_CLOUD_LOCATION=" in env_example
    assert "GOOGLE_API_KEY=" not in env_example


def test_readme_documents_vertex_ai_setup():
    readme = (Path(__file__).resolve().parents[1] / "README.md").read_text(encoding="utf-8")

    assert "gcloud auth application-default login" in readme
    assert "GOOGLE_GENAI_USE_VERTEXAI=true" in readme
    assert "GOOGLE_CLOUD_PROJECT" in readme
    assert "GOOGLE_CLOUD_LOCATION" in readme
    assert "GOOGLE_API_KEY" not in readme
