from __future__ import annotations

from pathlib import Path

import pytest
from fastapi.testclient import TestClient

import dubbing.web.app as app_module


PROJECT_ROOT = Path(__file__).resolve().parents[2]


def _spa_build(tmp_path: Path) -> Path:
    dist = tmp_path / "dist"
    assets = dist / "assets"
    assets.mkdir(parents=True)
    (dist / "index.html").write_text(
        '<!doctype html><html><body><div id="root">DubbLM SPA</div></body></html>',
        encoding="utf-8",
    )
    (assets / "app.js").write_text("window.DubbLM = true", encoding="utf-8")
    return dist


def test_spa_serves_assets_and_falls_back_to_index_for_browser_routes(tmp_path):
    app = app_module.create_app(root=tmp_path / "data", static_dir=_spa_build(tmp_path))

    with TestClient(app) as client:
        asset = client.get("/assets/app.js")
        root = client.get("/")
        nested = client.get("/jobs/current")

    assert asset.status_code == 200
    assert asset.text == "window.DubbLM = true"
    assert root.status_code == nested.status_code == 200
    assert root.text == nested.text
    assert '<div id="root">DubbLM SPA</div>' in nested.text


@pytest.mark.parametrize("path", ["/api", "/api/not-a-real-endpoint"])
def test_unknown_api_paths_remain_json_404s_when_spa_is_enabled(tmp_path, path):
    app = app_module.create_app(root=tmp_path / "data", static_dir=_spa_build(tmp_path))

    with TestClient(app) as client:
        response = client.get(path)

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")
    assert response.json()["code"] == "not_found"
    assert "DubbLM SPA" not in response.text


def test_static_serving_can_be_disabled_for_injected_api_tests(tmp_path):
    app = app_module.create_app(root=tmp_path / "data", static_dir=None)

    with TestClient(app) as client:
        response = client.get("/browser-route")

    assert response.status_code == 404
    assert response.headers["content-type"].startswith("application/json")


def test_missing_spa_build_has_an_actionable_startup_error(tmp_path):
    missing = tmp_path / "missing-dist"

    with pytest.raises(FileNotFoundError, match=r"npm --prefix frontend run build"):
        app_module.create_app(root=tmp_path / "data", static_dir=missing)


def test_web_entry_point_runs_the_production_app_with_uvicorn(monkeypatch):
    production_app = object()
    call = {}
    monkeypatch.setattr(app_module, "create_production_app", lambda: production_app)
    monkeypatch.setattr(
        app_module.uvicorn,
        "run",
        lambda application, **options: call.update(application=application, options=options),
    )

    app_module.main()

    assert call == {
        "application": production_app,
        "options": {"host": "127.0.0.1", "port": 8000},
    }


def test_launchers_expose_the_web_entry_point():
    launcher = (PROJECT_ROOT / "web_app.py").read_text(encoding="utf-8")
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")

    assert "from dubbing.web.app import main" in launcher
    assert 'dubblm-web = "dubbing.web.app:main"' in pyproject
