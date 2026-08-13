from pathlib import Path

import pytest


PROJECT_ROOT = Path(__file__).resolve().parents[1]


def test_cli_files_and_packaged_entry_point_are_removed():
    assert not (PROJECT_ROOT / "dubblm_cli.py").exists()
    assert not (PROJECT_ROOT / "src" / "dubbing" / "cli" / "main.py").exists()
    pyproject = (PROJECT_ROOT / "pyproject.toml").read_text(encoding="utf-8")
    assert 'dubblm = "dubbing.cli.main:main"' not in pyproject


def test_cli_only_config_construction_is_not_importable():
    from dubbing.core import config

    assert not hasattr(config.DubbingConfig, "load_from_cli")
    assert not hasattr(config.DubbingConfig, "_create_parser")
    assert not hasattr(config.DubbingConfig, "_removed_create_parser")
    assert not hasattr(config, "create_argument_parser")
    assert not hasattr(config, "create_config_from_args")
    assert not hasattr(config, "_removed_create_argument_parser")
    assert not hasattr(config, "_removed_create_config_from_args")

    with pytest.raises(ModuleNotFoundError, match=r"dubbing\.cli"):
        __import__("dubbing.cli.main")
