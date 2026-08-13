"""Client-pool and routing tests for SmartDubbing's per-voice profiles."""

from unittest.mock import patch

import dubbing.core.config as config_module
import dubbing.core.smart_dubbing as smart_dubbing_module
from dubbing.core.runner import build_config_from_overrides
from dubbing.core.smart_dubbing import SmartDubbing
from dubbing.core.voice_profiles import VoiceProfile
from tts.models import TTSSegmentData


class _StubTTSClient:
    """Minimal TTS client shim used to observe factory calls without touching real backends."""

    def __init__(self, *, tts_system, model=None, **kwargs):
        self.tts_system = tts_system
        self.model = model
        self.kwargs = kwargs

    def cleanup(self):
        pass


def _patch_projects_root(monkeypatch, tmp_path):
    projects_root = tmp_path / "prj"
    monkeypatch.setattr(config_module, "DEFAULT_PROJECTS_ROOT", projects_root, raising=False)
    return projects_root


def _skip_optional_init(dubber):
    """Bypass translator/transcriber init so tests focus on TTS wiring."""
    dubber._initialize_translator = lambda: setattr(dubber, "translator", None) or setattr(dubber, "translator_init_error", None)
    dubber._initialize_transcriber = lambda: setattr(dubber, "transcriber", None) or setattr(dubber, "transcriber_init_error", None)


def _install_stub_factory(monkeypatch, log):
    def fake_create_tts(*, tts_system, **kwargs):
        client = _StubTTSClient(
            tts_system=tts_system,
            model=kwargs.get("model"),
            **{k: v for k, v in kwargs.items() if k != "model"},
        )
        log.append(client)
        return client

    monkeypatch.setattr(
        smart_dubbing_module.TTSFactory,
        "create_tts",
        staticmethod(fake_create_tts),
    )


def test_pool_key_creates_one_client_per_unique_profile(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "voices": {
                "SPEAKER_00": {"tts_system": "gemini", "model": "flash"},
                "SPEAKER_01": {"tts_system": "gemini", "model": "flash"},   # same pool_key as SPEAKER_00
                "SPEAKER_02": {"tts_system": "openai", "model": "tts-1"},   # different backend
                "SPEAKER_03": {"tts_system": "gemini", "model": "pro"},     # same backend, different model
                "*": {"tts_system": "omnivoice"},
            },
        }
    )

    created = []
    _install_stub_factory(monkeypatch, created)

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    _skip_optional_init(dubber)

    with patch.object(SmartDubbing, "__init__", SmartDubbing.__init__):
        dubber.__init__(config)

    # Expect 4 unique clients: (gemini,flash), (openai,tts-1), (gemini,pro), (omnivoice,None)
    systems_by_pool = {(c.tts_system, c.model) for c in created}
    assert systems_by_pool == {
        ("gemini", "flash"),
        ("openai", "tts-1"),
        ("gemini", "pro"),
        ("omnivoice", None),
    }

    # Both SPEAKER_00 and SPEAKER_01 must map to the same pool_key
    key_00 = dubber._profile_pool_key(dubber._resolve_voice_profile("SPEAKER_00"))
    key_01 = dubber._profile_pool_key(dubber._resolve_voice_profile("SPEAKER_01"))
    key_02 = dubber._profile_pool_key(dubber._resolve_voice_profile("SPEAKER_02"))
    key_03 = dubber._profile_pool_key(dubber._resolve_voice_profile("SPEAKER_03"))
    assert key_00 == key_01
    assert key_00 != key_02
    assert key_00 != key_03
    assert dubber.tts_clients[key_00] is dubber.tts_clients[key_01]
    assert dubber.tts_clients[key_02] is not dubber.tts_clients[key_00]


def test_default_tts_uses_star_fallback_client(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "voices": {
                "SPEAKER_00": {"tts_system": "gemini"},
                "*": {"tts_system": "omnivoice"},
            },
        }
    )

    created = []
    _install_stub_factory(monkeypatch, created)
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    _skip_optional_init(dubber)
    dubber.__init__(config)

    assert dubber.default_tts.tts_system == "omnivoice"


def test_voice_matching_is_disabled_for_manual_voice_or_global_off(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "voice_auto_selection": True,
            "voices": {"SPEAKER_00": {"tts_system": "gemini", "voice_name": "Kore"}},
        }
    )
    created = []
    _install_stub_factory(monkeypatch, created)
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    _skip_optional_init(dubber)
    dubber.__init__(config)

    assert created[0].kwargs["enable_voice_matching"] is False

    config.voice_auto_selection = False
    created.clear()
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    _skip_optional_init(dubber)
    dubber.__init__(config)
    assert created[0].kwargs["enable_voice_matching"] is False


def test_unmapped_speaker_falls_through_to_star_profile(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    config = build_config_from_overrides(
        {
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "voices": {
                "*": {"tts_system": "omnivoice", "voice_name": "default"},
            },
        }
    )

    _install_stub_factory(monkeypatch, [])
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    _skip_optional_init(dubber)
    dubber.__init__(config)

    profile = dubber._resolve_voice_profile("SPEAKER_99")
    assert profile.tts_system == "omnivoice"
    assert profile.voice_name == "default"


def test_legacy_tts_system_mapping_still_routes(tmp_path, monkeypatch):
    """Existing configs using only tts_system_mapping keep working end-to-end."""
    import warnings

    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)

    with warnings.catch_warnings():
        warnings.simplefilter("ignore", DeprecationWarning)
        config = build_config_from_overrides(
            {
                "config": "",  # don't inherit repo dubbing_config.yml
                "input": str(video_path),
                "source_language": "en",
                "target_language": "be",
                "tts_system": "coqui",
                "tts_system_mapping": '{"SPEAKER_00": "gemini", "SPEAKER_01": "openai"}',
                "voice_prompt": '{"SPEAKER_00": "calm"}',
            }
        )

    _install_stub_factory(monkeypatch, [])
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    _skip_optional_init(dubber)
    dubber.__init__(config)

    assert dubber._get_tts_system_for_speaker("SPEAKER_00") == "gemini"
    assert dubber._get_tts_system_for_speaker("SPEAKER_01") == "openai"

    profile = dubber._resolve_voice_profile("SPEAKER_00")
    assert profile.style_prompt == "calm"


def test_pool_key_stable_across_param_ordering():
    a = VoiceProfile(tts_system="gemini", model="flash", params={"a": 1, "b": 2})
    b = VoiceProfile(tts_system="gemini", model="flash", params={"b": 2, "a": 1})
    assert a.pool_key() == b.pool_key()


def test_tts_factory_call_contains_only_selected_model(tmp_path, monkeypatch):
    video_path = tmp_path / "clip.mp4"
    video_path.write_bytes(b"video")
    _patch_projects_root(monkeypatch, tmp_path)
    config = build_config_from_overrides(
        {
            "config": "",
            "input": str(video_path),
            "source_language": "en",
            "target_language": "be",
            "voices": {
                "*": {
                    "tts_system": "gemini",
                    "model": "selected-model",
                    "fallback_model": "ignored-model",
                }
            },
            "tts_fallback_model": "also-ignored",
        }
    )
    calls = []

    def fake_create_tts(*, tts_system, **kwargs):
        calls.append((tts_system, kwargs))
        return _StubTTSClient(tts_system=tts_system, model=kwargs.get("model"))

    monkeypatch.setattr(
        smart_dubbing_module.TTSFactory,
        "create_tts",
        staticmethod(fake_create_tts),
    )
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = config
    _skip_optional_init(dubber)
    dubber.__init__(config)

    assert calls[0][1]["model"] == "selected-model"
    assert "fallback_model" not in calls[0][1]


def test_preflight_collects_all_pool_errors_before_any_synthesis():
    calls = []

    class Client:
        def __init__(self, issue):
            self.issue = issue

        def validate_segments(self, segments):
            calls.append(("validate", segments[0].segment_index))
            return [self.issue]

        def synthesize(self, *args, **kwargs):
            calls.append(("synthesize", None))

    dubber = SmartDubbing.__new__(SmartDubbing)
    clients = {
        ("a",): Client((8, "provider=a speaker=S2 segment=8 mode=speaker: missing a")),
        ("b",): Client((3, "provider=b speaker=S1 segment=3 mode=configured: missing b")),
    }
    pools = {
        ("a",): [TTSSegmentData(speaker="S2", text="x", segment_index=8)],
        ("b",): [TTSSegmentData(speaker="S1", text="y", segment_index=3)],
    }

    try:
        dubber._preflight_tts_pools(pools, clients)
    except ValueError as exc:
        message = str(exc)
    else:
        raise AssertionError("preflight errors must abort the TTS stage")

    assert calls == [("validate", 8), ("validate", 3)]
    assert message.index("segment=3") < message.index("segment=8")
    assert not any(call[0] == "synthesize" for call in calls)
