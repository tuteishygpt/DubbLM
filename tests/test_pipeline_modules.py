import inspect

import pytest

from dubbing.core.pipeline import transcription
from dubbing.core.smart_dubbing import SmartDubbing


TRANSCRIPTION_METHODS = {
    "_initialize_transcriber": "initialize_transcriber",
    "_require_transcriber": "require_transcriber",
    "_load_cached_diarize_and_transcribe": "load_cached_diarize_and_transcribe",
    "diarize_and_transcribe": "diarize_and_transcribe",
    "_semantic_classifier": "semantic_classifier",
    "_write_semantic_boundary_diagnostics": "write_semantic_boundary_diagnostics",
    "_diarize_and_transcribe_isolated": "diarize_and_transcribe_isolated",
    "_isolated_inner_kwargs": "isolated_inner_kwargs",
    "_restore_semantic_plan_fingerprint": "restore_semantic_plan_fingerprint",
}


def test_transcription_service_exports_every_owned_internal_callable():
    for helper_name in TRANSCRIPTION_METHODS.values():
        assert callable(getattr(transcription, helper_name, None)), helper_name


@pytest.mark.parametrize(
    ("facade_name", "helper_name"), TRANSCRIPTION_METHODS.items()
)
def test_every_owned_transcription_facade_method_is_a_thin_delegate(
    facade_name, helper_name
):
    source = inspect.getsource(SmartDubbing.__dict__[facade_name])

    assert f"transcription_helpers.{helper_name}(" in source
    assert len(source.splitlines()) <= 16


def test_transcription_facade_delegates_with_live_facade_and_arguments(monkeypatch):
    dubber = SmartDubbing.__new__(SmartDubbing)
    sentinel = object()
    calls = []

    def replacement(facade, inner_system):
        calls.append((facade, inner_system))
        return sentinel

    monkeypatch.setattr(transcription, "isolated_inner_kwargs", replacement)

    assert dubber._isolated_inner_kwargs("deepgram") is sentinel
    assert calls == [(dubber, "deepgram")]


def test_direct_transcription_helper_uses_live_config_at_call_time():
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"deepgram_model": "first"}

    assert transcription.isolated_inner_kwargs(dubber, "deepgram") == {
        "deepgram_model": "first"
    }

    dubber.config = {"deepgram_model": "patched"}
    assert transcription.isolated_inner_kwargs(dubber, "deepgram") == {
        "deepgram_model": "patched"
    }


def test_restore_semantic_plan_propagates_fingerprint_to_context_and_facade():
    segments = [
        {"semantic_plan_fingerprint": "plan-live"},
        {"semantic_plan_fingerprint": "plan-live"},
    ]

    class Cache:
        def cache_exists(self, step, key):
            return (step, key) == ("isolated_tracks_semantic_plan", "plan-key")

        def load_from_cache(self, step, key):
            return {"transcription": segments}

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "isolated_tracks": {"SPEAKER_00": "track.wav"},
        "semantic_split_enabled": True,
    }
    dubber.cache_manager = Cache()
    dubber._isolated_tracks_cache_key = lambda *_args: "plan-key"

    transcription.restore_semantic_plan_fingerprint(dubber, "audio.wav")

    assert dubber._semantic_plan_fingerprint == "plan-live"
    assert dubber._semantic_plan_cache_persistable is True


def test_cached_semantic_plan_keeps_segment_list_identity_and_restores_fingerprint():
    segments = [{"semantic_plan_fingerprint": "plan-live"}]
    diarization = {(0.0, 1.0): "SPEAKER_00"}

    class Cache:
        use_cache = True

        def cache_exists(self, *_args):
            return True

        def load_from_cache(self, *_args):
            return {
                "diarization": diarization,
                "transcription": segments,
                "semantic_diagnostics": [],
            }

    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {
        "semantic_split_enabled": True,
        "debug_info": False,
        "inner_transcription_system": "deepgram",
    }
    dubber.cache_manager = Cache()
    dubber.debug_data = {}
    dubber._isolated_tracks_cache_key = lambda *_args: "plan-key"
    saved = []
    dubber._save_transcription_file = lambda value: saved.append(value)

    speakers, result = transcription.diarize_and_transcribe_isolated(
        dubber, "audio.wav", {"SPEAKER_00": "track.wav"}
    )

    assert speakers is diarization
    assert result is segments
    assert saved == [segments]
    assert saved[0] is segments
    assert dubber._semantic_plan_fingerprint == "plan-live"
