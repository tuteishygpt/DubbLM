import importlib
import importlib.util
import inspect
from pathlib import Path

from dubbing.core.smart_dubbing import SmartDubbing
from dubbing.core.pipeline.context import PipelineRunContext


SYNTHESIS_METHODS = {
    "_default_tts_system": "default_tts_system",
    "_resolve_voice_profile": "resolve_voice_profile",
    "_profile_pool_key": "profile_pool_key",
    "_global_omnivoice_kwargs": "global_omnivoice_kwargs",
    "_build_tts_client": "build_tts_client",
    "_initialize_tts_systems": "initialize_tts_systems",
    "synthesize_speech": "synthesize_speech",
    "_synthesize_measured_candidates": "synthesize_measured_candidates",
    "_load_or_synthesize_candidate": "load_or_synthesize_candidate",
    "resynthesize_one_segment": "resynthesize_one_segment",
    "_get_tts_system_for_speaker": "get_tts_system_for_speaker",
    "_preflight_tts_pools": "preflight_tts_pools",
    "_cache_raw_tts_segment": "cache_raw_tts_segment",
    "_cached_segment_metadata": "cached_segment_metadata",
    "_cached_segment_contract": "cached_segment_contract",
}

AUDIO_ASSEMBLY_METHODS = {
    "_trim_trailing_silence": "trim_trailing_silence",
    "_measure_raw_tts_for_timing": "measure_raw_tts_for_timing",
    "_adjust_and_combine_audio_grouped_legacy": "adjust_and_combine_audio_grouped_legacy",
    "_adjust_and_combine_audio_grouped": "adjust_and_combine_audio_grouped",
    "rebuild_translated_audio_from_chunks": "rebuild_translated_audio_from_chunks",
}


def test_synthesis_and_audio_assembly_modules_exist():
    assert importlib.util.find_spec("dubbing.core.pipeline.synthesis") is not None
    assert importlib.util.find_spec("dubbing.core.pipeline.audio_assembly") is not None


def test_services_export_exact_owned_pool_candidate_resynthesis_cache_and_assembly_callables():
    synthesis = importlib.import_module("dubbing.core.pipeline.synthesis")
    assembly = importlib.import_module("dubbing.core.pipeline.audio_assembly")

    for helper_name in SYNTHESIS_METHODS.values():
        assert callable(getattr(synthesis, helper_name, None)), helper_name
    for helper_name in AUDIO_ASSEMBLY_METHODS.values():
        assert callable(getattr(assembly, helper_name, None)), helper_name


def test_every_owned_facade_method_is_an_exact_thin_delegate():
    for facade_name, helper_name in SYNTHESIS_METHODS.items():
        source = inspect.getsource(SmartDubbing.__dict__[facade_name])
        assert f"synthesis_helpers.{helper_name}(" in source
        assert len(source.splitlines()) <= 18

    for facade_name, helper_name in AUDIO_ASSEMBLY_METHODS.items():
        source = inspect.getsource(SmartDubbing.__dict__[facade_name])
        assert f"audio_assembly_helpers.{helper_name}(" in source
        assert len(source.splitlines()) <= 18


def test_synthesis_candidate_delegate_resolves_module_callable_at_call_time(monkeypatch):
    synthesis = importlib.import_module("dubbing.core.pipeline.synthesis")
    dubber = SmartDubbing.__new__(SmartDubbing)
    metadata = {"segment": object()}
    policy = object()
    calls = []

    monkeypatch.setattr(
        synthesis,
        "synthesize_measured_candidates",
        lambda facade, actual_metadata, actual_policy: calls.append(
            (facade, actual_metadata, actual_policy)
        ),
    )

    assert dubber._synthesize_measured_candidates(metadata, policy) is None
    assert calls == [(dubber, metadata, policy)]


def test_static_preflight_delegate_preserves_descriptor_and_arguments(monkeypatch):
    synthesis = importlib.import_module("dubbing.core.pipeline.synthesis")
    calls = []
    monkeypatch.setattr(
        synthesis,
        "preflight_tts_pools",
        lambda pools, clients, issues=None: calls.append((pools, clients, issues)),
    )
    pools = {("voice",): [object()]}
    clients = {("voice",): object()}
    issues = [(2, "bad reference")]

    assert isinstance(SmartDubbing.__dict__["_preflight_tts_pools"], staticmethod)
    assert SmartDubbing._preflight_tts_pools(pools, clients, issues) is None
    assert calls == [(pools, clients, issues)]


def test_audio_assembly_direct_contract_reorders_in_place_and_keeps_diagnostics_and_duration(tmp_path):
    assembly = importlib.import_module("dubbing.core.pipeline.audio_assembly")
    later = {
        "speaker": "S2",
        "start": 2.0,
        "end": 2.5,
        "text": "later",
        "translation": "later",
    }
    earlier = {
        "speaker": "S1",
        "start": 0.5,
        "end": 1.0,
        "text": "earlier",
        "translation": "earlier",
    }
    segments = [later, earlier]
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber.config = {"debug_info": False}
    dubber.audio_chunks_dir = tmp_path / "raw"
    dubber.su_audio_chunks_dir = tmp_path / "timed"
    dubber.audio_chunks_dir.mkdir()
    dubber.debug_data = {}
    dubber._timing_source_duration = 3.0
    dubber._pipeline_run_context = PipelineRunContext(timing_source_duration=4.0)

    combined, positions = assembly.adjust_and_combine_audio_grouped(dubber, segments)

    assert segments == [earlier, later]
    assert [row["segment_index"] for row in dubber.debug_data["timing_alignment"]] == [1, 0]
    assert [row["original_index"] for row in positions] == [1, 0]
    assert len(combined) == 4000


def test_raw_cache_service_preserves_audio_and_metadata_contract(tmp_path):
    synthesis = importlib.import_module("dubbing.core.pipeline.synthesis")
    source = tmp_path / "source.wav"
    cache = tmp_path / "cache.wav"
    source.write_bytes(b"raw-audio")
    dubber = SmartDubbing.__new__(SmartDubbing)
    dubber._segment_cache_metadata_path = lambda path: Path(f"{path}.json")

    synthesis.cache_raw_tts_segment(
        dubber, str(source), cache, synthesized_text="spoken text"
    )

    assert cache.read_bytes() == b"raw-audio"
    assert synthesis.cached_segment_contract(dubber, cache) == "anchor_raw_v2"
    assert synthesis.cached_segment_metadata(dubber, cache) == {
        "audio_contract": "anchor_raw_v2",
        "synthesized_text": "spoken text",
    }
