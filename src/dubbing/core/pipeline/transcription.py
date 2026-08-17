"""Internal transcription and semantic-planning stage implementations."""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional, Tuple

from transcription.transcription_factory import TranscriptionFactory

from ..log_config import get_logger
from .context import update_context as _update_pipeline_context

logger = get_logger(__name__)

def initialize_transcriber(facade) -> None:
    """Initialize transcriber based on configuration."""
    facade.transcriber = None
    facade.transcriber_init_error = None
    try:
        facade.transcriber = TranscriptionFactory.create_transcriber(
            transcription_system=facade.config.get('transcription_system', 'whisper'),
            source_language=facade.config.get('source_language'),
            device=facade.device,
            transcription_model=facade.config.get('transcription_model'),
            whisper_model=facade.config.get('whisper_model', 'large-v3'),
            gemini_transcription_model=facade.config.get('gemini_transcription_model', 'gemini-3-flash-preview'),
            deepgram_model=facade.config.get('deepgram_model', 'nova-3'),
            cache_manager=facade.cache_manager,
            artifacts_root=facade.config.get("artifacts_dir"),
        )
        logger.debug(f"Initialized {facade.transcriber.name} transcriber")
    except Exception as e:
        facade.transcriber_init_error = e
        logger.warning(f"Failed to initialize transcriber: {e}")

def require_transcriber(facade):
    """Return the initialized transcriber or raise an actionable error."""
    if facade.transcriber is not None:
        return facade.transcriber

    raise RuntimeError(
        facade._format_component_init_error(
            "Transcriber",
            facade.config.get('transcription_system', 'whisper'),
            getattr(facade, "transcriber_init_error", None),
        )
    )

def load_cached_diarize_and_transcribe(
    facade, audio_file: str
) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
    """Return a cached diarize+transcribe result or raise a clear error.

    Used by ``run_translate_only`` to guarantee we never re-run the
    transcriber during a resume step. Reproduces the exact cache_key that
    ``SmartDubbing.diarize_and_transcribe`` writes — a full-pipeline or
    ``transcribe_only`` run passes that key to the transcriber, so any
    cached result lives under it. If the transcriber does not advertise
    ``cache_step_name`` (legacy pyannote_openai with its multi-step cache
    layout), fall back to a normal ``diarize_and_transcribe`` call so its
    own internal cache is still consulted.
    """
    # Isolated-tracks path publishes its own step_name outside of any
    # transcriber. Reproduce the same cache_key the write path used;
    # miss → fall back to a fresh isolated run (its own cache is a no-op
    # then, so we just recompute).
    isolated_tracks = facade.config.get('isolated_tracks')
    if isolated_tracks:
        step_name = (
            "isolated_tracks_semantic_plan"
            if facade.config.get("semantic_split_enabled", True)
            else "isolated_tracks_transcription"
        )
        cache_key = facade._isolated_tracks_cache_key(audio_file, isolated_tracks)
        if not facade.cache_manager.cache_exists(step_name, cache_key):
            raise FileNotFoundError(
                f"run_step=translate_only requires cached isolated-tracks "
                f"diarization+transcription from a previous transcribe_only or "
                f"full pipeline run in the same project directory, but no cache "
                f"entry was found for step '{step_name}'. Run "
                f"--run_step transcribe_only (or the full pipeline) first."
            )
        return facade._diarize_and_transcribe_isolated(audio_file, isolated_tracks)

    transcriber = facade._require_transcriber()
    step_name = (getattr(transcriber, "cache_step_name", "") or "").strip()

    if not step_name:
        logger.warning(
            "Transcriber %s does not advertise a cache_step_name; falling back "
            "to a normal diarize_and_transcribe call (its own cache is still consulted).",
            transcriber.name,
        )
        speakers_rolls, transcription = transcriber.diarize_and_transcribe(
            audio_file=audio_file,
            cache_key=None,
            use_cache=True,
        )
        facade.debug_data["diarization"] = speakers_rolls
        facade.debug_data["transcription"] = transcription
        facade._save_transcription_file(transcription)
        return speakers_rolls, transcription

    # Use the SAME cache_key that SmartDubbing.diarize_and_transcribe would
    # compute on a full-pipeline / transcribe_only run — that's what any
    # existing cache entry was written under.
    cache_key = facade.cache_manager.generate_cache_key(
        audio_file,
        facade.config.get('source_language'),
        facade.config.get('target_language'),
        facade.config.get('whisper_model', 'large-v3'),
        facade.config.get('start_time'),
        facade.config.get('duration'),
    )

    if not facade.cache_manager.cache_exists(step_name, cache_key):
        raise FileNotFoundError(
            f"run_step=translate_only requires cached diarization+transcription "
            f"from a previous transcribe_only or full pipeline run in the same "
            f"project directory, but no cache entry was found for step "
            f"'{step_name}'. Run --run_step transcribe_only (or the full pipeline) "
            f"first."
        )

    speakers_rolls, transcription = transcriber.diarize_and_transcribe(
        audio_file=audio_file,
        cache_key=cache_key,
        use_cache=True,
    )
    facade.debug_data["diarization"] = speakers_rolls
    facade.debug_data["transcription"] = transcription
    facade._save_transcription_file(transcription)
    return speakers_rolls, transcription

def diarize_and_transcribe(facade, audio_file: str) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
    """Perform speaker diarization and transcription."""
    # Opt-in isolated-tracks path: only active when the user supplies
    # per-speaker isolated audio files. Skips the standard transcriber
    # entirely; the standard path below is left untouched.
    isolated_tracks = facade.config.get('isolated_tracks')
    if isolated_tracks:
        return facade._diarize_and_transcribe_isolated(audio_file, isolated_tracks)

    transcriber = facade._require_transcriber()

    # Generate cache key
    cache_key = facade.cache_manager.generate_cache_key(
        audio_file,
        facade.config.get('source_language'),
        facade.config.get('target_language'),
        facade.config.get('whisper_model', 'large-v3'),
        facade.config.get('start_time'),
        facade.config.get('duration')
    )

    # Perform diarization and transcription
    speakers_rolls, transcription = transcriber.diarize_and_transcribe(
        audio_file=audio_file,
        cache_key=cache_key,
        use_cache=facade.cache_manager.use_cache
    )

    # Store for debug
    facade.debug_data["diarization"] = speakers_rolls
    facade.debug_data["transcription"] = transcription

    # Save transcription to file
    facade._save_transcription_file(transcription)

    return speakers_rolls, transcription

def semantic_classifier(facade) -> Tuple[Optional[Any], str]:
    if facade.config.get("translator_type", "llm") != "llm":
        return None, "deterministic-only"
    translator = getattr(facade, "translator", None)
    classifier = getattr(translator, "classify_semantic_boundaries", None)
    if callable(classifier):
        return classifier, "ready"
    if getattr(facade, "translator_init_error", None) is not None:
        return None, "initialization_failed"
    return None, "unavailable"

def write_semantic_boundary_diagnostics(
    path: str, records: List[Dict[str, Any]]
) -> None:
    debug_path = Path(path)
    debug_path.parent.mkdir(parents=True, exist_ok=True)
    with debug_path.open("w", encoding="utf-8", newline="\n") as handle:
        for record in records:
            handle.write(
                json.dumps(record, ensure_ascii=False, sort_keys=True) + "\n"
            )

def diarize_and_transcribe_isolated(
    facade,
    audio_file: str,
    isolated_tracks: Dict[str, str],
) -> Tuple[Dict[Tuple[float, float], str], List[Dict]]:
    """Isolated-tracks branch of ``diarize_and_transcribe``.

    Delegates to ``dubbing.audio.isolated_tracks.run_isolated_tracks``
    and caches the result under ``isolated_tracks_transcription`` so
    ``translate_only`` / ``tts_to_end`` can resume without re-running
    VAD or the inner transcriber.
    """
    from dubbing.audio.isolated_tracks import (
        collect_isolated_tracks_raw,
        run_isolated_tracks,
    )

    semantic_enabled = facade.config.get("semantic_split_enabled", True)
    semantic_debug_path = (
        str(Path(facade.config.get("debug_dir")) / "semantic_boundaries.jsonl")
        if facade.config.get("debug_info", False) and facade.config.get("debug_dir")
        else None
    )
    step_name = (
        "isolated_tracks_semantic_plan"
        if semantic_enabled
        else "isolated_tracks_transcription"
    )
    cache_key = facade._isolated_tracks_cache_key(audio_file, isolated_tracks)

    if (
        facade.cache_manager.use_cache
        and facade.cache_manager.cache_exists(step_name, cache_key)
    ):
        logger.debug("Loading isolated-tracks diarization+transcription from cache...")
        cached = facade.cache_manager.load_from_cache(step_name, cache_key)
        if not isinstance(cached, dict):
            cached = None
        if cached is not None:
            if (
                semantic_enabled
                and semantic_debug_path
                and not isinstance(cached.get("semantic_diagnostics"), list)
            ):
                logger.info(
                    "Cached semantic plan predates boundary diagnostics; replanning from raw transcription."
                )
                cached = None
            if cached is None:
                logger.warning("Semantic-plan cache cannot satisfy debug diagnostics.")
            else:
                if semantic_enabled and semantic_debug_path:
                    facade._write_semantic_boundary_diagnostics(
                        semantic_debug_path,
                        cached["semantic_diagnostics"],
                    )
        if cached is not None:
            speakers_rolls = cached["diarization"]
            transcription = cached["transcription"]
            if transcription and semantic_enabled:
                fingerprints = {
                    segment.get("semantic_plan_fingerprint")
                    for segment in transcription
                    if segment.get("semantic_plan_fingerprint")
                }
                if (
                    len(fingerprints) != 1
                    or any(
                        segment.get("semantic_plan_fingerprint") not in fingerprints
                        for segment in transcription
                    )
                ):
                    logger.warning(
                        "Cached semantic plan has a missing or inconsistent fingerprint; replanning."
                    )
                    cached = None
                else:
                    _update_pipeline_context(
                        facade,
                        "semantic_plan_fingerprint",
                        next(iter(fingerprints)),
                    )
            if cached is not None:
                _update_pipeline_context(
                    facade, "semantic_plan_cache_persistable", True
                )
                facade.debug_data["diarization"] = speakers_rolls
                facade.debug_data["transcription"] = transcription
                facade._save_transcription_file(transcription)
                return speakers_rolls, transcription
        logger.warning("Corrupt isolated-tracks cache entry, recomputing.")

    inner_system = facade.config.get('inner_transcription_system', 'deepgram')
    logger.info(
        "Isolated-tracks path: %d tracks, inner_transcription_system=%s",
        len(isolated_tracks),
        inner_system,
    )

    raw_step_name = "isolated_tracks_raw_transcription"
    raw_cache_key = facade._isolated_tracks_raw_cache_key(isolated_tracks)
    raw_tracks_data = None
    if (
        facade.cache_manager.use_cache
        and facade.cache_manager.cache_exists(raw_step_name, raw_cache_key)
    ):
        raw_tracks_data = facade.cache_manager.load_from_cache(
            raw_step_name, raw_cache_key
        )
    if raw_tracks_data is None:
        raw_tracks_data = collect_isolated_tracks_raw(
            tracks=isolated_tracks,
            inner_system=inner_system,
            source_language=facade.config.get('source_language'),
            device=facade.config.get('device'),
            cache_manager=facade.cache_manager,
            inner_kwargs=facade._isolated_inner_kwargs(inner_system),
            start_time=facade.config.get('start_time'),
            duration=facade.config.get('duration'),
        )
        facade.cache_manager.save_to_cache(
            raw_step_name, raw_cache_key, raw_tracks_data
        )

    semantic_classifier, classifier_status = facade._semantic_classifier()
    classification_step = "semantic_boundary_classification"

    def load_classification(key: str) -> Any:
        if facade.cache_manager.cache_exists(classification_step, key):
            return facade.cache_manager.load_from_cache(classification_step, key)
        return None

    def save_classification(key: str, value: Any) -> None:
        facade.cache_manager.save_to_cache(classification_step, key, value)

    classifier_cache_context = {
        **facade._semantic_classifier_identity(),
        "timeout": 30.0,
        "batch_size": 50,
        "batch_characters": 12000,
    }
    semantic_diagnostics: List[Dict[str, Any]] = []
    speakers_rolls, transcription = run_isolated_tracks(
        tracks=isolated_tracks,
        inner_system=inner_system,
        source_language=facade.config.get('source_language'),
        device=facade.config.get('device'),
        cache_manager=facade.cache_manager,
        inner_kwargs=facade._isolated_inner_kwargs(inner_system),
        start_time=facade.config.get('start_time'),
        duration=facade.config.get('duration'),
        semantic_split_enabled=semantic_enabled,
        tts_preferred_segment_duration=facade.config.get(
            "tts_preferred_segment_duration", 15.0
        ),
        tts_hard_segment_duration=facade.config.get(
            "tts_hard_segment_duration", 35.0
        ),
        semantic_split_search_window=facade.config.get(
            "semantic_split_search_window", 10.0
        ),
        semantic_classifier=semantic_classifier,
        semantic_classifier_status=classifier_status,
        semantic_debug_path=semantic_debug_path,
        raw_tracks_data=raw_tracks_data,
        classification_cache_get=load_classification,
        classification_cache_set=save_classification,
        classifier_cache_context=classifier_cache_context,
        semantic_diagnostics_out=semantic_diagnostics,
    )

    plan_persistable = all(
        segment.get("_semantic_plan_cache_persistable", True)
        for segment in transcription
    )
    if semantic_enabled and not plan_persistable:
        logger.warning(
            "Semantic boundary classification used a transient fallback; "
            "caching the deterministic fallback plan for resume."
        )
    facade.cache_manager.save_to_cache(
        step_name,
        cache_key,
        {
            "diarization": speakers_rolls,
            "transcription": transcription,
            "semantic_diagnostics": semantic_diagnostics,
        },
    )
    if transcription and semantic_enabled:
        _update_pipeline_context(
            facade,
            "semantic_plan_fingerprint",
            transcription[0].get("semantic_plan_fingerprint"),
        )
        _update_pipeline_context(
            facade, "semantic_plan_cache_persistable", True
        )

    facade.debug_data["diarization"] = speakers_rolls
    facade.debug_data["transcription"] = transcription
    facade._save_transcription_file(transcription)
    return speakers_rolls, transcription

def isolated_inner_kwargs(facade, inner_system: str) -> Dict[str, Any]:
    """Extra kwargs for the inner transcriber used by isolated-tracks."""
    if inner_system == "deepgram":
        model = facade.config.get("deepgram_model")
        return {"deepgram_model": model} if model else {}
    if inner_system == "gemini":
        model = facade.config.get("gemini_transcription_model")
        return {"gemini_transcription_model": model} if model else {}
    if inner_system == "assemblyai":
        model = facade.config.get("transcription_model")
        return {"speech_model": model} if model else {}
    return {}

def restore_semantic_plan_fingerprint(facade, audio_file: str) -> None:
    """Restore the semantic identity needed by resume cache keys."""
    isolated_tracks = facade.config.get("isolated_tracks")
    if not isolated_tracks or not facade.config.get("semantic_split_enabled", True):
        return
    step_name = "isolated_tracks_semantic_plan"
    cache_key = facade._isolated_tracks_cache_key(audio_file, isolated_tracks)
    if not facade.cache_manager.cache_exists(step_name, cache_key):
        raise FileNotFoundError(
            "Semantic-plan cache is missing or stale; run transcribe_only or the "
            "full pipeline before resuming plan-dependent translation/TTS."
        )
    cached = facade.cache_manager.load_from_cache(step_name, cache_key)
    transcription = cached.get("transcription") if isinstance(cached, dict) else None
    fingerprints = {
        segment.get("semantic_plan_fingerprint")
        for segment in (transcription or [])
        if segment.get("semantic_plan_fingerprint")
    }
    if (
        len(fingerprints) != 1
        or any(
            segment.get("semantic_plan_fingerprint") not in fingerprints
            for segment in (transcription or [])
        )
    ):
        raise ValueError("Cached semantic plan has a missing or inconsistent fingerprint")
    _update_pipeline_context(
        facade, "semantic_plan_fingerprint", next(iter(fingerprints))
    )
    _update_pipeline_context(
        facade, "semantic_plan_cache_persistable", True
    )
