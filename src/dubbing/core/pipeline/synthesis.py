"""Internal TTS pool, candidate, resynthesis, and raw-cache implementations."""

import json
import os
import shutil
from pathlib import Path
from typing import Any, Dict, List, Optional

from pydub import AudioSegment
from tts.tts_factory import TTSFactory

from ..log_config import get_logger
from ..voice_profiles import FALLBACK_SPEAKER, VoiceProfile, resolve_profile
from .context import active_context
from .context import update_context as _update_pipeline_context

logger = get_logger(__name__)


def default_tts_system(facade) -> str:
    """The TTS backend used as fallback when a profile does not name one."""
    return facade.config.get('tts_system', 'coqui')


def resolve_voice_profile(facade, speaker: str) -> VoiceProfile:
    """Resolve a speaker to its effective VoiceProfile (with `"*"` fallback)."""
    return resolve_profile(
        facade.voice_profiles,
        speaker,
        tts_system_default=facade._default_tts_system(),
    )


def profile_pool_key(facade, profile: VoiceProfile) -> tuple:
    """Client-pool identity for a profile, taking global TTS defaults into account."""
    return (
        (profile.tts_system or facade._default_tts_system() or "").lower(),
        profile.model or (facade.config.get('tts_model') or ""),
        tuple(sorted(profile.params.items())),
    )


def global_omnivoice_kwargs(facade) -> Dict[str, Any]:
    """Backend-specific globals still sourced from top-level config (legacy)."""
    return {
        "space_id": facade.config.get('omnivoice_space_id'),
        "api_name": facade.config.get('omnivoice_api_name'),
        "lang": facade.config.get('omnivoice_lang'),
        "instruct": facade.config.get('omnivoice_instruct', ''),
        "num_steps": facade.config.get('omnivoice_num_steps'),
        "guidance_scale": facade.config.get('omnivoice_guidance_scale'),
        "denoise": facade.config.get('omnivoice_denoise'),
        "speed": facade.config.get('omnivoice_speed'),
        "duration": facade.config.get('omnivoice_duration'),
        "preprocess_prompt": facade.config.get('omnivoice_preprocess_prompt'),
        "postprocess_output": facade.config.get('omnivoice_postprocess_output'),
    }


def build_tts_client(facade, profile: VoiceProfile) -> Any:
    """Create a single TTS client for a normalized profile.

    Voice/style mapping is not attached here — those are per-segment and are
    threaded into TTSSegmentData at synth time. This function only handles
    provider construction (system, model, provider-specific bootstrap kwargs).
    """
    tts_system = profile.tts_system or facade._default_tts_system()
    model = profile.model or facade.config.get('tts_model')

    # Provider-specific bootstrap kwargs. Globals still come from the top-level
    # config; per-profile `params` override them.
    bootstrap: Dict[str, Any] = {}
    if tts_system.lower() == "omnivoice":
        bootstrap.update(facade._global_omnivoice_kwargs())
    bootstrap.update(profile.params or {})

    return TTSFactory.create_tts(
        tts_system=tts_system,
        device=facade.device,
        voice_config=None,  # per-segment via TTSSegmentData
        voice_prompt=None,  # per-segment via TTSSegmentData
        prompt_prefix=facade.config.get('tts_prompt_prefix'),
        # A manually assigned voice does not need catalog matching. This also
        # prevents OpenAI/Gemini from generating the catalog samples solely
        # for a profile whose voice is already known.
        enable_voice_matching=(
            facade.config.get('voice_auto_selection', True)
            and not bool(profile.voice_name)
        ),
        debug_tts=facade.config.get('debug_tts', False),
        model=model,
        default_reference_audio=facade.config.get('reference_audio'),
        **bootstrap,
    )


def initialize_tts_systems(facade) -> None:
    """Instantiate one TTS client per unique (system, model, params) profile."""
    # Client pool keyed by pool_key. Multiple speakers with identical settings
    # share the same client.
    facade.tts_clients: Dict[tuple, Any] = {}
    # Speaker -> pool_key resolution cache.
    facade._speaker_pool_key: Dict[str, tuple] = {}
    facade.default_tts = None
    facade.tts_init_error: Optional[Exception] = None

    # Build the effective set of profiles to instantiate: every named profile
    # plus a fallback (`"*"` or synthesised from global TTS defaults).
    profiles_to_build: Dict[tuple, VoiceProfile] = {}
    for speaker, profile in facade.voice_profiles.items():
        resolved = facade._resolve_voice_profile(speaker)
        key = facade._profile_pool_key(resolved)
        profiles_to_build.setdefault(key, resolved)
        facade._speaker_pool_key[speaker] = key

    fallback_profile = facade._resolve_voice_profile(FALLBACK_SPEAKER)
    fallback_key = facade._profile_pool_key(fallback_profile)
    profiles_to_build.setdefault(fallback_key, fallback_profile)

    first_error: Optional[Exception] = None
    for key, profile in profiles_to_build.items():
        try:
            client = facade._build_tts_client(profile)
            facade.tts_clients[key] = client
            logger.debug(
                "Initialized TTS client for pool_key=%s (system=%s, model=%s)",
                key, profile.tts_system, profile.model,
            )
        except Exception as exc:
            if first_error is None:
                first_error = exc
            logger.warning(
                "Failed to initialize TTS client for pool_key=%s (system=%s, model=%s): %s",
                key, profile.tts_system, profile.model, exc,
            )

    if first_error is not None and not facade.tts_clients:
        facade.tts_init_error = first_error

    # default_tts is the client for the "*" profile (or first available).
    if fallback_key in facade.tts_clients:
        facade.default_tts = facade.tts_clients[fallback_key]
    elif facade.tts_clients:
        facade.default_tts = next(iter(facade.tts_clients.values()))

    # Backwards-compat mirror: `facade.tts_systems` used to be a dict keyed by
    # backend name. A few older call sites/tests still touch it, so mirror the
    # default client under its backend name. New code should use tts_clients.
    facade.tts_systems: Dict[str, Any] = {}
    if facade.default_tts is not None:
        facade.tts_systems[fallback_profile.tts_system or facade._default_tts_system()] = facade.default_tts


def synthesize_speech(facade, segments: List[Dict], speakers_rolls: Dict, audio_file: str) -> str:
    """Synthesize measured text candidates within each recognized segment."""
    if not segments:
        raise ValueError("Cannot synthesize speech with no segments.")

    from ..timing import TimingPolicy, plan_anchor_windows, timing_cache_fingerprint
    from tts.models import TTSSegmentData

    try:
        _update_pipeline_context(
            facade,
            "timing_source_duration",
            len(AudioSegment.from_file(audio_file)) / 1000.0,
        )
    except Exception as exc:
        raise ValueError(f"Cannot measure processed source audio for timing: {audio_file}") from exc
    _update_pipeline_context(facade, "timing_source_audio_file", audio_file)

    planned = plan_anchor_windows(
        segments, active_context(facade).timing_source_duration
    )
    for item in planned:
        item.segment["_timing_original_index"] = item.original_index
        item.segment["_timing_available_window"] = item.available_window
        item.segment.pop("_timing_next_anchor", None)
    segments[:] = [item.segment for item in planned]

    policy = TimingPolicy.from_config(facade.config)
    _update_pipeline_context(
        facade,
        "plan_dependent_cache_allowed",
        active_context(facade).semantic_plan_cache_persistable,
    )
    facade.performance_tracker.start_timing("speech_synthesis")

    semantic_fingerprints = {
        segment.get("semantic_plan_fingerprint")
        for segment in segments
        if segment.get("semantic_plan_fingerprint")
    }
    semantic_segments = [
        segment for segment in segments if segment.get("semantic_unit_id")
    ]
    if semantic_segments and (
        len(semantic_fingerprints) != 1
        or any(
            segment.get("semantic_plan_fingerprint") not in semantic_fingerprints
            for segment in semantic_segments
        )
    ):
        raise ValueError(
            "Semantic TTS segments have a missing or inconsistent semantic_plan_fingerprint"
        )
    semantic_suffix = (
        f"_{next(iter(semantic_fingerprints))}"
        if len(semantic_fingerprints) == 1
        else ""
    )
    tts_fingerprint = facade._effective_tts_cache_fingerprint(
        segment["speaker"] for segment in segments
    )
    source_cache_key = facade.cache_manager.generate_cache_key(
        audio_file,
        facade.config.get("source_language"),
        facade.config.get("target_language"),
        facade.config.get("whisper_model", "large-v3"),
        facade.config.get("start_time"),
        facade.config.get("duration"),
    )
    selection_fingerprint = facade._tts_selection_cache_fingerprint(segments)
    has_segment_references = any(
        facade._resolve_voice_profile(segment["speaker"]).reference_mode == "segment"
        for segment in segments
    )
    cache_key = (
        f"{source_cache_key}_{facade.config.get('target_language')}_"
        f"{facade.config.get('tts_system')}_{timing_cache_fingerprint(policy)}_"
        f"{tts_fingerprint}_{selection_fingerprint}{semantic_suffix}"
    )
    aggregate_cache_dir = facade.cache_manager.get_cache_path("synthesized_speech")
    cached_audio_path = aggregate_cache_dir / f"{cache_key}.wav"
    output_path = facade.config.get("translated_audio_path")
    if (
        facade.cache_manager.use_cache
        and active_context(facade).plan_dependent_cache_allowed
        and not facade.config.get("debug_info", False)
        and not has_segment_references
        and cached_audio_path.exists()
    ):
        shutil.copy(cached_audio_path, output_path)
        facade.performance_tracker.end_timing("speech_synthesis")
        return output_path

    if not facade.tts_clients:
        raise ValueError("No TTS systems are initialized properly")

    segment_cache_dir = facade.cache_manager.get_cache_path("segment_synthesis")
    base_cache_prefix = f"{source_cache_key}_{tts_fingerprint}"
    facade.audio_chunks_dir.mkdir(parents=True, exist_ok=True)
    facade.su_audio_chunks_dir.mkdir(parents=True, exist_ok=True)

    segment_reference_min_duration = facade.config.get(
        "segment_reference_min_duration", 2.0
    )
    try:
        segment_reference_min_duration = max(
            0.0, float(segment_reference_min_duration)
        )
    except (TypeError, ValueError, OverflowError):
        segment_reference_min_duration = 2.0

    prepared: List[Dict[str, Any]] = []
    preflight_segments: Dict[tuple, List[Any]] = {}
    preflight_clients: Dict[tuple, Any] = {}
    reference_issues: List[tuple[int, str]] = []
    original_audio_segment = None
    segment_reference_audio_cache: Dict[tuple[str, float], AudioSegment] = {}

    for chronological_index, segment in enumerate(segments):
        speaker = segment["speaker"]
        profile = facade._resolve_voice_profile(speaker)
        pool_key = facade._profile_pool_key(profile)
        tts_instance = facade.tts_clients.get(pool_key) or getattr(
            facade, "default_tts", None
        )
        if tts_instance is None:
            raise ValueError(
                f"TTS client for {profile.tts_system or facade._default_tts_system()} is not available"
            )
        tts_system = profile.tts_system or facade._default_tts_system()
        original_index = facade._canonical_segment_index(
            segment, chronological_index
        )
        style_prompt = (
            (segment.get("style_prompt") or "").strip()
            or profile.style_prompt
        )
        voice_name = profile.voice_name
        if voice_name is None and isinstance(facade.config.get("voice_name"), str):
            voice_name = facade.config.get("voice_name")
        if facade.config.get("debug_info", False):
            facade.debug_data.setdefault("voices", {})[chronological_index] = {
                "speaker": speaker,
                "voice": voice_name,
                "style_prompt": style_prompt,
                "tts_system": tts_system,
                "model": profile.model,
            }
        base_args: Dict[str, Any] = {
            "speaker": speaker,
            "text": segment.get("translation", ""),
            "emotion": segment.get("emotion", "Neutral"),
            "style_prompt": style_prompt,
            "reference_audio_path": profile.reference_audio,
            "reference_text": profile.reference_text,
            "reference_mode": profile.reference_mode,
            "segment_index": original_index,
            "voice": voice_name,
            "speed": 1.0,
            "target_duration": segment["_timing_available_window"],
        }
        provider_capability = getattr(
            tts_instance, "reference_capability", "unsupported"
        )
        try:
            base_args, original_audio_segment = facade._resolve_segment_reference(
                tts_segment_data_args=base_args,
                segment_dict=segment,
                profile=profile,
                provider_capability=provider_capability,
                speaker=speaker,
                segment_index=original_index,
                original_audio_segment=original_audio_segment,
                segment_reference_min_duration=segment_reference_min_duration,
                processed_source_path=audio_file,
                decoded_audio_cache=segment_reference_audio_cache,
            )
        except Exception as exc:
            reference_issues.append(
                (
                    original_index,
                    f"provider={tts_system} speaker={speaker} segment={original_index} "
                    f"mode={profile.reference_mode or '<missing>'}: {exc}",
                )
            )

        final_path = str(facade.audio_chunks_dir / f"{chronological_index}.wav")
        initial_data = TTSSegmentData(
            **{
                **base_args,
                "text": segment.get("translation", ""),
                "output_path": final_path,
            }
        )
        preflight_segments.setdefault(pool_key, []).append(initial_data)
        preflight_clients[pool_key] = tts_instance
        prepared.append(
            {
                "index": chronological_index,
                "segment": segment,
                "profile": profile,
                "pool_key": pool_key,
                "tts_instance": tts_instance,
                "tts_system": tts_system,
                "base_args": base_args,
                "base_cache_prefix": base_cache_prefix,
                "segment_cache_dir": segment_cache_dir,
                "final_path": final_path,
                "style_prompt": style_prompt,
            }
        )

    facade._preflight_tts_pools(
        preflight_segments,
        preflight_clients,
        reference_issues,
    )

    for metadata in prepared:
        facade._synthesize_measured_candidates(metadata, policy)

    combined_audio, real_segment_positions = facade._adjust_and_combine_audio_grouped(
        segments
    )
    combined_audio.export(output_path, format="wav")
    facade.real_segment_positions = real_segment_positions

    if (
        facade.cache_manager.use_cache
        and active_context(facade).plan_dependent_cache_allowed
    ):
        shutil.copy(output_path, cached_audio_path)

    track_usage: Dict[str, int] = {}
    for segment in segments:
        variant = segment.get("selected_variant", "missing")
        track_usage[variant] = track_usage.get(variant, 0) + 1
    logger.debug("Measured TTS candidate usage: %s", track_usage)
    facade.performance_tracker.end_timing("speech_synthesis")
    return output_path


def synthesize_measured_candidates(
    facade,
    metadata: Dict[str, Any],
    policy,
) -> None:
    segment = metadata["segment"]
    target_duration = float(segment["_timing_available_window"])
    generated: List[Dict[str, Any]] = []
    tried_texts: set[str] = set()
    try:
        Path(metadata["final_path"]).unlink()
    except OSError:
        pass

    def generate(variant: str, attempts: int = 1) -> Optional[Dict[str, Any]]:
        value = segment.get(variant, "")
        text = value.strip() if isinstance(value, str) else ""
        if not text or text in tried_texts:
            return None
        tried_texts.add(text)
        candidate = facade._load_or_synthesize_candidate(
            metadata,
            variant=variant,
            text=text,
            attempts=attempts,
        )
        if candidate is not None:
            generated.append(candidate)
        return candidate

    normal = generate("translation", attempts=3)
    if normal is None:
        segment["synthesized_speech_len"] = 0.0
        segment["synthesized_speech_file"] = None
        segment["selected_variant"] = "missing"
        return

    if normal["duration"] < target_duration:
        generate("long_translation")
    elif normal["duration"] > target_duration + policy.max_overflow:
        short = generate("short_translation")
        if (
            short is not None
            and short["duration"] > target_duration + policy.max_overflow
        ):
            generate("very_short_translation")

    winner = generated[0]
    best_difference = abs(winner["duration"] - target_duration)
    for candidate in generated[1:]:
        difference = abs(candidate["duration"] - target_duration)
        if difference + 1e-9 < best_difference:
            winner = candidate
            best_difference = difference
    final_path = metadata["final_path"]
    shutil.copy(winner["path"], final_path)
    segment["_tts_cache_contract"] = "anchor_raw_v2"
    segment["synthesized_speech_len"] = winner["duration"]
    segment["synthesized_speech_file"] = final_path
    segment["synthesized_text"] = winner["text"]
    segment["selected_variant"] = winner["variant"]

    for candidate in generated:
        try:
            Path(candidate["path"]).unlink()
        except OSError:
            pass


def load_or_synthesize_candidate(
    facade,
    metadata: Dict[str, Any],
    *,
    variant: str,
    text: str,
    attempts: int,
) -> Optional[Dict[str, Any]]:
    from tts.models import TTSSegmentData

    segment = metadata["segment"]
    index = metadata["index"]
    candidate_path = str(
        facade.audio_chunks_dir / f"candidate_{index}_{variant}.wav"
    )
    cache_key = facade._raw_tts_segment_cache_key(
        base_cache_prefix=metadata["base_cache_prefix"],
        tts_system=metadata["tts_system"],
        segment=segment,
        speaker=segment["speaker"],
        translation=text,
        style_prompt=metadata["style_prompt"],
        reference_audio_path=metadata["base_args"].get("reference_audio_path"),
        reference_mode=metadata["base_args"].get("reference_mode"),
        reference_text=metadata["base_args"].get("reference_text"),
        client_pool_settings=metadata["pool_key"],
        legacy_index=segment.get("_timing_original_index", index),
        emotion=segment.get("emotion", "Neutral"),
        tts_prompt_prefix=facade.config.get("tts_prompt_prefix"),
        voice_prompt=None,
    )
    cache_path = metadata["segment_cache_dir"] / f"{cache_key}.wav"

    if (
        facade.cache_manager.use_cache
        and active_context(facade).plan_dependent_cache_allowed
        and cache_path.exists()
    ):
        try:
            shutil.copy(cache_path, candidate_path)
            duration = facade._measure_raw_tts_for_timing(candidate_path, index)
            if duration > 0:
                return {
                    "variant": variant,
                    "text": text,
                    "path": candidate_path,
                    "duration": duration,
                }
        except Exception as exc:
            logger.warning("Could not reuse cached TTS candidate %s: %s", variant, exc)
        try:
            cache_path.unlink()
            facade._segment_cache_metadata_path(cache_path).unlink()
        except OSError:
            pass

    segment_data = TTSSegmentData(
        **{
            **metadata["base_args"],
            "text": text,
            "output_path": candidate_path,
        }
    )
    last_error = None
    for attempt in range(max(1, attempts)):
        try:
            Path(candidate_path).unlink()
        except OSError:
            pass
        try:
            metadata["tts_instance"].synthesize(
                segments_data=[segment_data],
                language=facade.config.get("target_language"),
            )
            if not os.path.exists(candidate_path):
                continue
            duration = facade._measure_raw_tts_for_timing(candidate_path, index)
            if duration <= 0:
                continue
            if (
                facade.cache_manager.use_cache
                and active_context(facade).plan_dependent_cache_allowed
            ):
                facade._cache_raw_tts_segment(
                    candidate_path,
                    cache_path,
                    synthesized_text=text,
                )
            return {
                "variant": variant,
                "text": text,
                "path": candidate_path,
                "duration": duration,
            }
        except Exception as exc:
            last_error = exc
            logger.warning(
                "TTS candidate %s failed for segment %d (attempt %d/%d): %s",
                variant,
                index,
                attempt + 1,
                max(1, attempts),
                exc,
            )
    if last_error is not None:
        logger.error(
            "No usable %s candidate for segment %d: %s",
            variant,
            index,
            last_error,
        )
    try:
        Path(candidate_path).unlink()
    except OSError:
        pass
    return None


def resynthesize_one_segment(
    facade,
    segments: List[Dict],
    segment_index: int,
    override_text: Optional[str] = None,
) -> Dict[str, Any]:
    """Resynthesize a single segment (in-place) with the currently
    configured TTS and reference audio.

    Args:
        segments: The full list of cached translation segments.
        segment_index: Index of the segment to resynthesize.
        override_text: Optional text used instead of ``segment.translation``.

    Returns:
        The updated segment dict (also mutated in-place inside ``segments``).
    """
    from tts.models import TTSSegmentData

    if not (0 <= segment_index < len(segments)):
        raise IndexError(f"segment_index {segment_index} out of range (0..{len(segments)-1})")

    segment_dict = segments[segment_index]
    original_segment_index = facade._canonical_segment_index(
        segment_dict, segment_index
    )
    speaker = segment_dict.get("speaker") or "SPEAKER_00"

    profile = facade._resolve_voice_profile(speaker)
    pool_key = facade._profile_pool_key(profile)
    tts_instance = facade.tts_clients.get(pool_key) or facade.default_tts
    if tts_instance is None:
        raise RuntimeError(
            f"TTS client for '{profile.tts_system or facade._default_tts_system()}' is not initialised"
        )
    tts_system = profile.tts_system or facade._default_tts_system()

    text_to_synthesize = (override_text or segment_dict.get("translation") or "").strip()
    if not text_to_synthesize:
        raise ValueError("Cannot resynthesize a segment with empty text")

    voice_name = profile.voice_name
    if voice_name is None:
        voice_cfg = facade.config.get('voice_name')
        if isinstance(voice_cfg, str):
            voice_name = voice_cfg

    segment_style_override = (segment_dict.get("style_prompt") or "").strip()
    segment_style_prompt = segment_style_override or profile.style_prompt

    facade.audio_chunks_dir.mkdir(parents=True, exist_ok=True)
    output_path = str(facade.audio_chunks_dir / f"{segment_index}.wav")

    tts_segment_data_args: Dict[str, Any] = {
        "speaker": speaker,
        "text": text_to_synthesize,
        "emotion": segment_dict.get("emotion", "Neutral"),
        "style_prompt": segment_style_prompt,
        "reference_audio_path": profile.reference_audio,
        "reference_text": profile.reference_text,
        "reference_mode": profile.reference_mode,
        "segment_index": original_segment_index,
        "voice": voice_name,
        "speed": 1.0,
        "target_duration": segment_dict.get(
            "_timing_available_window",
            max(segment_dict.get("end", 0) - segment_dict.get("start", 0), 0.0),
        ),
    }

    segment_reference_min_duration = float(facade.config.get('segment_reference_min_duration', 2.0) or 0.0)
    provider_capability = getattr(
        tts_instance, "reference_capability", "unsupported"
    )
    tts_segment_data_args, _ = facade._resolve_segment_reference(
        tts_segment_data_args=tts_segment_data_args,
        segment_dict=segment_dict,
        profile=profile,
        provider_capability=provider_capability,
        speaker=speaker,
        segment_index=original_segment_index,
        original_audio_segment=None,
        segment_reference_min_duration=segment_reference_min_duration,
        for_resynthesis=True,
    )

    segment_data = TTSSegmentData(**{**tts_segment_data_args, "text": text_to_synthesize, "output_path": output_path})
    facade._preflight_tts_pools(
        {pool_key: [segment_data]},
        {pool_key: tts_instance},
    )

    # Remove any stale zero-byte file so `os.path.exists` reflects reality.
    try:
        if os.path.exists(output_path):
            os.remove(output_path)
    except OSError:
        pass
    if hasattr(facade, "su_audio_chunks_dir") and facade.su_audio_chunks_dir is not None:
        for stale_name in (
            f"tempo_{segment_index}.wav",
            f"tempo_in_{segment_index}.wav",
            f"timed_{segment_index}.wav",
        ):
            try:
                (facade.su_audio_chunks_dir / stale_name).unlink(missing_ok=True)
            except OSError:
                pass

    max_attempts = 3
    last_error: Optional[Exception] = None
    for attempt in range(max_attempts):
        logger.info(
            f"Resynthesizing segment {segment_index} (speaker={speaker}, tts={tts_system}) "
            f"attempt {attempt + 1}/{max_attempts}"
        )
        try:
            tts_instance.synthesize(
                segments_data=[segment_data],
                language=facade.config.get('target_language'),
            )
        except Exception as exc:
            last_error = exc
            logger.error(f"Resynthesize attempt {attempt + 1} failed for segment {segment_index}: {exc}")
            continue

        if os.path.exists(output_path):
            try:
                audio_info = AudioSegment.from_file(output_path)
            except Exception as exc:
                last_error = exc
                logger.error(f"Resynthesize produced unreadable audio for segment {segment_index}: {exc}")
                try:
                    os.remove(output_path)
                except OSError:
                    pass
                continue
            if len(audio_info) <= 0:
                try:
                    os.remove(output_path)
                except OSError:
                    pass
                continue

            segment_dict['_tts_cache_contract'] = 'anchor_raw_v2'
            segment_dict['synthesized_speech_len'] = facade._measure_raw_tts_for_timing(
                output_path,
                segment_index,
            )
            if segment_dict['synthesized_speech_len'] <= 0:
                last_error = RuntimeError("TTS produced only silence")
                continue
            segment_dict['synthesized_speech_file'] = output_path
            segment_dict['synthesized_text'] = text_to_synthesize
            if override_text:
                segment_dict['translation'] = text_to_synthesize
            return segment_dict

    raise RuntimeError(
        f"TTS ({tts_system}) did not produce an audio file for segment {segment_index} "
        f"after {max_attempts} attempts. Last error: {last_error!r}. "
        f"Reference audio: {segment_data.reference_audio_path!r}"
    )


def get_tts_system_for_speaker(facade, speaker_id: str) -> str:
    """Return the TTS backend name a speaker is routed to.

    Preserved for callers/tests that only need the backend name; new code
    should use :meth:`_resolve_voice_profile` to get the full profile.
    """
    return facade._resolve_voice_profile(speaker_id).tts_system or facade._default_tts_system()


def preflight_tts_pools(
    segments_by_pool: Dict[tuple, List[Any]],
    clients: Dict[tuple, Any],
    initial_issues: Optional[List[tuple[int, str]]] = None,
) -> None:
    """Validate every active pool before the first synthesis request."""
    issues = list(initial_issues or [])
    resolved_error_indices = {index for index, _ in issues}
    for pool_key, segments_data in segments_by_pool.items():
        if not segments_data:
            continue
        client = clients.get(pool_key)
        if client is None:
            continue
        validator = getattr(client, "validate_segments", None)
        if validator is None:
            continue
        for issue in validator(segments_data):
            if issue[0] not in resolved_error_indices:
                issues.append(issue)

    if issues:
        issues.sort(key=lambda item: item[0])
        details = "\n".join(f"- {message}" for _, message in issues)
        raise ValueError(f"TTS reference validation failed before synthesis:\n{details}")


def cache_raw_tts_segment(
    facade,
    source_path: str,
    cache_path: Path,
    *,
    synthesized_text: str = "",
) -> None:
    """Cache raw TTS plus a version marker, independent of timing policy."""
    shutil.copy(source_path, cache_path)
    metadata_path = facade._segment_cache_metadata_path(cache_path)
    metadata_path.write_text(
        json.dumps(
            {
                "audio_contract": "anchor_raw_v2",
                "synthesized_text": synthesized_text,
            },
            ensure_ascii=False,
            sort_keys=True,
        ),
        encoding="utf-8",
    )


def cached_segment_metadata(facade, cache_path: Path) -> Dict[str, Any]:
    metadata_path = facade._segment_cache_metadata_path(cache_path)
    try:
        data = json.loads(metadata_path.read_text(encoding="utf-8"))
        return data if isinstance(data, dict) else {}
    except (OSError, ValueError, TypeError):
        return {}


def cached_segment_contract(facade, cache_path: Path) -> str:
    contract = facade._cached_segment_metadata(cache_path).get("audio_contract")
    if contract in {"anchor_raw_v1", "anchor_raw_v2"}:
        return contract
    return "legacy"
