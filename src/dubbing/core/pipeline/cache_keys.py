"""Internal cache identity implementations for the dubbing pipeline."""

import hashlib
import json
import os
from pathlib import Path
from typing import Any, Dict, Iterable, List, Optional
from urllib.parse import quote


def cache_fingerprint(dimensions: Dict[str, Any]) -> str:
    serialized = json.dumps(
        dimensions,
        ensure_ascii=False,
        sort_keys=True,
        separators=(",", ":"),
        allow_nan=False,
    )
    return hashlib.sha256(serialized.encode("utf-8")).hexdigest()[:20]


def effective_tts_cache_fingerprint(facade: Any, speakers: Iterable[str]) -> str:
    profiles = []
    content_identities: Dict[str, str] = {}
    use_content_hashes = getattr(
        getattr(facade, "cache_manager", None), "use_cache", True
    )

    def content_identity(path: Optional[str]) -> str:
        identity_key = str(path or "")
        if not use_content_hashes:
            return identity_key
        if identity_key not in content_identities:
            content_identities[identity_key] = facade._file_content_identity(path)
        return content_identities[identity_key]

    for speaker in sorted({str(value) for value in speakers}):
        profile = facade._resolve_voice_profile(speaker)
        provider = profile.tts_system or facade._default_tts_system()
        voice_name = profile.voice_name
        if voice_name is None:
            global_voice_name = facade.config.get("voice_name")
            if isinstance(global_voice_name, str):
                voice_name = global_voice_name
        provider_params: Dict[str, Any] = {}
        if provider.lower() == "omnivoice":
            provider_params.update(
                {
                    key: value
                    for key, value in facade._global_omnivoice_kwargs().items()
                    if value is not None
                }
            )
        provider_params.update(profile.params or {})
        effective_reference_path = profile.reference_audio
        if profile.reference_mode == "speaker":
            speakers_dir = getattr(facade, "speakers_audio_dir", None)
            if speakers_dir is not None:
                effective_reference_path = str(Path(speakers_dir) / f"{speaker}.wav")
        segment_source_identity = None
        segment_source_offset = None
        if profile.reference_mode == "segment":
            isolated_tracks = facade.config.get("isolated_tracks")
            if isinstance(isolated_tracks, dict) and speaker in isolated_tracks:
                selected_source = isolated_tracks.get(speaker)
                segment_source_identity = content_identity(str(selected_source or ""))
                segment_source_offset = facade.config.get("start_time") or 0.0
            else:
                vocals_path, source_path = facade._segment_reference_artifact_paths()
                segment_source_identity = {
                    "vocals": content_identity(str(vocals_path))
                    if facade.config.get("keep_background", False)
                    else None,
                    "source": content_identity(str(source_path)),
                }
                segment_source_offset = 0.0
        profiles.append(
            {
                "speaker": speaker,
                "provider": provider,
                "model": profile.model or facade.config.get("tts_model"),
                "voice": voice_name,
                "style_prompt": profile.style_prompt,
                "reference_mode": profile.reference_mode,
                "reference_audio": profile.reference_audio,
                "reference_audio_identity": content_identity(effective_reference_path),
                "reference_text": profile.reference_text,
                "segment_source_identity": segment_source_identity,
                "segment_source_offset": segment_source_offset,
                "params": dict(sorted(provider_params.items())),
            }
        )
    return facade._cache_fingerprint(
        {"contract": "strict-reference-v1", "speakers": profiles}
    )


def file_content_identity(path: Optional[str]) -> str:
    identity = str(path or "")
    if not path:
        return identity
    try:
        file_path = Path(path)
        if not file_path.is_file():
            return identity
        digest = hashlib.sha256()
        with file_path.open("rb") as source_file:
            for chunk in iter(lambda: source_file.read(1024 * 1024), b""):
                digest.update(chunk)
        return f"{file_path.resolve(strict=False)}:{digest.hexdigest()}"
    except OSError:
        return identity


def shared_audio_transcription_identity(facade: Any, audio_file: str) -> str:
    return facade.cache_manager.generate_cache_key(
        audio_file,
        facade.config.get("source_language"),
        facade.config.get("target_language"),
        facade.config.get("whisper_model", "large-v3"),
        facade.config.get("start_time"),
        facade.config.get("duration"),
    )


def effective_translation_cache_dimensions(
    facade: Any,
    default_llm_models: Dict[str, Any],
    semantic_plan_fingerprint: Optional[str],
) -> Dict[str, Any]:
    primary_provider = facade.config.get("llm_provider") or "gemini"
    primary_model = facade.config.get("llm_model_name") or default_llm_models.get(
        primary_provider
    )
    primary_temperature = facade.config.get("llm_temperature", 0.5)
    if primary_temperature is None:
        primary_temperature = 0.5
    primary_max_tokens = facade.config.get("llm_max_tokens", 16384)
    if primary_max_tokens is None:
        primary_max_tokens = 16384
    refinement_provider = (
        facade.config.get("refinement_llm_provider") or primary_provider
    )
    refinement_model = facade.config.get("refinement_model_name") or primary_model
    refinement_temperature = facade.config.get("refinement_temperature", 1.0)
    if refinement_temperature is None:
        refinement_temperature = 1.0
    refinement_max_tokens = (
        facade.config.get("refinement_max_tokens") or primary_max_tokens
    )
    return {
        "schema_version": "translation_v2",
        "translator_type": facade.config.get("translator_type") or "llm",
        "primary": {
            "provider": primary_provider,
            "model": primary_model,
            "temperature": primary_temperature,
            "max_tokens": primary_max_tokens,
        },
        "refinement": {
            "provider": refinement_provider,
            "model": refinement_model,
            "temperature": refinement_temperature,
            "max_tokens": refinement_max_tokens,
            "persona": facade.config.get("refinement_persona") or "normal",
        },
        "glossary": facade.config.get("glossary") or {},
        "prompt_prefix": facade._build_translation_prompt_prefix(
            facade.config.get("translation_prompt_prefix")
        ),
        "semantic_plan_fingerprint": semantic_plan_fingerprint,
    }


def build_dubbing_text_snapshot_key(facade: Any, audio_file: str) -> str:
    audio_identity = facade._shared_audio_transcription_identity(audio_file)
    dimensions = {
        "schema_version": "dubbing_texts_v2",
        "prompt_prefix": facade._build_translation_prompt_prefix(
            facade.config.get("translation_prompt_prefix")
        ),
    }
    return f"dubbing_texts_v2_{audio_identity}_{facade._cache_fingerprint(dimensions)}"


def build_translation_cache_key(facade: Any, audio_file: str) -> str:
    snapshot_identity = facade._build_dubbing_text_snapshot_key(audio_file)
    settings = facade._effective_translation_cache_dimensions()
    return f"translation_v2_{snapshot_identity}_{facade._cache_fingerprint(settings)}"


def build_emotions_cache_key(
    facade: Any,
    audio_file: str,
    segments: List[Dict[str, Any]],
    provider: Optional[str],
    model: Optional[str],
    *,
    emotion_analysis_prompt: str,
    soft_style_by_emotion: Dict[str, str],
    semantic_plan_fingerprint: Optional[str],
) -> str:
    provider = str(provider or facade.config.get("emotion_provider") or "gemini").lower()
    model = str(model or facade.config.get("emotion_model") or "gemini-3.1-flash-lite")
    audio_identity = facade.cache_manager.generate_cache_key(
        audio_file, "", "", "analysis-audio-v2"
    )
    if provider == "gemini":
        namespace = f"gemini_{quote(model, safe='.-_')}"
        algorithm = {
            "prompt": emotion_analysis_prompt,
            "temperature": 0.2,
            "accepted_labels": ["Neutral", "Angry", "Happy", "Sad"],
        }
    elif provider == "speechbrain":
        namespace = "speechbrain"
        algorithm = {
            "source": "speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
            "pymodule_file": "custom_interface.py",
            "classname": "CustomEncoderWav2vec2Classifier",
            "label_mapping": {
                "neu": "Neutral", "ang": "Angry", "hap": "Happy",
                "sad": "Sad", "None": None,
            },
            "style_mapping": soft_style_by_emotion,
        }
    else:
        namespace = quote(provider, safe=".-_")
        algorithm = {"fallback_emotion": "Neutral"}
    dimensions = {
        "schema_version": "emotions_v2",
        "segments": segments,
        "semantic_plan_fingerprint": semantic_plan_fingerprint,
        "algorithm": algorithm,
    }
    return f"emotions_v2_{namespace}_{audio_identity}_{facade._cache_fingerprint(dimensions)}"


def isolated_tracks_cache_key(
    facade: Any, audio_file: str, isolated_tracks: Dict[str, str]
) -> str:
    inner_system = facade.config.get("inner_transcription_system", "deepgram")
    base_key = facade.cache_manager.generate_cache_key(
        audio_file,
        facade.config.get("source_language"),
        facade.config.get("target_language"),
        facade.config.get("whisper_model", "large-v3"),
        facade.config.get("start_time"),
        facade.config.get("duration"),
    )
    fingerprint = hashlib.md5()
    for speaker in sorted(isolated_tracks.keys()):
        path = isolated_tracks[speaker]
        fingerprint.update(speaker.encode("utf-8"))
        fingerprint.update(b"=")
        try:
            stat = os.stat(path)
            fingerprint.update(str(stat.st_size).encode("utf-8"))
            fingerprint.update(str(int(stat.st_mtime)).encode("utf-8"))
        except OSError:
            fingerprint.update(str(path).encode("utf-8"))
        fingerprint.update(b";")
    fingerprint.update(inner_system.encode("utf-8"))
    if facade.config.get("semantic_split_enabled", True):
        from ...audio.semantic_planner import SEMANTIC_PLANNER_VERSION

        fingerprint.update(facade._isolated_tracks_raw_cache_key(isolated_tracks).encode("utf-8"))
        semantic_payload = {
            "algorithm": SEMANTIC_PLANNER_VERSION,
            "incomplete_tail_rules": "incomplete_tail_en_v1",
            "prompt_parser": "semantic_boundary_prompt_v1",
            "semantic_split_enabled": True,
            "tts_preferred_segment_duration": facade.config.get("tts_preferred_segment_duration", 15.0),
            "tts_hard_segment_duration": facade.config.get("tts_hard_segment_duration", 35.0),
            "semantic_split_search_window": facade.config.get("semantic_split_search_window", 10.0),
            "source_language": facade.config.get("source_language"),
            "classifier": facade._semantic_classifier_identity(),
            "classifier_timeout": 30.0,
            "classifier_batch_size": 50,
            "classifier_batch_characters": 12000,
        }
        fingerprint.update(json.dumps(semantic_payload, sort_keys=True, separators=(",", ":")).encode("utf-8"))
    return f"{base_key}_isolated_{fingerprint.hexdigest()[:16]}"


def isolated_tracks_raw_cache_key(facade: Any, isolated_tracks: Dict[str, str]) -> str:
    payload: Dict[str, Any] = {
        "algorithm": "isolated_raw_v1",
        "source_language": facade.config.get("source_language"),
        "inner_transcription_system": facade.config.get("inner_transcription_system", "deepgram"),
        "start_time": facade.config.get("start_time"),
        "duration": facade.config.get("duration"),
        "deepgram_model": facade.config.get("deepgram_model"),
        "assemblyai_model": facade.config.get("transcription_model"),
        "gemini_model": facade.config.get("gemini_transcription_model"),
        "tracks": [],
    }
    for speaker in sorted(isolated_tracks):
        path = isolated_tracks[speaker]
        try:
            track_digest = hashlib.sha256()
            with open(path, "rb") as track_file:
                for chunk in iter(lambda: track_file.read(1024 * 1024), b""):
                    track_digest.update(chunk)
            identity = [speaker, track_digest.hexdigest()]
        except OSError:
            identity = [speaker, str(path), None, None]
        payload["tracks"].append(identity)
    digest = hashlib.sha256(json.dumps(payload, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:20]
    return f"isolated_raw_v1_{digest}"


def semantic_classifier_identity(facade: Any) -> Dict[str, Any]:
    _classifier, status = facade._semantic_classifier()
    if status == "deterministic-only":
        return {"mode": status}
    translator = getattr(facade, "translator", None)
    return {
        "mode": status,
        "provider": getattr(translator, "llm_provider", facade.config.get("llm_provider")),
        "model": getattr(translator, "model_name", facade.config.get("llm_model_name")),
        "temperature": getattr(translator, "temperature", facade.config.get("llm_temperature", 0.5)),
        "max_tokens": getattr(translator, "max_tokens", facade.config.get("llm_max_tokens", 16384)),
    }


def tts_selection_cache_fingerprint(
    facade: Any, segments: List[Dict[str, Any]]
) -> str:
    payload = {
        "selection_policy": "sequential_measured_v1",
        "tts_prompt_prefix": facade.config.get("tts_prompt_prefix"),
        # Retain the frozen cache dimension after removing legacy config input.
        "voice_prompt": None,
        "segments": [
            {
                "start": segment.get("start"), "end": segment.get("end"),
                "original_index": facade._canonical_segment_index(segment, index),
                "speaker": segment.get("speaker"), "reference_text": segment.get("text"),
                "emotion": segment.get("emotion", "Neutral"),
                "style_prompt": segment.get("style_prompt", ""),
                "translation": segment.get("translation", ""),
                "long_translation": segment.get("long_translation", ""),
                "short_translation": segment.get("short_translation", ""),
                "very_short_translation": segment.get("very_short_translation", ""),
            }
            for index, segment in enumerate(segments)
        ],
    }
    return hashlib.sha256(json.dumps(payload, ensure_ascii=False, sort_keys=True, separators=(",", ":")).encode("utf-8")).hexdigest()[:16]


def segment_cache_metadata_path(cache_path: Path) -> Path:
    return cache_path.with_suffix(cache_path.suffix + ".json")


def raw_tts_segment_cache_key(
    *, base_cache_prefix: str, tts_system: str, segment: Dict[str, Any],
    speaker: str, translation: str, style_prompt: str,
    reference_audio_path: Optional[str], reference_mode: Optional[str] = None,
    reference_text: Optional[str] = None, client_pool_settings: Any = None,
    legacy_index: Any = 0, emotion: Optional[str] = "Neutral",
    tts_prompt_prefix: Optional[str] = None, voice_prompt: Any = None,
    cache_fingerprint=cache_fingerprint, file_content_identity=file_content_identity,
) -> str:
    translation_hash = hashlib.md5(translation.encode()).hexdigest()[:8]
    target_duration = round(float(segment.get("_timing_available_window", 0.0)), 6)
    target_hash = hashlib.md5(str(target_duration).encode()).hexdigest()[:8]
    instruction_hash = hashlib.md5(cache_fingerprint({
        "style_prompt": style_prompt, "emotion": emotion or "Neutral",
        "tts_prompt_prefix": tts_prompt_prefix, "voice_prompt": voice_prompt,
    }).encode("utf-8")).hexdigest()[:8]
    reference_identity = file_content_identity(reference_audio_path)
    ref_audio_hash = hashlib.md5(reference_identity.encode()).hexdigest()[:8]
    reference_contract_hash = hashlib.md5(cache_fingerprint({
        "reference_mode": reference_mode, "reference_audio_path": reference_audio_path,
        "reference_text": reference_text, "segment_start": segment.get("start"),
        "segment_end": segment.get("end"), "client_pool_settings": client_pool_settings,
    }).encode("utf-8")).hexdigest()[:8]
    semantic_unit_identity = segment.get("semantic_unit_id", legacy_index)
    semantic_plan_identity = segment.get("semantic_plan_fingerprint", "legacy")
    return (
        f"{base_cache_prefix}_{tts_system}_{semantic_plan_identity}_"
        f"{semantic_unit_identity}_{speaker}_{translation_hash}_raw_candidate_v4_"
        f"{target_hash}_{instruction_hash}_{ref_audio_hash}_{reference_contract_hash}"
    )
