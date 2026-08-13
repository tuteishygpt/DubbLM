"""Internal translation and editor-persistence stage implementations."""

from typing import Dict, List, Optional

from translation.translator_factory import TranslatorFactory

from ..log_config import get_logger
from .context import update_context as _update_pipeline_context

logger = get_logger(__name__)


def initialize_translator(facade) -> None:
    """Initialize translator based on configuration."""
    facade.translator = None
    facade.translator_init_error = None
    try:
        settings = facade._effective_translation_cache_dimensions()
        primary = settings["primary"]
        refinement = settings["refinement"]
        facade.translator = TranslatorFactory.create_translator(
            translator_type=settings["translator_type"],
            llm_provider=primary["provider"],
            model_name=primary["model"],
            temperature=primary["temperature"],
            max_tokens=primary["max_tokens"],
            refinement_llm_provider=refinement["provider"],
            refinement_model_name=refinement["model"],
            refinement_temperature=refinement["temperature"],
            refinement_max_tokens=refinement["max_tokens"],
            refinement_persona=refinement["persona"],
            translation_prompt_prefix=facade.config.get('translation_prompt_prefix'),
            glossary=facade.config.get('glossary'),
            cache_manager=facade.cache_manager
        )
        logger.debug(f"Using {facade.config.get('translator_type', 'llm')} translator")
    except Exception as e:
        facade.translator_init_error = e
        logger.warning(f"Failed to initialize translator: {e}")


def require_translator(facade):
    """Return the initialized translator or raise an actionable error."""
    if facade.translator is not None:
        return facade.translator

    raise RuntimeError(
        facade._format_component_init_error(
            "Translator",
            facade.config.get('translator_type', 'llm'),
            getattr(facade, "translator_init_error", None),
        )
    )


def persist_dubbing_text_snapshot(
    facade, segments: List[Dict], audio_file: str
) -> None:
    """Persist the latest real segment state for the Dubbing Texts editor."""
    translation_cache_reusable = getattr(
        facade, "_semantic_plan_cache_persistable", True
    )
    try:
        snapshot_key = facade._build_dubbing_text_snapshot_key(audio_file)
        snapshot_payload = {
            "version": 1,
            "segments": segments,
            "translation_cache_reusable": translation_cache_reusable,
            "translation_cache_key": (
                facade._build_translation_cache_key(audio_file)
                if translation_cache_reusable
                else None
            ),
        }
        facade.cache_manager.save_to_cache(
            "dubbing_texts", snapshot_key, snapshot_payload
        )
    except Exception as e:
        logger.warning(f"Could not persist Dubbing Texts snapshot: {e}")


def persist_synthesis_results(
    facade, segments: List[Dict], audio_file: str
) -> None:
    """Persist post-synthesis editor state and reusable pipeline state."""
    facade._persist_dubbing_text_snapshot(segments, audio_file)

    translation_cache_reusable = getattr(
        facade, "_semantic_plan_cache_persistable", True
    )
    if not translation_cache_reusable:
        return
    try:
        cache_key = facade._build_translation_cache_key(audio_file)
        facade.cache_manager.save_to_cache("translation", cache_key, segments)
    except Exception as e:
        logger.warning(f"Could not persist synthesis results to translation cache: {e}")


def translate_segments(
    facade, transcription: List[Dict], audio_file: str
) -> List[Dict]:
    """Translate segments using the translator."""
    semantic_fingerprints = {
        segment.get("semantic_plan_fingerprint")
        for segment in transcription
        if segment.get("semantic_plan_fingerprint")
    }
    semantic_segments = [
        segment for segment in transcription if segment.get("semantic_unit_id")
    ]
    if semantic_segments and (
        len(semantic_fingerprints) != 1
        or any(
            segment.get("semantic_plan_fingerprint") not in semantic_fingerprints
            for segment in semantic_segments
        )
    ):
        raise ValueError(
            "Semantic transcription has a missing or inconsistent semantic_plan_fingerprint"
        )
    if len(semantic_fingerprints) > 1:
        raise ValueError("Transcription contains multiple semantic plan fingerprints")
    if semantic_fingerprints:
        _update_pipeline_context(
            facade,
            "semantic_plan_fingerprint",
            next(iter(semantic_fingerprints)),
        )
    cache_key = facade._build_translation_cache_key(audio_file)
    step_name = "translation"

    translated_segments = None
    if facade.cache_manager.cache_exists(step_name, cache_key):
        logger.debug("Loading translations from cache...")
        translated_segments = facade.cache_manager.load_from_cache(step_name, cache_key)
        if translated_segments is not None:
            facade._validate_plan_dependent_segments(translated_segments)
            facade.performance_tracker.record_metric("translation", 0.0)
        else:
            logger.warning("Found corrupted translation cache, re-translating.")

    if translated_segments is None:
        facade.performance_tracker.start_timing("translation")
        translator = facade._require_translator()

        if not translator.is_available():
            raise ValueError("No translator available")

        original_prompt_prefix = getattr(translator, "prompt_prefix", None)
        if hasattr(translator, "prompt_prefix"):
            translator.prompt_prefix = facade._build_translation_prompt_prefix(
                original_prompt_prefix
            )

        try:
            translated_segments = translator.translate(
                segments=transcription,
                source_language=facade.config.get('source_language'),
                target_language=facade.config.get('target_language'),
                refinement_persona=facade.config.get('refinement_persona', 'normal'),
                debug=facade.debug_data,
                debug_dir=facade.config.get("translation_debug_dir"),
                refinement_debug_dir=facade.config.get("translation_refinement_debug_dir"),
                timecodes_report_path=facade.config.get("timecodes_report_path"),
            )
        finally:
            if hasattr(translator, "prompt_prefix"):
                translator.prompt_prefix = original_prompt_prefix

        if getattr(facade, "_semantic_plan_cache_persistable", True):
            facade.cache_manager.save_to_cache(step_name, cache_key, translated_segments)

        elapsed_time = facade.performance_tracker.end_timing("translation")
        logger.info(f"Finished translation in {elapsed_time:.2f} seconds (≈ {elapsed_time/60:.2f} minutes)")

    facade.debug_data["translation"] = translated_segments
    facade._persist_dubbing_text_snapshot(translated_segments, audio_file)

    return translated_segments


def build_translation_prompt_prefix(
    facade, base_prompt_prefix: Optional[str], stress_marks_requirement: str
) -> str:
    """Combine a user prompt prefix with SmartDubbing TTS stress rules."""
    base_prompt = (base_prompt_prefix or "").strip()
    if "U+0301" in base_prompt or "каса́" in base_prompt:
        return base_prompt
    if base_prompt:
        return f"{base_prompt}\n\n{stress_marks_requirement}"
    return stress_marks_requirement
