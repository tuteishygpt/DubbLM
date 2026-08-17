"""Internal emotion-analysis stage implementations."""

import os
from typing import Dict, List

from ..log_config import get_logger

logger = get_logger(__name__)


def analyze_emotions(facade, segments: List[Dict], audio_file: str) -> List[Dict]:
    """Analyze emotions in the audio for each segment."""
    if not segments:
        return []

    provider = str(facade.config.get("emotion_provider") or "gemini").lower()
    model = str(facade.config.get("emotion_model") or "gemini-3.1-flash-lite")

    cache_key = facade._build_emotions_cache_key(
        audio_file, segments, provider, model
    )
    step_name = "emotions"

    if facade.cache_manager.cache_exists(step_name, cache_key):
        logger.debug("Loading emotion analysis from cache...")
        cached_segments = facade.cache_manager.load_from_cache(step_name, cache_key)
        if cached_segments is not None:
            try:
                facade._validate_plan_dependent_segments(cached_segments)
            except ValueError:
                logger.warning(
                    "Emotion cache does not match the active semantic plan; "
                    "re-analyzing."
                )
            else:
                return cached_segments
        logger.warning("Found corrupted emotion cache, re-analyzing.")

    logger.info("Analyzing speech emotions (provider=%s, model=%s)...", provider, model)
    facade.performance_tracker.start_timing("emotion_analysis")

    try:
        if provider == "gemini":
            facade._analyze_emotions_gemini(segments, audio_file, model)
        elif provider == "speechbrain":
            facade._analyze_emotions_speechbrain(segments, audio_file)
        else:
            logger.warning("Unknown emotion_provider '%s'; defaulting all segments to Neutral.", provider)
            for segment in segments:
                segment["emotion"] = "Neutral"
    finally:
        facade.performance_tracker.end_timing("emotion_analysis")

    facade.cache_manager.save_to_cache(step_name, cache_key, segments)
    return segments


def analyze_emotions_gemini(
    facade,
    segments: List[Dict],
    audio_file: str,
    model: str,
    emotion_analysis_prompt: str,
) -> None:
    """Classify each segment's emotion via a Gemini multimodal model on Vertex AI."""
    try:
        from google import genai
        from google.genai import types as genai_types
    except ImportError:
        logger.warning(
            "google-genai package unavailable; falling back to Neutral for all segments. "
            "Install google-genai or switch emotion_provider to 'speechbrain'."
        )
        for segment in segments:
            segment["emotion"] = "Neutral"
        return

    try:
        from google_vertex import get_vertex_ai_settings
        vertex_settings = get_vertex_ai_settings()
        client = genai.Client(**vertex_settings.genai_client_kwargs)
    except Exception as exc:
        logger.warning("Vertex AI unavailable for emotion analysis (%s); defaulting to Neutral.", exc)
        for segment in segments:
            segment["emotion"] = "Neutral"
        return

    import io
    from pydub import AudioSegment

    prompt = emotion_analysis_prompt
    config_obj = genai_types.GenerateContentConfig(temperature=0.2)
    allowed = {"Neutral", "Angry", "Happy", "Sad"}

    audio = AudioSegment.from_file(audio_file)
    for segment in segments:
        try:
            start = max(int(segment["start"] * 1000), 0)
            end = min(int(segment["end"] * 1000), len(audio))
            if end <= start:
                segment["emotion"] = "Neutral"
                segment.setdefault("style_prompt", "")
                continue

            segment_audio = audio[start:end]
            buffer = io.BytesIO()
            segment_audio.export(buffer, format="wav")
            audio_part = genai_types.Part.from_bytes(
                data=buffer.getvalue(),
                mime_type="audio/wav",
            )

            response = client.models.generate_content(
                model=model,
                contents=[audio_part, prompt],
                config=config_obj,
            )
            raw = (getattr(response, "text", None) or "").strip()
            style_text = ""
            emotion_label = "Neutral"
            for line in raw.splitlines():
                stripped = line.strip()
                if not stripped:
                    continue
                lower = stripped.lower()
                if lower.startswith("style:"):
                    style_text = stripped.split(":", 1)[1].strip().strip('"\'')
                elif lower.startswith("emotion:"):
                    tag = stripped.split(":", 1)[1].strip().split()[0]
                    tag = tag.strip(".,!?\"'").capitalize()
                    if tag in allowed:
                        emotion_label = tag
            if not style_text and raw:
                first_line = next((ln.strip() for ln in raw.splitlines() if ln.strip()), "")
                style_text = first_line.strip('"\'')
            segment["emotion"] = emotion_label
            segment["style_prompt"] = style_text
        except Exception as exc:
            logger.warning("Gemini emotion classification failed for segment: %s", exc)
            segment["emotion"] = "Neutral"
            segment.setdefault("style_prompt", "")


def analyze_emotions_speechbrain(
    facade,
    segments: List[Dict],
    audio_file: str,
    soft_style_by_emotion: Dict[str, str],
) -> None:
    """Legacy speechbrain-based classifier (IEMOCAP wav2vec2)."""
    try:
        from speechbrain.inference.interfaces import foreign_class
    except ModuleNotFoundError as exc:
        if exc.name != "speechbrain":
            raise
        logger.warning(
            "Skipping emotion analysis because optional dependency "
            "'speechbrain' is unavailable. Install it or switch emotion_provider to 'gemini'."
        )
        for segment in segments:
            segment["emotion"] = "Neutral"
        return

    classifier = foreign_class(
        source="speechbrain/emotion-recognition-wav2vec2-IEMOCAP",
        pymodule_file="custom_interface.py",
        classname="CustomEncoderWav2vec2Classifier",
        run_opts={"device": facade.torch_device},
    )
    emotion_dict = {
        'neu': 'Neutral',
        'ang': 'Angry',
        'hap': 'Happy',
        'sad': 'Sad',
        'None': None,
    }

    from pydub import AudioSegment
    audio = AudioSegment.from_file(audio_file)
    for segment in segments:
        try:
            start = int(segment["start"] * 1000)
            end = int(segment["end"] * 1000)
            segment_audio = audio[start:end]
            temp_segment_path = facade.config.get("temp_segment_audio_path")
            segment_audio.export(temp_segment_path, format="wav")
            out_prob, score, index, text_lab = classifier.classify_file(temp_segment_path)
            emotion = emotion_dict[text_lab[0]] or "Neutral"
            segment["emotion"] = emotion
            segment["style_prompt"] = soft_style_by_emotion.get(emotion, "")
            os.remove(temp_segment_path)
        except Exception as e:
            logger.warning(f"Error analyzing emotion: {e}")
            segment["emotion"] = "Neutral"
            segment.setdefault("style_prompt", "")
