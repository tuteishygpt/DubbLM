"""Smoke-test each configured voice profile by synthesising one sample.

Reads `dubbing_config.yml`, resolves the per-speaker VoiceProfile map, spins up
the corresponding TTS client for each, and writes one WAV per speaker into
`prj/_voice_profile_samples/`. Prints a summary so it's obvious which profile
succeeded and which fell back / failed.

Run:
    .\\.venv\\Scripts\\python.exe scripts/test_voice_profiles_synthesis.py
"""

from __future__ import annotations

import os
import sys
import traceback
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=True)

import warnings
warnings.simplefilter("ignore", DeprecationWarning)

import yaml

from dubbing.core.voice_profiles import FALLBACK_SPEAKER, normalize_voices, resolve_profile
from tts.models import TTSSegmentData
from tts.tts_factory import TTSFactory


SAMPLE_SENTENCES: dict[str, str] = {
    "en": "This is a short synthesis smoke test for the configured voice profile.",
    "ru": "Это короткий тестовый образец озвучки для настроенного профиля голоса.",
    "be": "Гэта кароткі тэст сінтэзу для наладжанага галасавога профілю.",
}


def _load_raw_config(path: Path) -> dict:
    with path.open("r", encoding="utf-8") as f:
        return yaml.safe_load(f) or {}


def _sample_text(target_lang: str) -> str:
    return SAMPLE_SENTENCES.get(target_lang, SAMPLE_SENTENCES["en"])


def _global_omnivoice_kwargs(cfg: dict) -> dict:
    return {
        "space_id": cfg.get("omnivoice_space_id"),
        "api_name": cfg.get("omnivoice_api_name"),
        "lang": cfg.get("omnivoice_lang") or "Russian",
        "instruct": cfg.get("omnivoice_instruct", ""),
        "num_steps": cfg.get("omnivoice_num_steps"),
        "guidance_scale": cfg.get("omnivoice_guidance_scale"),
        "denoise": cfg.get("omnivoice_denoise"),
        "speed": cfg.get("omnivoice_speed"),
        "duration": cfg.get("omnivoice_duration"),
        "preprocess_prompt": cfg.get("omnivoice_preprocess_prompt"),
        "postprocess_output": cfg.get("omnivoice_postprocess_output"),
    }


def _build_client_for_profile(profile, cfg: dict):
    tts_system = profile.tts_system or cfg.get("tts_system", "coqui")
    model = profile.model or cfg.get("tts_model")
    fallback_model = profile.fallback_model or cfg.get("tts_fallback_model")

    bootstrap = {}
    if tts_system.lower() == "omnivoice":
        bootstrap.update(_global_omnivoice_kwargs(cfg))
    bootstrap.update(profile.params or {})

    return TTSFactory.create_tts(
        tts_system=tts_system,
        device="cpu",
        voice_config=None,
        voice_prompt=None,
        prompt_prefix=None,                 # smoke test — don't inject a style prefix
        enable_voice_matching=False,        # skip embedding-based voice auto-selection
        enable_audio_validation=False,      # accept whatever the model returns
        debug_tts=False,
        model=model,
        fallback_model=fallback_model,
        default_reference_audio=cfg.get("reference_audio"),
        **bootstrap,
    )


def _synthesise_one(profile, speaker: str, client, target_lang: str, output_path: Path) -> None:
    text = _sample_text(target_lang)
    segment = TTSSegmentData(
        speaker=speaker,
        text=text,
        voice=profile.voice_name,
        style_prompt=profile.style_prompt,
        reference_audio_path=profile.reference_audio,
        reference_text=profile.reference_text,
        speed=1.0,
        target_duration=None,
        output_path=str(output_path),
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)
    client.synthesize(segments_data=[segment], language=target_lang)


def main() -> int:
    config_path = REPO_ROOT / "dubbing_config.yml"
    cfg = _load_raw_config(config_path)
    target_lang = cfg.get("target_language", "en")

    profiles = normalize_voices(cfg)
    if not profiles:
        print("No profiles resolved from dubbing_config.yml (missing 'voices' block?).")
        return 1

    # Optional CLI arg: `--only SPEAKER_00` runs just one profile.
    only_speaker: str | None = None
    for i, arg in enumerate(sys.argv[1:]):
        if arg == "--only" and i + 1 < len(sys.argv[1:]):
            only_speaker = sys.argv[1:][i + 1]
            break

    samples_dir = REPO_ROOT / "prj" / "_voice_profile_samples"
    samples_dir.mkdir(parents=True, exist_ok=True)

    results: list[tuple[str, str, str]] = []  # (speaker, status, detail)

    for speaker in sorted(profiles.keys()):
        if speaker == FALLBACK_SPEAKER:
            continue
        if only_speaker and speaker != only_speaker:
            continue

        profile = resolve_profile(profiles, speaker, tts_system_default=cfg.get("tts_system", "coqui"))
        tag = f"{profile.tts_system}"
        if profile.model:
            tag += f":{profile.model}"
        if profile.voice_name:
            tag += f"/{profile.voice_name}"

        out_path = samples_dir / f"{speaker}__{profile.tts_system}.wav"
        print(f"\n== {speaker} — {tag} ==")
        print(f"   output: {out_path}")

        try:
            client = _build_client_for_profile(profile, cfg)
            _synthesise_one(profile, speaker, client, target_lang, out_path)
        except Exception as exc:
            print(f"   FAIL: {exc}")
            traceback.print_exc()
            results.append((speaker, "FAIL", f"{tag} — {exc}"))
            continue

        if out_path.is_file() and out_path.stat().st_size > 0:
            size_kb = out_path.stat().st_size / 1024
            print(f"   OK ({size_kb:.1f} KB)")
            results.append((speaker, "OK", f"{tag} — {size_kb:.1f} KB"))
        else:
            print("   FAIL: no audio produced")
            results.append((speaker, "FAIL", f"{tag} — no audio produced"))

        try:
            client.cleanup()
        except Exception:
            pass

    print("\n=== Summary ===")
    for speaker, status, detail in results:
        print(f"  [{status}] {speaker}: {detail}")

    return 0 if all(status == "OK" for _, status, _ in results) else 2


if __name__ == "__main__":
    sys.exit(main())
