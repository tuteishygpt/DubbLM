"""Probe what the current Gemini TTS model actually returns.

Reproduces the exact call shape from the Google storytelling notebook
(https://github.com/GoogleCloudPlatform/generative-ai/blob/main/audio/speech/use-cases/storytelling/storytelling.ipynb)
so we can see the response's `inline_data.mime_type` and rule out that the
current wrapper is checking for a mime_type the new model does not emit.
"""

from __future__ import annotations

import sys
from pathlib import Path

REPO_ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(REPO_ROOT))
sys.path.insert(0, str(REPO_ROOT / "src"))

from dotenv import load_dotenv
load_dotenv(REPO_ROOT / ".env", override=True)

from google import genai
from google.genai import types as genai_types


MODEL = "gemini-3.1-flash-tts-preview"
VOICE = "Achird"
TEXT = "This is a short synthesis smoke test for the configured voice profile."


def main() -> int:
    client = genai.Client()

    config = genai_types.GenerateContentConfig(
        response_modalities=["AUDIO"],
        speech_config=genai_types.SpeechConfig(
            voice_config=genai_types.VoiceConfig(
                prebuilt_voice_config=genai_types.PrebuiltVoiceConfig(voice_name=VOICE)
            ),
        ),
    )

    print(f"Calling {MODEL} with voice={VOICE}")
    response = client.models.generate_content(model=MODEL, contents=TEXT, config=config)

    if not response.candidates:
        print("No candidates in response")
        return 1

    for i, part in enumerate(response.candidates[0].content.parts or []):
        inline = getattr(part, "inline_data", None)
        text = getattr(part, "text", None)
        if inline is not None:
            data = inline.data or b""
            print(f"  part[{i}]: inline_data mime_type={inline.mime_type!r}, bytes={len(data)}")
            if data:
                out = REPO_ROOT / "prj" / "_voice_profile_samples" / f"probe_{MODEL}_{VOICE}.wav"
                out.parent.mkdir(parents=True, exist_ok=True)
                # Wrap raw PCM into WAV if mime says PCM/L16
                if inline.mime_type and "pcm" in inline.mime_type.lower():
                    import wave
                    with wave.open(str(out), "wb") as wf:
                        wf.setnchannels(1)
                        wf.setsampwidth(2)
                        wf.setframerate(24000)
                        wf.writeframes(data)
                    print(f"  saved WAV to {out}")
                else:
                    out.write_bytes(data)
                    print(f"  saved raw bytes to {out}")
        elif text is not None:
            print(f"  part[{i}]: text={text[:120]!r}")
        else:
            print(f"  part[{i}]: {part}")

    return 0


if __name__ == "__main__":
    sys.exit(main())
