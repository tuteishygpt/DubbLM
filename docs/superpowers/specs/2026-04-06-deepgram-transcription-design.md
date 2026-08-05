# Deepgram Transcription Backend Design

**Date:** 2026-04-06

## Goal

Add Deepgram as a full transcription backend that can be selected through `transcription_system=deepgram` in runtime config, CLI, and Gradio UI.

## Scope

- Add a dedicated Deepgram transcriber module under `src/transcription/`.
- Use Deepgram utterances as the canonical segmentation source.
- Keep the downstream internal transcription contract unchanged.
- Expose backend selection in CLI, YAML, and Gradio UI.

## Architecture

- Implement `DeepgramTranscriber` as a `BaseTranscriber`.
- Authenticate via `DEEPGRAM_API_KEY` from environment variables.
- Submit local audio files to Deepgram with diarization and utterance segmentation enabled.
- Normalize the Deepgram response into:
  - `speakers_rolls: Dict[(start, end), speaker_id]`
  - `transcription: List[{"text","start","end","speaker",...}]`

## Parsing Rules

- Use `results.utterances` when present.
- Keep Deepgram utterance boundaries intact instead of applying local re-splitting.
- Normalize Deepgram speaker IDs into stable labels like `SPEAKER_00`, `SPEAKER_01`.
- Preserve confidence and word-level timing when available.
- Fall back to coarser transcript data only if `utterances` are missing.

## Non-Goals

- No hardcoded API key in source code.
- No URL-only integration path.
- No post-processing that changes natural phrase boundaries returned by Deepgram.
