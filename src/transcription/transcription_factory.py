"""
Factory for creating transcription and diarization services.
"""
from typing import Optional, Literal

from transcription.transcription_interface import TranscriptionInterface

class TranscriptionFactory:
    """Factory for creating transcription and diarization services."""
    
    @staticmethod
    def create_transcriber(
        transcription_system: Literal["pyannote_openai", "whisper", "openai", "whisperx", "assemblyai", "gemini", "deepgram"],
        source_language: str,
        device: Optional[str] = None,
        **kwargs
    ) -> TranscriptionInterface:
        """
        Create a transcription service implementation based on the specified system.
        
        Args:
            transcription_system: The transcription system to use
            source_language: Source language code (e.g., 'en')
            device: Compute device ('cuda' or 'cpu')
            **kwargs: Additional parameters for the specific implementation
            
        Returns:
            An initialized transcription service implementation
            
        Raises:
            ValueError: If the specified transcription system is not supported
        """
        model = kwargs.pop("transcription_model", None)

        if transcription_system == "whisperx":
            from transcription.whisperx_transcriber import WhisperXTranscriber

            whisperx_kwargs = dict(kwargs)
            if model:
                whisperx_kwargs["whisperx_model"] = model
            return WhisperXTranscriber(
                source_language=source_language,
                device=device,
                **whisperx_kwargs
            )
        elif transcription_system in {"pyannote_openai", "openai", "whisper"}:
            from transcription.pyannote_openai_transcriber import PyAnnoteOpenAITranscriber

            backend_kwargs = dict(kwargs)
            if transcription_system in {"openai", "whisper"}:
                backend_kwargs["transcription_system"] = transcription_system
            if model:
                backend_kwargs["whisper_model"] = model

            return PyAnnoteOpenAITranscriber(
                source_language=source_language,
                device=device,
                **backend_kwargs
            )
        elif transcription_system == "assemblyai":
            from transcription.assemblyai_transcriber import AssemblyAITranscriber

            assembly_kwargs = dict(kwargs)
            if model:
                assembly_kwargs["speech_model"] = model
            return AssemblyAITranscriber(
                source_language=source_language,
                device=device,
                **assembly_kwargs
            )
        elif transcription_system == "gemini":
            from transcription.gemini_transcriber import GeminiTranscriber

            gemini_kwargs = dict(kwargs)
            if model:
                gemini_kwargs["gemini_transcription_model"] = model
            return GeminiTranscriber(
                source_language=source_language,
                device=device,
                **gemini_kwargs
            )
        elif transcription_system == "deepgram":
            from transcription.deepgram_transcriber import DeepgramTranscriber

            deepgram_kwargs = dict(kwargs)
            deepgram_model = model or deepgram_kwargs.pop("deepgram_model", "nova-3")
            return DeepgramTranscriber(
                source_language=source_language,
                device=device,
                model=deepgram_model,
                **deepgram_kwargs
            )
        else:
            raise ValueError(f"Unsupported transcription system: {transcription_system}")
