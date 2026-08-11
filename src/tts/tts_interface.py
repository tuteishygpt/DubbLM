from abc import ABC, abstractmethod
from pathlib import Path
from typing import Optional, Dict, Any, List

from .models import TTSSegmentData, SegmentAlignment # Import the Pydantic models

class TTSInterface(ABC):
    """
    Abstract base class for text-to-speech systems.
    All TTS implementations should inherit from this class.
    """

    reference_capability = "unsupported"
    provider_name: Optional[str] = None

    def validate_segments(
        self,
        segments_data: List[TTSSegmentData],
    ) -> List[tuple[int, str]]:
        """Return deterministic, user-facing reference contract violations."""
        issues: List[tuple[int, str]] = []
        capability = self.reference_capability
        provider = self.provider_name or self.__class__.__name__.lower()
        allowed_modes = {"configured", "segment", "speaker", "none"}

        for segment in segments_data:
            mode = segment.reference_mode
            index = segment.segment_index
            prefix = (
                f"provider={provider} speaker={segment.speaker} segment={index} "
                f"mode={mode or '<missing>'}: "
            )
            reason: Optional[str] = None

            if mode is not None and mode not in allowed_modes:
                reason = f"unknown reference_mode '{mode}'"
            elif capability == "unsupported":
                if mode is not None:
                    reason = "does not support voice-cloning references"
            elif capability in {"optional", "required"}:
                if mode is None:
                    reason = (
                        "explicit reference_mode is required; add reference_mode "
                        "under the new-style voices: profile"
                    )
                elif capability == "required" and mode == "none":
                    reason = "reference_mode 'none' is not allowed"
                elif mode != "none":
                    reference_path = (
                        Path(segment.reference_audio_path).expanduser()
                        if segment.reference_audio_path
                        else None
                    )
                    if reference_path is None or not reference_path.is_file():
                        missing = segment.reference_audio_path or "<missing>"
                        reason = f"reference file does not exist: {missing}"
            else:
                reason = f"invalid provider reference capability '{capability}'"

            if reason:
                issues.append((index, prefix + reason))

        return issues

    def require_valid_segments(self, segments_data: List[TTSSegmentData]) -> None:
        """Raise one aggregate error when reference validation fails."""
        issues = sorted(self.validate_segments(segments_data), key=lambda item: item[0])
        if issues:
            details = "\n".join(f"- {message}" for _, message in issues)
            raise ValueError(
                f"TTS reference validation failed before synthesis:\n{details}"
            )
    
    @abstractmethod
    def __init__(self, **kwargs: Any) -> None:
        """Initialize the TTS system with provider-specific arguments."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def initialize(self) -> None:
        """Perform any necessary setup for the TTS client (e.g., API connections, model loading)."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def synthesize(
        self,
        segments_data: List[TTSSegmentData], # Use the Pydantic model here
        language: str = "en", # Global language for the synthesis batch (if applicable)
        # Removed global reference_audio_path and reference_text
        **kwargs: Any # For any other global parameters a specific TTS system might need
    ) -> List[SegmentAlignment]:
        """
        Synthesize speech for a list of text segments.

        Args:
            segments_data: A list of TTSSegmentData objects. Each object contains
                           speaker ID, text, and optional per-segment parameters like
                           emotion, speed, voice override, style_prompt,
                           reference_audio_path, reference_text, and output_path.
            language: Target language code (e.g., "en"). Applied globally if the TTS supports it.
            **kwargs: Additional global parameters for the specific TTS system.
            
        Returns:
            List of SegmentAlignment objects mapping original segments to synthesized audio
        """
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def estimate_audio_segment_length(
        self,
        segment_data: TTSSegmentData,
        language: str = "en"
    ) -> Optional[float]:
        """
        Estimate the duration in seconds for a given text segment.
        
        Args:
            segment_data: TTSSegmentData object containing text and voice parameters
            language: Target language code (e.g., "en")
            
        Returns:
            Estimated duration in seconds, or None if estimation is not possible
        """
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def is_available(self) -> bool:
        """Check if the TTS system is available and properly initialized."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def set_voice_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a global mapping of speaker IDs to voice names for the TTS system."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def set_voice_prompt_mapping(self, mapping: Dict[str, str]) -> None:
        """Set a global mapping of speaker IDs to voice style prompts for the TTS system."""
        pass # Specific implementation in derived classes
    
    @abstractmethod
    def cleanup(self) -> None:
        """
        Clean up any resources or temporary files used by the TTS system.
        This should be called when the TTS system is no longer needed.
        """
        pass # Specific implementation in derived classes 
