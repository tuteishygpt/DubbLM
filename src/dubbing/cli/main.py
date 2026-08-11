"""CLI entry point for the refactored Smart Dubbing system."""

# --- Suppress CUDA/GPU library warnings BEFORE any imports ---
import os
# Set environment variables to suppress noisy CUDA/GPU warnings
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'  # Suppress TensorFlow logs
os.environ['TF_ENABLE_ONEDNN_OPTS'] = '0'  # Disable oneDNN custom operations
os.environ['CUDA_LAUNCH_BLOCKING'] = '0'  # Async CUDA operations
# --- End suppression block ---

import sys
import logging
from dotenv import load_dotenv

from ..core.config import create_argument_parser, create_config_from_args
from ..core.smart_dubbing import SmartDubbing
from ..core.runner import _extract_output_path, _run_combine_video_step
from ..core.log_config import setup_logging, get_logger

# Setup logging
setup_logging()
logger = get_logger(__name__)


def main():
    """Main function to run the dubbing tool."""
    # Load environment variables
    load_dotenv(override=True)
    
    # Create argument parser and parse arguments
    parser = create_argument_parser()
    args = parser.parse_args()
    
    # Create configuration from arguments
    config = create_config_from_args(args)
    
    # Print configuration info
    glossary = config.get('glossary', {})
    if glossary:
        logger.debug(f"Using translation glossary with {len(glossary)} entries")
    
    voice_prompt = config.get('voice_prompt', {})
    if voice_prompt:
        logger.debug(f"Using voice prompts for {len(voice_prompt)} speakers")
        for speaker, prompt in voice_prompt.items():
            p = f"{prompt[:50]}..." if len(prompt) > 50 else prompt
            logger.debug(f"  {speaker}: {p}")
    
    # Initialize and run the dubbing system
    try:
        dubber = SmartDubbing(config)
        
        if config.get('generate_speaker_report'):
            logger.info("Generating speaker report...")
            try:
                report_path, samples_path = dubber.generate_diarization_report()
                logger.info(f"Speaker report generated: {report_path}")
                logger.info(f"Voice samples copied to: {samples_path}")
            except Exception as e:
                logger.error(f"Error generating speaker report: {e}")
                sys.exit(1)
            sys.exit(0)
            
        elif config.get('run_step') == 'combine_video':
            logger.info("Running only the 'combine_audio_with_video' step...")
            try:
                output_video_path = _extract_output_path(
                    _run_combine_video_step(dubber, config)
                )
                logger.info(f"Video combination complete. Output saved to: {output_video_path}")
            except Exception as e:
                logger.error(f"Error during 'combine_video' step: {e}", exc_info=True)
                sys.exit(1)
            sys.exit(0)
            
        elif config.get('run_step') == 'transcribe_only':
            logger.info("Running only diarization and transcription...")
            output_path = dubber.run_transcribe_only(
                save_original_subtitles=config.get('save_original_subtitles', False),
            )
            logger.info(f"Transcription complete. Saved to: {output_path}")

        elif config.get('run_step') == 'translate_only':
            logger.info("Running diarization, transcription, and translation...")
            output_path = dubber.run_translate_only(
                save_original_subtitles=config.get('save_original_subtitles', False),
                save_translated_subtitles=config.get('save_translated_subtitles', False),
            )
            logger.info(f"Translation complete. Saved to: {output_path}")

        elif config.get('run_step') == 'tts_to_end':
            logger.info("Resuming pipeline from TTS step...")
            output_path = dubber.run_from_tts(
                save_original_subtitles=config.get('save_original_subtitles', False),
                save_translated_subtitles=config.get('save_translated_subtitles', False),
            )
            logger.info(f"TTS resume complete. Output saved to: {output_path}")

        elif config.get('run_step') == 'from_scratch':
            logger.info("Running from scratch: clearing cache and reprocessing everything...")
            output_path = dubber.run_from_scratch(
                save_original_subtitles=config.get('save_original_subtitles', False),
                save_translated_subtitles=config.get('save_translated_subtitles', False),
            )
            logger.info(f"Video dubbing complete. Output saved to: {output_path}")

        else:
            # Run the full pipeline
            output_path = dubber.run_pipeline(
                save_original_subtitles=config.get('save_original_subtitles', False),
                save_translated_subtitles=config.get('save_translated_subtitles', False)
            )
            logger.info(f"Video dubbing complete. Output saved to: {output_path}")
            
    except Exception as e:
        logger.error(f"Error in dubbing system: {e}", exc_info=True)
        sys.exit(1)


if __name__ == "__main__":
    main() 
