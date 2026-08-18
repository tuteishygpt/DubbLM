import sys
import os
from pathlib import Path
from dubbing.core.runner import run_dubbing_job

def main():
    try:
        print("Starting dubbing job (FULL RUN) to verify pipeline services...")
        # Point to a small valid file, we will use our source file
        input_video = r"D:\CodexPRJ\DubbLM\prj\Nature\artifacts\audio\source.wav"
        
        result = run_dubbing_job({
            "project_name": "Test_Full_Pipeline_Run",
            "source_language": "en",
            "target_language": "es",
            "input": input_video,
            "transcribe_only": False,
            "save_translated_subtitles": True,
            "save_original_subtitles": True,
        })
        print(f"Status: {result.status}")
        print(f"Output File: {result.output_file}")
        print(f"Logs: {result.logs}")
    except Exception as e:
        print(f"Pipeline failed: {e}")
        import traceback
        traceback.print_exc()

if __name__ == '__main__':
    main()
