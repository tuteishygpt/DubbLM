from dubbing.core.smart_dubbing import SmartDubber
from dubbing.core.config import DubbingConfig
import os

prj_dir = r'D:\CodexPRJ\DubbLM\prj'
target_dir = None
for d in os.listdir(prj_dir):
    if len(d) == 4 and d != 'clip':
        target_dir = os.path.join(prj_dir, d, 'artifacts')
        break

config = DubbingConfig({'project_name': target_dir.replace('artifacts', ''), 'audio_artifacts_dir': os.path.join(target_dir, 'audio')})
dubber = SmartDubber(config)

import pickle  # noqa: E402
cache_dir = r'D:\CodexPRJ\DubbLM\cache\7cb74ef6\dubbing_texts'
import glob  # noqa: E402
pkl_files = glob.glob(os.path.join(cache_dir, '*.pkl'))
pkl_files.sort(key=os.path.getmtime, reverse=True)

with open(pkl_files[0], 'rb') as f:
    data = pickle.load(f)
    segments = data.get('segments', []) if isinstance(data, dict) else data

from dubbing.core.timing import plan_anchor_windows  # noqa: E402
source_audio_path = os.path.join(target_dir, 'audio', 'source.wav')
from pydub import AudioSegment  # noqa: E402
source_duration = len(AudioSegment.from_file(source_audio_path)) / 1000.0

planned = plan_anchor_windows(segments, source_duration)
for p in planned[2:6]:
    print(f"Segment #{p.original_index}: Start: {p.start}, End: {p.end}, Available Window: {p.available_window}")
    seg_file = p.segment.get('synthesized_speech_file')
    if not seg_file:
        cand = dubber.audio_chunks_dir / f"{p.original_index}.wav"
        if cand.exists():
            seg_file = str(cand)
    print(f"  File: {os.path.basename(seg_file) if seg_file else None}")
    if seg_file and os.path.exists(seg_file):
        aud = AudioSegment.from_file(seg_file)
        print(f"  Raw Duration: {len(aud)/1000.0}s")
