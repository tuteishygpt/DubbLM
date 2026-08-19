import sys
import io
import json

sys.stdout = io.TextIOWrapper(sys.stdout.buffer, encoding='utf-8', errors='replace')
sys.stderr = io.TextIOWrapper(sys.stderr.buffer, encoding='utf-8', errors='replace')

from gradio_client import Client  # noqa: E402

c = Client('k2-fsa/OmniVoice')
d = c.view_api(return_format='dict')
with open('api_info.json', 'w', encoding='utf-8') as f:
    json.dump(d, f, indent=2, ensure_ascii=False)
print("Done - saved to api_info.json")
