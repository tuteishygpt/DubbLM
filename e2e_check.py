import subprocess
import time
import urllib.request
import json
import sys
import os

print('Starting uvicorn server...')
env = os.environ.copy()
env['PYTHONPATH'] = 'src'
proc = subprocess.Popen(['python', '-m', 'uvicorn', 'dubbing.web.app:create_app', '--factory', '--port', '8081'], stdout=subprocess.PIPE, stderr=subprocess.PIPE, text=True, env=env)

try:
    time.sleep(5)
    
    if proc.poll() is not None:
        print('Server crashed. Output:')
        print(proc.stdout.read())
        print(proc.stderr.read())
        sys.exit(1)
        
    req = urllib.request.Request('http://localhost:8081/api/projects')
    with urllib.request.urlopen(req) as response:
        data = json.loads(response.read().decode())
        print('API Response Status:', response.status)
        print('Projects found:', len(data.get('projects', [])))
        if response.status == 200:
            print('Integration check PASSED!')
        else:
            print('Integration check FAILED!')
            sys.exit(1)
            
except Exception as e:
    print('Failed to communicate with API:', e)
    print('Checking server logs:')
    print(proc.stderr.read())
    print(proc.stdout.read())
    sys.exit(1)
finally:
    proc.terminate()
    proc.wait()
