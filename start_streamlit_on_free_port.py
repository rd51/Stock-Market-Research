import socket
import subprocess
import os
import sys

# Find a free localhost port
s = socket.socket()
s.bind(('127.0.0.1', 0))
port = s.getsockname()[1]
s.close()

print(f"STARTING_STREAMLIT_PORT:{port}")
sys.stdout.flush()

cmd = [sys.executable, '-m', 'streamlit', 'run', 'app.py', '--server.port', str(port)]
# Launch Streamlit and forward its output
proc = subprocess.Popen(cmd, cwd=os.getcwd(), stdout=subprocess.PIPE, stderr=subprocess.STDOUT, universal_newlines=True)
try:
    for line in proc.stdout:
        print(line, end='')
except KeyboardInterrupt:
    proc.terminate()
    proc.wait()

sys.exit(proc.returncode)
