import subprocess
import sys
import os

# HuggingFace Spaces needs app.py at root
# This just launches the actual server
if __name__ == "__main__":
    subprocess.run([
        sys.executable, "-m", "uvicorn",
        "server.app:app",
        "--host", "0.0.0.0",
        "--port", "7860"
    ])
