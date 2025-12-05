import subprocess
import webbrowser
import time
import os

def main():
    port = "8501"
    cmd = [
        "streamlit", "run", 
        os.path.join("dance_ai_app", "pro_app.py"),
        "--server.headless=true",
        f"--server.port={port}"
    ]

    subprocess.Popen(cmd, stdout=subprocess.PIPE, stderr=subprocess.PIPE)
    webbrowser.open(f"http://localhost:{port}")

if __name__ == "__main__":
    main()
