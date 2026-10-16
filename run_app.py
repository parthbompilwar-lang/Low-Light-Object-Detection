"""
One-click automatic launcher for Low-Light Enhancement & Object Detection App.
Automatically starts the Streamlit server and opens http://localhost:8501 in your default browser.
"""

import os
import sys
import time
import socket
import subprocess
import webbrowser
import threading

def is_port_in_use(port=8501):
    with socket.socket(socket.AF_INET, socket.SOCK_STREAM) as s:
        s.settimeout(0.5)
        return s.connect_ex(('127.0.0.1', port)) == 0

def free_port(port=8501):
    try:
        cmd = f'netstat -ano | findstr :{port}'
        output = subprocess.check_output(cmd, shell=True, stderr=subprocess.DEVNULL).decode()
        pids = set()
        for line in output.strip().splitlines():
            parts = line.split()
            if len(parts) >= 5 and "LISTENING" in parts:
                pids.add(parts[-1])
        for pid in pids:
            subprocess.run(f"taskkill /F /PID {pid}", shell=True, stdout=subprocess.DEVNULL, stderr=subprocess.DEVNULL)
        time.sleep(1.0)
    except Exception:
        pass

def open_browser():
    time.sleep(2.5)
    print("--> Automatically opening browser at http://localhost:8501 ...", flush=True)
    webbrowser.open("http://localhost:8501")

def main():
    print("=" * 65)
    print("  🌙 Low-Light Image Enhancement & Detection - Auto Launcher  ")
    print("=" * 65)
    print(f"--> Python: {sys.executable}")
    
    # Check PyTorch & GPU
    try:
        import torch
        if torch.cuda.is_available():
            print(f"--> Hardware Acceleration: CUDA GPU ({torch.cuda.get_device_name(0)})")
        else:
            print("--> Hardware: CPU (CUDA not active)")
    except Exception as e:
        print(f"[!] Warning checking PyTorch: {e}")

    # Check and free port 8501 if previously occupied
    if is_port_in_use(8501):
        print("--> Port 8501 is occupied by an earlier session. Resetting port 8501...")
        free_port(8501)

    # Launch browser in background thread
    threading.Thread(target=open_browser, daemon=True).start()

    # Launch Streamlit
    script_dir = os.path.dirname(os.path.abspath(__file__))
    app_path = os.path.join(script_dir, "src", "app.py")
    
    cmd = [
        sys.executable, "-m", "streamlit", "run", app_path,
        "--server.headless", "true",
        "--server.port", "8501"
    ]
    
    print("--> Starting Streamlit server on http://localhost:8501 ...\n")
    try:
        subprocess.run(cmd, check=True)
    except KeyboardInterrupt:
        print("\n--> Server closed by user.")
    except Exception as e:
        print(f"\n[!] Error running Streamlit: {e}")

if __name__ == "__main__":
    main()
