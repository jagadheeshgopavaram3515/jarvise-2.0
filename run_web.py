"""Run the local FastAPI/Next.js interface with one command."""
from __future__ import annotations

import shutil
import socket
import subprocess
import sys
import time
import urllib.request
import webbrowser
from pathlib import Path


def main():
    root = Path(__file__).resolve().parent
    npm = shutil.which("npm.cmd") or shutil.which("npm")
    if not npm:
        raise SystemExit("Install Node.js (20.9 or newer), then run this launcher again.")
    for port in (3000, 8765):
        with socket.socket() as probe:
            try:
                probe.bind(("127.0.0.1", port))
            except OSError:
                raise SystemExit(f"Port {port} is already in use. Stop the existing server first.") from None
    try:
        import fastapi, uvicorn  # noqa: F401
    except ImportError:
        raise SystemExit(f'Install web dependencies: "{sys.executable}" -m pip install -r requirements-web.txt') from None
    if not (root / "frontend/node_modules/next/package.json").exists():
        subprocess.run([npm, "ci", "--no-audit", "--no-fund"], cwd=root / "frontend", check=True)
    processes = []
    try:
        processes.append(subprocess.Popen([sys.executable, "-m", "uvicorn", "assistant.web.app:app", "--host", "127.0.0.1", "--port", "8765"], cwd=root))
        processes.append(subprocess.Popen([npm, "run", "dev"], cwd=root / "frontend"))
        for _ in range(90):
            if any(process.poll() is not None for process in processes):
                raise SystemExit("A web server exited. Check its output above.")
            try:
                with urllib.request.urlopen("http://127.0.0.1:3000", timeout=2) as response:
                    if response.status == 200:
                        break
            except (OSError, TimeoutError):
                time.sleep(1)
        else:
            raise SystemExit("Frontend did not become ready within 90 attempts.")
        webbrowser.open("http://127.0.0.1:3000")
        print("JARVIS: http://127.0.0.1:3000 — click Start JARVIS. Ctrl+C stops both servers.")
        while all(process.poll() is None for process in processes):
            time.sleep(1)
    except KeyboardInterrupt:
        pass
    finally:
        if processes and processes[0].poll() is None:
            try:
                request = urllib.request.Request("http://127.0.0.1:8765/api/shutdown", data=b"{}",
                                                 headers={"Content-Type": "application/json"}, method="POST")
                with urllib.request.urlopen(request, timeout=6):
                    pass
            except OSError:
                pass
        # Terminate only the process trees created by this launcher.
        for process in reversed(processes):
            if process.poll() is None:
                if sys.platform == "win32":
                    subprocess.run(["taskkill", "/PID", str(process.pid), "/T", "/F"], capture_output=True)
                else:
                    process.terminate()
                process.wait(timeout=10)


if __name__ == "__main__":
    main()
