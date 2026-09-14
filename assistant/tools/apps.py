"""
App control (Phase 2) — open / close / list desktop applications on Windows.

Allow-list only: every app Jarvis can touch is declared in ``APPS`` below. An
unknown app name is refused with a friendly message — there is no path to
launching or killing an arbitrary executable.

Launching uses ``subprocess`` (never ``os.system`` for app launch) and prefers
``shutil.which`` / well-known install locations. Closing uses ``psutil`` to find
and terminate the matching processes gracefully (``terminate()`` then, only if
needed, ``kill()``), so we never shell out to ``taskkill`` blindly.
"""
from __future__ import annotations

import os
import shutil
import subprocess
from typing import Dict, List, Optional

try:
    import psutil
except Exception:  # pragma: no cover - psutil is a declared dependency
    psutil = None  # type: ignore

from assistant.core.log import get
from assistant.tools.schemas import fail, ok

log = get("tools")


# --------------------------------------------------------------------------- #
# Allow-list of controllable apps.
#   aliases      : spoken/typed names that map to this app
#   launch       : argv list OR a list of candidate executables to try in order
#   uri          : optional Windows shell command (for Store/UWP apps that have
#                  no plain .exe path, e.g. Calculator)
#   processes    : process-name substrings used to find/close it with psutil
#   closable     : whether close_app may terminate it
# --------------------------------------------------------------------------- #
APPS: Dict[str, dict] = {
    "chrome": {
        "aliases": ["chrome", "google chrome"],
        "launch": ["chrome", "chrome.exe",
                   r"C:\Program Files\Google\Chrome\Application\chrome.exe",
                   r"C:\Program Files (x86)\Google\Chrome\Application\chrome.exe"],
        "processes": ["chrome.exe"],
        "closable": True,
    },
    "edge": {
        "aliases": ["edge", "microsoft edge", "msedge"],
        "launch": ["msedge", "msedge.exe",
                   r"C:\Program Files (x86)\Microsoft\Edge\Application\msedge.exe"],
        "processes": ["msedge.exe"],
        "closable": True,
    },
    "vs code": {
        "aliases": ["vs code", "vscode", "visual studio code", "code"],
        "launch": ["code", "code.cmd",
                   os.path.expandvars(
                       r"%LOCALAPPDATA%\Programs\Microsoft VS Code\Code.exe")],
        "processes": ["Code.exe"],
        "closable": True,
    },
    "notepad": {
        "aliases": ["notepad"],
        "launch": ["notepad.exe"],
        "processes": ["notepad.exe"],
        "closable": True,
    },
    "calculator": {
        "aliases": ["calculator", "calc"],
        # Calculator is a UWP app — launch via its shell URI.
        "uri": "calculator:",
        "launch": ["calc.exe"],
        "processes": ["Calculator.exe", "CalculatorApp.exe"],
        "closable": False,   # UWP, not reliably terminable by name
    },
    "explorer": {
        "aliases": ["explorer", "file explorer", "files", "windows explorer"],
        "launch": ["explorer.exe"],
        "processes": ["explorer.exe"],
        # Never close Explorer — it owns the desktop/taskbar.
        "closable": False,
    },
}


def _resolve(app_name: str) -> Optional[str]:
    """Map a spoken/typed app name to a canonical key in ``APPS``."""
    if not app_name:
        return None
    needle = app_name.strip().lower()
    if needle in APPS:
        return needle
    for key, meta in APPS.items():
        if needle in meta["aliases"] or any(needle in a or a in needle
                                            for a in meta["aliases"]):
            return key
    return None


def _first_launchable(candidates: List[str]) -> Optional[List[str]]:
    """Return an argv list for the first candidate that exists/resolves."""
    for cand in candidates:
        if not cand:
            continue
        # An absolute path that exists, or a bare name resolvable on PATH.
        if os.path.isabs(cand) and os.path.exists(cand):
            return [cand]
        found = shutil.which(cand)
        if found:
            return [found]
    return None


# --------------------------------------------------------------------------- #
# Public tool handlers (return canonical result dicts via ok()/fail()).
# --------------------------------------------------------------------------- #
def open_app(app: str) -> dict:
    """Launch a known desktop application by name."""
    key = _resolve(app)
    if key is None:
        return fail(f"I can't open '{app}', sir — it's not in my allowed apps.",
                    error="unknown_app", data={"requested": app})
    meta = APPS[key]
    try:
        # UWP/shell-URI apps (Calculator) launch via 'cmd /c start'.
        if meta.get("uri"):
            subprocess.Popen(["cmd", "/c", "start", "", meta["uri"]],
                             shell=False)
            log.info("[APPS] opened %s via uri %s", key, meta["uri"])
            return ok(f"{key.title()} opened.", data={"app": key})

        argv = _first_launchable(meta.get("launch", []))
        if argv is None:
            return fail(f"I couldn't find {key} installed on this PC, sir.",
                        error="not_installed", data={"app": key})
        subprocess.Popen(argv, shell=False)
        log.info("[APPS] opened %s via %s", key, argv[0])
        return ok(f"{key.title()} opened.", data={"app": key, "path": argv[0]})
    except Exception as e:
        log.exception("[APPS] failed to open %s", key)
        return fail(f"I couldn't open {key}, sir: {e}", error=str(e))


def close_app(app: str) -> dict:
    """Terminate a known, closable desktop application by name."""
    key = _resolve(app)
    if key is None:
        return fail(f"I can't close '{app}', sir — it's not in my allowed apps.",
                    error="unknown_app", data={"requested": app})
    meta = APPS[key]
    if not meta.get("closable", False):
        return fail(f"For safety I won't close {key}, sir.", error="not_closable",
                    data={"app": key})
    if psutil is None:
        return fail("Process control isn't available (psutil missing), sir.",
                    error="psutil_missing")

    targets = {p.lower() for p in meta["processes"]}
    killed = 0
    victims = []
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name in targets:
                victims.append(proc)
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    for proc in victims:
        try:
            proc.terminate()
            killed += 1
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue
    # Give them a moment; force-kill any stragglers.
    if victims:
        gone, alive = psutil.wait_procs(victims, timeout=2.0)
        for proc in alive:
            try:
                proc.kill()
            except (psutil.NoSuchProcess, psutil.AccessDenied):
                pass

    if killed:
        log.info("[APPS] closed %s (%d process(es))", key, killed)
        return ok(f"Closed {key}.", data={"app": key, "closed": killed})
    return ok(f"{key.title()} wasn't running, sir.", data={"app": key, "closed": 0})


def list_running_apps() -> dict:
    """List which of the known apps are currently running."""
    if psutil is None:
        return fail("Process listing isn't available (psutil missing), sir.",
                    error="psutil_missing")
    # name(lower) -> canonical app key
    name_to_app = {}
    for key, meta in APPS.items():
        for pname in meta["processes"]:
            name_to_app[pname.lower()] = key

    running = set()
    for proc in psutil.process_iter(["name"]):
        try:
            name = (proc.info.get("name") or "").lower()
            if name in name_to_app:
                running.add(name_to_app[name])
        except (psutil.NoSuchProcess, psutil.AccessDenied):
            continue

    running_list = sorted(running)
    if not running_list:
        return ok("None of your known apps are running, sir.", data=[])
    pretty = ", ".join(k.title() for k in running_list)
    return ok(f"Currently running: {pretty}.", data=running_list)
