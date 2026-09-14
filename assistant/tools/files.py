"""
File access (Phase 3) — STRICTLY READ-ONLY.

There is NO delete, rename, move, write or overwrite anywhere in this module —
by design. Jarvis can only:

  * find_file(filename)   — locate files by (partial) name
  * search_files(keyword) — locate files whose name contains a keyword
  * open_file(path)       — open a file with its default app (launch, not edit)
  * open_folder(path)     — reveal a folder in Explorer
  * recent_files()        — list recently modified files

Searches are confined to a configured set of safe roots (the user's home and
its common subfolders by default, see ``config.TOOLS_FILE_ROOTS``) so we never
crawl the whole disk or system directories.
"""
from __future__ import annotations

import os
import re
import subprocess
import sys
import time
from pathlib import Path
from typing import List, Optional

from assistant import config
from assistant.core.log import get
from assistant.tools.schemas import fail, ok

log = get("tools")

# Folders we never descend into while searching (noise + huge + sensitive).
_SKIP_DIRS = {
    "node_modules", "__pycache__", ".git", "venv", "env", ".venv",
    "site-packages", "AppData", "$Recycle.Bin", "Windows", "Program Files",
    "Program Files (x86)", ".cache", "dist", "build", ".next",
}
# Cap how much we scan so a voice command can't hang on a giant tree.
_MAX_SCAN_ENTRIES = 60000
_MAX_RESULTS = 25


def _safe_roots() -> List[Path]:
    """Resolved, existing search roots (from config; defaults to home folders)."""
    roots: List[Path] = []
    for raw in config.TOOLS_FILE_ROOTS:
        try:
            p = Path(os.path.expandvars(os.path.expanduser(raw))).resolve()
            if p.exists() and p.is_dir():
                roots.append(p)
        except Exception:
            continue
    # De-dup while preserving order.
    seen = set()
    uniq = []
    for p in roots:
        if p not in seen:
            seen.add(p)
            uniq.append(p)
    return uniq


def _is_within_roots(path: Path, roots: List[Path]) -> bool:
    """True if ``path`` lives inside one of the allowed roots (no escaping)."""
    try:
        rp = path.resolve()
    except Exception:
        return False
    for root in roots:
        try:
            rp.relative_to(root)
            return True
        except ValueError:
            continue
    return False


def _walk(roots: List[Path]):
    """Yield files under the roots, skipping noisy/system dirs, bounded."""
    scanned = 0
    for root in roots:
        for dirpath, dirnames, filenames in os.walk(root):
            # Prune skip-dirs in place so os.walk doesn't descend into them.
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS
                           and not d.startswith(".")]
            for fn in filenames:
                scanned += 1
                if scanned > _MAX_SCAN_ENTRIES:
                    log.info("[FILES] scan cap reached (%d entries)", _MAX_SCAN_ENTRIES)
                    return
                yield Path(dirpath) / fn


def _match(filename: str, needle: str) -> bool:
    return needle in filename.lower()


# --------------------------------------------------------------------------- #
# Public tool handlers.
# --------------------------------------------------------------------------- #
def find_file(filename: str) -> dict:
    """Find files whose name matches ``filename`` (substring, case-insensitive)."""
    needle = (filename or "").strip().lower()
    if not needle:
        return fail("What file should I look for, sir?", error="empty_query")
    roots = _safe_roots()
    if not roots:
        return fail("I have no folders I'm allowed to search, sir.",
                    error="no_roots")
    matches: List[str] = []
    for path in _walk(roots):
        if _match(path.name, needle):
            matches.append(str(path))
            if len(matches) >= _MAX_RESULTS:
                break
    if not matches:
        return fail(f"I couldn't find anything matching '{filename}', sir.",
                    error="not_found", data=[])
    head = Path(matches[0]).name
    msg = (f"Found {head}." if len(matches) == 1
           else f"Found {len(matches)} files matching '{filename}', "
                f"the first is {head}.")
    return ok(msg, data=matches)


def search_files(keyword: str) -> dict:
    """Search file names for ``keyword`` (alias of find_file, friendlier message)."""
    kw = (keyword or "").strip()
    if not kw:
        return fail("What keyword should I search for, sir?", error="empty_query")
    res = find_file(kw)
    if res["success"]:
        count = len(res.get("data") or [])
        res["message"] = f"Found {count} file(s) matching '{kw}', sir."
    return res


def open_file(path: str) -> dict:
    """Open a file with its default application (does NOT modify it)."""
    if not path:
        return fail("Which file should I open, sir?", error="empty_path")
    p = Path(os.path.expandvars(os.path.expanduser(path)))
    if not p.exists():
        return fail(f"I can't find that file, sir: {path}", error="not_found")
    if not p.is_file():
        return fail(f"That's not a file, sir: {path}", error="not_a_file")
    if not _is_within_roots(p, _safe_roots()):
        return fail("That file is outside the folders I'm allowed to open, sir.",
                    error="outside_roots")
    try:
        # os.startfile is the read-only "open with default app" call on Windows.
        os.startfile(str(p))  # type: ignore[attr-defined]
        log.info("[FILES] opened file %s", p)
        return ok(f"Opening {p.name}.", data={"path": str(p)})
    except Exception as e:
        log.exception("[FILES] open_file failed")
        return fail(f"I couldn't open that file, sir: {e}", error=str(e))


def open_folder(path: str) -> dict:
    """Open a folder in Windows Explorer. Accepts a path or a known folder name."""
    if not path:
        return fail("Which folder should I open, sir?", error="empty_path")

    target = _resolve_folder(path)
    if target is None:
        return fail(f"I couldn't find a folder called '{path}', sir.",
                    error="not_found")
    try:
        # Use explorer.exe (subprocess, not os.system) to reveal the folder.
        subprocess.Popen(["explorer", str(target)], shell=False)
        log.info("[FILES] opened folder %s", target)
        return ok(f"Opening {target.name or str(target)}.", data={"path": str(target)})
    except Exception as e:
        log.exception("[FILES] open_folder failed")
        return fail(f"I couldn't open that folder, sir: {e}", error=str(e))


def _resolve_folder(path: str) -> Optional[Path]:
    """Resolve a folder path, or a friendly name, to an existing directory."""
    raw = Path(os.path.expandvars(os.path.expanduser(path)))
    if raw.exists() and raw.is_dir():
        return raw

    roots = _safe_roots()
    name = path.strip().lower()
    # Common shortcut names first.
    home = Path.home()
    shortcuts = {
        "documents": home / "Documents",
        "downloads": home / "Downloads",
        "desktop": home / "Desktop",
        "pictures": home / "Pictures",
        "music": home / "Music",
        "videos": home / "Videos",
        "home": home,
    }
    if name in shortcuts and shortcuts[name].exists():
        return shortcuts[name]

    # Otherwise look for a directory whose name matches under the safe roots.
    for root in roots:
        for dirpath, dirnames, _ in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS
                           and not d.startswith(".")]
            for d in dirnames:
                if d.lower() == name or name in d.lower():
                    return Path(dirpath) / d
    return None


def recent_files(limit: int = 10) -> dict:
    """List the most recently modified files within the safe roots."""
    try:
        limit = max(1, min(int(limit), 50))
    except (TypeError, ValueError):
        limit = 10
    roots = _safe_roots()
    if not roots:
        return fail("I have no folders I'm allowed to search, sir.", error="no_roots")

    scored: List[tuple] = []
    for path in _walk(roots):
        try:
            mtime = path.stat().st_mtime
        except OSError:
            continue
        scored.append((mtime, str(path)))
    if not scored:
        return ok("I didn't find any recent files, sir.", data=[])
    scored.sort(reverse=True)
    top = [p for _, p in scored[:limit]]
    newest = Path(top[0]).name
    when = time.strftime("%b %d %H:%M", time.localtime(scored[0][0]))
    return ok(f"Your most recent file is {newest}, modified {when}.", data=top)


# --------------------------------------------------------------------------- #
# Phase 1 Safe Project-Bound Inspection & Testing Tools
# --------------------------------------------------------------------------- #
PROJECT_ROOT = Path(__file__).resolve().parent.parent.parent

_SEARCHABLE_CODE_EXTENSIONS = {
    ".py", ".json", ".md", ".txt", ".yaml", ".yml", ".toml",
    ".bat", ".sh", ".ini", ".cfg", ".sql", ".html", ".css", ".js"
}


def _resolve_project_path(path_str: str) -> Optional[Path]:
    """Resolve a path strictly within PROJECT_ROOT, preventing traversal."""
    if not path_str:
        return PROJECT_ROOT
    cleaned = path_str.strip().strip("'\"").replace("\\", "/")
    if not cleaned or cleaned.lower() in {".", "project_root", "root", "(project root)", "jarvis"}:
        return PROJECT_ROOT

    # Strip any leading 'd:/jarvis' or 'd:\\jarvis' if model passes full root prefix
    root_str = str(PROJECT_ROOT).replace("\\", "/").lower()
    if cleaned.lower().startswith(root_str):
        cleaned = cleaned[len(root_str):].lstrip("/")
    if cleaned.lower().startswith("d:/jarvis") or cleaned.lower().startswith("d:\\jarvis"):
        cleaned = cleaned[9:].lstrip("/\\")

    p = Path(cleaned)
    if not p.is_absolute():
        resolved = (PROJECT_ROOT / p).resolve()
    else:
        resolved = p.resolve()

    # Direct match within project root
    try:
        resolved.relative_to(PROJECT_ROOT)
        if resolved.exists():
            return resolved
    except ValueError:
        return None

    # Fallback: check if subfolder exists under assistant/ or tests/
    if not p.is_absolute():
        for sub in ["assistant", "tests"]:
            candidate = (PROJECT_ROOT / sub / p).resolve()
            try:
                candidate.relative_to(PROJECT_ROOT)
                if candidate.exists():
                    return candidate
            except ValueError:
                pass

    try:
        resolved.relative_to(PROJECT_ROOT)
        return resolved
    except ValueError:
        return None


def read_file(path: str, offset: int = 1, limit: int = 100) -> dict:
    """Read a text file within the project root, bounded by offset and line limit."""
    target = _resolve_project_path(path)
    if target is None:
        return fail("Access denied: path is outside the project root.", error="path_traversal")
    if not target.exists():
        return fail(f"File not found: {path}", error="not_found")
    if not target.is_file():
        return fail(f"Target is a directory, not a file: {path}", error="not_a_file")

    try:
        offset = max(1, int(offset))
        limit = max(1, min(int(limit), 200))
    except (TypeError, ValueError):
        offset, limit = 1, 100

    try:
        with open(target, "r", encoding="utf-8", errors="replace") as f:
            lines = f.readlines()
    except Exception as e:
        return fail(f"Could not read file: {e}", error=str(e))

    total = len(lines)
    start_idx = offset - 1
    end_idx = min(start_idx + limit, total)
    selected = lines[start_idx:end_idx]

    rel_path = target.relative_to(PROJECT_ROOT)
    summary = f"Read {len(selected)} lines from {rel_path} (lines {offset}-{offset + len(selected) - 1} of {total})"
    return ok(
        summary,
        data={
            "path": str(rel_path),
            "offset": offset,
            "limit": limit,
            "total_lines": total,
            "content": "".join(selected),
        },
    )


def list_directory(path: str = "") -> dict:
    """List directory contents within the project root."""
    target = _resolve_project_path(path)
    if target is None:
        return fail("Access denied: path is outside the project root.", error="path_traversal")
    if not target.exists():
        return fail(f"Directory not found: {path}", error="not_found")
    if not target.is_dir():
        return fail(f"Target is not a directory: {path}", error="not_a_directory")

    entries = []
    try:
        for entry in os.scandir(target):
            if entry.name in _SKIP_DIRS or entry.name.startswith("."):
                continue
            is_dir = entry.is_dir()
            size = entry.stat().st_size if not is_dir else None
            entries.append({
                "name": entry.name,
                "type": "dir" if is_dir else "file",
                "size_bytes": size,
            })
            if len(entries) >= 50:
                break
    except Exception as e:
        return fail(f"Failed listing directory: {e}", error=str(e))

    # Sort dirs first, then files alphabetically
    entries.sort(key=lambda x: (0 if x["type"] == "dir" else 1, x["name"].lower()))
    rel_path = target.relative_to(PROJECT_ROOT)
    display_path = str(rel_path) if str(rel_path) != "." else "(project root)"
    return ok(
        f"Found {len(entries)} items in {display_path}",
        data={"path": str(rel_path), "entries": entries},
    )


def grep_code(query: str, path: str = "") -> dict:
    """Search for a text pattern in source files within the project root."""
    if not query or not query.strip():
        return fail("Search query cannot be empty.", error="empty_query")
    target = _resolve_project_path(path)
    if target is None:
        return fail("Access denied: path is outside the project root.", error="path_traversal")
    if not target.exists():
        return fail(f"Search path not found: {path}", error="not_found")

    needle = query.strip().lower()
    raw_needles = [n.strip().lower() for n in query.split("|") if n.strip()]
    if not raw_needles:
        raw_needles = [query.strip().lower()]
    matches = []
    scanned_files = 0
    max_matches = 30

    search_roots = [target] if target.is_dir() else [target.parent]
    for root in search_roots:
        for dirpath, dirnames, filenames in os.walk(root):
            dirnames[:] = [d for d in dirnames if d not in _SKIP_DIRS and not d.startswith(".")]
            for fn in filenames:
                ext = os.path.splitext(fn)[1].lower()
                if ext not in _SEARCHABLE_CODE_EXTENSIONS:
                    continue
                file_path = Path(dirpath) / fn
                scanned_files += 1
                try:
                    with open(file_path, "r", encoding="utf-8", errors="replace") as f:
                        for line_no, line in enumerate(f, 1):
                            line_lower = line.lower()
                            if any(n in line_lower for n in raw_needles):
                                rel = str(file_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
                                matches.append({
                                    "file": rel,
                                    "line": line_no,
                                    "content": line.strip()[:150],
                                })
                                if len(matches) >= max_matches:
                                    break
                except Exception:
                    continue
                if len(matches) >= max_matches:
                    break
            if len(matches) >= max_matches:
                break

    return ok(
        f"Found {len(matches)} matches for '{query}' across {scanned_files} files",
        data={"query": query, "matches": matches, "scanned_files": scanned_files},
    )


_TEST_TARGET_SAFE_RE = re.compile(r"^[a-zA-Z0-9_\-\.\/\\:]+$")


def run_project_tests(test_target: str = "") -> dict:
    """Safely execute approved project test suites using Python subprocess."""
    cleaned = (test_target or "").strip().strip("'\"")
    if cleaned and not _TEST_TARGET_SAFE_RE.match(cleaned):
        return fail(f"Invalid test target syntax: '{cleaned}'", error="invalid_target")

    # Only allow test targets within tests/ directory or module names
    if cleaned and not (cleaned.startswith("tests") or "test" in cleaned):
        return fail(f"Target '{cleaned}' is not an approved project test module.", error="disallowed_target")

    python_bin = sys.executable
    if not cleaned or cleaned.lower() in {"all", "tests"}:
        cmd = [python_bin, "-m", "unittest", "discover", "-s", "tests", "-p", "test_*.py"]
        display_target = "all discovered tests"
    elif cleaned.endswith(".py"):
        # Resolve within project root
        target_path = _resolve_project_path(cleaned)
        if target_path is None or not target_path.exists():
            return fail(f"Test file not found: {cleaned}", error="not_found")
        rel_str = str(target_path.relative_to(PROJECT_ROOT)).replace("\\", "/")
        cmd = [python_bin, "-m", "unittest", rel_str]
        display_target = rel_str
    else:
        # Module path
        cmd = [python_bin, "-m", "unittest", cleaned]
        display_target = cleaned

    env = os.environ.copy()
    env["PYTHONPATH"] = str(PROJECT_ROOT)
    env["PYTHONIOENCODING"] = "utf-8"

    try:
        proc = subprocess.run(
            cmd,
            cwd=str(PROJECT_ROOT),
            capture_output=True,
            encoding="utf-8",
            errors="replace",
            env=env,
            timeout=45.0,
        )
        stdout_str = proc.stdout or ""
        stderr_str = proc.stderr or ""
        combined_output = (stdout_str + "\n" + stderr_str).strip()
        # Truncate output to prevent prompt overflow
        if len(combined_output) > 2500:
            combined_output = combined_output[:2500] + "\n...[output truncated]"

        if proc.returncode == 0:
            return ok(
                f"Tests passed for {display_target}",
                data={"target": display_target, "exit_code": 0, "output": combined_output},
            )
        else:
            return fail(
                f"Tests failed for {display_target} (exit code {proc.returncode})",
                error="test_failure",
                data={"target": display_target, "exit_code": proc.returncode, "output": combined_output},
            )
    except subprocess.TimeoutExpired:
        return fail(f"Tests timed out after 45 seconds for {display_target}", error="timeout")
    except Exception as e:
        return fail(f"Failed to execute tests: {e}", error=str(e))

