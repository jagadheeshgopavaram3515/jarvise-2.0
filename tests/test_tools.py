"""
Tests for the Jarvis V2 Desktop Tools Layer.

Network/OS side effects are avoided: app launches and browser navigation are
monkey-patched, so the tests verify the FRAMEWORK (registry, executor, timeout,
allow-list, routing, logging) deterministically. The single live browser test
(Playwright) is skipped automatically when Playwright or its browser isn't
installed.

Run with:   pytest -q tests/test_tools.py
       or:  python tests/test_tools.py     (self-contained runner, no pytest)
"""
import os
import sys
import time

sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.abspath(__file__))))

from assistant import config  # noqa: E402
from assistant.tools import get_executor, get_registry  # noqa: E402
from assistant.tools.registry import ToolRegistry, build_default_registry  # noqa: E402
from assistant.tools.executor import ToolExecutor  # noqa: E402
from assistant.tools.schemas import ToolSpec, ok, fail  # noqa: E402
from assistant.tools import apps, browser, files, router  # noqa: E402


# --------------------------------------------------------------- registry ----
def test_registry_loads():
    """The default registry loads with all expected desktop tools."""
    reg = build_default_registry()
    names = set(reg.names())
    expected = {
        "open_app", "close_app", "list_running_apps",
        "find_file", "search_files", "open_file", "open_folder", "recent_files",
        "open_url", "google_search", "youtube_search", "read_page", "get_page_title",
    }
    assert expected <= names, f"missing tools: {expected - names}"
    # get_tool / list_tools / membership
    assert reg.get_tool("open_app") is not None
    assert reg.get_tool("does_not_exist") is None
    assert "open_app" in reg
    assert len(reg.list_tools()) == len(names)
    # Function declarations are well-formed for Gemini tool-calling.
    decls = reg.to_function_declarations()
    assert all("name" in d and "parameters" in d for d in decls)


def test_register_and_get_custom_tool():
    reg = ToolRegistry()
    reg.register_tool(ToolSpec(name="ping", description="ping",
                               handler=lambda: ok("pong")))
    assert "ping" in reg
    assert reg.get_tool("ping").handler()["message"] == "pong"


# --------------------------------------------------------------- executor ----
def test_executor_runs_tool():
    reg = ToolRegistry()
    reg.register_tool(ToolSpec(name="echo", description="echo",
                               handler=lambda value: ok(f"got {value}")))
    ex = ToolExecutor(reg, timeout=2)
    res = ex.execute("echo", {"value": "hi"})
    assert res["success"] is True
    assert res["message"] == "got hi"


def test_executor_unknown_tool_is_blocked():
    """Allow-list: an unregistered tool name can never run (Phase 6)."""
    ex = ToolExecutor(ToolRegistry(), timeout=2)
    res = ex.execute("rm_rf_everything", {"path": "/"})
    assert res["success"] is False
    assert res["error"] == "unknown_tool"


def test_executor_handles_tool_exception():
    """A throwing tool becomes a clean failure, never a crash."""
    def boom():
        raise RuntimeError("kaboom")
    reg = ToolRegistry()
    reg.register_tool(ToolSpec(name="boom", description="boom", handler=boom))
    ex = ToolExecutor(reg, timeout=2)
    res = ex.execute("boom", {})
    assert res["success"] is False
    assert "kaboom" in (res.get("error") or "") or "kaboom" in res["message"]


def test_executor_timeout_handling():
    """A tool that overruns the timeout returns a timeout failure (Phase 1)."""
    def slow():
        time.sleep(5)
        return ok("done")
    reg = ToolRegistry()
    reg.register_tool(ToolSpec(name="slow", description="slow", handler=slow))
    ex = ToolExecutor(reg, timeout=0.3)
    started = time.perf_counter()
    res = ex.execute("slow", {})
    elapsed = time.perf_counter() - started
    assert res["success"] is False
    assert res["error"] == "timeout"
    assert elapsed < 2.0, "execute() did not return promptly on timeout"


def test_executor_writes_log(tmp_path=None):
    """Phase 7: every call is appended to the tools audit log."""
    reg = ToolRegistry()
    reg.register_tool(ToolSpec(name="noop", description="noop",
                               handler=lambda: ok("ok")))
    ex = ToolExecutor(reg, timeout=2)
    ex.execute("noop", {"a": 1})
    # flush handlers and read the configured log file
    for h in list(getattr(ex, "_audit").handlers):
        try:
            h.flush()
        except Exception:
            pass
    path = config.TOOLS_LOG_FILE
    assert os.path.exists(path), f"audit log not created at {path}"
    content = open(path, encoding="utf-8").read()
    assert "noop" in content and "success=true" in content


# ------------------------------------------------------------------ apps ----
def test_open_app_allow_list():
    """Unknown apps are refused; known apps resolve (launch is patched)."""
    res = apps.open_app("definitely-not-an-app")
    assert res["success"] is False
    assert res["error"] == "unknown_app"


def test_open_app_known(monkeypatch=None):
    """Open a known app with the OS launch patched out."""
    calls = {}

    class _FakePopen:
        def __init__(self, argv, **kw):
            calls["argv"] = argv

    _patch(apps.subprocess, "Popen", _FakePopen)
    try:
        # Force a resolvable launcher so we don't depend on Chrome being present.
        _patch(apps, "_first_launchable", lambda cands: ["notepad.exe"])
        res = apps.open_app("chrome")
        assert res["success"] is True
        assert "argv" in calls
    finally:
        _unpatch_all()


def test_close_app_protects_explorer():
    """Safety: Explorer is non-closable."""
    res = apps.close_app("explorer")
    assert res["success"] is False
    assert res["error"] == "not_closable"


def test_list_running_apps_shape():
    res = apps.list_running_apps()
    assert res["success"] is True
    assert isinstance(res["data"], list)


# ----------------------------------------------------------------- files ----
def test_find_file_search(tmp_path=None):
    """find_file locates a file we create under a temp root."""
    import tempfile
    d = tempfile.mkdtemp(prefix="jarvis_tools_")
    fpath = os.path.join(d, "my_unique_resume_123.txt")
    with open(fpath, "w", encoding="utf-8") as f:
        f.write("hello")
    _patch(config, "TOOLS_FILE_ROOTS", [d])
    try:
        res = files.find_file("unique_resume_123")
        assert res["success"] is True
        assert any("my_unique_resume_123" in p for p in res["data"])
        # search_files is the friendlier alias.
        res2 = files.search_files("unique_resume_123")
        assert res2["success"] is True
    finally:
        _unpatch_all()


def test_find_file_not_found():
    import tempfile
    d = tempfile.mkdtemp(prefix="jarvis_tools_empty_")
    _patch(config, "TOOLS_FILE_ROOTS", [d])
    try:
        res = files.find_file("no_such_file_xyz")
        assert res["success"] is False
        assert res["error"] == "not_found"
    finally:
        _unpatch_all()


def test_files_are_read_only():
    """The files module exposes NO destructive operations (Phase 3 safety)."""
    forbidden = {"delete", "remove", "rename", "move", "write", "overwrite",
                 "rmtree", "unlink"}
    public = {n for n in dir(files) if not n.startswith("_")}
    assert not (forbidden & public), f"destructive op exposed: {forbidden & public}"


# --------------------------------------------------------------- browser ----
def test_browser_search_builds_urls():
    """youtube_search / google_search build correct URLs (browser open patched)."""
    opened = {}
    _patch(browser.webbrowser, "open", lambda u: opened.update(url=u))
    try:
        res = browser.youtube_search("raspberry pi projects")
        assert res["success"] is True
        assert "youtube.com/results" in opened["url"]
        assert "raspberry+pi+projects" in opened["url"]

        res = browser.google_search("hello world")
        assert res["success"] is True
        assert "google.com/search" in opened["url"]
    finally:
        _unpatch_all()


def test_read_page_live():
    """Live Playwright read of example.com — skipped if Playwright unavailable."""
    try:
        from playwright.sync_api import sync_playwright  # noqa: F401
    except Exception:
        print("    (skipped: Playwright not installed)")
        return
    res = browser.get_page_title("https://example.com")
    if not res["success"] and res.get("error") in {"playwright_missing"}:
        print("    (skipped: Playwright browser not installed)")
        return
    # example.com title is the stable "Example Domain".
    assert res["success"] is True, res
    assert "Example" in res.get("data", {}).get("title", "")


# ------------------------------------------------------ router (Phase 5) ----
def test_router_deterministic_matches():
    cases = {
        "Open VS Code": ("open_app", {"app": "vs code"}),
        "open chrome": ("open_app", {"app": "chrome"}),
        "close notepad": ("close_app", {"app": "notepad"}),
        "Search YouTube for raspberry pi projects":
            ("youtube_search", {"query": "raspberry pi projects"}),
        "open my Bluye folder": ("open_folder", {"path": "bluye"}),
    }
    for phrase, expected in cases.items():
        got = router._deterministic(phrase)
        assert got == expected, f"{phrase!r} -> {got}, expected {expected}"


def test_router_find_strips_possessive():
    got = router._deterministic("find my resume")
    assert got == ("find_file", {"filename": "resume"}), got


def test_router_conversation_is_not_a_tool():
    """Plain chat must NOT match a tool (so the LLM still answers)."""
    assert router._deterministic("how are you today") is None
    assert router._deterministic("what's the meaning of life") is None


def test_try_tools_executes_and_speaks():
    """End-to-end (deterministic path): try_tools runs a tool and speaks."""
    spoken = []

    class FakeBus:
        def speak(self, text, perf=None):
            spoken.append(text)

    # Patch the underlying app launch and disable the natural-response API call.
    _patch(config, "TOOLS_NATURAL_RESPONSE", False)
    _patch(apps, "_first_launchable", lambda cands: ["notepad.exe"])

    class _FakePopen:
        def __init__(self, argv, **kw):
            pass
    _patch(apps.subprocess, "Popen", _FakePopen)
    try:
        handled = router.try_tools("open notepad", FakeBus())
        assert handled is True
        assert spoken and "notepad" in spoken[-1].lower()
    finally:
        _unpatch_all()


def test_try_tools_passes_through_conversation():
    """Non-tool input returns False so the conversation continues to the LLM."""
    class FakeBus:
        def speak(self, *a, **k):
            raise AssertionError("should not speak for conversation")
    # Disable Gemini selection so this is purely the deterministic check.
    _patch(config, "TOOLS_GEMINI_SELECTION", False)
    try:
        assert router.try_tools("tell me a joke about cats", FakeBus()) is False
    finally:
        _unpatch_all()


def test_tools_disabled_flag():
    _patch(config, "TOOLS_ENABLED", False)
    try:
        assert router.try_tools("open chrome", object()) is False
    finally:
        _unpatch_all()


# ----------------------------------------------------- tiny patch helpers ----
# A minimal monkeypatch so the suite runs under plain `python` too (no pytest).
_patches = []


def _patch(obj, name, value):
    _patches.append((obj, name, getattr(obj, name)))
    setattr(obj, name, value)


def _unpatch_all():
    while _patches:
        obj, name, old = _patches.pop()
        setattr(obj, name, old)


# ----------------------------------------------------------- script runner ---
if __name__ == "__main__":
    import traceback
    tests = [v for k, v in sorted(globals().items())
             if k.startswith("test_") and callable(v)]
    passed = 0
    for fn in tests:
        try:
            fn()
            print(f"PASS  {fn.__name__}")
            passed += 1
        except Exception:
            print(f"FAIL  {fn.__name__}:\n{traceback.format_exc()}")
        finally:
            _unpatch_all()
    print(f"\n{passed}/{len(tests)} passed")
    sys.exit(0 if passed == len(tests) else 1)
