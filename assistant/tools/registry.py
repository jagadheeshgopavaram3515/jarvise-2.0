"""
ToolRegistry — the single catalogue of callable tools.

This is the allow-list (Phase 6): a tool that is not registered here can never
be executed. Registration is explicit, so arbitrary command execution is
structurally impossible — the executor only ever dispatches to a ``ToolSpec``
that lives in this registry.

The registry is deliberately tiny and reusable (no Jarvis imports): Bluye can
instantiate its own ``ToolRegistry`` and register a different tool set against
the same ``ToolExecutor``.
"""
from __future__ import annotations

from typing import Dict, List, Optional

from assistant.core.log import get
from assistant.tools.schemas import ToolSpec

log = get("tools")


class ToolRegistry:
    """An ordered, name-keyed collection of ``ToolSpec`` objects."""

    def __init__(self) -> None:
        self._tools: Dict[str, ToolSpec] = {}

    # ------------------------------------------------------------------ CRUD
    def register_tool(self, spec: ToolSpec) -> None:
        """Add (or replace) a tool. Last registration wins for a given name."""
        if not isinstance(spec, ToolSpec):
            raise TypeError(f"register_tool expects a ToolSpec, got {type(spec)!r}")
        if not spec.name or not callable(spec.handler):
            raise ValueError("ToolSpec needs a non-empty name and a callable handler")
        if spec.name in self._tools:
            log.info("[REGISTRY] overriding existing tool %r", spec.name)
        self._tools[spec.name] = spec

    def register(self, *specs: ToolSpec) -> None:
        """Convenience: register several specs at once."""
        for spec in specs:
            self.register_tool(spec)

    def get_tool(self, name: str) -> Optional[ToolSpec]:
        """Return the spec for ``name`` or ``None`` if it is not registered."""
        return self._tools.get(name)

    def list_tools(self) -> List[ToolSpec]:
        """All registered specs, in registration order."""
        return list(self._tools.values())

    def names(self) -> List[str]:
        return list(self._tools.keys())

    def __contains__(self, name: str) -> bool:
        return name in self._tools

    def __len__(self) -> int:
        return len(self._tools)

    # -------------------------------------------------------- LLM advertising
    def to_function_declarations(self) -> List[dict]:
        """Gemini-style function declarations for every registered tool."""
        return [spec.to_function_declaration() for spec in self._tools.values()]

    def describe(self) -> str:
        """A compact, human/LLM-readable listing used in tool-selection prompts."""
        lines = []
        for spec in self._tools.values():
            props = (spec.parameters or {}).get("properties", {})
            args = ", ".join(props.keys()) if props else ""
            lines.append(f"- {spec.name}({args}): {spec.description}")
        return "\n".join(lines)


def build_default_registry() -> ToolRegistry:
    """Construct the registry with Jarvis's desktop tool set wired in.

    Imports of app/file/browser handlers are done HERE (lazily, inside the
    function) rather than at module top so that importing ``registry`` alone
    stays cheap and free of optional deps (psutil/playwright). Browser tools
    register even when Playwright isn't installed — the handler reports a clean,
    speakable error at call time instead of breaking registration.
    """
    from assistant.tools import apps, browser, files
    from assistant.tools import apps, browser, files, location, weather

    reg = ToolRegistry()

    # ---- Real-world Location & Weather tools (Phase 1.5) -------------------
    reg.register_tool(ToolSpec(
        name="location",
        description="Determine the user's current geographic location (city, region, country, latitude, longitude).",
        handler=location.get_location,
        parameters={
            "type": "object",
            "properties": {
                "purpose": {"type": "string", "description": "Purpose for requesting location (e.g. 'weather', 'general')."}
            },
            "required": [],
        },
        examples=["where am I", "get current location", "what city am I in"],
    ))
    reg.register_tool(ToolSpec(
        name="weather",
        description="Fetch current weather conditions, temperature, humidity, wind, and rain for given coordinates or city.",
        handler=weather.get_weather,
        parameters={
            "type": "object",
            "properties": {
                "latitude": {"type": "number", "description": "Latitude float coordinate (optional if city provided)."},
                "longitude": {"type": "number", "description": "Longitude float coordinate (optional if city provided)."},
                "city": {"type": "string", "description": "City or locality name (e.g. 'Hyderabad', 'London')."},
            },
            "required": [],
        },
        examples=["what is the weather in Hyderabad", "weather at current location", "will it rain today"],
    ))

    # ---- App control (Phase 2) ---------------------------------------------
    reg.register_tool(ToolSpec(
        name="open_app",
        description="Open/launch a desktop application by name on Windows "
                    "(chrome, edge, vs code, notepad, calculator, explorer).",
        handler=apps.open_app,
        parameters={
            "type": "object",
            "properties": {
                "app": {"type": "string",
                        "description": "Application to open, e.g. 'chrome', "
                                       "'vs code', 'notepad', 'calculator'."}
            },
            "required": ["app"],
        },
        examples=["open chrome", "open vs code", "launch notepad", "open calculator"],
    ))
    reg.register_tool(ToolSpec(
        name="close_app",
        description="Close a running desktop application by name "
                    "(chrome, edge, vs code, notepad).",
        handler=apps.close_app,
        parameters={
            "type": "object",
            "properties": {
                "app": {"type": "string",
                        "description": "Application to close, e.g. 'chrome', 'notepad'."}
            },
            "required": ["app"],
        },
        examples=["close chrome", "close notepad"],
    ))
    reg.register_tool(ToolSpec(
        name="list_running_apps",
        description="List the known desktop applications currently running.",
        handler=apps.list_running_apps,
        parameters={"type": "object", "properties": {}, "required": []},
        examples=["what apps are open", "list running apps"],
    ))

    # ---- File access (Phase 3, read-only) ----------------------------------
    reg.register_tool(ToolSpec(
        name="find_file",
        description="Find files whose name matches the given filename "
                    "(read-only search of common user folders).",
        handler=files.find_file,
        parameters={
            "type": "object",
            "properties": {
                "filename": {"type": "string",
                             "description": "Name (or part of a name) to find, "
                                            "e.g. 'resume', 'budget.xlsx'."}
            },
            "required": ["filename"],
        },
        examples=["find my resume", "find budget.xlsx"],
    ))
    reg.register_tool(ToolSpec(
        name="search_files",
        description="Search common user folders for files containing a keyword "
                    "in their name (read-only).",
        handler=files.search_files,
        parameters={
            "type": "object",
            "properties": {
                "keyword": {"type": "string",
                            "description": "Keyword to look for in file names."}
            },
            "required": ["keyword"],
        },
        examples=["search files for invoice", "search for python projects"],
    ))
    reg.register_tool(ToolSpec(
        name="open_file",
        description="Open a file with its default application (read-only: only "
                    "opens, never edits or deletes).",
        handler=files.open_file,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Full path to the file."}
            },
            "required": ["path"],
        },
        examples=["open that file", "open C:/Users/me/resume.pdf"],
    ))
    reg.register_tool(ToolSpec(
        name="open_folder",
        description="Open a folder in Windows Explorer (read-only navigation).",
        handler=files.open_folder,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string",
                         "description": "Folder path or known name, e.g. 'Bluye', "
                                        "'Documents', 'Downloads'."}
            },
            "required": ["path"],
        },
        examples=["open my Bluye folder", "open downloads"],
    ))
    reg.register_tool(ToolSpec(
        name="recent_files",
        description="List the most recently modified files in common user folders.",
        handler=files.recent_files,
        parameters={
            "type": "object",
            "properties": {
                "limit": {"type": "integer",
                          "description": "How many recent files to return (default 10)."}
            },
            "required": [],
        },
        examples=["what did I work on recently", "show recent files"],
    ))

    # ---- Project inspection and testing (Phase 1 Agent Tools) -------------
    reg.register_tool(ToolSpec(
        name="read_file",
        description="Read lines from a text file within the project root (bounded by offset and limit).",
        handler=files.read_file,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Relative or project-bound path to the file to read."},
                "offset": {"type": "integer", "description": "Starting line number (1-indexed, default 1)."},
                "limit": {"type": "integer", "description": "Number of lines to read (max 200, default 100)."},
            },
            "required": ["path"],
        },
        examples=["read assistant/config.py", "read lines 1 to 50 of main.py"],
    ))
    reg.register_tool(ToolSpec(
        name="list_directory",
        description="List files and folders within the project root directory.",
        handler=files.list_directory,
        parameters={
            "type": "object",
            "properties": {
                "path": {"type": "string", "description": "Subdirectory to list (relative to project root; empty for root)."},
            },
            "required": [],
        },
        examples=["list directory assistant", "list project root"],
    ))
    reg.register_tool(ToolSpec(
        name="grep_code",
        description="Search for a text pattern in code and config files across the project.",
        handler=files.grep_code,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "Text pattern or symbol to search for."},
                "path": {"type": "string", "description": "Subdirectory to restrict search to (empty for full project)."},
            },
            "required": ["query"],
        },
        examples=["search code for CLASSIFIER_TIMEOUT_S", "find where build_routing_graph is defined"],
    ))
    reg.register_tool(ToolSpec(
        name="run_project_tests",
        description="Execute approved project test modules safely using Python subprocess.",
        handler=files.run_project_tests,
        parameters={
            "type": "object",
            "properties": {
                "test_target": {"type": "string", "description": "Specific test file or module (e.g. 'tests/test_orchestration.py' or empty for all)."},
            },
            "required": [],
        },
        examples=["run tests", "run tests/test_orchestration.py"],
    ))

    # ---- Browser control (Phase 4, Playwright, DOM-only) -------------------
    reg.register_tool(ToolSpec(
        name="open_url",
        description="Open a web page URL in the default browser.",
        handler=browser.open_url,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to open."}
            },
            "required": ["url"],
        },
        examples=["open github.com", "go to wikipedia"],
    ))
    reg.register_tool(ToolSpec(
        name="google_search",
        description="Open a Google search for the given query in the browser.",
        handler=browser.google_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search for."}
            },
            "required": ["query"],
        },
        examples=["google the weather in London"],
    ))
    reg.register_tool(ToolSpec(
        name="youtube_search",
        description="Open a YouTube search for the given query in the browser.",
        handler=browser.youtube_search,
        parameters={
            "type": "object",
            "properties": {
                "query": {"type": "string", "description": "What to search YouTube for."}
            },
            "required": ["query"],
        },
        examples=["search youtube for raspberry pi projects"],
    ))
    reg.register_tool(ToolSpec(
        name="read_page",
        description="Fetch a web page (headless) and return a snippet of its "
                    "readable text. DOM-based, read-only.",
        handler=browser.read_page,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL to read."}
            },
            "required": ["url"],
        },
        examples=["read this page", "what does this article say"],
    ))
    reg.register_tool(ToolSpec(
        name="get_page_title",
        description="Fetch a web page (headless) and return its <title>. "
                    "DOM-based, read-only.",
        handler=browser.get_page_title,
        parameters={
            "type": "object",
            "properties": {
                "url": {"type": "string", "description": "The URL whose title to read."}
            },
            "required": ["url"],
        },
        examples=["read the title of this webpage"],
    ))

    return reg
