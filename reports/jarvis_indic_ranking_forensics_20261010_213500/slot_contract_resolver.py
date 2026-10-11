"""Canonical Argument Contract and Slot Contract Resolver for JARVIS Brain.

Provides:
1. Authoritative capability argument schemas (required vs optional).
2. Machine-readable argument identifiers vs human-readable clarification messages.
3. Strict decision contract validation and consistency enforcement.
4. Admissibility filtering against advertised capabilities.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from typing import Any, Mapping, Sequence

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
    DecisionValidationError,
)

DEICTIC_TOKENS = frozenset({
    "this", "that", "it", "these", "those", "them",
    "use", "isse", "iska", "iski", "isko", "usko",
    "idi", "adi", "danni", "veeti", "vaati",
    "ithu", "athu", "athai", "ithai",
    "यह", "वह", "इसे", "उसे", "इसको", "उसको",
    "ఇది", "అది", "దీన్ని", "దాన్ని",
    "இது", "அது", "இதை", "அதை",
})


@dataclass(frozen=True)
class CanonicalCapabilitySchema:
    name: str
    description: str
    required_arguments: tuple[str, ...]
    optional_arguments: tuple[str, ...] = ()
    argument_aliases: dict[str, str] = field(default_factory=dict)
    clarification_prompts: dict[str, str] = field(default_factory=dict)


CANONICAL_CAPABILITY_SCHEMAS: dict[str, CanonicalCapabilitySchema] = {
    "open_app": CanonicalCapabilitySchema(
        name="open_app",
        description="Launch or open an installed desktop application.",
        required_arguments=("app",),
        optional_arguments=(),
        argument_aliases={"name": "app"},
        clarification_prompts={"app": "Which application would you like to open?"},
    ),
    "close_app": CanonicalCapabilitySchema(
        name="close_app",
        description="Close a running application.",
        required_arguments=("app",),
        optional_arguments=(),
        argument_aliases={"name": "app"},
        clarification_prompts={"app": "Which application would you like to close?"},
    ),
    "list_running_apps": CanonicalCapabilitySchema(
        name="list_running_apps",
        description="List running desktop applications.",
        required_arguments=(),
        optional_arguments=(),
    ),
    "find_file": CanonicalCapabilitySchema(
        name="find_file",
        description="Find local files matching a filename or query pattern.",
        required_arguments=("filename",),
        optional_arguments=(),
        argument_aliases={"query": "filename"},
        clarification_prompts={"filename": "What file name or pattern are you looking for?"},
    ),
    "search_files": CanonicalCapabilitySchema(
        name="search_files",
        description="Search user folders for files matching keyword.",
        required_arguments=("keyword",),
        optional_arguments=(),
        argument_aliases={"query": "keyword"},
        clarification_prompts={"keyword": "What keyword would you like to search for?"},
    ),
    "open_file": CanonicalCapabilitySchema(
        name="open_file",
        description="Open a file with default system application.",
        required_arguments=("path",),
        optional_arguments=(),
        argument_aliases={"file_path": "path"},
        clarification_prompts={"path": "Which file path should I open?"},
    ),
    "open_folder": CanonicalCapabilitySchema(
        name="open_folder",
        description="Open a folder in Windows Explorer.",
        required_arguments=("path",),
        optional_arguments=(),
        argument_aliases={"folder_path": "path"},
        clarification_prompts={"path": "Which folder path would you like to open?"},
    ),
    "delete_file": CanonicalCapabilitySchema(
        name="delete_file",
        description="Delete a specified file permanently.",
        required_arguments=("path",),
        optional_arguments=(),
        argument_aliases={"file_path": "path"},
        clarification_prompts={"path": "Which file path should be deleted?"},
    ),
    "delete_folder": CanonicalCapabilitySchema(
        name="delete_folder",
        description="Delete a specified folder permanently.",
        required_arguments=("path",),
        optional_arguments=(),
        argument_aliases={"folder_path": "path"},
        clarification_prompts={"path": "Which folder path should be deleted?"},
    ),
    "read_file": CanonicalCapabilitySchema(
        name="read_file",
        description="Read lines from a project text file.",
        required_arguments=("path",),
        optional_arguments=("offset", "limit"),
        argument_aliases={"file_path": "path"},
        clarification_prompts={"path": "Which file should I read?"},
    ),
    "read_page": CanonicalCapabilitySchema(
        name="read_page",
        description="Read content from a specified URL.",
        required_arguments=("url",),
        optional_arguments=(),
        argument_aliases={"web_url": "url"},
        clarification_prompts={"url": "Which webpage URL should I read?"},
    ),
    "open_url": CanonicalCapabilitySchema(
        name="open_url",
        description="Open a URL in default browser.",
        required_arguments=("url",),
        optional_arguments=(),
        argument_aliases={"web_url": "url"},
        clarification_prompts={"url": "Which webpage URL should I open?"},
    ),
    "google_search": CanonicalCapabilitySchema(
        name="google_search",
        description="Perform web search for a query.",
        required_arguments=("query",),
        optional_arguments=(),
        argument_aliases={"search_query": "query"},
        clarification_prompts={"query": "What would you like me to search for?"},
    ),
}


class CanonicalArgumentContract:
    """Authoritative schema definition and argument validation."""

    def __init__(self, schemas: dict[str, CanonicalCapabilitySchema] | None = None) -> None:
        self._schemas = schemas or CANONICAL_CAPABILITY_SCHEMAS

    def get_schema(self, capability_name: str) -> CanonicalCapabilitySchema | None:
        return self._schemas.get(capability_name)

    def validate_argument(self, capability_name: str, argument_name: str) -> bool:
        schema = self.get_schema(capability_name)
        if schema is None:
            return False
        if argument_name in schema.required_arguments or argument_name in schema.optional_arguments:
            return True
        return argument_name in schema.argument_aliases

    def canonical_arg_name(self, capability_name: str, argument_name: str) -> str:
        schema = self.get_schema(capability_name)
        if schema is None:
            return argument_name
        return schema.argument_aliases.get(argument_name, argument_name)


class SlotContractResolver:
    """Enforces slot contract consistency, argument schema alignment, and decision safety."""

    def __init__(self, contract: CanonicalArgumentContract | None = None) -> None:
        self.contract = contract or CanonicalArgumentContract()

    def format_missing_item(self, name: str, description: str) -> dict[str, str]:
        return {"name": name, "description": description}

    def create_missing_info(self, items: Sequence[tuple[str, str]]) -> list[dict[str, str]]:
        return [self.format_missing_item(n, d) for n, d in items]

    def resolve_decision(
        self,
        raw_decision: Mapping[str, Any],
        request: DecisionRequest,
    ) -> dict[str, Any]:
        """Validates and enforces canonical contract rules on untrusted raw decision output."""
        decision = dict(raw_decision)
        advertised_caps = {c.name for c in request.capabilities} if request.capabilities else set()

        # 1. Capability Admissibility Check
        selected_cap = decision.get("selected_capability")
        if selected_cap is not None and advertised_caps and selected_cap not in advertised_caps:
            # Model hallucinated / selected an unadvertised capability
            decision["selected_capability"] = None
            decision["execution_path"] = "escalate"
            decision["needs_escalation"] = True
            decision["needs_clarification"] = False
            decision["missing_information"] = []
            decision["intent"] = "unsupported_request"

        # 2. Escalation Consistency
        if decision.get("execution_path") == "escalate" or decision.get("needs_escalation"):
            decision["execution_path"] = "escalate"
            decision["needs_escalation"] = True
            decision["needs_clarification"] = False
            decision["missing_information"] = []
            decision["selected_capability"] = None
            if not decision.get("intent", "").startswith("unsupported"):
                decision["intent"] = "unsupported_request"

        # 3. Clarification Consistency
        elif decision.get("execution_path") == "clarify" or decision.get("needs_clarification"):
            decision["execution_path"] = "clarify"
            decision["needs_clarification"] = True
            decision["needs_escalation"] = False
            decision["selected_capability"] = None
            decision["intent"] = "clarify_request"

            # Check missing_information
            missing = list(decision.get("missing_information") or [])
            if not missing:
                # Need to generate missing_information item
                words = set(re.findall(r"\w+", request.user_text.lower()))
                if words & DEICTIC_TOKENS or not request.context:
                    name = "target"
                    desc = "The requested target is not identifiable"
                else:
                    name = "required_information"
                    desc = "Required information is not available in the supplied request or context"
                decision["missing_information"] = [self.format_missing_item(name, desc)]
            else:
                decision["missing_information"] = missing

        # 4. Action / Non-clarification consistency
        else:
            decision["needs_clarification"] = False
            decision["missing_information"] = []

        # Validate contract conformance
        ClassifierDecision.from_mapping(decision)
        return decision

