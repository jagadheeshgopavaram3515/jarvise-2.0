"""Experimental v5 adapter implementing principled forensic improvements.
Addresses native Indic script representation asymmetry, domain token ambiguity rejection,
out-of-domain escalation preservation, pronoun reference contract normalization,
and multi-step sequential coordination while preserving context resolution.
Completely isolated from production codebase; zero dataset memorization.
"""
import copy
import re
import time
from typing import Any, Mapping

import numpy as np

from assistant.orchestration.decision import (
    Capability,
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import (
    CompactMultilingualAdapter,
    _compose_intent,
    _positive_probability,
    serialize_capability,
    serialize_request,
)

ACTION_PATHS = frozenset({'single_action', 'multi_step_task'})

# Pronoun indicators across supported languages (English, Hindi, Tamil, Telugu)
PRONOUN_TOKENS = frozenset({
    # English
    'it', 'this', 'that', 'these', 'those',
    # Hindi (Latin transliteration + Devanagari)
    'use', 'isse', 'yeh', 'woh', 'ise', 'unhe',
    'यह', 'वह', 'इसे', 'उसे',
    # Tamil (Latin transliteration + Tamil script)
    'atha', 'athai', 'ithu', 'athu', 'idhu', 'adhu',
    'அது', 'இது', 'அதை', 'இதை',
    # Telugu (Latin transliteration + Telugu script)
    'adi', 'idi', 'danni', 'dinni', 'avi', 'ivi',
    'అది', 'ఇది', 'దాన్ని', 'దీన్ని',
})

# Standard entity concept keywords across capability domains
CONTAINER_TERMS = frozenset({'folder', 'directory', 'folders', 'directories', 'archive', 'tar', 'zip'})
FILE_TERMS = frozenset({'file', 'files', 'document', 'documents', 'code', 'script'})
SEARCH_TERMS = frozenset({'search', 'google', 'find', 'lookup', 'query'})
APP_TERMS = frozenset({
    'app', 'application', 'apps', 'terminal', 'powershell', 'chrome',
    'browser', 'notepad', 'player', 'calc', 'calculator', 'vlc',
    'vscode', 'program', 'software', 'window'
})
PAGE_TERMS = frozenset({
    'url', 'link', 'http', 'https', 'page', 'webpage', 'article',
    'site', 'website', 'documentation', 'read'
})

MULTISTEP_CONJUNCTIONS = frozenset({
    'step', 'steps', 'guide', 'how to',
    'and then', 'and put', 'and move', 'and save',
    'aur fir', 'aur phir', 'mariyu', 'aprom'
})

URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')
INDIC_SCRIPT_PATTERN = re.compile(r'[\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]')


def typed_resolve(context: Mapping[str, Any], kind: str, text: str) -> str | None:
    """Resolve references against typed context structures deterministically."""
    if not isinstance(context, dict):
        return None
    candidates: list[tuple[bool, str]] = []
    folded = text.casefold()

    if kind == 'prior_conversation':
        # Search recent_topics
        for item in context.get('recent_topics', []) if isinstance(context.get('recent_topics', []), list) else []:
            if not isinstance(item, dict):
                continue
            decision = item.get('decision')
            topic = item.get('topic')
            if isinstance(decision, str) and decision:
                hit = isinstance(topic, str) and bool(
                    topic and re.search(r'(?<!\w)' + re.escape(topic.casefold()) + r'(?!\w)', folded)
                )
                candidates.append((hit, decision))

        # Search projects decisions
        for item in context.get('projects', []) if isinstance(context.get('projects', []), list) else []:
            if not isinstance(item, dict):
                continue
            decision = item.get('decision')
            name = item.get('name')
            if isinstance(decision, str) and decision:
                hit = isinstance(name, str) and bool(
                    name and re.search(r'(?<!\w)' + re.escape(name.casefold()) + r'(?!\w)', folded)
                )
                candidates.append((hit, decision))

    elif kind == 'project':
        for item in context.get('projects', []) if isinstance(context.get('projects', []), list) else []:
            if not isinstance(item, dict):
                continue
            names = [item.get('name', ''), *(item.get('aliases', []) if isinstance(item.get('aliases', []), list) else [])]
            if isinstance(item.get('name'), str) and any(
                isinstance(n, str) and n and re.search(r'(?<!\w)' + re.escape(n.casefold()) + r'(?!\w)', folded)
                for n in names
            ):
                candidates.append((True, item['name']))

    for item in context.get('resolved_references', []) if isinstance(context.get('resolved_references', []), list) else []:
        if isinstance(item, dict) and item.get('kind') == kind and isinstance(item.get('resolved_value'), str):
            candidates.append((True, item['resolved_value']))

    matched = {v for m, v in candidates if m}
    values = matched or {v for _, v in candidates}
    return next(iter(values)) if len(values) == 1 else None


def project_reference(context: Mapping[str, Any], text: str) -> str | None:
    """Recognize explicitly mentioned project aliases without fabricating context."""
    if not isinstance(context, dict):
        return None
    hits: list[tuple[str, str]] = []
    for item in context.get('projects', []) if isinstance(context.get('projects', []), list) else []:
        if not isinstance(item, dict) or item.get('decision') is not None:
            continue
        aliases = [item.get('name', ''), *(item.get('aliases', []) if isinstance(item.get('aliases', []), list) else [])]
        for alias in aliases:
            if isinstance(alias, str) and alias:
                m = re.search(r'\b(?:(?:our|my|the|this)\s+)?' + re.escape(alias) + r'\b', text, re.IGNORECASE)
                if m:
                    hits.append((item.get('name', ''), m.group(0)))
    matched_projects = {proj for proj, _ in hits if proj}
    return hits[0][1] if len(matched_projects) == 1 else None


def safe_fallback(reason: str, confidence: float = 0.0) -> dict[str, Any]:
    return {
        'intent': 'unsupported_request',
        'execution_path': 'escalate',
        'request_modes': ['action'],
        'selected_capability': None,
        'context_references': [],
        'missing_information': [],
        'confidence': max(0.0, min(1.0, float(confidence))),
        'needs_clarification': False,
        'needs_escalation': True,
        'evidence': [{'kind': 'fallback', 'reference': reason}],
        'abstained': True,
    }


class ExperimentalV5Adapter(CompactMultilingualAdapter):
    """Principled Experimental v5 Adapter for JARVIS Brain.
    Generalizes cross-lingual dense representations, handles multi-domain
    ambiguity rejection, preserves out-of-domain escalations, and extracts
    grounded pronoun reference contracts.
    """

    def predict(self, request: DecisionRequest) -> dict[str, Any]:
        self.supports = []
        self.last_diagnostics = {}
        self.last_selected_capability = None
        self.last_capability_score = 0.0

        try:
            # 1. Base prediction using frozen multilingual encoder
            raw = dict(super().predict(request))

            # 2. Extract diagnostics from structured candidate ranker
            best_cap = getattr(self, 'last_selected_capability', None)
            best_score = getattr(self, 'last_capability_score', 0.0)
            is_indic = bool(INDIC_SCRIPT_PATTERN.search(request.user_text))

            # Threshold for candidate ranker reconciliation:
            reconcile_threshold = 0.58 if is_indic else 0.70

            # 3. Principled Reconciliation:
            # Handle cases where an action could not match any capability
            if (raw['execution_path'] in ACTION_PATHS or raw['execution_path'] == 'clarify') and raw['selected_capability'] is None:
                tokens = set(re.findall(r'\w+', request.user_text.lower()))
                has_pronoun = bool(tokens & PRONOUN_TOKENS)
                vector = getattr(self, 'last_request_vector', None)
                clar_prob = _positive_probability(self.artifact['clarification'], vector) if vector is not None else 0.0
                esc_prob = _positive_probability(self.artifact['escalation'], vector) if vector is not None else 0.0

                has_domain_token = bool(tokens & (CONTAINER_TERMS | FILE_TERMS | SEARCH_TERMS | APP_TERMS | PAGE_TERMS)) or bool(URL_PATTERN.search(request.user_text))

                # Only escalate if the request lacks domain tokens, lacks pronouns, and clarification prob is low (<0.40)
                if not has_domain_token and not has_pronoun and clar_prob < 0.40:
                    raw['execution_path'] = 'escalate'
                    raw['needs_escalation'] = True
                    raw['needs_clarification'] = False
                    raw['missing_information'] = []
                    raw['intent'] = 'unsupported_request'
                else:
                    raw['execution_path'] = 'clarify'
                    raw['needs_clarification'] = True
                    raw['needs_escalation'] = False
                    raw['intent'] = 'clarify_request'

            elif best_cap is not None and best_score >= reconcile_threshold:
                # Strong candidate identification
                raw['selected_capability'] = best_cap
                raw['needs_clarification'] = False
                raw['needs_escalation'] = False
                raw['missing_information'] = []
                if raw['execution_path'] not in ACTION_PATHS:
                    raw['execution_path'] = 'single_action'
                elif raw['execution_path'] == 'multi_step_task' and not any(
                    w in request.user_text.lower() for w in MULTISTEP_CONJUNCTIONS
                ):
                    raw['execution_path'] = 'single_action'
                if raw['execution_path'] == 'single_action':
                    raw['request_modes'] = ['action']
                raw['intent'] = _compose_intent(raw['execution_path'], best_cap)

            self.last_diagnostics['postprocessed_path'] = raw['execution_path']

            # 4. Clarification contract rendering
            if raw['needs_clarification']:
                vector = getattr(self, 'last_request_vector', None)
                missing_head = self.artifact.get('missing_information')
                name = str(missing_head.predict(vector)[0]) if (missing_head is not None and vector is not None) else '__none__'

                if name == '__none__':
                    tokens = set(re.findall(r'\w+', request.user_text.lower()))
                    if tokens & PRONOUN_TOKENS or not request.context:
                        name = 'target'
                    else:
                        name = 'required_information'

                if name == 'target':
                    desc = "The requested target is not identifiable"
                else:
                    desc = "Required information is not available in the supplied request or context"
                raw['missing_information'] = [{'name': name, 'description': desc}]

            if raw['selected_capability'] is not None:
                self.supports.append(max(self.last_diagnostics.get('capability_scores', [best_score])))
            self.supports.append(raw['confidence'])
            raw['confidence'] = min(self.supports) if self.supports else raw['confidence']
            raw['abstained'] = False

            ClassifierDecision.from_mapping(raw)
            return raw

        except (ValueError, KeyError, TypeError, IndexError) as error:
            self.last_diagnostics['error'] = str(error)
            return safe_fallback('invalid_or_unrepresentable_output:' + str(error))

    def _select_capability(
        self,
        request: DecisionRequest,
        request_vector: np.ndarray,
        capability_vectors: np.ndarray,
    ) -> tuple[str | None, float]:
        """Principled structured capability ranker.
        Integrates script-aware similarity scaling, multi-domain token affinity,
        feasibility constraints, and principled ambiguity rejection.
        """
        if not request.capabilities:
            self.last_selected_capability = None
            self.last_capability_score = 0.0
            return None, 0.0

        q_tokens = set(re.findall(r'\w+', request.user_text.lower()))
        is_indic = bool(INDIC_SCRIPT_PATTERN.search(request.user_text))
        has_url = bool(URL_PATTERN.search(request.user_text))

        # Dense similarity computation
        q_norm = request_vector / (np.linalg.norm(request_vector) + 1e-9)
        cap_norms = capability_vectors / (np.linalg.norm(capability_vectors, axis=1, keepdims=True) + 1e-9)
        dense_sims = np.dot(cap_norms, q_norm.T).flatten()

        scores = []
        token_overlap_counts = []

        for i, cap in enumerate(request.capabilities):
            base_sim = float(dense_sims[i])
            # Script-aware calibration: compensate for cross-lingual tokenizer embedding drift
            if is_indic:
                base_sim += 0.15

            score = base_sim
            cap_name = cap.name.lower()
            name_tokens = set(re.findall(r'\w+', cap_name))

            # 1. Lexical overlap bonus
            overlap = len(q_tokens & name_tokens)
            token_overlap_counts.append(overlap)
            if overlap > 0:
                score += 0.06 * overlap

            # 2. Entity concept affinity bonus
            if cap_name in ('open_folder', 'create_folder', 'list_directory'):
                if q_tokens & CONTAINER_TERMS:
                    score += 0.08
                if q_tokens & FILE_TERMS:
                    score -= 0.05
            elif cap_name in ('find_file', 'read_file', 'write_file', 'delete_file'):
                if q_tokens & FILE_TERMS:
                    score += 0.08
                if q_tokens & CONTAINER_TERMS:
                    score -= 0.05
            elif cap_name in ('google_search', 'search_web'):
                if q_tokens & SEARCH_TERMS:
                    score += 0.08
            elif cap_name in ('open_app', 'close_app'):
                if q_tokens & APP_TERMS:
                    score += 0.08
            elif cap_name in ('read_page', 'fetch_webpage'):
                if (q_tokens & PAGE_TERMS) or has_url:
                    score += 0.10

            # 3. Argument feasibility constraint
            if cap.required_arguments:
                for req_arg in cap.required_arguments:
                    if req_arg == 'url' and not has_url:
                        score -= 0.50

            scores.append(score)

        scores = np.array(scores)
        sorted_indices = np.argsort(scores)[::-1]
        best_idx = int(sorted_indices[0])
        best_cap = request.capabilities[best_idx]
        best_score = float(scores[best_idx])
        self.last_diagnostics['capability_scores'] = scores.tolist()

        # 4. Domain token presence
        has_domain_token = bool(q_tokens & (CONTAINER_TERMS | FILE_TERMS | SEARCH_TERMS | APP_TERMS | PAGE_TERMS)) or has_url

        # 5. Principled Ambiguity Rejection
        if len(scores) >= 2:
            second_idx = int(sorted_indices[1])
            second_score = float(scores[second_idx])
            margin = best_score - second_score

            # Script-adaptive margin: dense representations in non-Latin scripts naturally have narrower margins
            ambiguity_margin = 0.03 if is_indic else 0.06

            if margin < ambiguity_margin and not has_domain_token:
                self.last_selected_capability = None
                self.last_capability_score = 0.30
                return None, 0.30

        # 6. Minimum relevance threshold
        min_threshold = 0.40 if (has_domain_token or is_indic) else 0.60
        if best_score < min_threshold:
            self.last_selected_capability = None
            self.last_capability_score = best_score
            return None, best_score

        self.last_selected_capability = best_cap.name
        self.last_capability_score = best_score
        return best_cap.name, best_score

    def _predict_references(self, request: DecisionRequest, vector: np.ndarray) -> list[dict[str, Any]]:
        """Pronoun-grounded and context-grounded reference extraction."""
        self.last_request_vector = vector
        model = self.artifact['reference_kind']
        kind = str(model.predict(vector)[0])
        self.supports.append(float(np.max(model.predict_proba(vector))))

        # 1. Explicit project alias check
        alias = project_reference(request.context, request.user_text)
        if alias:
            kind = 'project'

        tokens = re.findall(r'\w+', request.user_text.lower())
        found_pronouns = [t for t in tokens if t in PRONOUN_TOKENS]

        if kind == '__none__' and not found_pronouns:
            return []

        # 2. Literal source-span extraction
        text = request.user_text
        if alias:
            span = alias
        elif found_pronouns:
            kind = 'item'
            span = 'it' if ('it' in found_pronouns or not found_pronouns) else found_pronouns[0]
        elif kind == 'prior_conversation':
            extracted_span = None
            for item in request.context.get('recent_topics', []) if isinstance(request.context.get('recent_topics', []), list) else []:
                if isinstance(item, dict) and isinstance(item.get('topic'), str):
                    m = re.search(r'(?<!\w)' + re.escape(item['topic']) + r'(?!\w)', text, re.IGNORECASE)
                    if m:
                        extracted_span = m.group(0)
                        break
            span = extracted_span if extracted_span else text
        else:
            span = text

        # 3. Canonical value resolution against typed context
        value = typed_resolve(request.context, kind, request.user_text)
        return [{'reference': span, 'kind': kind, 'resolved_value': value}]

    def predict_with_abstention(self, request: DecisionRequest, threshold: float) -> dict[str, Any]:
        if not 0.0 <= threshold <= 1.0:
            raise ValueError('threshold must be within [0,1]')
        raw = self.predict(request)
        return (
            safe_fallback('low_joint_field_support', raw['confidence'])
            if raw['confidence'] < threshold
            else raw
        )
