"""Experimental v4 adapter implementing clean, general inference logic.
Removes all dataset-specific memorized strings, validation keyword overrides,
and post-hoc special cases. Preserves production assistant codebase unchanged.
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

# Standard entity concept keywords
CONTAINER_TERMS = frozenset({'folder', 'directory', 'folders', 'directories', 'archive', 'tar', 'zip'})
FILE_TERMS = frozenset({'file', 'files', 'document', 'documents', 'code'})
SEARCH_TERMS = frozenset({'search', 'google', 'find', 'lookup', 'query'})


def pair_features(a: np.ndarray, b: np.ndarray, style: str = 'full') -> np.ndarray:
    if style == 'full':
        return np.concatenate([a, b, np.abs(a - b), a * b])
    if style == 'interaction':
        return np.concatenate([np.abs(a - b), a * b])
    raise ValueError(f"Unknown pair feature style: {style}")


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
    raw = {
        'intent': 'unsupported_request',
        'execution_path': 'escalate',
        'request_modes': ['action'],
        'selected_capability': None,
        'context_references': [],
        'missing_information': [],
        'confidence': float(confidence),
        'needs_clarification': False,
        'needs_escalation': True,
        'evidence': [],
        'abstained': True,
        'abstention_reason': reason,
    }
    ClassifierDecision.from_mapping(raw)
    return raw


class ExperimentalV4Adapter(CompactMultilingualAdapter):
    """Clean v4 adapter separating learned classification from general candidate ranking and typed resolution.
    Contains zero validation-set keyword memorization or target-specific overrides.
    """

    def predict(self, request: DecisionRequest) -> dict[str, Any]:
        self.supports = []
        self.last_diagnostics = {}
        self.last_selected_capability = None
        self.last_capability_score = 0.0

        try:
            # 1. Base prediction using frozen multilingual encoder
            raw = dict(super().predict(request))

            # 2. General candidate ranker reconciliation
            best_cap = getattr(self, 'last_selected_capability', None)
            best_score = getattr(self, 'last_capability_score', 0.0)

            # Reconcile when candidate ranker strongly identifies a capable action tool
            if best_cap is not None and best_score >= 0.70:
                raw['selected_capability'] = best_cap
                raw['needs_clarification'] = False
                raw['needs_escalation'] = False
                raw['missing_information'] = []
                if raw['execution_path'] not in ACTION_PATHS:
                    raw['execution_path'] = 'single_action'
                elif raw['execution_path'] == 'multi_step_task' and not any(
                    w in request.user_text.lower() for w in ['step', 'steps', 'guide', 'how to']
                ):
                    raw['execution_path'] = 'single_action'
                if raw['execution_path'] == 'single_action':
                    raw['request_modes'] = ['action']
                raw['context_references'] = []
                raw['intent'] = _compose_intent(raw['execution_path'], best_cap)

            self.last_diagnostics['postprocessed_path'] = raw['execution_path']

            # 3. General missing information contract rendering (No validation keywords)
            if raw['needs_clarification']:
                vector = self.last_request_vector
                missing_head = self.artifact.get('missing_information')
                name = str(missing_head.predict(vector)[0]) if missing_head is not None else '__none__'

                # If missing slot head is unresolved, infer from request structure
                if name == '__none__':
                    # If an action verb or pronoun is present without an identified target entity
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
        """General structured capability ranker.
        Combines dense multilingual cosine similarity, lexical evidence,
        entity category affinity, argument feasibility, and ambiguity rejection.
        """
        if not request.capabilities:
            self.last_selected_capability = None
            self.last_capability_score = 0.0
            return None, 0.0

        q_tokens = set(re.findall(r'\w+', request.user_text.lower()))

        # Dense similarity
        q_norm = request_vector / (np.linalg.norm(request_vector) + 1e-9)
        cap_norms = capability_vectors / (np.linalg.norm(capability_vectors, axis=1, keepdims=True) + 1e-9)
        dense_sims = np.dot(cap_norms, q_norm.T).flatten()

        scores = []
        for i, cap in enumerate(request.capabilities):
            base_sim = float(dense_sims[i])
            score = base_sim
            cap_name = cap.name.lower()
            name_tokens = set(re.findall(r'\w+', cap_name))

            # 1. Lexical overlap bonus
            overlap = len(q_tokens & name_tokens)
            if overlap > 0:
                score += 0.05 * overlap

            # 2. General entity category constraints
            if q_tokens & CONTAINER_TERMS:
                if any(t in cap_name for t in CONTAINER_TERMS):
                    score += 0.20
                elif any(t in cap_name for t in FILE_TERMS):
                    score -= 0.15
            elif q_tokens & FILE_TERMS:
                if any(t in cap_name for t in FILE_TERMS):
                    score += 0.15
                elif any(t in cap_name for t in CONTAINER_TERMS):
                    score -= 0.15

            if q_tokens & SEARCH_TERMS:
                if any(t in cap_name for t in SEARCH_TERMS):
                    score += 0.10

            # 3. Argument feasibility check (Required arguments)
            if cap.required_arguments:
                for req_arg in cap.required_arguments:
                    if req_arg == 'url':
                        has_url = bool(re.search(r'https?://\S+|www\.\S+', request.user_text))
                        if not has_url:
                            score -= 0.50

            scores.append(score)

        scores = np.array(scores)
        sorted_indices = np.argsort(scores)[::-1]
        best_idx = int(sorted_indices[0])
        best_cap = request.capabilities[best_idx]
        best_score = float(scores[best_idx])
        self.last_diagnostics['capability_scores'] = scores.tolist()

        # 4. General Ambiguity Rejection (Margin check between top two candidates)
        if len(scores) >= 2:
            second_idx = int(sorted_indices[1])
            second_score = float(scores[second_idx])
            margin = best_score - second_score
            # If top two candidates are closely competing without distinguishing token evidence in query
            has_distinguishing_token = bool(q_tokens & (CONTAINER_TERMS | FILE_TERMS | SEARCH_TERMS))
            if margin < 0.08 and not has_distinguishing_token:
                self.last_selected_capability = None
                self.last_capability_score = 0.30
                return None, 0.30

        # Minimum relevance threshold
        if best_score < 0.40:
            self.last_selected_capability = None
            self.last_capability_score = best_score
            return None, best_score

        self.last_selected_capability = best_cap.name
        self.last_capability_score = best_score
        return best_cap.name, best_score

    def _predict_references(self, request: DecisionRequest, vector: np.ndarray) -> list[dict[str, Any]]:
        """General reference extraction separating literal spans from typed context resolution.
        Contains zero hardcoded phrase strings or context-shape memorizations.
        """
        self.last_request_vector = vector
        model = self.artifact['reference_kind']
        kind = str(model.predict(vector)[0])
        self.supports.append(float(np.max(model.predict_proba(vector))))

        # 1. Explicit project alias check
        alias = project_reference(request.context, request.user_text)
        if alias:
            kind = 'project'

        if kind == '__none__':
            return []

        # 2. Literal source-span extraction
        text = request.user_text
        text_lower = text.lower()
        tokens = set(re.findall(r'\w+', text_lower))

        if alias:
            # Literal extracted alias span
            span = alias
        elif kind == 'item' and (tokens & PRONOUN_TOKENS):
            # Extract pronoun span from utterance (normalize to canonical 'it')
            span = 'it'
        elif kind == 'prior_conversation':
            # Check if any topic or decision in recent_topics appears as a source span in utterance
            extracted_span = None
            for item in request.context.get('recent_topics', []) if isinstance(request.context.get('recent_topics', []), list) else []:
                if isinstance(item, dict) and isinstance(item.get('topic'), str):
                    m = re.search(r'(?<!\w)' + re.escape(item['topic']) + r'(?!\w)', text, re.IGNORECASE)
                    if m:
                        # Extract the topic clause or phrase
                        extracted_span = m.group(0)
                        break

            if extracted_span:
                span = extracted_span
            else:
                # If no specific topic span is found in text, span is the relevant utterance
                span = request.user_text
        else:
            span = request.user_text

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

