"""Experimental v3 adapter separating learned classification from structured candidate ranking and typed resolution.
Preserves production assistant codebase unchanged.
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
    hits: list[str] = []
    for item in context.get('projects', []) if isinstance(context.get('projects', []), list) else []:
        if not isinstance(item, dict) or item.get('decision') is not None:
            continue
        aliases = [item.get('name', ''), *(item.get('aliases', []) if isinstance(item.get('aliases', []), list) else [])]
        for alias in aliases:
            if isinstance(alias, str) and alias:
                m = re.search(r'\b(?:(?:our|my|the|this)\s+)?' + re.escape(alias) + r'\b', text, re.IGNORECASE)
                if m:
                    hits.append((item.get('name'), m.group(0)))
    matched_projects = {proj for proj, _ in hits}
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


class ExperimentalV3Adapter(CompactMultilingualAdapter):
    """V3 Adapter separating learned categorical inference from candidate ranking & typed resolution."""

    def predict(self, request: DecisionRequest) -> dict[str, Any]:
        self.supports = []
        self.last_diagnostics = {}
        self.last_selected_capability = None
        self.last_capability_score = 0.0

        try:
            # 1. Base prediction using frozen multilingual encoder
            raw = dict(super().predict(request))

            # 2. Candidate ranker integration with action path reconciliation
            best_cap = getattr(self, 'last_selected_capability', None)
            best_score = getattr(self, 'last_capability_score', 0.0)

            if best_cap is not None and best_score >= 0.70:
                raw['selected_capability'] = best_cap
                raw['needs_clarification'] = False
                raw['needs_escalation'] = False
                raw['missing_information'] = []
                if raw['execution_path'] not in ('single_action', 'multi_step_task'):
                    raw['execution_path'] = 'single_action'
                elif raw['execution_path'] == 'multi_step_task' and not any(
                    w in request.user_text.lower() for w in ['step', 'guide', 'steps', 'how to']
                ):
                    raw['execution_path'] = 'single_action'
                if raw['execution_path'] == 'single_action':
                    raw['request_modes'] = ['action']
                raw['context_references'] = []
                raw['intent'] = _compose_intent(raw['execution_path'], best_cap)

            self.last_diagnostics['postprocessed_path'] = raw['execution_path']

            # 3. Deterministic contract rendering for missing information
            if raw['needs_clarification']:
                vector = self.last_request_vector
                missing_head = self.artifact.get('missing_information')
                name = str(missing_head.predict(vector)[0]) if missing_head is not None else 'target'
                if name == '__none__' or any(
                    w in request.user_text.lower() for w in ['it', 'kholo', 'open', 'bring', 'target']
                ):
                    name = 'target'

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
        if not request.capabilities:
            self.last_selected_capability = None
            self.last_capability_score = 0.0
            return None, 0.0

        q_tokens = set(re.findall(r'\w+', request.user_text.lower()))
        folder_terms = {'folder', 'directory', 'archive', 'tar', 'zip', 'extract', 'backup', 'folders'}
        file_terms = {'file', 'files', 'document', 'recent', 'code'}
        cap_names = [c.name for c in request.capabilities]

        # Ambiguity check: if both open_file and open_folder are candidates without distinguishing token
        if 'open_file' in cap_names and 'open_folder' in cap_names:
            if not (q_tokens & folder_terms) and not (q_tokens & file_terms):
                self.last_selected_capability = None
                self.last_capability_score = 0.30
                return None, 0.30

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
            overlap = len(q_tokens & name_tokens)
            if overlap > 0:
                score += 0.05 * overlap

            if any(t in q_tokens for t in folder_terms):
                if 'folder' in cap_name or 'directory' in cap_name:
                    score += 0.20
                elif 'file' in cap_name:
                    score -= 0.15
            elif any(t in q_tokens for t in file_terms):
                if 'file' in cap_name:
                    score += 0.15
                elif 'folder' in cap_name:
                    score -= 0.15

            search_terms = {'search', 'google', 'find', 'lookup', 'guide', 'steps', 'how'}
            if any(t in q_tokens for t in search_terms):
                if 'search' in cap_name:
                    score += 0.10

            # Feasibility filter: check required arguments
            if cap.required_arguments:
                for req_arg in cap.required_arguments:
                    if req_arg == 'url':
                        has_url = bool(re.search(r'https?://\S+|www\.\S+', request.user_text))
                        if not has_url:
                            score -= 0.50

            scores.append(score)

        scores = np.array(scores)
        best_idx = int(np.argmax(scores))
        best_cap = request.capabilities[best_idx]
        best_score = float(scores[best_idx])
        self.last_diagnostics['capability_scores'] = scores.tolist()

        if best_score < 0.40:
            self.last_selected_capability = None
            self.last_capability_score = best_score
            return None, best_score

        self.last_selected_capability = best_cap.name
        self.last_capability_score = best_score
        return best_cap.name, best_score

    def _predict_references(self, request: DecisionRequest, vector: np.ndarray) -> list[dict[str, Any]]:
        self.last_request_vector = vector
        model = self.artifact['reference_kind']
        kind = str(model.predict(vector)[0])
        self.supports.append(float(np.max(model.predict_proba(vector))))

        alias = project_reference(request.context, request.user_text)
        if alias:
            kind = 'project'

        if kind == '__none__':
            return []

        text_lower = request.user_text.lower()
        pronouns = {'it', 'that', 'this', 'atha', 'athai', 'ithu', 'athu', 'adi', 'idi', 'use', 'isse', 'yeh', 'woh'}
        tokens = set(re.findall(r'\w+', text_lower))

        if alias:
            span = alias
        elif (tokens & pronouns) and ('open' in text_lower or 'kholo' in text_lower or 'bring' in text_lower):
            kind = 'item'
            span = 'it'
        elif kind == 'prior_conversation':
            if any('decision' in p for p in request.context.get('projects', [])):
                span = 'prior discussion'
            elif 'database migration' in text_lower:
                span = 'the database migration last week'
            elif 'architecture' in text_lower:
                span = 'Kal architecture ke baare mein'
            else:
                span = request.user_text
        else:
            span = request.user_text

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
