r"""Experimental v7 Adapter: Symmetrized Indic Semantic Ranker with Unicode Tokenization.

Preserves the exact battle-tested v5 arbitration architecture, but resolves the confirmed root causes:
1. Replaces broken r'\w+' that shattered Indic matras/viramas with Unicode-safe word extraction.
2. Symmetrizes domain concept keywords across Hindi, Telugu, and Tamil.
3. Fixes false escalation of Indic deictic requests by correctly detecting Indic pronouns.
4. Corrects search affinity competition between find_file and google_search.
5. Integrates canonical slot contract resolver.
"""
from __future__ import annotations

import re
import sys
import time
from pathlib import Path
from typing import Any, Mapping

# Ensure V5 and Slot Resolver directories are importable
V5_DIR = Path("D:/JARVIS/reports/jarvis_brain_forensic_improvement_20261010_183000")
SLOT_DIR = Path("D:/JARVIS/reports/jarvis_slot_contract_fix_20261010_211600")
if str(V5_DIR) not in sys.path:
    sys.path.insert(0, str(V5_DIR))
if str(SLOT_DIR) not in sys.path:
    sys.path.insert(0, str(SLOT_DIR))

import numpy as np

from assistant.orchestration.decision import (
    ClassifierDecision,
    DecisionRequest,
)
from assistant.orchestration.compact_adapter import (
    _decision_to_mapping,
    _positive_probability,
    serialize_capability,
    serialize_request,
)
from experimental_adapter_v5 import (
    ExperimentalV5Adapter,
    _compose_intent,
    safe_fallback,
    project_reference,
)
from slot_contract_resolver import (
    SlotContractResolver,
)

ACTION_PATHS = frozenset({'single_action', 'multi_step_task'})

# Regex for Unicode-safe word extraction (preserves combining marks/matras/viramas)
UNICODE_WORD_PATTERN = re.compile(r'[^\s.,?!:;\'"()\[\]{}]+')

# Symmetrized Pronoun indicators across supported languages
INDIC_PRONOUN_TOKENS = frozenset({
    # English
    'it', 'this', 'that', 'these', 'those',
    # Hindi (Latin transliteration + Devanagari)
    'use', 'isse', 'yeh', 'woh', 'ise', 'unhe', 'isko', 'usko',
    'यह', 'वह', 'इसे', 'उसे', 'इसको', 'उसको', 'इन्हें', 'उन्हें',
    # Tamil (Latin transliteration + Tamil script)
    'atha', 'athai', 'ithu', 'athu', 'idhu', 'adhu', 'ithai',
    'அது', 'இது', 'அதை', 'இதை', 'இவை', 'அவை',
    # Telugu (Latin transliteration + Telugu script)
    'adi', 'idi', 'danni', 'dinni', 'avi', 'ivi',
    'అది', 'ఇది', 'దాన్ని', 'దీన్ని', 'అవి', 'ఇవి',
})

CONTAINER_TERMS = frozenset({
    # English
    'folder', 'directory', 'folders', 'directories', 'archive', 'tar', 'zip',
    # Hindi
    'फ़ोल्डर', 'फोल्डर', 'डायरेक्टरी',
    # Telugu
    'ఫోల్డర్', 'డైరెక్టరీ',
    # Tamil
    'கோப்புறை',
})

FILE_TERMS = frozenset({
    # English
    'file', 'files', 'document', 'documents', 'code', 'script', 'spreadsheet', 'pdf', 'invoice', 'report',
    # Hindi
    'फाइल', 'फ़ाइल', 'दस्तावेज़', 'दस्तावेज', 'पीडीएफ', 'रसीद',
    # Telugu
    'ఫైల్', 'పత్రం', 'పిడిఎఫ్', 'సర్టిఫికెట్',
    # Tamil
    'கோப்பு', 'ஆவணம்', 'ரசீது',
})

SEARCH_TERMS = frozenset({
    # English
    'search', 'google', 'find', 'lookup', 'query', 'web', 'internet',
    # Hindi
    'ढूंढो', 'तलाशो', 'खोजो', 'ढूंढ', 'वेब', 'गूगल',
    # Telugu
    'వెతుకు', 'శోధించు', 'గూగుల్', 'వెతకండి',
    # Tamil
    'தேடு', 'இணையம்', 'கூகுள்', 'தேடவும்',
})

APP_TERMS = frozenset({
    # English
    'app', 'application', 'apps', 'terminal', 'powershell', 'chrome',
    'browser', 'notepad', 'player', 'calc', 'calculator', 'vlc',
    'vscode', 'program', 'software', 'window', 'whatsapp', 'spotify',
    # Hindi
    'ऐप', 'एप्लिकेशन', 'नोटपैड',
    # Telugu
    'యాప్', 'యాప్‌',
    # Tamil
    'செயலி',
})

PAGE_TERMS = frozenset({
    # English
    'url', 'link', 'http', 'https', 'page', 'webpage', 'article',
    'site', 'website', 'documentation', 'read',
    # Indic
    'वेबपेज', 'వెబ్‌పేజీ', 'பக்கம்',
})

MULTISTEP_CONJUNCTIONS = frozenset({
    'step', 'steps', 'guide', 'how to',
    'and then', 'and put', 'and move', 'and save',
    'aur fir', 'aur phir', 'mariyu', 'aprom',
    'మరియు', 'और फिर', 'பிறகு'
})

INDIC_SCRIPT_PATTERN = re.compile(r'[\u0900-\u097F\u0B80-\u0BFF\u0C00-\u0C7F]')
URL_PATTERN = re.compile(r'https?://\S+|www\.\S+')


class ExperimentalV7Adapter(ExperimentalV5Adapter):
    """v7 Adapter with Unicode-safe parsing, symmetrized Indic lexicons, and contract resolution."""

    def __init__(self, artifact_path: str | Path, encoder: Any) -> None:
        super().__init__(artifact_path, encoder)
        self._adapter_id = "jarvis-compact-bert-v7-indic-ranking"
        self.contract_resolver = SlotContractResolver()
        self.last_contract_metrics: dict[str, Any] = {}

    def predict(self, request: DecisionRequest) -> Mapping[str, Any]:
        """Runs calibrated inference matching v5's structure with Unicode Indic fixes."""
        started = time.perf_counter_ns()
        before_calls = self.encoder.forward_calls
        texts = [serialize_request(request)] + [
            serialize_capability(cap.name, cap.description, cap.required_arguments)
            for cap in request.capabilities
        ]
        vectors = self.encoder.encode(texts)
        if self.encoder.forward_calls - before_calls != 1:
            raise RuntimeError("Decision inference must use exactly one encoder forward pass")
        self.inference_count += 1
        self.last_diagnostics = {}
        self.supports = []

        try:
            request_vector = vectors[0:1]
            path_model = self.artifact['execution_path']
            path = str(path_model.predict(request_vector)[0])
            path_probabilities = path_model.predict_proba(request_vector)[0]
            confidence = float(np.max(path_probabilities))

            mode_threshold = self.artifact.get('mode_threshold', 0.5)
            modes = [
                mode
                for mode, model in self.artifact['request_modes'].items()
                if _positive_probability(model, request_vector) >= mode_threshold
            ]
            if not modes:
                modes = [self.artifact.get('path_default_mode', {}).get(path, 'conversational')]

            clar_prob = _positive_probability(self.artifact['clarification'], request_vector)
            esc_prob = _positive_probability(self.artifact['escalation'], request_vector)
            needs_clarification = path == 'clarify' or clar_prob >= 0.50
            needs_escalation = path == 'escalate' or esc_prob >= 0.50

            if needs_clarification and needs_escalation:
                if clar_prob >= esc_prob:
                    needs_escalation = False
                    path = 'clarify'
                else:
                    needs_clarification = False
                    path = 'escalate'
            elif needs_clarification:
                path = 'clarify'
            elif needs_escalation:
                path = 'escalate'

            # 1. Multilingual capability selection
            selected_capability, capability_score = self._select_capability(
                request, request_vector[0], vectors[1:]
            )
            if path not in ACTION_PATHS:
                selected_capability = None

            raw = {
                'intent': 'unknown',
                'execution_path': path,
                'request_modes': sorted(set(modes)),
                'selected_capability': selected_capability,
                'needs_clarification': needs_clarification,
                'needs_escalation': needs_escalation,
                'missing_information': [],
                'context_references': self._predict_references(request, request_vector),
                'confidence': confidence,
                'evidence': [{'kind': 'input_span', 'reference': request.user_text[:160]}],
            }

            # 2. Extract ranker diagnostics
            best_cap = getattr(self, 'last_selected_capability', None)
            best_score = getattr(self, 'last_capability_score', 0.0)
            is_indic = bool(INDIC_SCRIPT_PATTERN.search(request.user_text))
            reconcile_threshold = 0.58 if is_indic else 0.70

            # 3. Principled Reconciliation with Unicode-safe tokenization
            if (raw['execution_path'] in ACTION_PATHS or raw['execution_path'] == 'clarify') and raw['selected_capability'] is None:
                # Unicode-safe tokens preserve Indic characters, matras, and viramas
                tokens = set(UNICODE_WORD_PATTERN.findall(request.user_text.lower()))
                has_pronoun = bool(tokens & INDIC_PRONOUN_TOKENS)
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
            raw['intent'] = _compose_intent(raw['execution_path'], raw['selected_capability'])

            # 4. Canonical Slot Contract Resolution
            resolved = self.contract_resolver.resolve_decision(raw, request)
            validated_decision = ClassifierDecision.from_mapping(resolved)

            total_ms = (time.perf_counter_ns() - started) / 1_000_000.0
            self.last_inference_metrics = {
                'total_ms': total_ms,
                'capability_score': capability_score,
            }
            return _decision_to_mapping(validated_decision)

        except Exception as error:
            self.last_diagnostics['error'] = str(error)
            return safe_fallback('invalid_or_unrepresentable_output:' + str(error))

    def _select_capability(
        self,
        request: DecisionRequest,
        request_vector: np.ndarray,
        capability_vectors: np.ndarray,
    ) -> tuple[str | None, float]:
        """Symmetrized capability ranker with Unicode-safe tokens."""
        if not request.capabilities:
            self.last_selected_capability = None
            self.last_capability_score = 0.0
            return None, 0.0

        # Unicode-safe tokens
        q_tokens = set(UNICODE_WORD_PATTERN.findall(request.user_text.lower()))
        is_indic = bool(INDIC_SCRIPT_PATTERN.search(request.user_text))
        has_url = bool(URL_PATTERN.search(request.user_text))

        # Dense similarity
        q_norm = request_vector / (np.linalg.norm(request_vector) + 1e-9)
        cap_norms = capability_vectors / (np.linalg.norm(capability_vectors, axis=1, keepdims=True) + 1e-9)
        dense_sims = np.dot(cap_norms, q_norm.T).flatten()

        scores = []
        token_overlap_counts = []

        for i, cap in enumerate(request.capabilities):
            base_sim = float(dense_sims[i])
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

            # 2. Entity concept affinity bonus (symmetrized across English, Hindi, Telugu, Tamil)
            if cap_name in ('open_folder', 'create_folder', 'list_directory'):
                if q_tokens & CONTAINER_TERMS:
                    score += 0.08
                if q_tokens & FILE_TERMS:
                    score -= 0.06
                if q_tokens & SEARCH_TERMS:
                    score -= 0.06
            elif cap_name in ('find_file', 'read_file', 'write_file', 'delete_file'):
                if q_tokens & FILE_TERMS:
                    score += 0.08
                if (q_tokens & SEARCH_TERMS) and (q_tokens & FILE_TERMS):
                    score += 0.04
                if q_tokens & CONTAINER_TERMS:
                    score -= 0.05
            elif cap_name in ('google_search', 'search_web'):
                if q_tokens & SEARCH_TERMS:
                    score += 0.10
                if q_tokens & CONTAINER_TERMS:
                    score -= 0.05
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

        # 5. Ambiguity rejection
        if len(scores) >= 2:
            second_idx = int(sorted_indices[1])
            second_score = float(scores[second_idx])
            margin = best_score - second_score
            ambiguity_margin = 0.03 if is_indic else 0.06

            if margin < ambiguity_margin and not has_domain_token:
                self.last_selected_capability = None
                self.last_capability_score = 0.30
                return None, 0.30

        # 6. Minimum threshold
        min_threshold = 0.40 if (has_domain_token or is_indic) else 0.60
        if best_score < min_threshold:
            self.last_selected_capability = None
            self.last_capability_score = best_score
            return None, best_score

        self.last_selected_capability = best_cap.name
        self.last_capability_score = best_score
        return best_cap.name, best_score
