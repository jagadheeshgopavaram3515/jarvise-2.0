"""
Central configuration.

All secrets come from environment variables (loaded from a .env file).
NOTHING sensitive is hard-coded here. See .env.example for the keys.
"""
import os
from dataclasses import dataclass, field
from dotenv import load_dotenv, find_dotenv

# Load the nearest .env walking up the tree (project root or jarvis/ folder).
load_dotenv(find_dotenv(usecwd=True))


def _get(key: str, default: str = "") -> str:
    val = os.getenv(key, default)
    if val:
        val = val.split("#")[0].strip()
    return val if val else default


# ---------------------------------------------------------------- API keys
GOOGLE_API_KEY = _get("GOOGLE_API_KEY")
SERPAPI_KEY = _get("SERPAPI_KEY")
CALENDARIFIC_API_KEY = _get("CALENDARIFIC_API_KEY")

# ---------------------------------------------------------------- LLM
GEMINI_MODEL = _get("GEMINI_MODEL", "gemini-2.5-flash")
# Comma-separated models tried (in order) when the primary hits a 429 rate-limit.
GEMINI_FALLBACK_MODELS = [m.strip() for m in _get(
    "GEMINI_FALLBACK_MODELS", "gemini-2.5-flash-lite,gemini-flash-lite-latest"
).split(",") if m.strip()]
ASSISTANT_NAME = _get("ASSISTANT_NAME", "Jarvis")
# Seconds to back off after a 429 before falling through to the next model.
GEMINI_RATE_LIMIT_DELAY = float(_get("GEMINI_RATE_LIMIT_DELAY", "0.5"))
# Max number of recent history messages included in the Gemini prompt.
GEMINI_MAX_HISTORY = int(_get("GEMINI_MAX_HISTORY", "5"))
# Minimum seconds between LLM dispatches — throttles Gemini to dodge 429s.
GEMINI_MIN_DISPATCH_INTERVAL = float(_get("GEMINI_MIN_DISPATCH_INTERVAL", "3.0"))

# Provider selection: "gemini" (default) or "ollama"
LLM_PROVIDER = _get("LLM_PROVIDER", "gemini").lower()

# Local Ollama / Qwen model configuration
OLLAMA_BASE_URL = _get("OLLAMA_BASE_URL", "http://localhost:11434").rstrip("/")
OLLAMA_MODEL = _get("OLLAMA_MODEL", "qwen2.5-coder:3b")
OLLAMA_TIMEOUT_S = float(_get("OLLAMA_TIMEOUT_S", "45.0"))
OLLAMA_KEEP_ALIVE = _get("OLLAMA_KEEP_ALIVE", "10m")

# LangGraph Orchestration & Fast Classifier
LLM_ROUTING_ENABLED = _get("LLM_ROUTING_ENABLED", "true").lower() == "true"
CLASSIFIER_TIMEOUT_S = float(_get("CLASSIFIER_TIMEOUT_S", "5.0"))
CLASSIFIER_MODEL = _get("CLASSIFIER_MODEL", OLLAMA_MODEL)

# ---------------------------------------------------------------- STT
# faster-whisper model: tiny / base / small / medium / large-v3
STT_MODEL_SIZE = _get("STT_MODEL_SIZE", "small")
STT_FAST_MODEL_SIZE = _get("STT_FAST_MODEL_SIZE", "tiny")
# device: "cuda" (GTX 1650 ok with int8) or "cpu"
STT_DEVICE = _get("STT_DEVICE", "auto")
STT_COMPUTE_TYPE = _get("STT_COMPUTE_TYPE", "")  # "" = auto pick
STT_GPU_COMPUTE_TYPE = _get("STT_GPU_COMPUTE_TYPE", "float16")
STT_CPU_COMPUTE_TYPE = _get("STT_CPU_COMPUTE_TYPE", "int8")
STT_SAMPLE_RATE = 16000
STT_BEAM_SIZE = int(_get("STT_BEAM_SIZE", "1"))  # 1 = greedy = fastest
STT_CPU_THREADS = int(_get("STT_CPU_THREADS", "0"))  # 0 = CTranslate2 auto
STT_NUM_WORKERS = int(_get("STT_NUM_WORKERS", "1"))  # parallel CTranslate2 workers
# VAD pre-filter (skip whisper on silence) + language-detection cache.
STT_VAD_PREFILTER = _get("STT_VAD_PREFILTER", "true").lower() == "true"
STT_PREFILTER_RMS = float(_get("STT_PREFILTER_RMS", "0.005"))  # normalized-audio RMS gate
STT_LANG_CACHE_REUSE = int(_get("STT_LANG_CACHE_REUSE", "8"))
# Cache a detected language once it's at least this confident (lower than the old
# 0.75 — on CPU/short audio, clear English only scores ~0.55-0.70, so 0.75 never
# cached and every final re-paid the ~1.7s detection cost).
STT_LANG_CACHE_MIN_PROB = float(_get("STT_LANG_CACHE_MIN_PROB", "0.50"))
# Drop a FINAL that has waited longer than this in the queue (backlog) — better
# to skip a stale utterance than answer it 25s late.
STT_MAX_STALE_S = float(_get("STT_MAX_STALE_S", "8.0"))
# Reject noise hallucinations: drop finals below this language confidence, and
# unexpected languages unless very confident. User speaks te/hi/en (+ta).
STT_MIN_LANG_PROB = float(_get("STT_MIN_LANG_PROB", "0.25"))
STT_EXPECTED_LANGS = {s.strip() for s in _get("STT_EXPECTED_LANGS", "te,hi,en,ta").split(",") if s.strip()}
# Force a language ("en"/"hi"/"te") or set "auto" for multilingual detect.
_STT_LANGUAGE_RAW = _get("STT_LANGUAGE", "en").lower()
STT_LANGUAGE = None if _STT_LANGUAGE_RAW in {"", "auto"} else _STT_LANGUAGE_RAW
STT_LANGUAGE_LABEL = "auto" if STT_LANGUAGE is None else STT_LANGUAGE
# When auto-detecting, skip the costly per-partial language detection: partials
# decode as English; only final utterances run real detection.
STT_LANGUAGE_DETECT_ONLY_ON_FINAL = _get(
    "STT_LANGUAGE_DETECT_ONLY_ON_FINAL", "true").lower() == "true"
STT_PARTIAL_LANGUAGE = _get("STT_PARTIAL_LANGUAGE", "en")
STT_WARMUP_SECONDS = float(_get("STT_WARMUP_SECONDS", "0.35"))
STT_STARTUP_WAIT_SECONDS = float(_get("STT_STARTUP_WAIT_SECONDS", "90"))
STT_ENABLE_PARTIALS = _get("STT_ENABLE_PARTIALS", "true").lower() == "true"
STT_PARTIAL_INTERVAL_MS = int(_get("STT_PARTIAL_INTERVAL_MS", "500"))
STT_PARTIAL_MIN_AUDIO_MS = int(_get("STT_PARTIAL_MIN_AUDIO_MS", "700"))
STT_PROFILE_WINDOW = int(_get("STT_PROFILE_WINDOW", "10"))

# ---------------------------------------------------------------- VAD
VAD_AGGRESSIVENESS = int(_get("VAD_AGGRESSIVENESS", "2"))   # 0..3, higher = stricter
VAD_FRAME_MS = 30                                           # 10/20/30 supported
# Adaptive endpointing: a short utterance ends after VAD_SILENCE_MS of silence
# (snappy for quick questions); the longer you speak, the more silence is
# tolerated before ending (so mid-sentence thinking pauses don't cut you off),
# up to VAD_SILENCE_MAX_MS. This is what lets long sentences be heard fully.
VAD_SILENCE_MS = int(_get("VAD_SILENCE_MS", "500"))         # base end-of-turn gap
VAD_SILENCE_MAX_MS = int(_get("VAD_SILENCE_MAX_MS", "1300"))  # cap for long speech
VAD_SILENCE_GROW_MS_PER_SEC = int(_get("VAD_SILENCE_GROW_MS_PER_SEC", "180"))
VAD_MIN_UTTERANCE_MS = int(_get("VAD_MIN_UTTERANCE_MS", "250"))
VAD_MAX_UTTERANCE_MS = int(_get("VAD_MAX_UTTERANCE_MS", "30000"))  # allow long turns
SEMANTIC_ENDPOINTING = _get("SEMANTIC_ENDPOINTING", "true").lower() == "true"
SEMANTIC_ENDPOINT_MIN_MS = int(_get("SEMANTIC_ENDPOINT_MIN_MS", "900"))
SEMANTIC_ENDPOINT_SILENCE_MS = int(_get("SEMANTIC_ENDPOINT_SILENCE_MS", "240"))

# ---------------------------------------------------------------- Hybrid STT (Phase 1)
# Re-transcribe CONVERSATIONAL finals with Gemini audio for accuracy; commands
# stay on Whisper; any failure falls back to Whisper. OFF by default — each
# conversational turn costs one extra Gemini call (mind the rate limits).
HYBRID_STT = _get("HYBRID_STT", "false").lower() == "true"
HYBRID_STT_MODEL = _get("HYBRID_STT_MODEL", "gemini-2.5-flash-lite")
HYBRID_STT_TIMEOUT_S = float(_get("HYBRID_STT_TIMEOUT_S", "3.0"))

# ---------------------------------------------------------------- TTS
# Backend: "edge" (Edge-TTS, default) or "piper" (local, calm British JARVIS).
TTS_BACKEND = _get("TTS_BACKEND", "edge")
# Path to a Piper .onnx voice (e.g. en_GB-alan-medium.onnx); its .json sits beside it.
PIPER_MODEL_PATH = _get("PIPER_MODEL_PATH", "")
# Per-language Edge-TTS voices. Romanised/code-switched text falls back to EN.
VOICE_MAP = {
    "en": _get("TTS_VOICE_EN", "en-GB-ThomasNeural"),   # softer British (Edge fallback)
    "hi": _get("TTS_VOICE_HI", "hi-IN-MadhurNeural"),
    "te": _get("TTS_VOICE_TE", "te-IN-MohanNeural"),
    "ta": _get("TTS_VOICE_TA", "ta-IN-PallaviNeural"),  # Tamil (user speaks ta too)
}
DEFAULT_VOICE = VOICE_MAP["en"]
# Optional per-language tone (Edge rate/pitch). Empty = use the default
# emotion-based style, i.e. no special per-language override.
TTS_TE_RATE = _get("TTS_TE_RATE", "")
TTS_TE_PITCH = _get("TTS_TE_PITCH", "")
TTS_HI_RATE = _get("TTS_HI_RATE", "")
TTS_HI_PITCH = _get("TTS_HI_PITCH", "")

# ---------------------------------------------------------------- Barge-in
# Words that always interrupt speech, regardless of echo.
INTERRUPT_WORDS = {"stop", "cancel", "wait", "enough", "quiet", "halt", "shush",
                   "ruko", "aap ruk", "aagu", "aapu", ASSISTANT_NAME.lower()}
# If True, ONLY interrupt-words can barge-in while speaking (robust against
# the mic picking up the assistant's own voice / echo). If False, any
# confident user utterance barges in. Set True if you hear self-triggering.
BARGE_IN_KEYWORDS_ONLY = _get("BARGE_IN_KEYWORDS_ONLY", "true").lower() == "true"

# Full-duplex (listen while speaking). Best with headphones or hardware echo
# cancellation; ENABLE_AEC adds a software guard for normal speakers.
FULL_DUPLEX = _get("FULL_DUPLEX", "false").lower() == "true"
# Optional acoustic echo-control hook. If a WebRTC/Speex APM binding is not
# installed, Jarvis falls back to a conservative echo suppressor while speaking.
ENABLE_AEC = _get("ENABLE_AEC", "true").lower() == "true"
# How long to keep ignoring the mic after speech ends (echo tail), milliseconds.
SPEECH_TAIL_GUARD_MS = int(_get("SPEECH_TAIL_GUARD_MS", "400"))

# ---------------------------------------------------------------- Files
DATA_DIR = _get("DATA_DIR", os.path.dirname(os.path.dirname(os.path.abspath(__file__))))
NAME_FILE = os.path.join(DATA_DIR, "user_name.txt")
MEMORY_FILE = os.path.join(DATA_DIR, "memory.json")
CONVERSATION_HISTORY_FILE = os.path.join(DATA_DIR, "conversation_history.json")
HISTORY_TURNS = int(_get("HISTORY_TURNS", "8"))
SESSION_SUMMARY_FILE = os.path.join(DATA_DIR, "session_summary.json")
VECTOR_MEMORY_FILE = os.path.join(DATA_DIR, "vector_memory.json")
MOOD_FILE = os.path.join(DATA_DIR, "mood_state.json")  # Phase 7: emotional continuity
# performance_log.json is pure instrumentation — cap it so it never bloats memory/disk.
# ~16 serialized lines per record, so 300 records ≈ 4.8k lines, safely under 5k.
PERF_LOG_MAX_LINES = int(_get("PERF_LOG_MAX_LINES", "5000"))
PERF_LOG_LINES_PER_RECORD = int(_get("PERF_LOG_LINES_PER_RECORD", "16"))
MEMORY_TOP_K = int(_get("MEMORY_TOP_K", "5"))
# Minimum cosine similarity for a retrieved memory to be injected. Balanced so
# relevant memories surface naturally but an off-topic past chat doesn't hijack a
# reply. When the user explicitly asks to recall, the lower RECALL threshold is
# used so Jarvis can actually remember.
MEMORY_MIN_SIMILARITY = float(_get("MEMORY_MIN_SIMILARITY", "0.18"))
MEMORY_RECALL_MIN_SIMILARITY = float(_get("MEMORY_RECALL_MIN_SIMILARITY", "0.05"))
SUMMARY_EVERY_TURNS = int(_get("SUMMARY_EVERY_TURNS", "10"))

# ------------------------------------------------- Long-term memory v2 (engine)
# All new memory state lives under this folder (working/profile/projects/
# episodes/emotions/relationship JSON + the semantic index .npy/.meta.json).
MEMORY_DIR = _get("MEMORY_DIR", os.path.join(DATA_DIR, "memory_store"))
# Embedding backend: auto | sentence-transformers | gemini | hash.
#   auto  → sentence-transformers (local, free) → gemini (reuses key) → hash.
MEMORY_EMBED_BACKEND = _get("MEMORY_EMBED_BACKEND", "auto")
MEMORY_EMBED_MODEL_ST = _get("MEMORY_EMBED_MODEL_ST", "all-MiniLM-L6-v2")
MEMORY_EMBED_MODEL_GEMINI = _get("MEMORY_EMBED_MODEL_GEMINI", "text-embedding-004")
# Layer 1: working-memory turns kept (always injected). Tip: also raise
# GEMINI_MAX_HISTORY in .env to widen the verbatim recent-turn window.
MEMORY_WORKING_TURNS = int(_get("MEMORY_WORKING_TURNS", "24"))
# Consolidation (LLM extraction) cadence and importance gate.
MEMORY_CONSOLIDATE_EVERY = int(_get("MEMORY_CONSOLIDATE_EVERY", "20"))
MEMORY_IMPORTANCE_THRESHOLD = int(_get("MEMORY_IMPORTANCE_THRESHOLD", "6"))
MEMORY_RETRIEVAL_TOP_K = int(_get("MEMORY_RETRIEVAL_TOP_K", "6"))
MEMORY_CONTEXT_MAX_CHARS = int(_get("MEMORY_CONTEXT_MAX_CHARS", "2600"))
MEMORY_VECTOR_MAX_ENTRIES = int(_get("MEMORY_VECTOR_MAX_ENTRIES", "5000"))
# LLM-based extraction/consolidation (small/fast model — never the turn model).
MEMORY_EXTRACTION = _get("MEMORY_EXTRACTION", "true").lower() == "true"
MEMORY_EXTRACTION_MODEL = _get("MEMORY_EXTRACTION_MODEL", "gemini-2.5-flash-lite")

# ---------------------------------------------------------------- Natural voice
ENABLE_FILLERS = _get("ENABLE_FILLERS", "false").lower() == "true"  # legacy emotion fillers
# Phase 3: occasional, rate-limited human lead-ins on Jarvis's own replies.
ENABLE_HUMANIZER = _get("ENABLE_HUMANIZER", "true").lower() == "true"
STREAM_MIN_WORDS = int(_get("STREAM_MIN_WORDS", "7"))
STREAM_MAX_WORDS = int(_get("STREAM_MAX_WORDS", "18"))

# ---------------------------------------------------------------- Backchannel
# Dynamic, LLM-generated "listening" sounds ("hmm"/"yeah"/"I see") played while
# the USER is still speaking. A separate fast model — never the main turn LLM.
ENABLE_BACKCHANNEL = _get("ENABLE_BACKCHANNEL", "true").lower() == "true"
BACKCHANNEL_MODEL = _get("BACKCHANNEL_MODEL", "gemini-2.5-flash-lite")
BACKCHANNEL_VOLUME = float(_get("BACKCHANNEL_VOLUME", "0.7"))     # 0..1 softer than speech
BACKCHANNEL_MIN_GAP_S = float(_get("BACKCHANNEL_MIN_GAP_S", "4.0"))
BACKCHANNEL_MAX_PER_TURN = int(_get("BACKCHANNEL_MAX_PER_TURN", "3"))
BACKCHANNEL_MIN_WORDS = int(_get("BACKCHANNEL_MIN_WORDS", "8"))
BACKCHANNEL_PAUSE_MS = int(_get("BACKCHANNEL_PAUSE_MS", "800"))   # mid-speech pause trigger
BACKCHANNEL_LLM_TIMEOUT_S = float(_get("BACKCHANNEL_LLM_TIMEOUT_S", "0.5"))
# Phase 4: after the user has spoken this long in one utterance, drop a brief
# "I'm listening, sir" / "go on" acknowledgment.
BACKCHANNEL_LONG_SPEECH_MS = int(_get("BACKCHANNEL_LONG_SPEECH_MS", "10000"))

# ----------------------------------------------- Real-time knowledge layer
# Additive layer: when a query needs *current* internet data (news/today/latest/
# stock/weather/sports/…), Jarvis searches the web, extracts the top articles and
# injects a sanitised briefing into the Gemini prompt so it answers in its own
# voice. Definitional/timeless queries ("what is Python") never search.
REALTIME_ENABLED = _get("REALTIME_ENABLED", "true").lower() == "true"
# Search backend: auto (SerpAPI → DuckDuckGo fallback) | serpapi | duckduckgo.
REALTIME_PROVIDER = _get("REALTIME_PROVIDER", "auto").lower()
REALTIME_MAX_RESULTS = int(_get("REALTIME_MAX_RESULTS", "5"))
REALTIME_GL = _get("REALTIME_GL", "IN")   # SerpAPI geo (country)
REALTIME_HL = _get("REALTIME_HL", "en")   # SerpAPI host language
# Article extraction (readability-lxml / trafilatura; bs4 helper only).
REALTIME_EXTRACT = _get("REALTIME_EXTRACT", "true").lower() == "true"
REALTIME_MAX_EXTRACT_ARTICLES = int(_get("REALTIME_MAX_EXTRACT_ARTICLES", "3"))
REALTIME_FETCH_TIMEOUT_S = float(_get("REALTIME_FETCH_TIMEOUT_S", "5.0"))
REALTIME_EXTRACT_MAX_CHARS = int(_get("REALTIME_EXTRACT_MAX_CHARS", "1200"))
REALTIME_CONTEXT_MAX_CHARS = int(_get("REALTIME_CONTEXT_MAX_CHARS", "3500"))
# Optional pre-summarisation pass on a small/fast model (extra LLM call — OFF by
# default to protect rate limits; Gemini already summarises the injected context).
REALTIME_SUMMARIZE = _get("REALTIME_SUMMARIZE", "false").lower() == "true"
REALTIME_SUMMARIZER_MODEL = _get("REALTIME_SUMMARIZER_MODEL", "gemini-2.5-flash-lite")
REALTIME_SUMMARIZER_TIMEOUT_S = float(_get("REALTIME_SUMMARIZER_TIMEOUT_S", "6.0"))
# Disk cache of fetched context, keyed by query hash. TTL per query class.
REALTIME_CACHE_DIR = _get("REALTIME_CACHE_DIR", os.path.join(DATA_DIR, "cache"))
REALTIME_TTL_NEWS_S = int(_get("REALTIME_TTL_NEWS_S", "1800"))       # news   = 30 min
REALTIME_TTL_WEATHER_S = int(_get("REALTIME_TTL_WEATHER_S", "900"))  # weather= 15 min
REALTIME_TTL_GENERAL_S = int(_get("REALTIME_TTL_GENERAL_S", "43200"))  # general= 12 h

# ----------------------------------------------- Desktop Tools Layer (V2)
# Additive: app/file/browser tools dispatched through a single ToolExecutor,
# selected by Gemini tool-calling. Fully backward-compatible — set
# TOOLS_ENABLED=false to disable the whole layer and behave exactly as before.
TOOLS_ENABLED = _get("TOOLS_ENABLED", "true").lower() == "true"
# Per-tool execution timeout (seconds) enforced by the ToolExecutor.
TOOLS_TIMEOUT_S = float(_get("TOOLS_TIMEOUT_S", "25.0"))
# Structured audit log of every tool call (Phase 7).
TOOLS_LOG_FILE = _get("TOOLS_LOG_FILE", os.path.join(DATA_DIR, "logs", "tools.log"))
# Use Gemini to SELECT a tool + fill args (Phase 5). When false, only the fast
# deterministic intent matcher runs (no extra LLM call).
TOOLS_GEMINI_SELECTION = _get("TOOLS_GEMINI_SELECTION", "true").lower() == "true"
# Small/fast model for tool selection — never the main turn model.
TOOLS_SELECTION_MODEL = _get("TOOLS_SELECTION_MODEL", "gemini-2.5-flash-lite")
TOOLS_SELECTION_TIMEOUT_S = float(_get("TOOLS_SELECTION_TIMEOUT_S", "4.0"))
# After a tool runs, ask Gemini to phrase the spoken reply in Jarvis's voice.
# When false (or on any failure) the tool's own message is spoken directly.
TOOLS_NATURAL_RESPONSE = _get("TOOLS_NATURAL_RESPONSE", "true").lower() == "true"
TOOLS_RESPONSE_MODEL = _get("TOOLS_RESPONSE_MODEL", "gemini-2.5-flash-lite")
TOOLS_RESPONSE_TIMEOUT_S = float(_get("TOOLS_RESPONSE_TIMEOUT_S", "3.0"))

# ---- Browser tools (Playwright) ----
TOOLS_BROWSER_HEADLESS = _get("TOOLS_BROWSER_HEADLESS", "true").lower() == "true"
TOOLS_BROWSER_TIMEOUT_S = float(_get("TOOLS_BROWSER_TIMEOUT_S", "20.0"))
TOOLS_BROWSER_MAX_CHARS = int(_get("TOOLS_BROWSER_MAX_CHARS", "1200"))

# ---- File tools (read-only) ----
# Comma-separated roots the read-only file tools may search. Defaults to the
# user's home folder + its common subfolders. Nothing outside these is touched.
_DEFAULT_FILE_ROOTS = ",".join([
    os.path.expanduser("~"),
])
TOOLS_FILE_ROOTS = [r.strip() for r in _get("TOOLS_FILE_ROOTS", _DEFAULT_FILE_ROOTS).split(",")
                    if r.strip()]

# ----------------------------------------------- Reliability / Health (V2)
# Additive resilience layer: a HealthMonitor thread heartbeats every service and
# auto-restarts any that die, so no single exception can take Jarvis down. All
# toggles default to safe values; set HEALTH_MONITOR_ENABLED=false to behave
# exactly as before (services still run, just without supervision).
HEALTH_MONITOR_ENABLED = _get("HEALTH_MONITOR_ENABLED", "true").lower() == "true"
HEALTH_INTERVAL_S = float(_get("HEALTH_INTERVAL_S", "15"))      # heartbeat cadence
HEALTH_AUTO_RECOVERY = _get("HEALTH_AUTO_RECOVERY", "true").lower() == "true"
# Cap automatic restarts per service so a permanently-broken dependency (e.g. no
# mic, no model) can't spin in a tight restart loop. Counter is per service.
HEALTH_MAX_RESTARTS = int(_get("HEALTH_MAX_RESTARTS", "5"))
# Seconds to wait after a restart before counting the service healthy again.
HEALTH_RESTART_BACKOFF_S = float(_get("HEALTH_RESTART_BACKOFF_S", "2.0"))
# Log a [HEALTH] heartbeat EVERY cycle. Default false → the monitor is silent in
# steady state and only logs on a state change or an actual failure (zero log I/O
# while healthy). Set true for verbose per-cycle heartbeats.
HEALTH_LOG_HEARTBEAT = _get("HEALTH_LOG_HEARTBEAT", "false").lower() == "true"

# ---------------------------------------------------------------- Misc
SLEEP_AFTER_FAILURES = int(_get("SLEEP_AFTER_FAILURES", "5"))


@dataclass
class RuntimeFlags:
    """Mutable flags shared across services."""
    demo_mode: bool = field(default=_get("DEMO_MODE", "false").lower() == "true")


FLAGS = RuntimeFlags()
