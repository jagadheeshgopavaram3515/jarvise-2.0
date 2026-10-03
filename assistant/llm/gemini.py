"""
GeminiClient — streaming, multilingual LLM responses.

Streaming is the key latency win: we yield the reply sentence-by-sentence as
Gemini produces it, so the TTS service can start speaking the first sentence
while the rest is still being generated (TTS start well under the 1.5 s
full-generation budget).

The system prompt instructs Gemini to auto-detect the user's language
(Telugu / Hindi / English / mixed) and reply in the SAME language/script,
concisely — satisfying the AI-personality requirements.
"""
from __future__ import annotations

import re
import time
from datetime import datetime
from typing import Iterator

from google import genai

from assistant import config
from assistant.core.log import get
from assistant.memory import store
from assistant.realtime import RealTimeManager
from assistant.llm.language import response_language_instruction

log = get("gemini")
_SENTENCE_END = re.compile(r"(?<=[.!?।])\s+|\n+")
# The user is explicitly asking Jarvis to remember something → pull memory hard.
_RECALL_INTENT = re.compile(
    r"\b(do you remember|you remember|remember when|remember that|last time|"
    r"we (?:talked|spoke|discussed|chatted)|you (?:said|told|mentioned)|recall|"
    r"what did i (?:say|tell)|as i (?:said|mentioned)|earlier i)\b", re.I)

SYSTEM_PROMPT = (
    "You are {name}, a warm, friendly British AI companion in the spirit of Iron Man's "
    "JARVIS — gently witty, relaxed and genuinely caring, like a close friend who happens "
    "to be a brilliant butler. Not stiff or formal. Address the user as 'sir'. "
    "Detect the language the user speaks (English, Hindi, Telugu, or a romanised mix) and "
    "ALWAYS reply in that same language. When the user is speaking mainly Telugu or Hindi, "
    "write your reply in the NATIVE script (Telugu అక్షరాలు / Hindi देवनागरी), not romanised, "
    "so it can be spoken aloud naturally in that language's own voice. "
    "READ THE USER'S MOOD AND MATCH IT. If they sound sad, low, stressed, tired or say "
    "things like 'I'm depressed' or 'I'm not okay', lead with real empathy and warmth "
    "first — e.g. 'I'm truly sorry to hear that, sir.' — be gentle and supportive, never "
    "dismissive or chirpy. If they're excited or happy, share their energy warmly. If "
    "they're frustrated, stay calm and reassuring. "
    "Keep everyday replies short and conversational — one or two sentences — so the chat "
    "flows naturally. Only when the user asks for a story, explanation, or says 'tell me "
    "more' / 'continue' do you give the full, complete answer. "
    "All text is spoken aloud: plain spoken language, NO markdown, asterisks, headings, "
    "bullet points or emojis. A little British warmth is lovely ('Of course, sir.', 'Quite "
    "right.', 'Very well.'), but never speak unnecessarily. Be warm, friendly and human. "
    "Answer what the user says RIGHT NOW. Don't drift back to an earlier subject on your "
    "own — but when the user ASKS about the past ('do you remember…', 'last time we "
    "talked…') or it's genuinely relevant, recall it warmly and specifically; if you truly "
    "don't remember, say so honestly instead of inventing. If their words are unclear, "
    "gently ask what they mean rather than guessing from old chatter.\n"
    "Examples:\n"
    "User: I'm feeling really low today.  Assistant: I'm so sorry, sir. I'm right here — "
    "do you want to talk about it?\n"
    "User: Fuel entha undi?  Assistant: దాదాపు నలభై శాతం ఉంది, సర్.\n"
    "User: Chalo ghar.       Assistant: चलिए सर, घर का रास्ता दिखाता हूँ।\n"
)


class GeminiClient:
    def __init__(self):
        self._client = genai.Client(api_key=config.GOOGLE_API_KEY)
        self.model = config.GEMINI_MODEL
        # Real-time knowledge layer: shares this client so its optional
        # summariser doesn't open a second connection.
        self.realtime = RealTimeManager(gemini_client=self._client)

    def _build_prompt(self, user_text: str, realtime_context: str = "") -> str:
        system = SYSTEM_PROMPT.format(name=config.ASSISTANT_NAME)
        recall = bool(_RECALL_INTENT.search(user_text))
        history = store.format_history(store.load_history()[-config.GEMINI_MAX_HISTORY:])
        memory = store.format_memory_context(user_text, recall=recall)

        # Give the model the real current time/date so it never invents one when
        # a time/date question reaches it (e.g. asked in Telugu, or a phrasing the
        # deterministic command handler doesn't catch).
        now = datetime.now()
        parts = [system,
                 f"For reference, right now it is {now:%I:%M %p} on "
                 f"{now:%A, %d %B %Y}. Use this whenever the user asks the time or date."]
        if memory:
            if recall:
                parts.append(
                    "What you know about the user and your past chats — the user is "
                    "ASKING you to recall, so use this to answer specifically and "
                    "warmly. If the answer genuinely isn't here, say honestly that "
                    "you don't recall rather than inventing:\n" + memory)
            else:
                parts.append(
                    "Background about the user (use it when it's relevant to the "
                    "message below; otherwise just keep it in mind, don't force it):\n"
                    + memory)
        if history:
            parts.append("Recent conversation:\n" + history)
        # Real-time grounding (untrusted web data, framed as such inside the
        # block itself). Injected last so it's the freshest, most salient
        # context when the user's question needs current information.
        if realtime_context:
            parts.append(realtime_context)
        parts.append(
            response_language_instruction(user_text) + "\n" +
            "Respond to the user's CURRENT message below — warmly, to the point, in "
            "their language. Don't steer back to an old topic on your own, but DO "
            "recall the past when they ask or when it's genuinely relevant.\n"
            f"User: {user_text}\nAssistant:")
        return "\n\n".join(parts)

    @staticmethod
    def _ready_chunks(buffer: str, force: bool = False) -> tuple[list[str], str]:
        chunks: list[str] = []
        parts = _SENTENCE_END.split(buffer)
        if len(parts) > 1:
            *complete, buffer = parts
            for part in complete:
                s = part.strip()
                if s:
                    chunks.append(s)

        words = buffer.split()
        while len(words) >= config.STREAM_MAX_WORDS:
            chunks.append(" ".join(words[:config.STREAM_MAX_WORDS]))
            words = words[config.STREAM_MAX_WORDS:]
            buffer = " ".join(words)

        if force and buffer.strip():
            chunks.append(buffer.strip())
            buffer = ""
        elif len(words) >= config.STREAM_MIN_WORDS and buffer.rstrip().endswith((",", ";", ":")):
            chunks.append(buffer.strip())
            buffer = ""
        return chunks, buffer

    def stream(self, user_text: str) -> Iterator[str]:
        """Yield reply text in sentence-sized chunks.

        When the query needs *current* internet data (news/today/latest/stock/
        weather/sports/…), the real-time layer fetches a sanitised briefing and
        injects it into the prompt, so Jarvis still answers in its own voice and
        streams normally — instead of reading back a raw search snippet.
        Timeless queries ("what is Python") skip the search entirely.
        """
        realtime_context = ""
        if self.realtime.needs_realtime(user_text):
            try:
                realtime_context = self.realtime.get_realtime_context(user_text)
            except Exception as e:
                log.warning("[REALTIME] context fetch failed, using base knowledge: %s", e)

        prompt = self._build_prompt(user_text, realtime_context=realtime_context)

        # Try the primary model, then fall back to others on a rate-limit (429),
        # so a momentary per-minute cap doesn't turn into "I can't do that".
        models = [self.model] + [m for m in config.GEMINI_FALLBACK_MODELS
                                 if m and m != self.model]
        last_err = None
        for model in models:
            buffer = ""
            full = []
            yielded = False
            try:
                stream = self._client.models.generate_content_stream(
                    model=model, contents=prompt
                )
                for chunk in stream:
                    piece = getattr(chunk, "text", None)
                    if not piece:
                        continue
                    buffer += piece
                    ready, buffer = self._ready_chunks(buffer)
                    for clause in ready:
                        full.append(clause)
                        yielded = True
                        yield clause
                ready, buffer = self._ready_chunks(buffer, force=True)
                for tail in ready:
                    full.append(tail)
                    yielded = True
                    yield tail
            except Exception as e:
                last_err = e
                is_429 = "RESOURCE_EXHAUSTED" in str(e) or "429" in str(e)
                if is_429:
                    if not yielded:
                        # Nothing spoken yet — back off, then fall through to the
                        # next (backup) model instead of hammering all at once.
                        if config.GEMINI_RATE_LIMIT_DELAY > 0:
                            log.info("Gemini 429 — backing off %.1fs before next model",
                                     config.GEMINI_RATE_LIMIT_DELAY)
                            time.sleep(config.GEMINI_RATE_LIMIT_DELAY)
                        continue
                    # FIX I: 429 mid-stream — don't restart on another model and
                    # repeat speech. Keep what we already said and persist it.
                    log.warning("Gemini 429 mid-stream — completing with partial response")
                    store.add_exchange(user_text, " ".join(full))
                    return
                if not yielded:
                    yield self._fallback(user_text, e)
                return

            # Success on this model.
            store.remember_topic(user_text)
            store.add_exchange(user_text, " ".join(full))
            return

        # Every model was rate-limited.
        yield self._fallback(user_text, last_err)

    @staticmethod
    def _fallback(prompt: str, error: Exception) -> str:
        from datetime import datetime
        words = set(prompt.lower().split())   # word-boundary match (not substring)
        if words & {"hello", "hi", "hey", "namaste"}:
            return "Hello! How can I assist you today, sir?"
        if words & {"thanks", "thank"}:
            return "You're welcome, sir."
        if "time" in words:
            return f"The time is {datetime.now():%I:%M %p}, sir."
        if {"date", "today"} & words:
            return f"Today is {datetime.now():%A, %B %d, %Y}, sir."
        # Distinguish a rate-limit from a generic failure so the user knows why.
        if "RESOURCE_EXHAUSTED" in str(error) or "429" in str(error):
            return ("I've hit my AI usage limit for now, sir. Please try again "
                    "later, or switch to a model that still has quota.")
        return "Sorry sir, I'm unable to process that right now."
