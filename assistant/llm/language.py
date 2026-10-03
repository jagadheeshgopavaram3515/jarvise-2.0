"""Local response-language guidance; no provider calls or conversation state."""
import re

_TELUGU = re.compile(r"[\u0c00-\u0c7f]")
_ROMAN_TELUGU = re.compile(
    r"\b(?:nenu|neenu|neene|nen|nuvvu|nuvve|ninnu|naaku|naku|neeku|niku|"
    r"manam|naatho|natho|neetho|cheppu|cheppandi|cheppamani|"
    r"chepthunnanu|cheyalo|cheyyi|chesthunnav|chestunnav|"
    r"matladu|matladedi|matladaadu|matladuthunnanu|matladuthunnavu|"
    r"matladukundam|vinpistunda|vinipisthundha|vinipisthundi|"
    r"adugutunnanu|aduguthunnanu|anundhi|anipisthundi)\b", re.I)
_ENGLISH_REQUEST = re.compile(
    r"\b(?:speak|talk|reply|respond|answer)\s+(?:to me\s+)?in\s+english\b", re.I)
_TELUGU_REQUEST = re.compile(
    r"\b(?:speak|talk|reply|respond|answer)\s+(?:to me\s+)?in\s+telugu\b|"
    r"\btelugu\s*(?:lo|lone)\s+matladu\b", re.I)


def response_language_instruction(user_text: str) -> str:
    """Prefer an explicit language request, then current-turn Telugu evidence."""
    english = list(_ENGLISH_REQUEST.finditer(user_text))
    telugu = list(_TELUGU_REQUEST.finditer(user_text))
    if english or telugu:
        use_telugu = bool(telugu) and (not english or telugu[-1].start() > english[-1].start())
    else:
        use_telugu = bool(_TELUGU.search(user_text) or _ROMAN_TELUGU.search(user_text))
    if use_telugu:
        return (
            "Response language for THIS turn: Telugu. The user may write Telugu in "
            "Latin letters; that is still Telugu, not English or Hindi. Write every "
            "spoken sentence in native Telugu script (తెలుగు), never romanized Telugu. "
            "Use Telugu script from the first word, including acknowledgements. "
            "Keep necessary names and technical terms. Do not translate the user's input."
        )
    if english:
        return "Response language for THIS turn: English. Write every spoken sentence in English."
    return (
        "Match the language of the CURRENT user message, not earlier conversation. "
        "For English, reply in English. For Telugu written in Latin letters, reply "
        "in native Telugu script, never romanized Telugu or Hindi. For Hindi, use "
        "Devanagari. An explicit request to answer in English takes precedence."
    )
