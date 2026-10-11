"""Lightweight, rule-based phonemic transliterator for Indic scripts (Telugu, Devanagari, Tamil).
Preserves URLs, file paths, numbers, code, and Latin-script entities verbatim.
Pure offline implementation with zero external dependencies.
"""
import re
from typing import NamedTuple

# Unicode block ranges
DEVANAGARI_RANGE = (0x0900, 0x097F)
TELUGU_RANGE = (0x0C00, 0x0C7F)
TAMIL_RANGE = (0x0B80, 0x0BFF)

URL_OR_PATH_REGEX = re.compile(
    r'(https?://\S+|www\.\S+|[a-zA-Z]:\\[\w\\\.\-]+|/[\w/\.\-]+|[\w\-]+\.[a-zA-Z]{2,4})'
)

# Devanagari Phonemic Map
DEV_VOWELS = {
    '\u0905': 'a', '\u0906': 'aa', '\u0907': 'i', '\u0908': 'ee',
    '\u0909': 'u', '\u090a': 'oo', '\u090b': 'ri', '\u090e': 'e',
    '\u090f': 'e', '\u0910': 'ai', '\u0912': 'o', '\u0913': 'o',
    '\u0914': 'au',
}
DEV_MATRAS = {
    '\u093e': 'aa', '\u093f': 'i', '\u0940': 'ee', '\u0941': 'u',
    '\u0942': 'oo', '\u0943': 'ri', '\u0946': 'e', '\u0947': 'e',
    '\u0948': 'ai', '\u094a': 'o', '\u094b': 'o', '\u094c': 'au',
    '\u0945': 'e', '\u0949': 'o',
}
DEV_CONSONANTS = {
    '\u0915': 'k', '\u0916': 'kh', '\u0917': 'g', '\u0918': 'gh', '\u0919': 'ng',
    '\u091a': 'ch', '\u091b': 'chh', '\u091c': 'j', '\u091d': 'jh', '\u091e': 'ny',
    '\u091f': 't', '\u0920': 'th', '\u0921': 'd', '\u0922': 'dh', '\u0923': 'n',
    '\u0924': 't', '\u0925': 'th', '\u0926': 'd', '\u0927': 'dh', '\u0928': 'n',
    '\u092a': 'p', '\u092b': 'ph', '\u092c': 'b', '\u092d': 'bh', '\u092e': 'm',
    '\u092f': 'y', '\u0930': 'r', '\u0931': 'r', '\u0932': 'l', '\u0933': 'l',
    '\u0935': 'v', '\u0936': 'sh', '\u0937': 'sh', '\u0938': 's', '\u0939': 'h',
    # Nukta consonants
    '\u0958': 'q', '\u0959': 'kh', '\u095a': 'gh', '\u095b': 'z',
    '\u095c': 'r', '\u095d': 'rh', '\u095e': 'f', '\u095f': 'y',
}

# Telugu Phonemic Map
TEL_VOWELS = {
    '\u0c05': 'a', '\u0c06': 'aa', '\u0c07': 'i', '\u0c08': 'ee',
    '\u0c09': 'u', '\u0c0a': 'oo', '\u0c0b': 'ru', '\u0c0e': 'e',
    '\u0c0f': 'e', '\u0c10': 'ai', '\u0c12': 'o', '\u0c13': 'o',
    '\u0c14': 'au',
}
TEL_MATRAS = {
    '\u0c3e': 'aa', '\u0c3f': 'i', '\u0c40': 'ee', '\u0c41': 'u',
    '\u0c42': 'oo', '\u0c43': 'ru', '\u0c46': 'e', '\u0c47': 'e',
    '\u0c48': 'ai', '\u0c4a': 'o', '\u0c4b': 'o', '\u0c4c': 'au',
}
TEL_CONSONANTS = {
    '\u0c15': 'k', '\u0c16': 'kh', '\u0c17': 'g', '\u0c18': 'gh', '\u0c19': 'ng',
    '\u0c1a': 'ch', '\u0c1b': 'chh', '\u0c1c': 'j', '\u0c1d': 'jh', '\u0c1e': 'ny',
    '\u0c1f': 't', '\u0c20': 'th', '\u0c21': 'd', '\u0c22': 'dh', '\u0c23': 'n',
    '\u0c24': 't', '\u0c25': 'th', '\u0c26': 'd', '\u0c27': 'dh', '\u0c28': 'n',
    '\u0c2a': 'p', '\u0c2b': 'ph', '\u0c2c': 'b', '\u0c2d': 'bh', '\u0c2e': 'm',
    '\u0c2f': 'y', '\u0c30': 'r', '\u0c31': 'r', '\u0c32': 'l', '\u0c33': 'l',
    '\u0c35': 'v', '\u0c36': 'sh', '\u0c37': 'sh', '\u0c38': 's', '\u0c39': 'h',
}

# Tamil Phonemic Map
TAM_VOWELS = {
    '\u0b85': 'a', '\u0b86': 'aa', '\u0b87': 'i', '\u0b88': 'ee',
    '\u0b89': 'u', '\u0b8a': 'oo', '\u0b8e': 'e', '\u0b8f': 'e',
    '\u0b90': 'ai', '\u0b92': 'o', '\u0b93': 'o', '\u0b94': 'au',
}
TAM_MATRAS = {
    '\u0bbe': 'aa', '\u0bbf': 'i', '\u0bc0': 'ee', '\u0bc1': 'u',
    '\u0bc2': 'oo', '\u0bc6': 'e', '\u0bc7': 'e', '\u0bc8': 'ai',
    '\u0bca': 'o', '\u0bcb': 'o', '\u0bcc': 'au',
}
TAM_CONSONANTS = {
    '\u0b95': 'k', '\u0b99': 'ng', '\u0b9a': 's', '\u0b9c': 'j',
    '\u0b9e': 'ny', '\u0b9f': 't', '\u0ba3': 'n', '\u0ba4': 'th',
    '\u0ba8': 'n', '\u0ba9': 'n', '\u0baa': 'p', '\u0bae': 'm',
    '\u0baf': 'y', '\u0bb0': 'r', '\u0bb1': 'r', '\u0bb2': 'l',
    '\u0bb3': 'l', '\u0bb4': 'zh', '\u0bb5': 'v', '\u0bb7': 'sh',
    '\u0bb8': 's', '\u0bb9': 'h',
}


class TransliterationResult(NamedTuple):
    original_text: str
    transliterated_text: str
    detected_scripts: list[str]
    transformation_applied: bool


def detect_scripts(text: str) -> list[str]:
    scripts = []
    has_dev = any(DEVANAGARI_RANGE[0] <= ord(c) <= DEVANAGARI_RANGE[1] for c in text)
    has_tel = any(TELUGU_RANGE[0] <= ord(c) <= TELUGU_RANGE[1] for c in text)
    has_tam = any(TAMIL_RANGE[0] <= ord(c) <= TAMIL_RANGE[1] for c in text)
    has_lat = bool(re.search(r'[a-zA-Z]', text))

    if has_dev:
        scripts.append('devanagari')
    if has_tel:
        scripts.append('telugu')
    if has_tam:
        scripts.append('tamil')
    if has_lat:
        scripts.append('latin')
    return scripts


def transliterate_indic_segment(text: str) -> str:
    """Convert an Indic-script text segment to Latin phonemic form."""
    out = []
    n = len(text)
    i = 0

    while i < n:
        c = text[i]
        code = ord(c)

        # 1. Devanagari Block (0x0900 - 0x097F)
        if DEVANAGARI_RANGE[0] <= code <= DEVANAGARI_RANGE[1]:
            if c in DEV_VOWELS:
                out.append(DEV_VOWELS[c])
            elif c in DEV_CONSONANTS:
                base = DEV_CONSONANTS[c]
                # Check next character for virama or matra
                if i + 1 < n and text[i + 1] == '\u094d':  # Virama / halant
                    out.append(base)
                    i += 1  # Skip virama
                elif i + 1 < n and text[i + 1] in DEV_MATRAS:  # Matra
                    out.append(base + DEV_MATRAS[text[i + 1]])
                    i += 1  # Skip matra
                elif i + 1 < n and text[i + 1] == '\u093c':  # Nukta alone
                    out.append(base)
                    i += 1
                else:
                    # In Hindi/Devanagari, final consonant schwa deletion (halant deletion at word end)
                    # If end of word or followed by space/punctuation, usually no trailing 'a'
                    is_word_end = (i + 1 == n) or text[i + 1].isspace() or not (DEVANAGARI_RANGE[0] <= ord(text[i + 1]) <= DEVANAGARI_RANGE[1])
                    out.append(base if is_word_end else base + 'a')
            elif c in DEV_MATRAS:
                out.append(DEV_MATRAS[c])
            elif c in ('\u0902', '\u0901'):  # Anusvara / Chandrabindu
                out.append('n')
            elif c == '\u0903':  # Visarga
                out.append('h')
            elif c == '\u094d':  # Stray virama
                pass
            else:
                out.append(c)

        # 2. Telugu Block (0x0C00 - 0x0C7F)
        elif TELUGU_RANGE[0] <= code <= TELUGU_RANGE[1]:
            if c in TEL_VOWELS:
                out.append(TEL_VOWELS[c])
            elif c in TEL_CONSONANTS:
                base = TEL_CONSONANTS[c]
                if i + 1 < n and text[i + 1] == '\u0c4d':  # Virama
                    out.append(base)
                    i += 1
                elif i + 1 < n and text[i + 1] in TEL_MATRAS:
                    out.append(base + TEL_MATRAS[text[i + 1]])
                    i += 1
                else:
                    # In Telugu, schwa is pronounced as 'a'
                    out.append(base + 'a')
            elif c in TEL_MATRAS:
                out.append(TEL_MATRAS[c])
            elif c == '\u0c02':  # Anusvara
                out.append('m')
            elif c == '\u0c03':  # Visarga
                out.append('h')
            elif c == '\u0c4d':
                pass
            else:
                out.append(c)

        # 3. Tamil Block (0x0B80 - 0x0BFF)
        elif TAMIL_RANGE[0] <= code <= TAMIL_RANGE[1]:
            if c in TAM_VOWELS:
                out.append(TAM_VOWELS[c])
            elif c in TAM_CONSONANTS:
                base = TAM_CONSONANTS[c]
                if i + 1 < n and text[i + 1] == '\u0bcd':  # Pulli / virama
                    out.append(base)
                    i += 1
                elif i + 1 < n and text[i + 1] in TAM_MATRAS:
                    out.append(base + TAM_MATRAS[text[i + 1]])
                    i += 1
                else:
                    out.append(base + 'a')
            elif c in TAM_MATRAS:
                out.append(TAM_MATRAS[c])
            elif c == '\u0bcd':
                pass
            else:
                out.append(c)

        else:
            out.append(c)

        i += 1

    return ''.join(out)


def transliterate_text(text: str) -> TransliterationResult:
    """Transliterate text preserving URLs, file paths, numbers, code, and Latin terms."""
    if not text:
        return TransliterationResult(text, text, [], False)

    scripts = detect_scripts(text)
    indic_scripts = [s for s in scripts if s in ('devanagari', 'telugu', 'tamil')]

    if not indic_scripts:
        # Already entirely Latin / punctuation; preserve verbatim
        return TransliterationResult(text, text, scripts, False)

    # Protect URLs, file paths, and identifiers
    placeholders = {}

    def mask_match(m):
        key = f"__PROTECTED_ENTITY_{len(placeholders)}__"
        placeholders[key] = m.group(0)
        return key

    masked_text = URL_OR_PATH_REGEX.sub(mask_match, text)

    # Transliterate the masked text
    transliterated = transliterate_indic_segment(masked_text)

    # Restore protected tokens
    for key, val in placeholders.items():
        transliterated = transliterated.replace(key, val)

    # Clean double spaces
    transliterated = re.sub(r'\s+', ' ', transliterated).strip()

    return TransliterationResult(
        original_text=text,
        transliterated_text=transliterated,
        detected_scripts=scripts,
        transformation_applied=True
    )

