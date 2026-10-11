"""Unit test suite verifying correctness, entity preservation, and script handling of transliteration."""
import re
import pytest
from pathlib import Path
import sys

ROOT = Path('D:/JARVIS')
EXP_DIR = ROOT / 'reports/jarvis_transliteration_experiment_20261010_205500'
sys.path.insert(0, str(EXP_DIR))

from indic_transliterator import transliterate_text, detect_scripts

def test_english_and_latin_unaltered():
    """English requests must not be altered."""
    samples = [
        "Open PowerShell terminal application",
        "Find the quarterly financial report spreadsheet",
        "What did we decide about our Phoenix project?",
        "Delete that completely",
    ]
    for s in samples:
        res = transliterate_text(s)
        assert res.transliterated_text == s
        assert res.transformation_applied is False
        assert res.detected_scripts == ['latin']

def test_url_and_filepath_preservation():
    """URLs and file paths must not have their characters transliterated or mangled."""
    samples = [
        ("Read this https://docs.python.org/3/library/os.html page", "https://docs.python.org/3/library/os.html"),
        ("ये फ़ाइल C:\\Users\\Projects\\report.pdf खोलो", "C:\\Users\\Projects\\report.pdf"),
        ("ఈ లింక్ www.google.com ఓపెన్ చెయ్యి", "www.google.com"),
    ]
    for text, entity in samples:
        res = transliterate_text(text)
        assert entity in res.transliterated_text, f"Entity '{entity}' was corrupted in '{res.transliterated_text}'"

def test_telugu_native_script_transliteration():
    """Telugu script must map phonemically to Latin."""
    t1 = "నా డౌన్‌లోడ్స్ ఫోల్డర్ ఓపెన్ చెయ్యి"
    res1 = transliterate_text(t1)
    assert res1.transformation_applied is True
    assert 'telugu' in res1.detected_scripts
    # Must not contain Telugu Unicode characters
    assert not re.search(r'[\u0C00-\u0C7F]', res1.transliterated_text)
    # Check key phonetic elements
    assert 'foldar' in res1.transliterated_text or 'fold' in res1.transliterated_text or 'daun' in res1.transliterated_text

def test_devanagari_hindi_transliteration():
    """Devanagari script must map phonemically to Latin."""
    t1 = "डाउनलोड्स फ़ोल्डर खोलो"
    res1 = transliterate_text(t1)
    assert res1.transformation_applied is True
    assert 'devanagari' in res1.detected_scripts
    assert not re.search(r'[\u0900-\u097F]', res1.transliterated_text)
    assert 'kholo' in res1.transliterated_text

def test_tamil_native_script_transliteration():
    """Tamil script must map phonemically to Latin."""
    t1 = "டவுன்லோட்ஸ் கோப்புறையை திறக்கவும்"
    res1 = transliterate_text(t1)
    assert res1.transformation_applied is True
    assert 'tamil' in res1.detected_scripts
    assert not re.search(r'[\u0B80-\u0BFF]', res1.transliterated_text)

def test_mixed_code_switching_preservation():
    """Latin words in code-switched queries must remain intact."""
    text = "Chrome browser application बंद करो"
    res = transliterate_text(text)
    assert res.transformation_applied is True
    assert "Chrome browser application" in res.transliterated_text
    assert "band" in res.transliterated_text or "karo" in res.transliterated_text

def test_numbers_and_punctuation_preservation():
    """Numbers, spaces, and punctuation must remain intact."""
    text = "बेडरूम का तापमान 22 डिग्री पर सेट करो"
    res = transliterate_text(text)
    assert "22" in res.transliterated_text

