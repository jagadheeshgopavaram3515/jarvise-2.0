"""Privacy-conscious STT forensic helper.

No provider call is possible unless --execute-live is supplied explicitly.
By default this reports metadata, hashes, energy, and script classes only.
"""
from __future__ import annotations

import argparse
import hashlib
import json
import os
import sys
import wave
from pathlib import Path

import numpy as np


ROOTS = {
    "active": Path(r"D:\JARVIS"),
    "reference": Path(r"D:\GenAi_Project\Dhasha avatar project v1\jarvis"),
    "bluye": Path(r"D:\Bluye\backend"),
}


def script_counts(text: str) -> dict[str, int]:
    ranges = {
        "latin": ((0x0041, 0x005A), (0x0061, 0x007A)),
        "devanagari": ((0x0900, 0x097F),),
        "telugu": ((0x0C00, 0x0C7F),),
        "kannada": ((0x0C80, 0x0CFF),),
        "tamil": ((0x0B80, 0x0BFF),),
    }
    counts = {name: 0 for name in ranges}
    counts["other_letters"] = 0
    for char in text:
        code = ord(char)
        found = False
        for name, spans in ranges.items():
            if any(lo <= code <= hi for lo, hi in spans):
                counts[name] += 1
                found = True
                break
        if not found and char.isalpha():
            counts["other_letters"] += 1
    return counts


def classification(text: str) -> dict:
    counts = script_counts(text)
    present = [name for name, count in counts.items() if count]
    app_label = "te" if counts["telugu"] else "hi" if counts["devanagari"] else \
        "ta" if counts["tamil"] else "en"
    return {
        "sha256_utf8": hashlib.sha256(text.encode("utf-8")).hexdigest(),
        "character_count": len(text),
        "scripts_present": present,
        "script_counts": counts,
        "active_app_label": app_label,
        "note": "Latin text is labeled en; romanized Telugu cannot be inferred from script alone.",
    }


def load_wav(path: Path) -> tuple[np.ndarray, dict]:
    raw = path.read_bytes()
    with wave.open(str(path), "rb") as wav:
        meta = {
            "sha256_file": hashlib.sha256(raw).hexdigest(),
            "channels": wav.getnchannels(),
            "sample_width_bytes": wav.getsampwidth(),
            "sample_rate_hz": wav.getframerate(),
            "frame_count": wav.getnframes(),
            "compression": wav.getcomptype(),
        }
        if meta["channels"] != 1 or meta["sample_width_bytes"] != 2 or meta["compression"] != "NONE":
            raise ValueError("Expected uncompressed mono 16-bit PCM WAV; no conversion is performed")
        pcm = np.frombuffer(wav.readframes(wav.getnframes()), dtype="<i2")
    audio = pcm.astype(np.float32) / 32768.0
    meta["duration_s"] = round(audio.size / meta["sample_rate_hz"], 6)
    return audio, meta


def energy(audio: np.ndarray, rate: int) -> dict:
    def stats(part: np.ndarray) -> dict:
        if not part.size:
            return {"samples": 0, "rms": 0.0, "peak": 0.0}
        return {
            "samples": int(part.size),
            "rms": round(float(np.sqrt(np.mean(part * part))), 7),
            "peak": round(float(np.max(np.abs(part))), 7),
        }

    result = {"whole": stats(audio), "tail_1000ms": stats(audio[-rate:]),
              "tail_500ms": stats(audio[-rate // 2:]), "tail_100ms_bins": []}
    bin_size = max(1, rate // 10)
    tail = audio[-rate:]
    start_ms = -1000
    for start in range(0, tail.size, bin_size):
        item = stats(tail[start:start + bin_size])
        item["relative_start_ms"] = start_ms + round(start * 1000 / rate)
        result["tail_100ms_bins"].append(item)
    return result


def analyze(args) -> None:
    audio, meta = load_wav(args.wav)
    print(json.dumps({"wav": meta, "energy": energy(audio, meta["sample_rate_hz"])}, indent=2))


def classify(args) -> None:
    print(json.dumps(classification(args.text), indent=2, ensure_ascii=True))


def live(args) -> None:
    if not args.execute_live:
        raise SystemExit("Refusing provider call: add --execute-live only after explicit authorization")
    audio, meta = load_wav(args.wav)
    if meta["sample_rate_hz"] != 16000:
        raise SystemExit("Existing paths expect 16 kHz audio; refusing implicit resampling")
    root = ROOTS[args.system]
    os.chdir(root)
    sys.path.insert(0, str(root))
    from assistant.stt.cloud import GeminiAudioSTT  # imported from selected root
    provider = GeminiAudioSTT()
    try:
        text = provider.transcribe(audio) or ""
    finally:
        close = getattr(provider, "close", None)
        if close:
            close()
    output = {"system": args.system, "input_sha256": meta["sha256_file"],
              "model": provider.model, "result": classification(text), "nonempty": bool(text)}
    if args.show_transcript:
        output["transcript"] = text
    print(json.dumps(output, indent=2, ensure_ascii=True))


def main() -> None:
    parser = argparse.ArgumentParser()
    sub = parser.add_subparsers(dest="command", required=True)
    p_analyze = sub.add_parser("analyze", help="Inspect WAV metadata and tail energy; never calls a provider")
    p_analyze.add_argument("wav", type=Path)
    p_analyze.set_defaults(func=analyze)
    p_classify = sub.add_parser("classify", help="Classify scripts without printing text back")
    p_classify.add_argument("text")
    p_classify.set_defaults(func=classify)
    p_live = sub.add_parser("live", help="Call one selected implementation; gated and off by default")
    p_live.add_argument("system", choices=ROOTS)
    p_live.add_argument("wav", type=Path)
    p_live.add_argument("--execute-live", action="store_true")
    p_live.add_argument("--show-transcript", action="store_true",
                        help="Opt in to printing sensitive transcript text")
    p_live.set_defaults(func=live)
    args = parser.parse_args()
    args.func(args)


if __name__ == "__main__":
    main()
