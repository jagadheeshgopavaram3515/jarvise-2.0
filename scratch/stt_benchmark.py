"""
STT Benchmark Runner — Isolated offline comparison of Faster-Whisper and GeminiAudioSTT.

This script executes offline benchmarks against provided WAV audio files WITHOUT:
- Starting the microphone or sounddevice capture
- Enqueuing items to the production STT, Dispatcher, LLM, or TTS pipelines
- Modifying production files or environment variables
- Exposing secrets/API keys
"""
from __future__ import annotations

import argparse
import io
import json
import os
import sys
import time
import wave
from dataclasses import asdict, dataclass
from typing import Any, Dict, List, Optional, Tuple

import numpy as np

# Ensure JARVIS project root is on sys.path
PROJECT_ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from assistant import config


# ----------------------------------------------------------------------
# Script & Language Detection Helpers (pure Unicode analysis)
# ----------------------------------------------------------------------
_DEVANAGARI_RANGE = (0x0900, 0x097F)
_TELUGU_RANGE = (0x0C00, 0x0C7F)
_TAMIL_RANGE = (0x0B80, 0x0BFF)
_LATIN_RANGES = [
    (0x0041, 0x005A),  # A-Z
    (0x0061, 0x007A),  # a-z
]


def classify_scripts(text: str) -> Dict[str, Any]:
    """Analyze character scripts in the transcript text using Unicode ranges.

    Note: Having characters in Devanagari script indicates Devanagari Unicode
    codepoints, but does NOT uniquely prove Hindi (it could be Marathi, Sanskrit,
    or a phonetically collapsed representation of another language).
    """
    if not text:
        return {
            "has_telugu": False,
            "has_devanagari": False,
            "has_latin": False,
            "has_tamil": False,
            "primary_script": "empty",
            "script_summary": "Empty transcript",
        }

    counts = {
        "telugu": 0,
        "devanagari": 0,
        "latin": 0,
        "tamil": 0,
        "other": 0,
    }

    for ch in text:
        o = ord(ch)
        if not ch.isalpha():
            continue
        if _TELUGU_RANGE[0] <= o <= _TELUGU_RANGE[1]:
            counts["telugu"] += 1
        elif _DEVANAGARI_RANGE[0] <= o <= _DEVANAGARI_RANGE[1]:
            counts["devanagari"] += 1
        elif _TAMIL_RANGE[0] <= o <= _TAMIL_RANGE[1]:
            counts["tamil"] += 1
        elif any(start <= o <= end for start, end in _LATIN_RANGES):
            counts["latin"] += 1
        else:
            counts["other"] += 1

    total_alpha = sum(counts.values())
    if total_alpha == 0:
        primary = "non-alphabetic"
    else:
        primary = max(counts, key=lambda k: counts[k])

    return {
        "has_telugu": counts["telugu"] > 0,
        "has_devanagari": counts["devanagari"] > 0,
        "has_latin": counts["latin"] > 0,
        "has_tamil": counts["tamil"] > 0,
        "counts": counts,
        "primary_script": primary,
        "script_summary": f"Primary: {primary.capitalize()} (telugu={counts['telugu']}, devanagari={counts['devanagari']}, latin={counts['latin']})",
    }


# ----------------------------------------------------------------------
# Audio Loading & Format Normalization
# ----------------------------------------------------------------------
@dataclass
class AudioMetadata:
    path: str
    sample_rate: int
    channels: int
    sample_width: int
    duration_s: float
    num_frames: int
    format_notes: str


def load_and_normalize_wav(path: str, target_sr: int = 16000) -> Tuple[np.ndarray, AudioMetadata]:
    """Read a WAV file and normalize it to float32 16kHz mono, matching AudioInputService."""
    if not os.path.isfile(path):
        raise FileNotFoundError(f"Audio file not found: {path}")

    with wave.open(path, "rb") as wf:
        channels = wf.getnchannels()
        sampwidth = wf.getsampwidth()
        framerate = wf.getframerate()
        nframes = wf.getnframes()
        raw_bytes = wf.readframes(nframes)

    duration_s = nframes / float(framerate) if framerate > 0 else 0.0

    notes = []
    # Read samples into numpy array according to sample width
    if sampwidth == 2:
        samples = np.frombuffer(raw_bytes, dtype=np.int16).astype(np.float32) / 32768.0
    elif sampwidth == 1:
        samples = (np.frombuffer(raw_bytes, dtype=np.uint8).astype(np.float32) - 128.0) / 128.0
        notes.append("Converted 8-bit unsigned PCM to float32")
    elif sampwidth == 4:
        samples = np.frombuffer(raw_bytes, dtype=np.int32).astype(np.float32) / 2147483648.0
        notes.append("Converted 32-bit PCM to float32")
    else:
        raise ValueError(f"Unsupported WAV sample width: {sampwidth} bytes")

    # Downmix multi-channel to mono
    if channels > 1:
        samples = samples.reshape(-1, channels)
        samples = samples.mean(axis=1)
        notes.append(f"Downmixed {channels} channels to mono")

    # Resample if sample rate does not match target_sr
    if framerate != target_sr:
        # Simple high-quality linear/polyphase resample without requiring extra packages
        duration = len(samples) / float(framerate)
        target_len = int(duration * target_sr)
        orig_indices = np.linspace(0, len(samples) - 1, target_len)
        samples = np.interp(orig_indices, np.arange(len(samples)), samples).astype(np.float32)
        notes.append(f"Resampled from {framerate}Hz to {target_sr}Hz")

    meta = AudioMetadata(
        path=os.path.abspath(path),
        sample_rate=framerate,
        channels=channels,
        sample_width=sampwidth,
        duration_s=round(duration_s, 3),
        num_frames=nframes,
        format_notes="; ".join(notes) if notes else "16kHz mono PCM (no conversion needed)",
    )
    return samples, meta


# ----------------------------------------------------------------------
# Recognizer Wrappers (Isolated, non-interfering)
# ----------------------------------------------------------------------
@dataclass
class EngineBenchmarkResult:
    engine_name: str
    is_cloud: bool
    status: str  # "success", "empty", "error", "skipped"
    transcript: str
    reported_language: Optional[str]
    reported_confidence: Optional[float]
    elapsed_time_s: float
    script_analysis: Dict[str, Any]
    error_message: Optional[str] = None
    execution_notes: str = ""


class FasterWhisperRunner:
    def __init__(self, model_size: Optional[str] = None):
        from assistant.stt.recognizer import FasterWhisperSTT
        self.model_size = model_size or config.STT_MODEL_SIZE
        # Temporarily suppress language cache side-effects by keeping a local instance
        self.engine = FasterWhisperSTT()

    def run(self, audio: np.ndarray) -> EngineBenchmarkResult:
        t0 = time.perf_counter()
        try:
            # Transcribe with language=None to allow genuine auto-detection
            # beam_size=config.STT_BEAM_SIZE, vad_filter=False, condition_on_previous_text=False
            res = self.engine._transcribe_once(
                audio=audio,
                model_size=self.model_size,
                queue_wait=0.0,
                is_partial=False,
                language=None,  # Do not force or use cached language for benchmark
            )
            elapsed = time.perf_counter() - t0
            text = (res.text or "").strip()
            script_info = classify_scripts(text)
            status = "success" if text else "empty"

            return EngineBenchmarkResult(
                engine_name=f"FasterWhisper ({self.model_size}, {self.engine.device}, {self.engine.compute_type})",
                is_cloud=False,
                status=status,
                transcript=text,
                reported_language=res.language,
                reported_confidence=round(res.language_probability, 4) if res.language_probability else None,
                elapsed_time_s=round(elapsed, 4),
                script_analysis=script_info,
                execution_notes="Local machine-dependent execution; forced language=None (unbiased auto-detect)",
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return EngineBenchmarkResult(
                engine_name=f"FasterWhisper ({self.model_size})",
                is_cloud=False,
                status="error",
                transcript="",
                reported_language=None,
                reported_confidence=None,
                elapsed_time_s=round(elapsed, 4),
                script_analysis=classify_scripts(""),
                error_message=str(e),
                execution_notes="Local machine-dependent execution",
            )


class GeminiAudioRunner:
    def __init__(self):
        from assistant.stt.cloud import GeminiAudioSTT
        self.engine = GeminiAudioSTT()
        self.model_name = self.engine.model

    def run(self, audio: np.ndarray) -> EngineBenchmarkResult:
        t0 = time.perf_counter()
        try:
            # Call cloud transcription directly
            text = self.engine.transcribe(audio)
            elapsed = time.perf_counter() - t0
            cleaned_text = (text or "").strip()
            script_info = classify_scripts(cleaned_text)
            status = "success" if cleaned_text else "empty"

            # In Gemini multimodal audio API, no raw language confidence float is returned;
            # script detection can infer script, but not acoustic language probability.
            return EngineBenchmarkResult(
                engine_name=f"GeminiAudioSTT ({self.model_name})",
                is_cloud=True,
                status=status,
                transcript=cleaned_text,
                reported_language=None,  # Gemini multimodal prompt outputs verbatim text without lang code
                reported_confidence=None,  # Gemini API does not report language probability
                elapsed_time_s=round(elapsed, 4),
                script_analysis=script_info,
                execution_notes="Cloud network-dependent execution; verbatim multilingual prompt",
            )
        except Exception as e:
            elapsed = time.perf_counter() - t0
            return EngineBenchmarkResult(
                engine_name=f"GeminiAudioSTT ({self.model_name})",
                is_cloud=True,
                status="error",
                transcript="",
                reported_language=None,
                reported_confidence=None,
                elapsed_time_s=round(elapsed, 4),
                script_analysis=classify_scripts(""),
                error_message=str(e),
                execution_notes="Cloud network-dependent execution",
            )


# ----------------------------------------------------------------------
# Main Runner & Reporting
# ----------------------------------------------------------------------
def run_benchmark(wav_paths: List[str], output_json: Optional[str] = None):
    print("=" * 80)
    print(" JARVIS 2.0 — ISOLATED OFFLINE STT BENCHMARK")
    print("=" * 80)
    print(f"Target sample count: {len(wav_paths)}")
    print("Verifying recognizer initializations without starting pipeline...")

    # Initialize local engine
    print("[1/2] Initializing FasterWhisperSTT...")
    whisper_runner = None
    try:
        whisper_runner = FasterWhisperRunner()
        print(f"  -> FasterWhisper initialized ({whisper_runner.model_size}, device={whisper_runner.engine.device})")
    except Exception as e:
        print(f"  -> Failed to initialize FasterWhisper: {e}")

    # Initialize cloud engine
    print("[2/2] Initializing GeminiAudioSTT...")
    cloud_runner = None
    try:
        cloud_runner = GeminiAudioRunner()
        print(f"  -> GeminiAudioSTT initialized (model={cloud_runner.model_name})")
    except Exception as e:
        print(f"  -> Failed to initialize GeminiAudioSTT: {e}")

    if not whisper_runner and not cloud_runner:
        print("\nERROR: Neither recognizer could be initialized. Aborting benchmark.")
        return

    benchmark_records = []

    for idx, path in enumerate(wav_paths, 1):
        print("\n" + "-" * 80)
        print(f"Sample {idx}/{len(wav_paths)}: {os.path.basename(path)}")
        print("-" * 80)

        try:
            audio_data, meta = load_and_normalize_wav(path)
            print(f"  Path: {meta.path}")
            print(f"  Duration: {meta.duration_s}s | Sample Rate: {meta.sample_rate}Hz | Channels: {meta.channels}")
            print(f"  Notes: {meta.format_notes}")
        except Exception as e:
            print(f"  Failed reading audio file: {e}")
            benchmark_records.append({
                "sample_path": path,
                "error": str(e),
            })
            continue

        sample_record = {
            "metadata": asdict(meta),
            "results": {},
        }

        # --- Run Faster-Whisper ---
        if whisper_runner:
            print("\n  [Running FasterWhisper CPU...]")
            fw_res = whisper_runner.run(audio_data)
            sample_record["results"]["faster_whisper"] = asdict(fw_res)
            print(f"    Status:     {fw_res.status.upper()}")
            print(f"    Elapsed:    {fw_res.elapsed_time_s:.3f}s (machine-dependent)")
            print(f"    Language:   {fw_res.reported_language} (prob: {fw_res.reported_confidence})")
            print(f"    Script:     {fw_res.script_analysis['script_summary']}")
            print(f"    Transcript: {repr(fw_res.transcript)}")
            if fw_res.error_message:
                print(f"    Error:      {fw_res.error_message}")
        else:
            print("  [FasterWhisper skipped - not available]")

        # --- Run Gemini Audio ---
        if cloud_runner:
            print("\n  [Running Gemini Cloud STT...]")
            cloud_res = cloud_runner.run(audio_data)
            sample_record["results"]["gemini_cloud"] = asdict(cloud_res)
            print(f"    Status:     {cloud_res.status.upper()}")
            print(f"    Elapsed:    {cloud_res.elapsed_time_s:.3f}s (network-dependent)")
            print(f"    Language:   (Multimodal direct verbatim output)")
            print(f"    Script:     {cloud_res.script_analysis['script_summary']}")
            print(f"    Transcript: {repr(cloud_res.transcript)}")
            if cloud_res.error_message:
                print(f"    Error:      {cloud_res.error_message}")
        else:
            print("  [Gemini Cloud STT skipped - not available]")

        benchmark_records.append(sample_record)

    # Save JSON results
    if output_json is None:
        scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
        os.makedirs(scratch_dir, exist_ok=True)
        timestamp = time.strftime("%Y%m%d_%H%M%S")
        output_json = os.path.join(scratch_dir, f"stt_benchmark_{timestamp}.json")

    with open(output_json, "w", encoding="utf-8") as f:
        json.dump(benchmark_records, f, indent=2, ensure_ascii=False)

    print("\n" + "=" * 80)
    print(" BENCHMARK COMPLETE")
    print(f" Full results written to: {output_json}")
    print("=" * 80)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Isolated offline STT benchmark for FasterWhisper vs GeminiAudioSTT.")
    parser.add_argument("wav_files", nargs="*", help="Paths to WAV audio files to benchmark.")
    parser.add_argument("--output", "-o", default=None, help="Optional output JSON path.")
    args = parser.parse_args()

    if not args.wav_files:
        print("Usage: python scratch/stt_benchmark.py sample1.wav sample2.wav ...")
        print("\nNo WAV files provided. Checking scratch directory for test recordings...")
        scratch_dir = os.path.join(PROJECT_ROOT, "scratch")
        found = [
            os.path.join(scratch_dir, f)
            for f in os.listdir(scratch_dir)
            if f.lower().endswith(".wav")
        ]
        if found:
            print(f"Found {len(found)} WAV file(s) in scratch/:")
            for p in found:
                print(f" - {p}")
            args.wav_files = found
        else:
            print(f"No WAV files found in '{scratch_dir}'.")
            print("Please provide one or more .wav files to run the benchmark.")
            sys.exit(0)

    run_benchmark(args.wav_files, args.output)

