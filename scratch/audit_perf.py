import json

with open("performance_log.json", "r", encoding="utf-8") as f:
    records = json.load(f)

with open("jarvis.log", "r", encoding="utf-8", errors="ignore") as f:
    log_lines = f.readlines()

print("=== DETAILED LATENCY & PROVIDER AUDIT ===")

for r in records[-10:]:
    tid = r["trace_id"]
    ts = r["timestamp"]
    time_part = ts.split("T")[1] if "T" in ts else ts

    # Find log lines mentioning this trace_id
    tid_lines = [l.strip() for l in log_lines if tid in l]
    
    # Extract query and response if present
    query = "Unknown"
    response_start = ""
    for l in tid_lines:
        if "stt_done" in l and "text=" in l:
            query = l.split("text=")[-1]
        elif "gemini_first_text" in l and "text=" in l:
            response_start = l.split("text=")[-1]

    # Find routing decision around that time
    route_info = "Not found"
    minute = time_part[:5]
    for l in log_lines:
        if minute in l and "[ROUTER] Classified" in l:
            route_info = l.split("[ROUTER]")[-1].strip()
        elif minute in l and "[REALTIME]" in l:
            pass

    # Check which backend was called around that minute:
    # 1. Gemini API (generativelanguage.googleapis.com)
    # 2. Local Ollama (qwen)
    # 3. SerpAPI
    backends = []
    for l in log_lines:
        if minute in l:
            if "generativelanguage.googleapis.com" in l and "Gemini API" not in backends:
                backends.append("Gemini API (Cloud)")
            if "realtime" in l and "provider=serpapi" in l and "SerpAPI (Google Search)" not in backends:
                backends.append("SerpAPI (Google Search)")
            if "WARMUP" in l or "Ollama" in l and "Ollama Local" not in backends:
                backends.append("Ollama Local")

    print(f"\nTrace ID: {tid} | Time: {ts}")
    print(f"  User Query: {query}")
    print(f"  First Spoken Text: {response_start}")
    print(f"  Routing Event: {route_info}")
    print(f"  Network/Backend Calls in window: {', '.join(backends) if backends else 'Local only (Ollama/Internal)'}")
    print(f"  --- Latency Breakdown ---")
    print(f"    1. Audio Speech Duration : {r.get('speech_duration', 0.0):.2f}s")
    print(f"    2. STT Transcription     : {r.get('stt', 0.0):.2f}s")
    print(f"    3. LLM Wait (First Token): {r.get('gemini', 0.0):.2f}s")
    print(f"    4. TTS Audio Synthesis   : {r.get('tts_synth', 0.0):.2f}s")
    print(f"    5. TTS Playback Duration : {r.get('tts_play', 0.0):.2f}s")
    print(f"    * TOTAL Turn Latency     : {r.get('total', 0.0):.2f}s (User stopped speaking -> Assistant spoke)")

