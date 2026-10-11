# STT reference investigation — 2026-10-03

## Outcome

The relevant reference Gemini behavior is already present in the working tree.
No provider, model, prompt, encoding, segmentation, or language rewrite is justified
by this comparison. This task adds privacy-conscious rejection measurements and
synthetic/mocked contract tests. It does **not** establish improved recognition
quality or resolve a measured live recognition discrepancy.

## Reference call chain and exact behavior

All paths in this section are relative to the read-only
`D:\GenAi_Project\Dhasha avatar project v1\jarvis`.

1. `main.py:main` creates `Bus`; `run_gui` calls
   `assistant/core/pipeline.py:start_services`. The factory starts `STTService`,
   waits for readiness, then starts `AudioInputService`. Entry-point and cloud
   module descriptions are stale: follow the actual recognizer implementation.
2. `assistant/audio/capture.py:run` opens sounddevice `RawInputStream`, 16,000 Hz,
   mono, int16, 480 samples per 30ms callback. `_callback` copies bytes into an
   unbounded microphone frame queue. No microphone-device override is supplied.
3. `_segment_loop` applies half-duplex speaking/tail guarding, then
   `EchoCanceller.process_audio` and `VAD.is_speech`.
   `assistant/audio/aec.py` optionally uses WebRTC audio processing with noise
   suppression, otherwise playback-conditioned echo suppression. The fallback
   can zero quiet microphone frames during playback (900 int16 RMS without a
   speaker reference; 1.6 times reference RMS with one).
   `assistant/audio/vad.py` uses WebRTC VAD aggressiveness 2, or an int16 RMS
   fallback threshold of 500 when the dependency is unavailable.
4. Speech begins on the first voiced frame, retaining up to 16 preroll frames
   (480ms). Endpointing uses 500ms silence, grows by 180ms per second of buffered
   audio, caps at 1300ms; semantic short-utterance endpointing can reduce this
   to 240ms once buffered duration reaches 900ms and remains below 3500ms.
   Integer frame rounding applies. Minimum buffered utterance is 250ms configured
   (8 frames); maximum is 30 seconds (1000 frames). Very short blips can instead
   become filler signals. Partial snapshots occur only while the assistant speaks.
5. `_emit` concatenates **all** buffered frames, including preroll and trailing
   silence, converts int16 to float32 by dividing by 32768, and queues `Utterance`
   with ID, sequence, timing, and during-speech metadata. There is no silence
   trimming or resampling. The input bus queue has capacity 64; capture rejects
   when full. Timing metadata subtracts trailing silence, but audio retains it.
6. `assistant/stt/recognizer.py:STTService.run` initializes FasterWhisper even when
   hybrid STT is enabled, then optionally initializes Gemini. The reference `.env`
   enables hybrid STT. `_transcribe_and_emit` rejects stale finals, applies the
   prefilter, and tries Gemini first for **every** eligible final. The command
   heuristic in cloud.py does not select the provider on this path.
7. `_has_real_speech` optionally loads Silero (threshold 0.5, minimum speech
   duration 200ms), otherwise uses whole-segment normalized RMS > 0.005. Thus the
   reference also rejects RMS-low segments if Silero is absent. Its actual runtime
   dependency/backend state was not supplied and cannot be inferred from source.
8. `assistant/stt/cloud.py:GeminiAudioSTT` uses `google.genai.Client`, Google API-key
   authentication, and `models.generate_content`. Both requirement files specify
   `google-genai>=0.3.0`; the test interpreter has 2.11.0. This does not establish
   the SDK version used by the reference application's original runtime.
   Model default is `gemini-2.5-flash-lite`; both `.env` files select
   **`gemini-flash-lite-latest`**. No generation config or forced language is sent.
9. `_encode_wav` clips float samples to [-1,1], multiplies by 32767, casts to
   little-endian signed int16, and creates mono PCM WAV at 16kHz. Contents are
   `Part.from_bytes(data=wav, mime_type="audio/wav")` followed by exactly:

   > Transcribe this audio verbatim in its original language (English, Hindi, Telugu, or a mix). Output ONLY the transcript text — no quotes, labels, or commentary.

   This is local source text, not an external copyrighted excerpt.
10. The response `.text` is stripped; empty, timeout, or error returns None.
    A single-worker executor waits four seconds per the reference `.env` (default
    three). It sets no explicit SDK HTTP timeout or retry options. Timed-out calls
    are not cancelled and further submissions can queue behind them. There is no
    explicit cloud retry in this wrapper; SDK defaults are not constrained by it.
    Cloud failure triggers Whisper fallback, configured `base`, language `auto`
    (partial model defaults `tiny`, partial language English), beam size 1,
    `vad_filter=False`, `condition_on_previous_text=False`, with language caching
    and confidence/language rejection. No added silence trim occurs there.
11. `_post_transcript` preserves utterance association and posts to `transcript_q`;
    the dispatcher executes the existing command/LLM flow. Script detection marks
    Telugu script `te`, Devanagari `hi`, Tamil `ta`, otherwise `en`. Romanized Telugu
    is not separately detected or transliterated; mixed speech is passed through
    as returned. A language probability of 1.0 is assigned to cloud text by the
    application, not supplied as calibrated confidence by Gemini. Reference
    partials use Whisper; partial/final interruption keywords are handled by the
    dispatcher. There is no Gemini streaming transcription.

## Current flow and concrete differences

`AudioInputService` → complete `Utterance` → RMS prefilter → bounded pending FIFO
→ one final worker → one outstanding Gemini SDK call → `Transcript` → existing
`Dispatcher._route` → existing configured orchestration/commands → TTS.

Capture, VAD, and AEC match the reference; capture's only code differences are
the existing interruption-generation tagging. Same 16kHz PCM conversion, silence
retention, boundaries, WAV encoding, request prompt, configured model, and script
language inference. Both `.env` files have half-duplex mode and semantic
endpointing enabled, auto legacy language, and RMS threshold 0.005.

The active implementation intentionally forces Gemini-only STT, disables Whisper
fallback and partials regardless of obsolete `.env` switches, avoids Silero/model
loading, and preserves Phase 1 bounded FIFO admission, expiry, newest-overflow
rejection, duplicate-ID suppression, interruption generations, dispatch-time
stale-generation rejection, and non-waiting shutdown. Current configured caller
timeout is 45 seconds versus reference four; current SDK HTTP timeout is explicit
and automatic retry attempts are restricted to one. The SDK slot remains occupied
after caller timeout until completion, preventing duplicate calls and executor
backlog. Default FIFO capacity is four pending finals, default expiry eight seconds;
requests can therefore cause waiting finals to expire before a 45-second timeout.
These policies affect availability and barge-in latency, not the audio prompt.

## Root-cause evidence and quiet speech

No live paired recordings, provider responses, or error-rate measurements were
provided. Therefore **the real-world root cause is not established**. Request
construction and capture differences are not supported as causes by source
comparison. Fallback availability, partial latency, Silero presence, and queue
expiry are concrete differences that could explain different observed behavior.

Synthetic evidence: 100ms at normalized amplitude 0.006 passes RMS alone; adding
400ms silence lowers whole-segment RMS to approximately 0.002683 and rejects it.
This proves silence dilution is possible, not that a synthetic constant signal is
speech, or that the reference accepted the same audio. The fallback capture VAD
threshold itself is approximately 0.0153 normalized RMS, another limitation when
WebRTC VAD is unavailable. No thresholds or backend selection were changed.

Available scratch logs contain 51 `vad_prefilter_no_speech` lines in
`filtered_log.txt`, with no per-rejection RMS measurements; `log_slice.txt` has no
matching rejections. They do not independently reproduce the reported 256-count
run or identify which rejections contained real speech. Missing evidence is
backend/runtime information and per-utterance energy/duration measurements tied
to labeled utterances. High-energy noise can pass an RMS gate: these tests cannot
establish general noise/speech discrimination.

## Changes made by this task

- `assistant/stt/recognizer.py`: rejection-only diagnostics add maximum 30ms-frame
  RMS and total duration of frames above the **unchanged** threshold. Partial
  frames use their actual sample count. No audio trimming, transcription text,
  credentials, or raw audio is added to logs; no admission behavior changes.
- `tests/test_stt_reference_contract.py`: eight tests covering exact PCM clipping,
  silence retention, WAV metadata, original-language prompt and response
  preservation across English/Telugu/romanized/mixed text, capture association,
  padded quiet bursts, silence/low-energy noise, and conversational delivery through
  the existing dispatcher/orchestrator interface.
- `scratch/stt_recovery_baseline.json`: before-edit SHA-256 record for inspected
  reference and corresponding active source files.
- This report records evidence, boundaries, and validation.

No new provider or fallback was added. Already migrated Gemini behavior was
retained. Reference fallback, model startup, Silero loading, partials, and unsafe
unbounded cloud submissions were intentionally not copied. No Phase 2 work.

## Validation

Exact command:

```powershell
python -m pytest -q tests/test_stt_reference_contract.py tests/test_stt_migration.py tests/test_stt_fifo.py tests/test_cloud_stt_lifecycle.py tests/test_resilience.py tests/test_story_pipeline.py
git diff --check
```

Final result: **109 passed, 0 failed** in 24.07 seconds. Initial run: 107 passed,
one new test failed because its expected above-threshold frame duration was 100ms
rather than 90ms; corrected the expectation to account for a silence-containing
boundary frame, then added the conversational integration test and reran all six
files. Existing suites exercise FIFO ordering/capacity/expiry, timeout and recovery,
interruption and stale-result prevention, pending shutdown, duplicate prevention,
no Whisper/Silero loading, endpointing, dispatch, and audio/turn lifecycle.

`git diff --check`: exit 0; only pre-existing LF-to-CRLF conversion warnings.
Providers are mocked and audio synthetic. No live STT API calls, microphones,
or real recognition quality tests. The test interpreter lacks sounddevice,
webrtcvad, silero-vad, and faster-whisper; installed optional backend behavior
and Telugu/code-switch accuracy remain unverified.

## Change control

Before editing, `git status --short`, `git diff --stat`, and relevant diffs were
inspected. The working tree already contained substantial Phase 1 and unrelated
changes, including cloud/recognizer/config/capture, dispatcher/events/pipeline,
orchestration, memory engine, benchmark artifacts, and tests. Those changes were
preserved; the new production edit is confined to rejection diagnostics in the
already modified recognizer. Reference hashes were rechecked and all inspected
reference files remained identical. No writes were made to the reference project,
`D:\Bluye`, `.env`, persistent data, or unrelated working-tree files. No Git reset,
commit, formatting pass, provider replacement, or architecture change occurred.
