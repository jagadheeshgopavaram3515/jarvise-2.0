# Active STT failure diagnosis — 2026-10-03

## Finding

The first proven configuration divergence capable of losing an utterance tail is
endpointing. Active JARVIS currently uses `VAD_SILENCE_MS=350`,
`VAD_SILENCE_MAX_MS=700`, and `VAD_SILENCE_GROW_MS_PER_SEC=75`. Old JARVIS uses
500/1300/180. Bluye's Python microphone path uses 500/1300/180, while the Bluye
dashboard path evidenced by its trace log uses a separate browser VAD: 650ms base,
up to 2000ms after longer speech, with no semantic 240ms shortcut.

This proves behavioral non-equivalence and a plausible cut point. It does not yet
prove that a particular missing ending was absent from the active WAV, because no
authorized failed-utterance WAV exists. Old and active both enable the same
semantic shortcut, so endpoint configuration alone does not explain every reported
difference between those two systems.

Active runtime evidence also proves that Romanized Telugu reaches the dispatcher
after a successful Gemini call. The provider response is stripped but otherwise
passed unchanged through `Transcript`; `_lang_from_script` only labels it and does
not transliterate it. Therefore the observed Romanized output originates in
Gemini's response, not the dispatcher or agentic orchestration. No captured active
example contains Devanagari, so the reported Hindi-script case remains unobserved.

No production files were changed. No live provider calls were made.

## Three-system comparison

| Stage | Active JARVIS | Old JARVIS | Bluye evidenced runtime |
|---|---|---|---|
| Capture | sounddevice raw int16, mono 16kHz, 30ms | same | dashboard Web Audio float32 chunks; Python path also exists |
| Start/stop detector | WebRTC VAD mode 2 | same | browser RMS start 0.035 / stop 0.018 |
| Base/max/growth silence | 350/700/75ms | 500/1300/180ms | browser 650/2000/180ms after 1.5s |
| Semantic shortcut | enabled, 240ms for buffered 0.9–3.5s | same | absent in dashboard path |
| Final assembly | preroll + speech + trailing silence, no trim | same | frames from onset through trailing silence, no preroll |
| Prefilter | whole-utterance RMS > 0.005 only | Silero if available, otherwise same RMS | `_has_real_speech`; backend depends on environment |
| Gemini request | alias `gemini-flash-lite-latest`, original-language prompt | same | same alias, but prompt adds explicit silence/noise instruction |
| Failure behavior | no Whisper fallback | Whisper fallback | Whisper fallback in Python service; WebSocket calls Gemini wrapper directly |
| Queue | bounded FIFO, expiry, one outstanding SDK call | executor submissions can backlog | WebSocket turn ownership; no active-style FIFO |

The Bluye log cannot validate Bluye's Python `AudioInputService`: its `TURN-XXXX`,
`AUDIO RECEIVED`, and `STT START/RESULT` records are emitted by
`backend/api/websocket.py`, after `VoiceController.tsx` performs browser VAD and
sends a completed float32 utterance. Bluye is therefore a useful working system,
but not an identical capture-path control.

## Active utterance trace

1. PortAudio supplies 480 mono int16 samples per 30ms callback. `_callback` puts
   each byte frame into `_ring`; frames shorter than 960 bytes are ignored.
2. In half-duplex mode, frames are discarded while TTS speaks and for the tail
   guard afterward. Remaining frames pass through AEC, then WebRTC VAD.
3. The first voiced frame starts an utterance and copies the preroll. Every later
   frame is appended. Silence is counted in 30ms units. A resumed voiced frame
   resets the count only if endpointing has not already fired.
4. `_emit` concatenates every retained frame and converts int16 to normalized
   float32. It does not trim the tail. `Perf.t_start` subtracts trailing silence
   for timing only; that subtraction does not affect audio samples.
5. The STT consumer assigns/carries an interruption epoch, rejects cancelled or
   expired work, applies RMS prefilter before FIFO admission, rejects the newest
   final when pending capacity is full, and runs one final at a time.
6. `_encode_wav` clips the full float array, scales by 32767, writes mono 16-bit
   PCM at 16kHz, and sends those exact bytes as `audio/wav`. There is no resample,
   language conversion, or silence trim between `Utterance.audio` and WAV creation.
7. Gemini `.text` is stripped. On success the recognizer strips again, derives a
   label from Unicode script, and publishes the same string with the same
   utterance ID. Latin-only Telugu is labeled `en`; Telugu script is `te`;
   Devanagari is `hi`. This label does not alter text.
8. Dispatcher rejects stale interruption generations, then calls `.strip()` and
   routes the same string through existing command/orchestration logic.

Potential tail-loss points are: half-duplex/tail-guard frame discard, VAD marking
quiet final speech as silence, endpoint firing before speech resumes, capture queue
or malformed-frame loss, prefilter rejection of the complete final, FIFO overflow,
FIFO expiry, interruption cancellation, provider omission, and provider timeout.
Only VAD/endpointing can remove a suffix *within* an otherwise submitted final.
The prefilter and FIFO reject whole utterances; `_encode_wav` preserves the entire
accepted array. Gemini can omit words even when the samples are present.

## Existing runtime evidence

The current `D:\JARVIS\jarvis.log` run reports WebRTC VAD available and the
Realtek microphone opened at 16kHz. It contains 24 `no_speech`, eight FIFO
`queued`, six `delivered`, and no recorded queue overflow, expiry, stale,
duplicate, cancellation, timeout, or provider-error category. Consequently the
FIFO is not supported as the cause in this captured run.

Several rejected finals contain only 30–60ms above the RMS threshold even though
their buffered durations range up to about two seconds. They were rejected before
Gemini. Audio content was not retained, so those events cannot be labeled speech,
noise, or missing suffixes.

Successful provider events preserve utterance IDs through delivery. The existing
dispatcher log then shows multiple Latin-script Telugu/code-switched utterances.
Because the sole success path is `resp.text.strip()` → `text.strip()` →
`Transcript.text` → dispatcher `tr.text.strip()`, with no transliteration or
translation operation, later stages are excluded as the source of Romanization.
The current privacy posture is imperfect: dispatcher/performance logs already
contain transcript excerpts. New diagnostics should omit them by default.

## Observation plan for one failed utterance

Use one correlation ID across the following records. Wall-clock timestamp and
monotonic elapsed time are both needed.

| Boundary | Required fields, excluding content |
|---|---|
| Capture start | utterance ID, interrupt epoch, device, backend, sample rate, frame size, start timestamp |
| Endpoint | timestamp, reason (`silence`/`max_length`), total frames, voiced frames, final silence frames, required silence frames, semantic-shortcut flag |
| Assembly | utterance ID, sample count, duration, PCM/WAV SHA-256, whole RMS/peak, final 1000/500/100ms-bin RMS and peak |
| Prefilter | pass/reject, threshold, whole RMS, max frame RMS, above-threshold duration |
| Admission | captured/admitted timestamps, FIFO depth, active request ID, outcome (`queued`/overflow/expired/cancelled) |
| Provider | request start/end, WAV hash and byte count, model, prompt/config fingerprint, HTTP/result category, latency; no key or text |
| Response | empty/nonempty, UTF-8 SHA-256, character count, Unicode script counts, derived app language label; text only with explicit opt-in |
| Publication | utterance ID, interrupt epoch, transcript hash, enqueue timestamp |
| Dispatcher | utterance ID, transcript hash, dequeue timestamp, accepted/cancelled outcome |

Existing hooks cover speech start/end, frame count, queue category/depth, provider
start/end/status, interruption ID, and dispatcher text receipt. Missing fields are
endpoint decision inputs, utterance ID at capture logs, WAV/sample hash and byte
count, tail energy, queue wait as a direct field, response script/hash, transcript
publication timestamp/hash, and dispatcher ID/hash. No production instrumentation
was added in this investigation.

## Authorized WAV procedure

Active JARVIS does not currently save the exact submitted WAV. It can do so safely
only as an explicit, single-utterance diagnostic: opt-in arm, random/nonsemantic
filename under `D:\JARVIS\scratch`, restrictive local permissions, size/duration
limit, immediate disarm, and deletion after review. Implementing that requires a
temporary production capture hook and was prohibited here.

If the user independently places an explicitly authorized 16kHz mono PCM WAV in
scratch, the added harness reads it without copying or playing it:

```powershell
python scratch/stt_forensic_harness.py analyze scratch/authorized_failure.wav
```

It reports duration, format, file hash, whole energy, final 1000ms and 500ms
energy, and 100ms tail bins. Presence of voiced-looking energy is still not proof
of the intended words; listening or a labeled waveform review requires explicit
authorization. To prove the ending reached Gemini, the saved file's SHA-256 must
match the hash of the exact WAV bytes passed to `Part.from_bytes`.

## Identical-audio comparison plan

The harness refuses provider access unless `--execute-live` is supplied. No calls
were executed. With explicit authorization, run each command in a fresh process
against the same file and retain the input hash plus result hash/script counts:

```powershell
python scratch/stt_forensic_harness.py live active scratch/authorized_failure.wav --execute-live
python scratch/stt_forensic_harness.py live reference scratch/authorized_failure.wav --execute-live
python scratch/stt_forensic_harness.py live bluye scratch/authorized_failure.wav --execute-live
```

By default the harness does not print transcripts; `--show-transcript` is a
separate sensitive-data opt-in. Run repetitions because the alias and generative
decoding may vary. Record model resolved by each app if the API exposes it, HTTP
status, latency, output hash, and script counts. Active and reference compare the
same Gemini prompt. Bluye's existing prompt differs by its added no-speech clause,
so that run compares Bluye's actual Gemini path, not prompt-controlled provider
behavior. A controlled prompt comparison would require a separate experimental
request and must be labeled as such. Complete-application comparison is different:
old JARVIS and Bluye may invoke Whisper after Gemini failure, while active must
remain Gemini-only.

## Proven facts and unresolved hypotheses

Proven:

- Runtime endpoint settings are not equivalent; active waits less than old JARVIS
  and the evidenced Bluye browser path in non-semantic cases.
- Bluye's successful trace uses browser VAD, not its Python microphone capture.
- Active WAV encoding does not remove an accepted utterance's tail.
- The captured active run shows no FIFO expiry/overflow and many whole-utterance
  RMS rejections.
- Romanized output in that run originates at or before Gemini response handling;
  later text routing does not create it.

Unverified:

- Whether a reported missing suffix existed in microphone samples, survived AEC,
  was marked silent by VAD, or was present in the submitted WAV.
- Whether Hindi script was returned by Gemini in a failed case; no such event is
  present in the available active log.
- Whether old JARVIS had Silero installed when it worked.
- Whether model alias resolution or nondeterminism differs across calls/accounts.

The smallest candidate correction, contingent on an authorized trace proving the
suffix begins after active endpointing but before the old/Bluye window, is to align
endpoint timing with the proven working system. If the suffix is present in the
submitted WAV and Gemini omits it, endpoint changes are unjustified; the next test
is repeated identical-audio Gemini calls. If the whole utterance is rejected at
prefilter, diagnose that stage separately. No correction is implemented now.

## Regression-test plan

1. Deterministic frame sequences with pauses at 210, 240, 330, 360, 480, 510,
   650, and 710ms; assert whether one or two finals emerge and that all voiced
   suffix samples remain in the intended final.
2. Repeat for short (<0.9s), semantic-window (0.9–3.5s), long (>3.5s), and
   maximum-length utterances.
3. Verify AEC/half-duplex discard boundaries independently from endpointing.
4. Assert assembled PCM hash equals decoded provider WAV PCM hash and validate
   final 1000/500ms samples byte-for-byte.
5. Exercise prefilter pass/reject without changing samples; cover quiet suffixes,
   trailing silence, silence, and noise with labeled fixtures.
6. Retain Phase 1 FIFO order, capacity, expiry, timeout recovery, duplicate,
   interruption, stale-result, and shutdown tests.
7. Mock Gemini responses in Telugu, Devanagari, Latin/Romanized Telugu, and mixed
   scripts; assert response hash/text is unchanged through publication and
   dispatcher, while only the metadata label changes.
8. With explicit authorization, replay the same real failure WAV repeatedly to
   each Gemini path, then separately test full reference/Bluye fallback behavior.
