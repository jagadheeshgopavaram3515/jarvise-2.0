# STT regression fix assessment — 2026-10-03

## Outcome

No production correction was applied because the available evidence does not yet
identify the stage that removed a real failed utterance's ending. Applying an
endpoint or filter change now would be speculative and would conflict with the
instruction to avoid production changes while root cause remains unproven.

The endpoint difference is real, but it is not a source-code regression. Active
JARVIS's tracked defaults in `assistant/config.py` are already the same as the
read-only reference: 500ms base silence, 1300ms maximum, and 180ms growth per
second. The active runtime's shorter 350/700/75 values come from the local ignored
`.env`. That file is not tracked, has no Git history from which intent can be
established, and was explicitly excluded from modification by this task.

The Gemini-only provider configuration, agentic orchestration, bounded FIFO,
expiry, one-request constraint, interruption generations, and stale-result
protections remain unchanged. No live Gemini call was made.

## Proven facts

- Active runtime configuration is behaviorally less tolerant of pauses than old
  JARVIS outside the shared semantic-shortcut window.
- Both source trees default to 500/1300/180 and share the 240ms semantic shortcut.
- Both active and old `.env` files enable semantic endpointing. For buffered
  utterances between 900ms and 3500ms, both can endpoint after eight 30ms silent
  frames (240ms). That shared path cannot explain a difference between them.
- For a six-second non-semantic/long utterance, active runtime configuration
  reaches its 700ms cap and endpoints after 23 frames (690ms). The reference
  reaches its 1300ms cap and endpoints after 43 frames (1290ms). A continuation
  beginning between those boundaries would be split by active JARVIS and retained
  by the reference.
- `_emit` retains preroll and trailing silence. WAV encoding preserves every
  accepted float sample; it does not trim the suffix.
- The inspected active log contains 24 RMS `no_speech` rejections in its initial
  diagnostic interval and later many more low-energy segments. It does not retain
  audio, so those segments cannot be labeled as speech or noise.
- The same runtime log shows provider calls lasting roughly eleven seconds and at
  least one pending utterance expiring at eight seconds. This proves whole
  utterances can be lost under sustained provider latency, but not that a suffix
  was removed from an otherwise delivered transcript.
- Romanized Telugu returned by Gemini passes unchanged through active STT and the
  dispatcher. Endpoint alignment cannot force Telugu script or improve provider
  transcription of audio already present in the WAV.

## Still hypothetical

- VAD/endpointing cut a spoken continuation before it entered the submitted WAV.
- Whole-utterance RMS rejected actual quiet speech rather than noise, echo, or
  false VAD triggers.
- Gemini omitted an ending that was present in the submitted WAV.
- The old runtime had Silero installed and therefore admitted quiet speech that
  active's deliberately energy-only path rejects.
- Restoring Whisper would improve the reported examples rather than adding
  latency, model startup cost, inconsistent transcripts, and fallback-dependent
  behavior.

## Files and settings changed

No tracked production source, configuration, prompt, model, VAD threshold, FIFO,
provider, dispatcher, or orchestration file was changed in this task.

Created only this report under `scratch/`. The existing forensic harness and
diagnostic report remain available:

- `scratch/stt_forensic_harness.py`
- `scratch/active_stt_failure_diagnosis.md`

The local `.env`, reference JARVIS, Bluye, secrets, memory, logs, and persistent
data were read-only.

## Why no endpoint override was added

The smallest candidate correction would be to align the active runtime values to
500/1300/180 while preserving the 240ms semantic shortcut. However, the tracked
code already expresses those values. Adding clamps, a hidden compatibility
profile, or source constants that override explicit environment configuration
would make configuration misleading and would be a speculative production change.

The next decisive observation is one explicitly authorized failed utterance with
the exact WAV submitted to Gemini (or hashes proving identity), endpoint decision
fields, and final 1000ms energy. If the continuation starts after active's endpoint
but before the reference boundary, aligning the runtime settings is justified. If
the suffix is present in the submitted WAV, endpoint changes are wrong and the
same WAV must be replayed through Gemini. If the entire segment is rejected by RMS,
the prefilter must be investigated separately with labeled audio.

## RMS prefilter decision

The threshold remains 0.005. Existing logs show RMS values and frame-energy
measurements but no ground-truth label. Lowering the threshold would admit more
noise and Gemini hallucination opportunities without proving recovery of speech.

Missing observation: an authorized rejected segment labeled by the speaker as a
real utterance, with whole RMS, maximum frame RMS, above-threshold duration, and
the exact samples that reached the filter. A waveform/controlled playback test can
then distinguish silence dilution from false VAD activation and quiet speech.

## Whisper fallback assessment

Whisper was not restored. Old JARVIS initializes local FasterWhisper, uses it for
partials, and falls back when Gemini returns empty, errors, or exceeds its short
caller timeout. Benefits are continued operation during cloud failure and partial
barge-in. Costs are model memory/startup, local decode latency, a second transcript
policy, lower accuracy for some Telugu/code-switch speech on the configured base
model, and more concurrency/lifecycle complexity. The current active requirement
is Gemini-only with one outstanding request, and no evidence shows that a reported
bad transcript followed a Gemini failure. Restoring Whisper would not correct a
tail already removed during capture.

## Validation

Executed without network or microphone access:

```powershell
python -m pytest -q tests/test_stt_reference_contract.py tests/test_stt_migration.py tests/test_stt_fifo.py tests/test_cloud_stt_lifecycle.py tests/test_resilience.py tests/test_story_pipeline.py
git diff --check
```

Result: **109 passed, 0 failed** in 26.45 seconds. These tests cover endpoint to
Gemini to dispatcher flow, WAV/prompt contracts, quiet/silence boundaries, FIFO
ordering/capacity/expiry, timeout recovery, duplicate prevention, interruption,
stale-result rejection, shutdown, and Gemini-only behavior. No new regression test
was added because no behavior was changed and no real failure fixture exists.

`git diff --check` exited successfully with only pre-existing line-ending warnings.

## Manual validation plan

1. **Romanized Telugu:** arm a one-utterance authorized WAV capture, speak a fixed
   Romanized Telugu sentence without pauses, record expected words privately, and
   compare the submitted WAV hash, Gemini result hash/script class, and dispatcher
   hash. Repeat three times because the Gemini alias may vary.
2. **Pauses:** speak the same sentence with controlled pauses near 240ms, 350ms,
   500ms, 700ms, and 1300ms after both a two-second and six-second prefix. Record
   endpoint reason, required/final silence frames, and whether one or two utterance
   IDs were emitted.
3. **Silence/noise:** record labeled room silence, keyboard/fan noise, and one quiet
   acknowledgement. Confirm the first two are rejected and inspect whether quiet
   speech fails whole RMS despite sustained speech-like frames. Do not change the
   threshold during collection.
4. **Long utterance:** speak a 10–15 second sentence with natural thinking pauses.
   Verify the last spoken words are audible/present in the final 1000ms of the exact
   provider WAV before evaluating Gemini output.
5. If the authorized trace proves endpoint loss, change only the effective runtime
   endpoint values toward 500/1300/180, preserve the 240ms semantic shortcut, then
   rerun the same recordings plus FIFO/interruption tests. If it proves provider
   omission, run repeated identical-WAV Gemini comparisons instead.

## Unresolved risk

The active endpoint configuration remains shorter because modifying `.env` was
out of scope. The current eight-second pending-final expiry can also discard a
later utterance while an accepted Gemini request takes longer than eight seconds.
That behavior is an intentional Phase 1 bound and was preserved; changing it needs
a separate latency/availability decision, not an endpoint regression fix.
