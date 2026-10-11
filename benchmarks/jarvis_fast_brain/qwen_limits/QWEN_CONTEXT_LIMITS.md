# Qwen Context Limits

## Important distinction

Ollama `/api/tags` reported model metadata context length **32768**. After loading, `/api/ps` reported active context length **4096**. The active JARVIS runtime setting is therefore not 32K merely because the model metadata supports 32K.

## Probe

The probe repeated a meaningful marker and asked for exact JSON. It is a capacity/latency probe, not a full semantic degradation benchmark.

| Requested prompt target | Prompt chars | Status | Prompt tokens reported | Wall ms | Marker correct |
|---:|---:|---|---:|---:|---|
| 1K | 983 | success | 321 | 16,124.2 | YES |
| 2K | 1,886 | success | 594 | 3,374.3 | YES |
| 4K | 3,735 | success | 1,153 | 3,923.6 | YES |
| 8K | 7,390 | success | 2,258 | 5,439.9 | YES |
| 12K | 11,088 | success | 3,376 | 5,597.9 | YES |
| 16K | 14,743 | success | 2,050 | 7,474.3 | YES |
| 24K | 22,096 | success | 2,050 | 2,458.7 | YES |
| 32K | 29,406 | success | 2,050 | 2,606.3 | YES |

The non-monotonic times and repeated 2050-token counts at 16K–32K indicate truncation, capping, caching, or another runtime behavior. They are not evidence that the model used a full 32K prompt. No semantic distractor/recall matrix was completed, so the degradation threshold is **INSUFFICIENT EVIDENCE**.

## Practical conclusion

For current JARVIS, selectively bounded context is safer than sending maximum context. The normal target should be derived from actual active runtime configuration and task success, not the model card metadata. Full 32K should not be assumed usable until a controlled runtime setting, prompt-token verification, memory measurement, and semantic recall benchmark all pass.
