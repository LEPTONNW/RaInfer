# Persistent worker protocol

The public runtime entry point is:

```text
mobile_runner --serve TARGET.pte CPU_BODY.bin DRAFT_MANIFEST.json IO_CONTRACT.json
```

The qualified target exposes `kv_forward`, four input rows, 34 input tensors,
35 output tensors including four feature outputs, and a fixed 2K context.
The manifest must identify `mobile-dflash2-derived`, the matching architecture,
tokenizer and companion files. Runtime shape checks reject incompatible tensors.
File digests are supplied in reference metadata; the integrator must verify files
before invoking the runner. The runner is not a cryptographic model installer.

## Input and output

Use stdin/stdout pipes. Each request is one JSON line, at most 65,536 characters.
This is not an HTTP/OpenAI-compatible server. The process emits protocol records
prefixed with `@DFLASH2 `; other native libraries may write separate diagnostics.
Filter records by this prefix and preserve stderr for failures.

```json
{"id":1,"tokens":[2,105,2364,107,19102,7726,24370,12048,75818,118692,11973,230115,237456,21503,144211,117637,242332,236761,106,107,105,4368,107],"max_new_tokens":96,"cpu_workers":4,"body_workers":1,"fast_argmax":true,"compact_logits":true,"ngram":true,"gpu_head":false}
```

The token list above illustrates the wire format only. Generate actual prompt
IDs with the exact Gemma tokenizer and chat template, including correct BOS,
turn boundaries, assistant generation prefix and thinking settings. Raw user
text or an unrelated model's token IDs are invalid substitutes. Decode response
IDs with the same tokenizer. Token IDs must be in `[0,262144)`.

Requests are sequential. Model weights remain resident, but each request resets
KV/draft state; a conversation must be re-encoded by the host. The current worker
returns the complete result at once rather than incremental token events.

| Field | Recommended value / behavior |
|---|---|
| `id` | unsigned request identifier |
| `tokens` | nonempty prompt token array, within compiled capacity |
| `max_new_tokens` | positive generation budget, at most 2,048 and bounded by remaining capacity |
| `cpu_workers` | 4, CPU output-head pool; use 1..4 |
| `body_workers` | 1; increasing to four did not help the qualified device |
| `fast_argmax` | true, preserve first-index tie behavior |
| `compact_logits` | true, avoid materializing inactive draft logits |
| `full_vocabulary` | false; optional slower draft diagnostic |
| `ngram` | true in the app baseline; 19.1 t/s reference sample used false |
| `gpu_head` | false; Vulkan draft head is experimental |
| `gpu_check` | false; optional first-round full draft-head CPU/GPU parity check |
| `fixed_affinity` | false; avoid experimental fixed worker placement |
| `burst` | false; no default burst-power experiment |
| `profile` | false; optional overlapping host spans, not exclusive kernel times |

The runtime restricts worker activity to the performance cluster where permitted
by the SM8450 Android cpuset and reports an error on affinity failures. It cannot
override OS restrictions or guarantee all four cores remain available.

Example event envelopes (values abbreviated):

```json
{"event":"ready","protocol":1,"context":2048,"load_seconds":1.0}
{"event":"result","id":1,"report":{"tokens":[123],"accepted_draft_tokens":0,"proposed_draft_tokens":0}}
{"event":"error","id":1,"message":"Invalid request budget or compiled prompt capacity"}
```

Every envelope is preceded by `@DFLASH2 ` in stdout. The real report includes
timing, backend, CPU worker, n-gram and verification statistics. These envelopes
are format examples, not measured performance records.

Shutdown request:

```json
{"id":2,"command":"shutdown"}
```

EOF also exits the loop. There is no mid-request cancel command or concurrent
request scheduler. A host app can terminate the owned worker process to cancel,
then launch a new worker for subsequent requests. Native exceptions abort staged
target state before a recoverable request error is reported.

## Experimental image bundles

A vision-enabled PTE and matching IO contract add `vision_bundle_protocol: 1`
to the ready record. A text-only contract reports `0`. Supply `vision_bundle`
instead of `tokens` to load a locally encoded image/prompt bundle. Reports add
image-row injection counters and separately recorded encoder timing. The
experimental format and limited device checks are documented in [VISION.md](VISION.md).
Normal text requests retain the existing behavior. A ready flag reports the
selected contract; it is not a certification of the model's image accuracy.

## Matching outputs

The verifier uses greedy target selection. Accept only a consecutive prefix of
matching draft tokens. The token at the first mismatch comes from the target.
Only processed prefix rows are committed; the final emitted token is pending
until the next round. EOS IDs currently include `1` and `106` for this model.

Acceptance rate is `accepted_draft_tokens / proposed_draft_tokens` when the
denominator is positive. It is not a model-quality score. A wider block can lower
acceptance and increase verification cost; changing a JSON field cannot resize
a fixed-shape QNN graph.
