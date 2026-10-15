# Measured results and limits

Measurements below were collected on a Galaxy S22 Ultra / SM8450, Hexagon v69,
with the Gemma E2B fixed-2K mixed-precision NPU target and matching CPU drafter.
The qualified baseline uses AR4, CPU body workers 1, CPU head workers up to 4,
compact draft logits, greedy target selection, and GPU head disabled.

## Reference Java sample

Prompt:

> Java에서 두 정수를 곱해 반환하는 static 메서드와 main에서 3과 4를 넣어 출력하는 아주 짧은 실행 가능한 예제를 코드 블록 하나로 써줘.

The application measurement recorded 70 prompt tokens, 72 completion tokens,
3,770.364686 ms generation time and 2,244.620259 ms prefill. Application-reported
generation throughput was **19.096295980955936 t/s**. It accepted 42 of 90 proposed
draft tokens in 30 verification rounds. This specific sample had n-gram disabled.
The app's existing adaptive n-gram option is enabled by default and should be
reported separately in an attempted reproduction.

This is one observed sample, not a sustained average, universal benchmark, or
hardware limit. Later Java repeated-trial groups ranged approximately 15.7–18.0
t/s under changing thermal and measurement conditions. The original Java source
output was separately compiled and executed on a PC JDK, producing `12`.

## Timing definitions

The application reports completion tokens divided by its measured generation
duration. The native runner exposes `decode_tokens_per_second`, calculated from
`committed_tokens / decode_seconds` after prefill. The first token obtained while
prefilling is present in `tokens` but is not included in native `committed_tokens`.
These two counters/timing boundaries are not interchangeable. Always retain the
raw fields and distinguish model loading, prefill, draft priming and decoding.

Native `verification_execution_seconds` wraps target execution including host
and runtime costs. Profiling spans can overlap and must not be added as exclusive
accelerator/RPC/wait times. Detailed HTP per-operator profiling was not reliably
available on the qualified device, so no invented operator timing is reported.

## Experiments not selected as defaults

| Experiment | Measured result | Decision |
|---|---|---|
| NPU final graph: Conv2D → FC | Java pooled 17.53/17.14 vs original 17.97/17.58 t/s | No reliable improvement |
| NPU final graph: MatMul | Java pooled 16.08 vs paired original 16.06 t/s | No reliable improvement |
| LM Head only, four rows, NPU | 16.24 ms standalone | Component time, not full generation |
| Same original-weight LM Head, CPU | 17.54 ms standalone | No clear offload advantage |
| Same original-weight LM Head, portable Vulkan | 95.93 ms standalone | Slower on this path/device |
| AR5 Java, three requests pooled | 16.8684 t/s; 44/112 accepted per Java request | Not selected |
| AR6 Java, three requests pooled | 16.5333 t/s; 44/140 accepted per Java request | Not selected |

The split LM Head CPU/GPU test had maximum absolute logit error 0.0242594 against
QNN and matching argmax at 12/12 captured rows. FC/MatMul full U16 logits matched
exactly on checked captures. These are different validation scopes. Component
times cannot simply be added and converted into measured end-to-end t/s.

Separate CPU/GPU-only GGUF trials use different quantization and have no DFlash
feature connection; they are not directly equivalent speculative-decoding
comparisons. No CPU/GPU-only speed is presented as a limit of all future kernels.

## Publication scope

This source repository contains a summary and reference metadata rather than
private working-directory logs, full captures, model weights or training data.
The measurements predate this public source packaging. This packaging passed
host tests and was rebuilt for Android using existing dependencies; it is not
claimed to be a new on-device performance measurement. Average 25–30 t/s has not
been demonstrated. Original installed APK and weights were preserved.
