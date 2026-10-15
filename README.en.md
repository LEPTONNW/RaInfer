# RaInfer

[한국어](README.md) | [English](README.en.md)

**Android inference engine with CPU drafting and Qualcomm NPU block verification**

RaInfer is a C++17 mobile inference engine that connects a small draft model to a target model capable of verifying multiple proposed tokens in one pass. The currently validated configuration is **Gemma 4 E2B with a DFlash2-derived drafter trained for that specific NPU target**.

The engine has been tested on a Galaxy S22 Ultra with Snapdragon 8 Gen 1 (SM8450). The CPU generates draft candidates while reusing cached computations, and the Hexagon v69 NPU executes the target model through the ExecuTorch Qualcomm backend and QNN/QAIRT. An experimental Vulkan GPU draft output head is also implemented, but the selected deployment baseline is **CPU + NPU with the GPU draft head disabled**.

> This is an **engine source release**. Model weights, training data, the Qualcomm SDK/runtime, and APKs are not included. A fresh clone can run the host contract tests. Actual inference on a phone requires matching model assets and separately prepared dependencies as described below.

Model/DFlash2 assets and bilingual usage instructions are published separately as
[Gemma4-E2B-Dflash2-Android](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android).
**mmproj is not reuploaded.** Download Unsloth's GGUF conversion of Google's
vision weights [from its upstream provider](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/0314792d7f1f7e229411f620751375812bb9faf2/mmproj-F16.gguf);
the model package records its pinned revision and SHA256.

**Start with the [required model/DFlash2 files and startup guide](docs/MODEL_FILES.en.md).** It lists the target PTE, IO contract, six draft companion files and tokenizer by exact filename.

**To connect your own app, read the [app-independent integration guide](docs/INTEGRATION.md) (Korean).** Integration uses a JSONL process interface and does not require a particular chat app. Instead of a UI or application distribution, the repository provides a [Kotlin process adapter example](examples/android/RaInferProcess.kt).

**Image-input experiment:** an [experimental bridge](docs/VISION.md) runs a
separate CPU mmproj encoder and forwards projected image features to the NPU
target. S22 checks passed for red/blue answers, target-only versus DFlash2 output,
Java/Korean/English text regressions and zero native-logit error. This is limited
validation with two synthetic images, not general vision-quality certification.
The existing release APK is unchanged.

## What is this engine for?

Running a relatively large target model one token at a time on a mobile device repeatedly incurs the cost of processing its weights and execution graphs. RaInfer aims to confirm several tokens per target execution by verifying a batch of proposals from a smaller drafter.

Primary uses include:

- Integrating on-device text generation into an Android app.
- Running a specific NPU target alongside a drafter trained for that target's intermediate features.
- Measuring draft acceptance, target verification time, and CPU drafting costs separately.
- Researching mobile speculative decoding with previously converted and validated models.

The project does not provide a general-purpose LLM server, HTTP API server, chat UI, or universal model converter. The host app handles tokenization, chat templates, asset installation, process lifecycle, and the user interface. RaInfer performs generation using token IDs.

## Current support

| Item | Validated configuration |
|---|---|
| Phone | Samsung Galaxy S22 Ultra, Snapdragon 8 Gen 1 / SM8450 |
| NPU | Qualcomm Hexagon HTP v69 |
| Native ABI | Android `arm64-v8a` |
| Page size | 4 KiB on the tested device; other page sizes are unvalidated |
| Target | Text decoder converted from Google's Gemma 4 E2B instruction-tuned model |
| Target precision | Dedicated QNN graph with mixed W4/W8-class weights and 16-bit activations/KV |
| Compiled context | **2,048 tokens**; the original model's maximum context is not available automatically |
| Verification block | **4 input rows: 1 pending token + 3 draft candidates** |
| Feature outputs | Taps `[3, 13, 23, 33]` as recorded in the model contract; four features of width 1,536 |
| Drafter | Locally trained DFlash2-derived architecture, hidden size 384 / 3 layers / context 512 |
| Draft execution | CPU-cached FP32 body + dynamic INT8 output head + native selector |
| Sampling | **Greedy / temperature=0** |
| SDK | QAIRT `2.37.0.250724` |
| ExecuTorch | Installation based on `ca56e1ffdba1c902d9202021d7297982d3501439` |

Support is not claimed for every Gemma 4 size, Qwen, generic GGUF files, or arbitrary public DFlash/MTP weights. Other Snapdragon generations and NPUs require separate graph conversion and validation. The static-image experiment above is separate from the existing text path; audio and MTP are not provided.

### Which acceleration model does it use?

The accelerator is a **DFlash2-derived draft model that proposes candidates for the target to verify**. It does not replace the target as the authority for output tokens. Its operations are based on the public Z Lab DFlash2 implementation, with separate training to match the features and outputs of this Gemma NPU target.

You cannot connect another project's Gemma DFlash or MTP weights simply by renaming files. Intermediate target features, tokenizer, quantization, drafter dimensions, and cache contracts must match. This repository does not include trained weights or the complete training/conversion pipeline.

## Execution flow

```text
App / host: apply chat template -> tokenize
                 |
                 v
RaInfer persistent worker
  1. Process the prompt and collect features with the NPU target
  2. Generate 3 candidates with the CPU DFlash2 drafter
  3. Verify pending token + candidates in one 4-row NPU pass
  4. Accept only the consecutive prefix matching the target
  5. At the first mismatch, continue with the target's chosen token
  6. Commit only confirmed KV/features, then repeat
                 |
                 v
Token IDs, acceptance statistics, timings -> host decoding
```

Incorrect draft candidates are not directly committed as answers. Only the confirmed prefix is retained in the KV cache, and the final emitted token remains pending for the next round. Contract tests cover EOS handling, output limits, and rollback on errors.

Drafting and verification have data dependencies. The CPU, GPU, and NPU do not continuously execute the entire model simultaneously. Verification is relative to the **quantized NPU target's greedy output**, not a guarantee of identical logits to the original BF16 model.

## Implemented features

- A persistent worker that keeps models loaded between requests.
- Reuse of feature projections and context caches in the CPU drafter.
- ARM NEON and a dynamic INT8 draft output head.
- Up to four CPU workers; one draft-body worker by default.
- First-index-preserving U16 argmax and compact logits over selected draft vocabulary.
- Adaptive n-gram candidate reuse based only on already committed tokens.
- Target block verification, prefix acceptance, and KV/feature commit and rollback.
- An experimental draft output head using Vulkan 1.1 integer operations.
- Per-request draft/verification timings, acceptance counts, and generated token IDs.

The selected-vocabulary optimization affects draft proposals only. It does not skip full-vocabulary target verification. The n-gram optimization is separate from a trained large MoE model's PLE architecture: it reuses repeated patterns in tokens already confirmed during generation.

## Getting started from source

### 1. Clone and run host tests

```bash
git clone https://github.com/LEPTONNW/RaInfer.git
cd RaInfer
cmake -S . -B build/host -DBUILD_TESTING=ON
cmake --build build/host --config Release --parallel 4
ctest --test-dir build/host -C Release --output-on-failure
python tools/check_source_package.py
```

Host tests require a C++17 compiler and CMake 3.22 or later. Python example checks use Python 3.11 or later. These tests validate CPU-side logic such as token commits, cache contracts, and selection; they do not simulate NPU inference on a PC.

### 2. Build the Android executables

Prepare the Android NDK, Vulkan shader tools, the pinned ExecuTorch source and Android installation, and the matching QNN SDK separately. **All dependency paths are explicit**; the build does not automatically download SDKs or models.

```powershell
.\tools\build_android.ps1 `
  -Install 'D:\deps\executorch\build-android' `
  -ExecuTorchSource 'D:\deps\executorch' `
  -QnnSdk 'D:\deps\qairt\2.37.0.250724' `
  -Ndk "$env:LOCALAPPDATA\Android\Sdk\ndk\29.0.14206865" `
  -CMake 'cmake' -Ninja 'ninja'
```

Executables are generated in `build/android-release/`. The public build configuration was checked with NDK 29 on a Windows host. Your external installation must provide compatible CMake exports and libraries. See the [build documentation](docs/BUILD.md) for detailed dependencies and a direct CMake command.

### 3. Prepare matching assets

Prepare the following files on the device. **Do not mix an unrelated target PTE with arbitrary draft files.**

```text
assets/
  gemma4-e2b-dflash2-verify-2k.pte  # AR4 target with intermediate feature outputs
  io-contract.json                # actual target dtype/scale/zero-point values
  draft-manifest.json             # matching draft model/file/hash contract
  cpu-body-fp32.bin
  embedding-cpu-fp32.pte
  head-cpu-int8.bin
  selector-cpu-fp32.bin
  active-vocabulary.u32
```

`examples/draft-manifest.reference.json` and `examples/io-contract.reference.json` are reference metadata for the validated configuration. They are not weights and cannot be reused unchanged for a different model. File names and digests are listed in the [reference asset inventory](examples/reference-assets.json).

The [original Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it) safetensors or a generic NPU PTE do not automatically provide the four intermediate features and block-verification interface required by this runner. This source release does not provide a complete process to regenerate all matching assets from the original model.

### 4. Run the worker on a phone

Deploy the executable, compatible ExecuTorch QNN backend, shared libraries required by your build, properly obtained QNN v69 runtime, and model assets into your Android app or ADB test environment. For an app, package the executable in its APK and execute it from the installed native library directory. A typical ADB test invocation looks like this:

```sh
# Android shell: paths are examples; place your own dependencies there first.
export LD_LIBRARY_PATH=/data/local/tmp/rainfer/lib:/vendor/lib64
export ADSP_LIBRARY_PATH='/data/local/tmp/rainfer/lib;/vendor/lib/rfsa/adsp;/vendor/dsp'
/data/local/tmp/rainfer/mobile_runner --serve \
  /data/local/tmp/rainfer/assets/gemma4-e2b-dflash2-verify-2k.pte \
  /data/local/tmp/rainfer/assets/cpu-body-fp32.bin \
  /data/local/tmp/rainfer/assets/draft-manifest.json \
  /data/local/tmp/rainfer/assets/io-contract.json
```

After startup, send one JSON request per line through standard input. Prompts must be encoded with the exact Gemma tokenizer and chat template. The optional `make_request.py` helper accepts either local Hugging Face tokenizer assets or an existing whitespace-separated token ID file.

```sh
# Optional: a Gemma-4-compatible transformers installation is required only for text tokenization/decoding.
python tools/make_request.py --tokenizer-dir /path/to/local/gemma-tokenizer \
  --prompt "Java에서 두 정수를 곱하는 짧은 예제를 써줘." --output request.jsonl --ngram
# Or, without transformers:
python tools/make_request.py --tokens-file prompt.tokens --output request.jsonl --ngram
```

The Korean prompt above asks for a short Java example multiplying two integers. These tools do not download models or tokenizers. The token IDs in `examples/request.json` illustrate the format; replace them with your own encoded prompt.

```json
{"id":1,"tokens":[2,105,2364,107,19102,7726,24370,12048,75818,118692,11973,230115,237456,21503,144211,117637,242332,236761,106,107,105,4368,107],"max_new_tokens":96,"cpu_workers":4,"body_workers":1,"fast_argmax":true,"compact_logits":true,"ngram":true,"gpu_head":false}
```

Responses are JSON records prefixed with `@DFLASH2 `. Generation returns token IDs and statistics rather than decoded text, so the host must decode them with the same tokenizer. For a saved worker log, use `python tools/decode_response.py --tokenizer-dir /path/to/local/gemma-tokenizer --input worker.log`. Ready, result, error, and shutdown messages are described in the [protocol documentation](docs/PROTOCOL.md).

## Measured performance and limitations

The following values are actual measurements from the SM8450 S22 Ultra. `t/s` refers to generation throughput excluding prefill and model loading. The [benchmark notes](docs/BENCHMARKS.md) explain the exact counters used by the different measurement tools.

| Condition | Observed result | Interpretation |
|---|---:|---|
| CPU DFlash2 + NPU AR4, one Java sample | **19.0963 t/s** | 72 completion tokens / 3.770365 s; excludes 2.244620 s prefill; n-gram off |
| Repeated Java trials of the same configuration family | Approximately **15.7–18.0 t/s** | Varies with thermal state, trial, and measurement conditions |
| NPU AR5 experiment, pooled across three Java requests | 16.8684 t/s | Not enabled in the baseline |
| NPU AR6 experiment, pooled across three Java requests | 16.5333 t/s | Not enabled in the baseline |

**19.1 t/s is not a guaranteed average across prompts or the device's theoretical speed limit.** The goal of averaging 25–30 t/s has not been achieved. The GPU is not required to obtain the reported 19.1 t/s sample.

Token agreement was checked for identical Java, Korean, and English inputs, and a separate Java output was compiled and executed using a PC JDK. These are limited regression checks, not a general model-quality certification or a guarantee of agreement on every prompt.

The main bottleneck is NPU target verification. Diagnostics placed roughly 92–94% of total generation cost in that path, so making only the CPU drafter faster did not produce a large overall gain. Offloading the final graph's LM Head, replacing Conv2D with FC/MatMul, and increasing verification width were also tested without demonstrating a reliable advantage over the baseline.

## Repository layout

```text
include/mobile_dflash2/   Verification contracts, CPU cache/head/selector, Vulkan draft head
src/                     Persistent runner, QNN target integration, timing/diagnostic executables
tests/                   Host contract tests and numerical-comparison fixtures
tools/                   Android build, shader compilation, public example checks
examples/                Request format and matching asset metadata
docs/                    App-independent integration, build, protocol, and benchmark documentation
licenses/                Full third-party license texts
```

For compatibility with existing host integrations and models, the C++ namespace `mobile_dflash2`, executable name `mobile_runner`, and protocol prefix `@DFLASH2` are retained. The engine's official name is **RaInfer**.

## Licensing and attribution

Original RaInfer source is released under **MIT**. Third-party components retain their original licenses and copyrights.

| Component | Source / terms |
|---|---|
| RaInfer execution control, caching, integration, and tools | [MIT](LICENSE), LEPTONNW |
| DFlash2-based operations | [Z Lab DFlash](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a), MIT |
| Model execution framework | [ExecuTorch](https://github.com/pytorch/executorch), BSD-3-Clause and applicable dependency terms |
| Original target model | [Google Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it), Apache-2.0; weights not included |
| NPU SDK/runtime | Qualcomm QNN/QAIRT under a separate agreement; not included |

RaInfer does not implement a new Qualcomm SDK, HTP driver, or HMX kernel. It implements drafting, verification, caching, and host integration on top of public frameworks and a separately licensed NPU runtime. See [NOTICE.md](NOTICE.md) for detailed provenance and redistribution considerations.

When distributing your own binaries or apps, include the required notices for the linked ExecuTorch build and all third-party components. Independently check your actual SDK agreement before redistributing Qualcomm files. This project's MIT license does not replace SDK or model-weight licenses.

## Development and reproducibility

To integrate another compatible model, first define its block I/O, feature taps, tokenizer, and quantized cache contracts, then validate it with a drafter trained for those conditions. The current code checks Gemma E2B-specific dimensions and feature positions; there is no generic automatic model detection.

When proposing changes, compare identical inputs with greedy settings and report output tokens, logit error, acceptance, prefill time, and generation time separately. Record the device, SDK, compiled context, thermal state, and CPU thread count to make comparisons useful.
