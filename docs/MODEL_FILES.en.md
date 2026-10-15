# Model and DFlash2 download / startup guide

[한국어](MODEL_FILES.md) | [English](MODEL_FILES.en.md)

This guide prepares the tested **RaInfer + Gemma 4 E2B NPU target + locally
trained DFlash2-derived drafter** combination. The reference is S22 Ultra,
SM8450/Hexagon v69, Android ARM64, QAIRT `2.37.0.250724`, fixed 2K context.

## 1. Obtain the engine and model assets separately

- **Engine source:** [github.com/LEPTONNW/RaInfer](https://github.com/LEPTONNW/RaInfer)
- **Model/draft files:** [Gemma4-E2B-Dflash2-Android](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android)
- **Engine build:** [BUILD.md](BUILD.md). Prepare a matching ExecuTorch installation
  and licensed QNN runtime separately.

RaInfer is executable code; the Hugging Face repository contains model data.
The engine is distributed as C++ source and must be built for Android. This
package cannot be loaded simply by passing a model file to a generic GGUF loader.

## 2. Required target and DFlash2 files

Keep **all eight files together in `npu/`**. Links below download individual files.

| Component | File | Role |
|---|---|---|
| Gemma target | [gemma4-e2b-dflash2-vision-verify-2k.pte](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/gemma4-e2b-dflash2-vision-verify-2k.pte) | NPU verifier, about 3.36 GB; also accepts ordinary text requests |
| Target contract | [io-contract.vision.json](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/io-contract.vision.json) | Matching target IO, quantization and image contract |
| Draft body | [cpu-body-fp32.bin](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/cpu-body-fp32.bin) | FP32 CPU draft computation |
| Draft embedding | [embedding-cpu-fp32.pte](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/embedding-cpu-fp32.pte) | CPU token embeddings |
| Draft output head | [head-cpu-int8.bin](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/head-cpu-int8.bin) | CPU INT8 vocabulary projection |
| Draft selector | [selector-cpu-fp32.bin](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/selector-cpu-fp32.bin) | Candidate selection |
| Draft vocabulary | [active-vocabulary.u32](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/active-vocabulary.u32) | Active draft vocabulary |
| Draft manifest | [draft-manifest.json](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/draft-manifest.json) | Draft filenames, architecture and hashes |

The six draft files form one runtime bundle; weights are not embedded inside
RaInfer or the Gemma target. The embedding `.pte` executes on **CPU**, whereas
the Gemma target `.pte` executes on **NPU**. The target uses mixed W4/W8 weights.

## 3. Tokenizer and conversation template

Obtain `tokenizer/tokenizer.json`, `tokenizer/tokenizer_config.json`,
`tokenizer/chat_template.jinja` and `tokenizer/config.json`. Your application
formats/tokenizes prompts and decodes generated tokens. The weightless
`gemma4-e2b-tokenizer.gguf` is an additional asset for tokenization implementations
that use it.

On a PC with the Hugging Face CLI, this downloads the runtime directories,
including additional weightless GGUF metadata, totaling about 4 GB:

```sh
hf download RoundRobin237/Gemma4-E2B-Dflash2-Android --include "npu/*" --include "tokenizer/*" --local-dir ./model-assets
```

It does not download `original-bf16/` or `q4-gguf/`. BF16 is for training/conversion
and Q4_K_M GGUF is a separate llama.cpp reference; neither is required by this
NPU runtime. Verify sizes/SHA256 against
[files.json](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/blob/main/files.json).
Preserve filenames; do not mix these draft files with arbitrary targets.

## 4. Start text generation

Prepare the built `mobile_runner` and matching runtime on Android. Follow
[BUILD.md](BUILD.md) and [INTEGRATION.md](INTEGRATION.md) for shared-library/DSP
search paths and APK packaging. With `model-assets/` as the current directory
and the executable on PATH:

```sh
mobile_runner --serve npu/gemma4-e2b-dflash2-vision-verify-2k.pte npu/cpu-body-fp32.bin npu/draft-manifest.json npu/io-contract.vision.json
```

Wait for ready, then write one JSON line to that worker's stdin. Replace
`PROMPT_TOKEN_IDS` below with an actual array tokenized using the matching template:

```text
{"id":1,"tokens":PROMPT_TOKEN_IDS,"max_new_tokens":96,"cpu_workers":4,"body_workers":1,"fast_argmax":true,"compact_logits":true,"ngram":true,"gpu_head":false}
```

This is a schematic line, not valid JSON until the token array is substituted.
Parse JSON after `@DFLASH2 ` and decode `report.tokens`. N-gram is an engine
feature and needs no separate weights. See [PROTOCOL.md](PROTOCOL.md) and the
[Kotlin process adapter](../examples/android/RaInferProcess.kt) to integrate
your own application. No particular chat app is required.

## 5. Additional assets for images only

- `tokenizer/gemma4-e2b-vision-vocab.gguf`: exact tokenizer/model metadata for
  the image helper, included by the download command above. No decoder weights.
- **External mmproj:** [Unsloth mmproj-F16.gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/mmproj-F16.gguf),
  Unsloth's GGUF conversion of Google's vision weights. It is not reuploaded in
  our model repository. The pinned revision/hash are in `external-assets.json`.
- RaInfer's separately built `rainfer_vision_encoder` CPU executable.

Send the encoder's bundle path using `vision_bundle` instead of `tokens`. See
[VISION.md](VISION.md) for image formatting, encoding and execution. Text-only
use does not require mmproj or the image helper.
