# Experimental static-image input / 실험용 이미지 입력

This adds an **experimental** Gemma 4 E2B image bridge. On 2026-10-15, S22
tests passed for two synthetic color images, target-only versus DFlash2 tokens,
three text regressions and exact native U16 logits. The existing installed APK
and model were preserved. This limited suite does not certify arbitrary images,
OCR, general multimodal quality or other devices.

PC GGUF 비교에 이어 S22 NPU에서도 빨강·파랑 응답과 DFlash2 출력 일치,
텍스트 세 질문의 토큰·Logits 일치를 확인했습니다. 이미지→텍스트→다른 이미지
순서에서도 상태가 섞이지 않았습니다. 앱이나 APK는 제공하지 않습니다.

## Architecture

1. `rainfer_vision_encoder` uses llama.cpp **mtmd on CPU** to preprocess one
   static image and run its compatible Gemma 4 E2B `mmproj-F16.gguf`.
2. It writes a `rainfer-vision-v1` JSON manifest and a neighboring FP32 binary.
   This contains formatted prompt tokens, projected image vectors and positions.
3. `mobile_runner` reads the bundle. A portable ExecuTorch embedding hook merges
   the vectors into the target input before the existing QNN decoder graphs.
4. CPU DFlash2 drafting and NPU AR4 verification then use the normal transaction
   path. The draft was trained for text. The one-word image answers required
   only one post-prefill verification round and accepted 0/3 candidates; this
   does not establish the acceptance of longer image-conditioned answers.

The GPU is not used in this image path. The mmproj is not itself a QNN/NPU model.
Audio, video, other Gemma sizes, noncausal image attention and arbitrary
mmproj/target combinations are unsupported. Gemma 4 E2B's causal image handling
is required. The target remains compiled for 2,048 positions; image rows consume
that same budget. No larger-context claim follows from the original model.

## Separate assets and compatibility

The PC experiment used llama.cpp revision
`a11f57ba93797579a5d1855ee216a31f10242676` and the Unsloth Gemma 4 E2B
GGUF/mmproj revision `0314792d7f1f7e229411f620751375812bb9faf2`.

The mmproj was **not created by this project and is not redistributed in our
model package**. Google provides the [original model/vision weights](https://huggingface.co/google/gemma-4-E2B-it).
The exact tested GGUF conversion is from
[Unsloth](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/0314792d7f1f7e229411f620751375812bb9faf2/mmproj-F16.gguf);
[download the matching file upstream](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/mmproj-F16.gguf).
Verify the SHA256 below. The engine receives a local path to that external file.

| Asset | Purpose |
|---|---|
| Matching `mmproj-F16.gguf` | CPU vision encoder/projector; SHA256 `140be8d7849741f88c50757d529b84373ee8e27052cc2236855b537f4a8215fa` |
| Weightless Gemma 4 E2B GGUF | Exact tokenizer and model metadata, no decoder weights; create with `make_vision_vocab.py` |
| Vision-enabled target PTE | Original target with the portable embedding operator descriptor aliased |
| `io-contract.vision.json` | Matching target IO metadata plus experimental vision contract |
| Existing DFlash2 companions | CPU body, embedding, head, selector, vocabulary and manifest; unchanged weights |

`prepare_vision_pte.py` requires the exact source hash, refuses to overwrite
the original, and checks every other byte. It modifies only the shared portable
operator name from `quantized_decomposed::embedding_4bit.dtype_out` to
`rainfer::embedding_4bit.dtype_out`. QNN context bytes and weights do not change.
This PTE requires the new kernel; the old runtime will not load it. Merely adding
a vision flag to an old IO file does not make an old PTE support images. The
worker checks that all image rows reached both embedding boundaries.

mtmd emits projected image vectors that are already at the desired scale.
The existing QNN decoder multiplies its primary embedding by `sqrt(1536)`.
The bridge therefore supplies `image / sqrt(1536)`, while image positions use
pad token `0` in the PLE lookup. Normal text rows use the original registered
ExecuTorch dequantization kernel. With no image bundle, the hook delegates the
entire call unchanged. These semantics passed the limited physical-device suite.

## Build the optional encoder

Obtain the pinned llama.cpp source separately. The helper is a standalone
executable linked to llama/mtmd; it is not linked into the NPU worker.

```sh
cmake -S tools/vision -B build/vision-host \
  -DLLAMA_CPP_SOURCE=/path/to/llama.cpp -DCMAKE_BUILD_TYPE=Release
cmake --build build/vision-host --config Release --target rainfer_vision_encoder --parallel 4

cmake -S tools/vision -B build/vision-android -G Ninja \
  -DLLAMA_CPP_SOURCE=/path/to/llama.cpp \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DGGML_CPU_ARM_ARCH=armv8.2-a+dotprod
cmake --build build/vision-android --target rainfer_vision_encoder --parallel 4
```

The normal Android engine build also builds the new embedding kernel. Rebuild
the worker and probes using [BUILD.md](BUILD.md). Dependencies and SDK rights
remain separate. The helper uses 1..4 CPU threads and requests the performance
cluster on SM8450 when Android permits it; it does not override cpuset restrictions.

## Prepare model metadata and a prompt

`make_vision_vocab.py` requires Python, numpy and the pinned llama.cpp
`gguf-py` package. It copies GGUF metadata exactly and writes zero tensors.
`prepare_vision_pte.py` requires Python 3.11+, flatbuffers and generated
ExecuTorch `executorch_flatbuffer` schema bindings matching the source PTE.
`--schema-dir` is the parent of that Python package.

```sh
python tools/make_vision_vocab.py --gguf-python /path/to/llama.cpp/gguf-py \
  --source gemma-4-E2B-it-Q4_K_M.gguf --output gemma4-e2b-vision-vocab.gguf
python tools/prepare_vision_pte.py --schema-dir /path/to/schema-bindings \
  --source original-target.pte --expected-sha256 EXACT_SOURCE_SHA256 \
  --output experimental/gemma4-e2b-dflash2-vision-verify-2k.pte \
  --io-contract io-contract.json

python tools/make_vision_prompt.py --tokenizer-dir /path/to/local/tokenizer \
  --prompt "What is the dominant color in the image? Answer with one color word." \
  --output image-prompt.txt
```

The prompt tool uses the local Hugging Face chat template, no network access,
one `<__media__>` marker and thinking disabled. It requires a compatible
`transformers` tokenizer installation, not remote custom code. Do not manually
insert a variable number of placeholder IDs into a text-only prompt.

## Encode, then generate

All paths below are examples on the machine/device running each process.
In normal encoder mode, `--text-model` can be the weightless GGUF.

```sh
rainfer_vision_encoder --text-model gemma4-e2b-vision-vocab.gguf \
  --mmproj mmproj-F16.gguf --image red.png --prompt-file image-prompt.txt \
  --output session/image.bundle.json --threads 4 --image-max-tokens 256
```

For a PC GGUF comparison only, pass the full Q4 GGUF as `--text-model` and
add `--reference-max-new 16`. That reference does **not** execute QNN or DFlash2.
Encoder stdout is one JSON summary; diagnostics go to stderr. The manifest
records image load, preprocessing and encoding time separately. Total image
latency includes loading a fresh encoder process; those times are not generation
tokens/s. This first implementation does not keep the vision encoder resident.

Start the persistent NPU worker with the experimental PTE and matching IO:

```sh
mobile_runner --serve gemma4-e2b-dflash2-vision-verify-2k.pte \
  cpu-body-fp32.bin draft-manifest.json io-contract.vision.json
```

Wait for `ready` with `vision_bundle_protocol: 1`, then send JSONL:

```json
{"id":1,"vision_bundle":"/absolute/path/image.bundle.json","max_new_tokens":64,"cpu_workers":4,"body_workers":1,"fast_argmax":true,"compact_logits":true,"ngram":true,"gpu_head":false}
```

Supply either `tokens` or `vision_bundle`, never both. Bundle binary paths must
be basenames beside the manifest. The worker validates dimensions, finite values,
token IDs and nonoverlapping image spans. Treat all bundles as local trusted
assets and enforce your app's own storage access boundaries. The report's
`vision` object contains `image_tokens`, `primary_rows_injected` and
`ple_rows_padded`; all three must agree. `vision_encoder_timing` is carried from
the manifest, not a new measurement made by the worker.

The [Kotlin adapter](../examples/android/RaInferProcess.kt) accepts explicit
`targetFile` and `contractFile` arguments and adds `generateVision(bundle)`.
Your app must run the encoder, wait for successful exit, then invoke the worker.
Drain both processes' stderr, set deadlines and own their lifecycle. An Android
integrator may package the encoder executable as `libRaInferVisionEncoder.so`
in the extracted native-library directory, as with the worker; do not call
`System.loadLibrary` on it. Executable packaging is described in
[INTEGRATION.md](INTEGRATION.md). There is no required companion app.

## Qualification procedure and recorded results

1. Compare original text PTE and experimental PTE with identical Java,
   Korean and English token inputs: output tokens, prefill predictions and native
   U16 logits must match. `pte_probe ... IO.json --prefill-logits output.u16`
   dumps the last AR4 prefill block before commit.
2. For red/blue fixtures, encode images, then run target-only
   `pte_probe MODEL.pte BUNDLE.json 16 result.json features.f32 IO.json --vision-bundle`.
   Check both image boundaries were populated and that answers depend on the image.
3. Run the same bundles with DFlash2. Target-only and speculative output tokens
   must match under the same greedy policy. Use `tools/compare_vision_reports.py`.
4. Check request isolation by sending image, text, then a different image through
   one persistent worker. A previous image must not leak into a later request.
5. Record image encode, target prefill, decode, acceptance and thermal/foreground
   state separately. Report measured device data only. Recheck selected text
   speed against the same build/environment; historical 19.1 t/s is not a vision
   performance claim.

The original installed model and APK were preserved; tests used a separate
scratch model. Only the portable operator descriptor was toggled in that scratch
file between baseline and image modes, with full SHA256 checks after each change.

Recorded results are in [vision-sm8450-20261015.json](benchmarks/vision-sm8450-20261015.json).
The three text cases produced identical output/prefill predictions and zero
maximum/mean native U16 error across each captured last-prefill AR4 logit block.
The generated Java example compiled and printed the product `12`. Red/blue
answers were `Red`/`Blue`; each image injected 64 primary rows and 64 PLE pad rows.
Target-only and DFlash2 tokens matched, including a repeated red request after
an intervening Java and blue request. The Java request reported zero image rows.

| Device measurement | Result |
|---|---:|
| Java target-only, original PTE | 6.9460 native decode t/s |
| Same Java, vision-enabled PTE, target-only | 6.9749 native decode t/s |
| Same Java, vision-enabled PTE + CPU DFlash2 | 15.5555 native decode t/s; 83/183 candidates accepted |
| Red/blue CPU image encoding | 9.4474 / 9.4604 s |
| Red/blue encoder loading | 3.2702 / 3.1627 s |
| Red/blue NPU prefill, target-only | 3.4431 / 3.4776 s |

The Java answer had 145 tokens including EOS, with 144 counted in native decode
throughput. Four workers were requested, but CPU availability changed across
state checks: **4–5** initially, **4–5,7** around target-only runs, **4,6–7**
before the persistent run and **4–7** after it. These snapshots do not guarantee
four continuously available cores. Thermal status before the persistent run was
`2`. Image encoding
used four requested CPU threads, no GPU. These shell measurements are distinct
from the historical app-level 19.1 t/s example. Two-token color answers are not a
meaningful long-answer throughput benchmark. All data are single-run records;
thermal/scheduling variation and the narrow synthetic fixture scope apply.

## Licensing

RaInfer integration is MIT. llama.cpp/mtmd is MIT and is credited in
[NOTICE.md](../NOTICE.md). The helper links additional upstream components;
binary distributors must collect notices from their actual dependency build.
Model assets retain their own licenses. No SDK, model, app or runtime binary is
distributed in this source repository.
