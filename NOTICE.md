# RaInfer — attribution and licensing

RaInfer's original engine control, cache transactions, native integration,
optimization code and public packaging are Copyright (c) 2026 LEPTONNW and are
licensed under the MIT License in [LICENSE](LICENSE). Original third-party
copyrights and licenses are not replaced by that license.

## DFlash / DFlash2

The associated DFlash2-derived training implementation uses grouped dynamic
convolution and candidate-selection ideas/code adapted from Z Lab DFlash:

- https://github.com/z-lab/dflash
- Reference revision: `07ebd93db9f472af339b644bb70221ad8428328a`
- Reference file: `dflash/model.py`
- Copyright (c) 2026 Z Lab, MIT.
- Full notice: [licenses/DFlash-MIT.txt](licenses/DFlash-MIT.txt).

The CPU cached body and selector here execute the associated exported model.
RaInfer's wrapper, block validation and cache management are local implementations.
The upstream repository, its authors and its trained models are distinct from
this engine and the privately prepared, model-specific drafter. No upstream
endorsement, generic checkpoint compatibility, weights or training corpus is implied.

## ExecuTorch

Runtime dependency: https://github.com/pytorch/executorch

Qualified source revision: `ca56e1ffdba1c902d9202021d7297982d3501439`.
The build consumes an existing Android install and compiles its
`extension/llm/custom_ops/op_fallback.cpp` directly from the external checkout.
That file and linked libraries retain upstream copyrights and BSD terms.
Full notice: [licenses/ExecuTorch-BSD-3-Clause.txt](licenses/ExecuTorch-BSD-3-Clause.txt).
This public source package does not vendor that checkout or distribute binaries.
An integrator distributing binaries must also collect notices for all linked
third-party components from the actual ExecuTorch build, including relevant
tokenizer, Abseil, FlatBuffers, JSON and other dependencies.

## llama.cpp / mtmd (optional experimental image helper)

Source: https://github.com/ggml-org/llama.cpp

PC reference revision: `a11f57ba93797579a5d1855ee216a31f10242676`.
The separately built `rainfer_vision_encoder` links llama.cpp and mtmd from an
external source checkout. They retain the upstream MIT license; a copy is at
[licenses/llama.cpp-MIT.txt](licenses/llama.cpp-MIT.txt). This repository does not
vendor that source, model weights or generated binaries. Binary distributors
must also preserve notices for components linked from the actual checkout,
including its JSON, image-decoding and GGML dependencies. See [VISION.md](docs/VISION.md)
for the experimental bridge's validation limits.

## Gemma 4

Reference model: https://huggingface.co/google/gemma-4-E2B-it

Author: Google. Gemma 4 uses Apache-2.0; earlier Gemma-specific terms should not
be assumed to be the license for this model. License reference:
https://ai.google.dev/gemma/apache_2

A copy is provided at [licenses/Gemma4-Apache-2.0.txt](licenses/Gemma4-Apache-2.0.txt).
No Google model weights, tokenizer assets or compiled model are distributed here.
JSON examples describe input/output and companion-file contracts only.
Conversion, mixed-precision quantization, fixed-context compilation, feature
outputs and associated drafter training are LEPTONNW modifications/preparations;
they are not an official Google/Qualcomm model release.

## Qualcomm QNN / QAIRT

Qualified version: `2.37.0.250724`, SM8450 / HTP v69.

The actual NPU computation uses ExecuTorch's Qualcomm backend and Qualcomm's
proprietary SDK/runtime. RaInfer does not implement or relicense HTP drivers,
HMX kernels, SDK headers, DSP skeletons or QNN shared libraries. None of those
files is included in this source package. Obtain and use them under the
applicable Qualcomm agreement; independently verify redistribution rights before
shipping an application or runtime bundle. Downloading a runtime separately does
not by itself establish redistribution permission.

## Public snapshot

This repository packages the qualified engine runtime source into a separate,
portable-path build. It excludes workstation orchestration scripts, conversion
workspaces, captured data, SDKs, model files and generated binaries. CMake and
build tools are modified to accept explicit dependency paths and build the core
runner without the local profiling-backend copy pipeline. Existing namespace,
executable and protocol identifiers remain for compatibility. It does not include
the complete private training/export pipeline or the Android host application.
