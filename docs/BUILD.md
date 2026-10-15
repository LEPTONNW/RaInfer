# Building RaInfer

## Host-only tests

Use CMake 3.22+, a C++17 compiler, and the platform thread library. No weights,
Qualcomm SDK, ExecuTorch installation or Android device are needed.

```sh
cmake -S . -B build/host -DBUILD_TESTING=ON
cmake --build build/host --config Release --parallel 4
ctest --test-dir build/host -C Release --output-on-failure
python tools/check_source_package.py
```

Five CTest cases cover pending-token/prefix transactions, first-index argmax,
signed INT8 and selected-vocabulary projection, the hybrid n-gram fallback, and
image-row scaling, PLE padding and AR4 boundaries.
`preprocess_fixture` and `cached_body_fixture` are helper executables for external
numerical comparisons, not additional full-model tests.

## Android dependencies

The qualified setup used Windows, Android NDK `29.0.14206865`, arm64-v8a,
ExecuTorch revision `ca56e1ffdba1c902d9202021d7297982d3501439`, and QAIRT
`2.37.0.250724`. API 28 is the native binary build floor; actual supported-device
and application constraints are stricter. Only the SM8450 / v69 configuration has
device measurements. Other platforms are not validated by host CI.

Prepare an ExecuTorch source checkout and an **Android CMake installation** built
from the matching revision, with Qualcomm backend, portable/quantized operators,
extension module/tensor/data-loader/named-data-map support, ETDump, FlatCC and
Abseil CMake exports. Enable event tracing consistently with this runner. The
installed exported targets used in `CMakeLists.txt` must actually exist.

This repository does not bootstrap a complete ExecuTorch installation or patch an
arbitrary upstream release into the qualified local build. The revision and
required targets are compatibility requirements, not a claim that any stock
ExecuTorch install is sufficient. Incompatible headers, schema, delegate options
or compiled PTE files can fail even if the source compiles.

Obtain the SDK yourself under its applicable agreement. Point `QNN_SDK_ROOT` to
the directory containing `include/QNN/`. Do not copy the SDK into this repository.
Provide `glslc` and `spirv-val` through the NDK shader tools or explicit paths.
The optional Vulkan implementation is compiled but remains disabled in the
recommended runtime request.

## CMake command

```sh
cmake -S . -B build/android-release -G Ninja \
  -DCMAKE_TOOLCHAIN_FILE=/path/to/android-ndk/build/cmake/android.toolchain.cmake \
  -DANDROID_ABI=arm64-v8a -DANDROID_PLATFORM=android-28 \
  -DCMAKE_BUILD_TYPE=Release -DBUILD_TESTING=OFF \
  -DEXECUTORCH_INSTALL=/path/to/executorch/build-android \
  -DEXECUTORCH_SOURCE=/path/to/executorch \
  -DQNN_SDK_ROOT=/path/to/qairt/2.37.0.250724 \
  -DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH
cmake --build build/android-release --parallel 4
```

Set `JSON_INCLUDE_DIR` when `nlohmann/json.hpp` is not available under the pinned
ExecuTorch tokenizer dependency. Set `RAINFER_GLSLC` and `RAINFER_SPIRV_VAL` if the
NDK does not ship shader tools at the detected host path. Python 3.11+ is used for
shader compilation. Windows users may use `tools/build_android.ps1`; all local
dependency paths are explicit arguments.

Outputs are `mobile_runner`, `pte_probe`, `draft_probe`, and `target_capture`.
The build links the existing `qnn_executorch_backend`; it does not generate or
bundle a proprietary SDK. `op_fallback.cpp` is compiled from your own pinned
ExecuTorch checkout and retains its original copyright/license.

## Android runtime deployment

Prepare the matching QNN ExecuTorch shared backend, `libc++_shared.so`, and the
licensed QNN HTP/System/v69 stub and DSP skeleton runtime dependencies. Match
their versions to the compiled model. Use `LD_LIBRARY_PATH` for Android shared
libraries and `ADSP_LIBRARY_PATH` for DSP libraries. Ensure the DSP runtime can
actually initialize before interpreting a failed run as a model issue.

The public repository does not install or delete phone files, modify power or
thermal controls, or download SDKs. Device deployment and app packaging are the
integrator's responsibility. Packaging an executable inside an APK, process
cancellation, foreground execution, tokenization and detokenization belong to
the host application. Benchmark with the screen unlocked and record thermal
state; Android can restrict the performance-core cpuset in the background.

No conversion graph or full inference is run by host CI. The existing qualified
APK/model files were not replaced by preparing this source release.

## Optional experimental image helper

See [VISION.md](VISION.md) for the separately built llama.cpp/mtmd CPU encoder,
vision-specific PTE metadata preparation, app integration and pending device
qualification. Its successful build does not validate NPU image inference.
