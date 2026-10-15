# RaInfer

[한국어](README.md) | [English](README.en.md)

**Android용 CPU 초안 생성 + Qualcomm NPU 블록 검증 추론 엔진**

RaInfer는 작은 초안 모델이 다음 토큰 후보를 만들고, 실제 답변을 담당하는 모델이 여러 후보를 한 번에 검증하도록 연결하는 C++17 모바일 추론 엔진입니다. 현재 검증된 구성은 **Gemma 4 E2B + 해당 NPU 모델에 맞춰 학습한 DFlash2 파생 초안 모델**입니다.

Galaxy S22 Ultra의 Snapdragon 8 Gen 1(SM8450)에서 동작을 검증했습니다. CPU는 캐시를 재사용하면서 초안을 생성하고, Hexagon v69 NPU는 ExecuTorch Qualcomm backend와 QNN/QAIRT를 통해 본체 모델을 실행합니다. Vulkan GPU 초안 출력 헤드도 실험적으로 구현되어 있지만, 현재 배포 기준은 **GPU를 끈 CPU + NPU 구성**입니다.

> 이 저장소는 **엔진 소스 공개본**입니다. 모델 가중치, 학습 데이터, Qualcomm SDK·런타임, APK는 포함하지 않습니다. 처음 복제한 상태에서는 호스트 계약 테스트를 실행할 수 있습니다. 휴대폰의 실제 추론에는 아래에 설명한 서로 호환되는 모델 자산과 별도로 준비한 의존성이 필요합니다.

모델·DFlash2 자산과 한·영 사용 설명은 별도 [Gemma4-E2B-Dflash2-Android](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android)에 공개합니다. **mmproj는 재업로드하지 않습니다.** Google 원본 비전 가중치의 Unsloth GGUF 변환본을 [원 제공처](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/blob/0314792d7f1f7e229411f620751375812bb9faf2/mmproj-F16.gguf)에서 받도록 안내하고 고정 리비전·SHA256을 기록했습니다.

**먼저 [모델·DFlash2 필수 파일과 시작 방법](docs/MODEL_FILES.md)을 확인하세요.** 본체 PTE 1개, 입출력 규격 1개, 초안 파일 6개와 토크나이저를 정확한 파일명으로 안내합니다.

**자신의 앱에 연결하려면 [앱 독립 연동 가이드](docs/INTEGRATION.md)를 읽어보세요.** 특정 채팅 앱 설치 없이 JSONL 프로세스 인터페이스로 연결합니다. UI·앱 배포본 대신 [Kotlin 프로세스 어댑터 예제](examples/android/RaInferProcess.kt)를 제공합니다.

**이미지 입력 실험:** 별도 mmproj CPU 인코더의 이미지 특징을 NPU 본체에 전달하는 [경로](docs/VISION.md)를 추가했습니다. S22에서 빨강·파랑 응답, DFlash2 출력 일치, Java·한국어·영어 텍스트 회귀와 Logits 오차 0을 확인했습니다. 합성 이미지 두 장을 사용한 제한된 검증이며 일반 비전 품질을 보증하지 않습니다. 기존 APK는 변경하지 않았습니다.

## 어떤 용도의 엔진인가요?

모바일에서 비교적 큰 본체 모델을 한 토큰씩 실행하면, 매번 모델 가중치와 실행 그래프를 처리하는 비용이 반복됩니다. RaInfer는 작은 초안 모델의 제안을 본체가 묶어서 확인하여, 한 번의 본체 실행으로 여러 토큰을 확정하는 것을 목표로 합니다.

주요 용도는 다음과 같습니다.

- Android 앱에 온디바이스 텍스트 생성 기능을 연결합니다.
- 특정 NPU 본체 모델과 그 본체의 중간 특징에 맞춰 학습한 초안 모델을 함께 실행합니다.
- 초안 채택률, 본체 검증 시간, CPU 초안 비용을 분리해서 측정합니다.
- 이미 변환·검증된 모델을 사용하여 모바일 speculative decoding을 연구합니다.

일반 LLM 서버, HTTP API 서버, 채팅 UI 또는 범용 모델 변환기를 제공하는 프로젝트는 아닙니다. 앱은 토크나이저·대화 템플릿·파일 설치·프로세스 수명·사용자 인터페이스를 담당하고, RaInfer는 토큰 ID 기반 생성 작업을 수행합니다.

## 현재 지원 범위

| 항목 | 검증된 구성 |
|---|---|
| 휴대폰 | Samsung Galaxy S22 Ultra, Snapdragon 8 Gen 1 / SM8450 |
| NPU | Qualcomm Hexagon HTP v69 |
| 네이티브 ABI | Android `arm64-v8a` |
| 페이지 크기 | 검증 기기 4 KiB; 다른 페이지 크기 미검증 |
| 본체 | Google Gemma 4 E2B instruction-tuned 모델에서 변환한 텍스트 디코더 |
| 본체 정밀도 | 혼합 W4/W8 계열 가중치, 16비트 활성값·KV를 사용하는 전용 QNN 그래프 |
| 컴파일 문맥 | **2,048토큰**. 원본 모델의 최대 문맥을 그대로 사용할 수 없음 |
| 검증 블록 | **4개 입력 행: pending 토큰 1개 + 초안 3개** |
| 특징 출력 | 모델 계약에 기록된 `[3, 13, 23, 33]` 탭, 폭 1,536의 특징 4개 |
| 초안 | 자체 학습한 DFlash2 파생 구조, hidden 384 / 3 layers / 문맥 512 |
| 초안 실행 | CPU 캐시 본체 FP32 + 동적 INT8 출력 헤드 + 네이티브 선택기 |
| 샘플링 | **greedy / temperature=0** |
| SDK | QAIRT `2.37.0.250724` |
| ExecuTorch | `ca56e1ffdba1c902d9202021d7297982d3501439` 기반 설치본 |

전체 Gemma 4 크기, Qwen, 일반 GGUF, 임의의 공개 DFlash/MTP 가중치를 지원한다고 주장하지 않습니다. 다른 Snapdragon 세대와 다른 NPU도 별도의 그래프 변환 및 검증이 필요합니다. 텍스트 경로 외에 위 정지 이미지 실험 경로를 별도로 제공합니다. 오디오·MTP는 제공하지 않습니다.

### 어떤 가속 모델을 사용하나요?

RaInfer의 가속 모델은 본체 모델의 다음 토큰을 대신 확정하는 모델이 아니라, **본체가 확인할 후보를 만드는 DFlash2 파생 초안 모델**입니다. 공개 Z Lab DFlash2 코드의 연산 구조를 참고하고, 해당 Gemma NPU 본체의 특징과 출력에 맞춰 별도로 학습했습니다.

따라서 다른 프로젝트에서 받은 Gemma용 DFlash 가중치나 MTP 가중치를 파일명만 바꾸어 연결할 수 없습니다. 본체의 중간 특징, 토크나이저, 양자화, 초안 크기, 캐시 계약이 맞아야 합니다. 이 저장소에는 학습 완료 가중치나 전체 학습·변환 파이프라인이 포함되지 않습니다.

## 실행 구조

```text
앱 / 호스트: 대화 템플릿 적용 → 토큰화
                 │
                 ▼
RaInfer persistent worker
  1. NPU 본체로 프롬프트 처리 및 특징 수집
  2. CPU DFlash2 초안으로 후보 3개 생성
  3. NPU 본체가 pending + 후보를 4행으로 일괄 검증
  4. 본체와 일치하는 연속 후보만 채택
  5. 불일치 위치는 본체가 선택한 토큰으로 진행
  6. 확정된 부분만 KV·특징 캐시에 반영하고 반복
                 │
                 ▼
토큰 ID · 채택 통계 · 시간 측정 → 앱에서 디코딩
```

초안에서 틀린 후보가 나와도 이를 그대로 답변으로 확정하지 않습니다. KV 캐시를 확정된 prefix만큼만 반영하고, 미확정 토큰은 다음 라운드의 pending 토큰으로 관리합니다. EOS·출력 길이 제한·오류 시 되돌리기 처리는 계약 테스트로 확인합니다.

초안과 검증에는 데이터 의존성이 있습니다. CPU·GPU·NPU가 항상 동시에 전체 모델을 실행하는 구조는 아닙니다. 검증의 기준은 **양자화된 NPU 본체의 greedy 출력**이며, 원본 BF16 모델과 동일한 Logits를 보장한다는 뜻도 아닙니다.

## 구현한 기능

- 모델을 매 요청마다 다시 읽지 않는 persistent worker.
- CPU 초안의 특징 투영과 문맥 캐시 재사용.
- ARM NEON 및 동적 INT8 초안 출력 헤드.
- 최대 4개 CPU 작업자; 기본 초안 본체 작업자 1개.
- 첫 번째 최대값을 유지하는 U16 argmax 및 선택 어휘의 compact logits.
- 이미 확정된 토큰만 사용하는 적응형 n-gram 후보 재사용.
- 본체의 블록 검증, prefix 채택, KV·특징의 확정/되돌리기.
- Vulkan 1.1 정수 연산을 이용한 실험용 초안 출력 헤드.
- 요청별 초안·검증 시간, 채택 수, 생성 토큰 ID 기록.

초안의 선택 어휘 최적화는 후보 생성에만 영향을 줍니다. 본체의 전체 어휘 검증을 생략하지 않습니다. n-gram 역시 학습된 대형 MoE의 PLE 구조가 아니라, 이번 생성에서 확정한 토큰의 반복 패턴을 재사용하는 별도 최적화입니다.

## 소스부터 실행하기

### 1. 복제하고 호스트 테스트 실행

```bash
git clone https://github.com/LEPTONNW/RaInfer.git
cd RaInfer
cmake -S . -B build/host -DBUILD_TESTING=ON
cmake --build build/host --config Release --parallel 4
ctest --test-dir build/host -C Release --output-on-failure
python tools/check_source_package.py
```

호스트 테스트는 C++17 컴파일러와 CMake 3.22 이상만 있으면 실행할 수 있습니다. Python 예제 검사는 Python 3.11 이상을 사용합니다. 이 테스트는 토큰 확정·캐시 계약·선택기 등 CPU 측 로직의 검증이며, PC에서 NPU 추론을 시뮬레이션하는 기능은 아닙니다.

### 2. Android 실행 파일 빌드

Android NDK, Vulkan shader tools, pinned ExecuTorch 소스와 Android 설치본, 해당 버전의 QNN SDK를 별도로 준비합니다. **의존성 경로는 모두 직접 지정**하며 SDK나 모델을 자동 다운로드하지 않습니다.

```powershell
.\tools\build_android.ps1 `
  -Install 'D:\deps\executorch\build-android' `
  -ExecuTorchSource 'D:\deps\executorch' `
  -QnnSdk 'D:\deps\qairt\2.37.0.250724' `
  -Ndk "$env:LOCALAPPDATA\Android\Sdk\ndk\29.0.14206865" `
  -CMake 'cmake' -Ninja 'ninja'
```

실행 파일은 `build/android-release/`에 생성됩니다. NDK 29와 Windows 호스트에서 공개용 빌드 구성을 확인했습니다. 외부 설치본의 CMake export와 라이브러리 구성이 맞아야 하므로, 자세한 의존성과 일반 CMake 명령은 [빌드 문서](docs/BUILD.md)를 확인하세요.

### 3. 호환되는 자산 준비

기기에 아래 파일들을 준비합니다. **다른 본체 PTE와 무작위 초안 파일을 섞지 마세요.**

```text
assets/
  gemma4-e2b-dflash2-verify-2k.pte  # AR4 + 중간 특징 출력이 있는 본체
  io-contract.json                # 해당 본체의 실제 dtype/scale/zero-point
  draft-manifest.json             # 해당 초안의 모델·파일·해시 계약
  cpu-body-fp32.bin
  embedding-cpu-fp32.pte
  head-cpu-int8.bin
  selector-cpu-fp32.bin
  active-vocabulary.u32
```

`examples/draft-manifest.reference.json`과 `examples/io-contract.reference.json`은 검증 구성의 메타데이터 참고본입니다. 가중치가 아니며, 다른 모델에 그대로 적용할 수 없습니다. 파일명과 해시는 [기준 자산 목록](examples/reference-assets.json)에서 확인할 수 있습니다.

[원본 Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it)의 safetensors나 일반 NPU PTE만으로는 이 실행기에 필요한 4개 중간 특징과 블록 검증 인터페이스가 자동으로 생기지 않습니다. 현재 소스 공개본만으로 원본 모델에서 동일 자산 전체를 다시 생성하는 기능은 제공하지 않습니다.

### 4. 휴대폰에서 worker 실행

실행 파일, 호환되는 ExecuTorch QNN backend, 빌드에 필요한 공유 라이브러리, 합법적으로 준비한 QNN v69 런타임과 모델 자산을 Android 앱 또는 ADB 시험 환경에 배치합니다. 앱에서는 실행 파일을 APK에 포함하고 설치된 native library 경로에서 실행합니다. 일반적인 ADB 시험 실행 형태는 다음과 같습니다.

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

시작 후 표준 입력으로 JSON 한 줄씩 요청을 전달합니다. 정확한 Gemma 토크나이저와 대화 템플릿으로 질문을 인코딩한 토큰 ID가 필요합니다. 선택 도구 `make_request.py`는 로컬 Hugging Face 토크나이저를 사용하거나, 이미 인코딩한 공백 구분 토큰 ID 파일을 받아 요청을 만듭니다.

```sh
# Optional: a Gemma-4-compatible transformers installation is required only for text tokenization/decoding.
python tools/make_request.py --tokenizer-dir /path/to/local/gemma-tokenizer \
  --prompt "Java에서 두 정수를 곱하는 짧은 예제를 써줘." --output request.jsonl --ngram
# Or, without transformers:
python tools/make_request.py --tokens-file prompt.tokens --output request.jsonl --ngram
```

이 도구는 모델이나 토크나이저를 다운로드하지 않습니다. `examples/request.json`의 토큰은 형식 설명용이므로 자신의 질문을 인코딩한 토큰으로 바꾸세요.

```json
{"id":1,"tokens":[2,105,2364,107,19102,7726,24370,12048,75818,118692,11973,230115,237456,21503,144211,117637,242332,236761,106,107,105,4368,107],"max_new_tokens":96,"cpu_workers":4,"body_workers":1,"fast_argmax":true,"compact_logits":true,"ngram":true,"gpu_head":false}
```

응답은 `@DFLASH2 ` 접두사가 있는 JSON입니다. 생성 결과는 텍스트 대신 토큰 ID와 통계로 반환되므로 호스트가 동일한 토크나이저로 디코딩해야 합니다. 로그를 저장한 경우 `python tools/decode_response.py --tokenizer-dir /path/to/local/gemma-tokenizer --input worker.log`로 결과를 확인할 수 있습니다. 준비·결과·오류 이벤트와 종료 요청은 [프로토콜 문서](docs/PROTOCOL.md)에 설명합니다.

## 실측 성능과 한계

아래 수치는 SM8450 S22 Ultra의 실제 측정 기록입니다. `t/s`는 프리필과 모델 로딩을 제외한 생성 처리량이며, 측정 도구별 세부 분모는 [벤치마크 설명](docs/BENCHMARKS.md)에 기록했습니다.

| 조건 | 관측값 | 해석 |
|---|---:|---|
| CPU DFlash2 + NPU AR4, Java 단일 측정 | **19.0963 t/s** | 생성 72토큰 / 3.770365초, 프리필 2.244620초 제외, n-gram 꺼짐 |
| 같은 계열의 Java 반복 측정 | 약 **15.7~18.0 t/s** | 발열·실행 회차·계측 조건에 따라 변동 |
| NPU AR5 검증 실험, Java 3건 합산 | 16.8684 t/s | 기본 미적용 |
| NPU AR6 검증 실험, Java 3건 합산 | 16.5333 t/s | 기본 미적용 |

**19.1 t/s는 모든 질문에서 보장되는 평균이나 장치의 이론적 최고 속도가 아닙니다.** 평균 25~30 t/s 목표는 달성하지 못했습니다. GPU를 켜야 이 수치가 나오는 것도 아닙니다.

동일 입력의 Java·한국어·영어 비교에서 토큰 일치 여부를 검사했고, 별도의 Java 출력은 PC JDK로 컴파일·실행했습니다. 이는 제한된 회귀 검증이며 전체 모델의 품질 인증이나 모든 질문의 출력 일치를 의미하지 않습니다.

주요 병목은 NPU 본체 검증입니다. 진단에서 전체 생성 비용의 약 92~94%가 검증 경로에 있었으며, CPU 초안만 빠르게 하여 전체 속도를 크게 높이지는 못했습니다. 마지막 NPU 그래프의 LM Head 분리, Conv2D→FC/MatMul, 블록 폭 증가도 시험했지만 기본 구성보다 확실한 이득을 입증하지 못했습니다.

## 프로젝트 구성

```text
include/mobile_dflash2/   검증 계약, CPU 캐시·출력 헤드·선택기, Vulkan 초안 헤드
src/                     persistent runner, QNN 본체 연결, 측정·진단 실행기
tests/                   호스트 계약 테스트 및 수치 비교 fixture
tools/                   Android 빌드, shader 컴파일, 공개 예제 검사
examples/                요청 형식과 호환 자산의 메타데이터
docs/                    앱 독립 연동·빌드·프로토콜·벤치마크 설명
licenses/                외부 구성 요소의 라이선스 전문
```

기존 앱·모델과의 연결을 위해 C++ namespace `mobile_dflash2`, `mobile_runner` 실행 파일명, `@DFLASH2` 프로토콜 접두사는 유지합니다. 공식 엔진 이름은 **RaInfer**입니다.

## 라이선스와 출처

직접 작성한 RaInfer 소스는 **MIT**로 공개합니다. 외부 구성 요소의 원래 라이선스와 저작권은 그대로 유지합니다.

| 구성 요소 | 출처 / 조건 |
|---|---|
| RaInfer 실행 제어·캐시·연결·도구 | [MIT](LICENSE), LEPTONNW |
| DFlash2 기반 연산 구조 | [Z Lab DFlash](https://github.com/z-lab/dflash/tree/07ebd93db9f472af339b644bb70221ad8428328a), MIT |
| 모델 실행 기반 | [ExecuTorch](https://github.com/pytorch/executorch), BSD-3-Clause 및 해당 외부 의존성 조건 |
| 본체 원본 모델 | [Google Gemma 4 E2B](https://huggingface.co/google/gemma-4-E2B-it), Apache-2.0; 가중치 미포함 |
| NPU SDK·런타임 | Qualcomm QNN/QAIRT 별도 계약, 이 저장소에 미포함 |

RaInfer는 Qualcomm SDK, HTP 드라이버나 HMX 커널을 새로 제작한 프로젝트가 아닙니다. 공개 프레임워크와 별도 라이선스의 NPU 런타임 위에 초안 생성·검증·캐시·앱 연동 계층을 구현했습니다. 자세한 출처와 재배포 시 확인 사항은 [NOTICE.md](NOTICE.md)에 있습니다.

자체 바이너리나 앱을 배포할 때는 링크한 ExecuTorch 및 모든 외부 구성 요소의 고지도 함께 제공해야 합니다. Qualcomm 파일의 재배포 가능 범위는 실제 SDK 계약으로 별도 확인해야 합니다. 이 프로젝트의 MIT 라이선스가 SDK나 모델 가중치의 라이선스를 대체하지 않습니다.

## 개발 및 재현

호환 모델을 새로 연결하려면 본체의 블록 입출력·특징 탭·토크나이저·캐시 양자화 계약부터 정의하고, 그 조건으로 학습한 초안과 함께 검증해야 합니다. 현재 코드에는 Gemma E2B의 차원과 특징 위치를 검사하는 제한이 있으며, 범용 모델 자동 탐지 기능은 없습니다.

변경을 제안할 때는 동일한 입력과 greedy 설정에서 출력 토큰, Logits 오차, 채택률, 프리필·생성 시간을 분리해서 보고해주세요. 기기·SDK·컴파일 문맥·발열 상태·CPU 스레드 수도 함께 기록하면 비교에 도움이 됩니다.
