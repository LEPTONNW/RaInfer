# 모델과 DFlash2 다운로드·실행 안내

[한국어](MODEL_FILES.md) | [English](MODEL_FILES.en.md)

이 안내는 **RaInfer + Gemma 4 E2B NPU 본체 + 자체 학습 DFlash2 파생 초안**의
검증된 조합을 준비하는 방법입니다. 지원 기준은 S22 Ultra의 SM8450/Hexagon v69,
Android ARM64, QAIRT `2.37.0.250724`, 컴파일 문맥 2K입니다.

## 1. 엔진과 모델을 각각 준비하기

- **엔진 소스:** [github.com/LEPTONNW/RaInfer](https://github.com/LEPTONNW/RaInfer)
- **모델·초안 파일:** [Gemma4-E2B-Dflash2-Android](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android)
- **엔진 빌드:** [BUILD.md](BUILD.md). 호환 ExecuTorch 설치본과 QNN 런타임은 별도 준비합니다.

RaInfer는 실행 코드이고 Hugging Face 저장소는 모델 데이터입니다.
현재 엔진은 C++ 소스로 제공하며 Android 실행 파일을 빌드해야 합니다.
모델만 내려받아 일반 GGUF 로더에 넣으면 실행되는 구성이 아닙니다.

## 2. 본체와 DFlash2 필수 파일

아래 **8개를 모두 같은 `npu/` 폴더**에 유지합니다. 링크는 각 파일 다운로드입니다.

| 구분 | 파일 | 역할 |
|---|---|---|
| Gemma 본체 | [gemma4-e2b-dflash2-vision-verify-2k.pte](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/gemma4-e2b-dflash2-vision-verify-2k.pte) | NPU 검증 본체, 약 3.36 GB. 텍스트만 입력해도 사용 가능 |
| 본체 규격 | [io-contract.vision.json](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/io-contract.vision.json) | 위 PTE의 입출력·양자화·비전 규격 |
| DFlash2 본체 | [cpu-body-fp32.bin](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/cpu-body-fp32.bin) | CPU 초안 생성, FP32 |
| DFlash2 임베딩 | [embedding-cpu-fp32.pte](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/embedding-cpu-fp32.pte) | CPU 토큰 임베딩 |
| DFlash2 출력 헤드 | [head-cpu-int8.bin](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/head-cpu-int8.bin) | CPU INT8 토큰 점수 계산 |
| DFlash2 선택기 | [selector-cpu-fp32.bin](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/selector-cpu-fp32.bin) | 초안 후보 선택 |
| DFlash2 어휘 | [active-vocabulary.u32](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/active-vocabulary.u32) | 초안용 어휘 목록 |
| DFlash2 연결 설정 | [draft-manifest.json](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/resolve/main/npu/draft-manifest.json) | 초안 파일 이름·구조·해시 |

DFlash2 가중치는 엔진이나 Gemma 본체에 내장되어 있지 않습니다. 위 초안 6개 파일이
하나의 실행 묶음입니다. `.pte` 확장자의 임베딩도 **CPU**에서 실행하며, Gemma 본체
PTE는 **NPU**에서 실행합니다. 본체는 혼합 W4/W8 변환본입니다.

## 3. 토크나이저와 대화 템플릿

`tokenizer/`의 `tokenizer.json`, `tokenizer_config.json`, `chat_template.jinja`,
`config.json`을 준비합니다. 앱이 올바른 대화 형식으로 입력을 토큰화하고 결과를
문자열로 복호화하는 데 사용합니다. `gemma4-e2b-tokenizer.gguf`는 가중치 없는
추가 토크나이저 자산이며, 실제 토큰화 구현에서 사용할 때 선택합니다.

Hugging Face CLI가 설치된 PC에서는 아래 한 줄로 실행용 폴더를 받을 수 있습니다.
추가 GGUF 메타데이터도 함께 받아 약 4 GB의 모델 데이터를 준비합니다.

```sh
hf download RoundRobin237/Gemma4-E2B-Dflash2-Android --include "npu/*" --include "tokenizer/*" --local-dir ./model-assets
```

이 명령은 `original-bf16/`와 `q4-gguf/`를 받지 않습니다. 원본 BF16은 변환·학습용,
Q4_K_M GGUF는 별도 llama.cpp 비교용이므로 RaInfer NPU 실행에 필요하지 않습니다.
공개 [files.json](https://huggingface.co/RoundRobin237/Gemma4-E2B-Dflash2-Android/blob/main/files.json)의
크기·SHA256을 확인하고 파일명을 유지하세요. 초안 파일을 임의의 다른 모델과 섞지 않습니다.

## 4. 텍스트 생성 시작

빌드한 `mobile_runner`와 호환 런타임을 대상 Android에 준비합니다. 공유 라이브러리와
DSP 검색 경로는 [BUILD.md](BUILD.md), [INTEGRATION.md](INTEGRATION.md)를 따릅니다.
현재 디렉터리가 `model-assets/`이고 실행 파일이 PATH에 있는 예시입니다.

```sh
mobile_runner --serve npu/gemma4-e2b-dflash2-vision-verify-2k.pte npu/cpu-body-fp32.bin npu/draft-manifest.json npu/io-contract.vision.json
```

ready 이벤트를 기다린 뒤 같은 worker의 stdin에 JSON 한 줄을 보냅니다. 예시의
`PROMPT_TOKEN_IDS`는 실제 템플릿으로 토큰화한 배열로 바꿔야 합니다.

```text
{"id":1,"tokens":PROMPT_TOKEN_IDS,"max_new_tokens":96,"cpu_workers":4,"body_workers":1,"fast_argmax":true,"compact_logits":true,"ngram":true,"gpu_head":false}
```

위 줄은 토큰 배열을 넣기 전의 설명용 형식입니다. 응답은 `@DFLASH2 ` 뒤의 JSON이며,
`report.tokens`를 같은 토크나이저로 복호화합니다. n-gram은 엔진 기능이라 별도 모델
파일을 받지 않습니다. [프로토콜](PROTOCOL.md)과 [Kotlin 연동 예제](../examples/android/RaInferProcess.kt)를
참고해 자신의 앱에 연결할 수 있습니다. 특정 채팅 앱은 필요하지 않습니다.

## 5. 이미지 입력에만 추가할 파일

- `tokenizer/gemma4-e2b-vision-vocab.gguf`: 이미지 인코더용 어휘·모델 메타데이터.
  위 CLI 명령에 포함됩니다. 디코더 가중치가 든 Q4 GGUF가 아닙니다.
- **외부 mmproj:** [Unsloth mmproj-F16.gguf](https://huggingface.co/unsloth/gemma-4-E2B-it-GGUF/resolve/0314792d7f1f7e229411f620751375812bb9faf2/mmproj-F16.gguf).
  Google 비전 가중치의 Unsloth GGUF 변환본이며 우리 저장소에 재업로드하지 않습니다.
  고정 리비전·SHA256은 모델 저장소의 `external-assets.json`에 있습니다.
- RaInfer의 `rainfer_vision_encoder`: 별도 빌드한 CPU mmproj 인코더 실행 파일.

이미지 인코더가 만든 bundle 경로를 `tokens` 대신 `vision_bundle`로 전달합니다.
이미지 인코딩·템플릿·실행 순서는 [VISION.md](VISION.md)를 따릅니다. 텍스트만 쓸 때는
mmproj와 이미지 인코더를 준비하지 않아도 됩니다.
