# 다른 앱에 RaInfer 연결하기

RaInfer는 특정 채팅 앱을 필요로 하지 않습니다. 저장소에는 앱·Activity·화면·APK·대화 저장소가 없으며,
어떤 Android 앱이든 지원되는 기기와 호환 모델을 준비하면 동일한 프로세스 프로토콜로 연결할 수 있습니다.
Kotlin/Java에서는 `ProcessBuilder`, Flutter/React Native/Unity에서는 얇은 Android 네이티브 브리지를
통해 같은 방법을 사용할 수 있습니다. 이것이 다른 OS·NPU·임의 모델까지 지원한다는 의미는 아닙니다.

현재 제공하는 경계는 **실행 파일 + stdin/stdout JSONL 프로토콜**입니다. 바로 호출할 수 있는
JNI 함수, C ABI, Maven AAR, HTTP API, OpenAI 호환 서버는 아직 제공하지 않습니다.
C++ 코어 인터페이스는 `include/mobile_dflash2/engine.h`에 있으나, NPU runner를 같은 프로세스의
라이브러리로 사용하려면 별도의 래퍼와 수명 관리가 필요합니다. 현재 권장 방식은 프로세스 연결입니다.

## 역할 분담

| 앱이 담당 | RaInfer가 담당 |
|---|---|
| 화면, 메시지·파일·세션 저장 | 모델 로딩과 persistent worker |
| 모델/런타임 준비 및 해시 확인 | CPU 초안 및 QNN NPU 본체 실행 |
| 정확한 토크나이저와 대화 템플릿 | 후보 일괄 검증, prefix 확정, KV·특징 관리 |
| 입력 길이·출력 예산 결정 | 토큰 ID, 채택 통계와 시간 반환 |
| 결과 토큰 디코딩과 화면 표시 | 프로토콜 준비·결과·오류 이벤트 |
| 실행 기한·취소·앱 수명 관리 | 요청 내부 오류 시 미확정 캐시 중단 |

대화 전체를 엔진에 자동 저장하지 않습니다. 다음 요청에서는 앱이 필요한 대화 내역을 다시 토큰화하여
전달합니다. 문맥 한도는 현재 컴파일된 **2K**입니다. 여러 앱이 하나의 설치된 모델을 자동으로
공유하는 서비스도 아니며, 각 앱이 접근 가능한 파일을 직접 준비해야 합니다.

## 1. 실행 파일을 자신의 APK에 포함하기

[BUILD.md](BUILD.md)에 따라 Android ARM64 실행 파일을 빌드합니다. 앱 프로젝트의
`app/src/main/jniLibs/arm64-v8a/` 등에 `mobile_runner`를 **실제 파일로 복사**하고,
Android 패키징을 위한 이름을 `libRaInferRunner.so`로 지정합니다. 이는 실행 파일의 패키징 이름이며,
JNI 라이브러리로 변환된 것이 아니므로 `System.loadLibrary("RaInferRunner")`로 로드하지 않습니다.

```text
your-app/app/src/main/jniLibs/arm64-v8a/
  libRaInferRunner.so            # built mobile_runner executable
  libgemma4_qnn_backend.so       # exact backend SONAME used by the qualified build
  ...                           # your build's other required shared libraries
```

실제 의존 라이브러리 이름은 빌드에 따라 달라질 수 있습니다. NDK의 `llvm-readelf -d mobile_runner`에서
`NEEDED` 항목을 확인하고, backend의 전이 의존성도 확인하세요. `libc++_shared.so`가 필요한 빌드라면
그 파일도 맞춰 제공합니다. Android의 시스템 `libc`·`libm`·`libvulkan` 등을 앱에 복사하지 않습니다.
QNN 런타임 파일의 포함·별도 설치 방법과 재배포 권한은 사용하는 SDK 계약에 맞게 결정해야 합니다.

AGP Kotlin DSL 예시:

```kotlin
android {
    defaultConfig { ndk { abiFilters += "arm64-v8a" } }
    packaging {
        jniLibs {
            useLegacyPackaging = true
            keepDebugSymbols += "**/libRaInferRunner.so"
        }
    }
}
```

실행 파일이 실제 설치 경로에 추출되도록 패키징하고, 실행할 때
`context.applicationInfo.nativeLibraryDir`에서 찾습니다. Android 10 이상에서는 앱의 쓰기 가능한
`filesDir`에 실행 파일을 내려받아 실행하는 방법을 사용하지 않습니다. 모델 데이터의 저장 위치와
실행 코드의 설치 위치는 구분합니다. APK와 AAB의 실제 추출 결과는 대상 기기에서 확인해야 합니다.
근거: [Android 실행 파일 제약](https://developer.android.com/about/versions/10/behavior-changes-10),
[AGP jniLibs 패키징](https://developer.android.com/reference/tools/gradle-api/8.13/com/android/build/api/dsl/JniLibsPackaging).

SM8450의 vendor RPC 접근을 사용하는 앱에서는 `<application>` 안에 아래 선언도 검토합니다.

```xml
<uses-native-library android:name="libcdsprpc.so" android:required="false" />
<uses-native-library android:name="libadsprpc.so" android:required="false" />
```

`required="false"`는 없는 장치에도 앱 설치를 허용한다는 뜻이며, 그 장치에서 NPU가 동작하도록
만드는 옵션은 아닙니다. 앱은 지원 여부를 확인하고 적절한 오류나 자체 fallback을 제공해야 합니다.
[Android vendor native library 설명](https://developer.android.com/guide/topics/manifest/uses-native-library-element).

## 2. 모델과 런타임 경로 지정

먼저 [다운로드 파일 안내](MODEL_FILES.md)의 본체·초안·토크나이저를 준비합니다.
앱 전용 저장소 등 앱이 읽을 수 있는 실제 경로에 공개 모델의 폴더 구성을 유지합니다.

```text
modelDirectory/
  gemma4-e2b-dflash2-vision-verify-2k.pte
  io-contract.vision.json
  draft-manifest.json
  cpu-body-fp32.bin
  embedding-cpu-fp32.pte
  head-cpu-int8.bin
  selector-cpu-fp32.bin
  active-vocabulary.u32
```

여기서 `modelDirectory`는 다운로드 폴더의 **`npu/`**입니다. 토크나이저는 그 옆의
`tokenizer/`에서 사용합니다. 본체와 초안은 [README의 지원 조건](../README.md)에 맞아야 합니다.
엔진 Git 저장소에는 가중치가 없으며 연결된 Hugging Face 저장소에서 받습니다.
`io-contract.vision.json`의 scale·zero-point는 임의로 만들거나 다른 PTE에서
복사하지 마세요. 로딩 전에 파일 해시를 확인하는 것은 앱/설치기의 책임입니다.

호스트는 `LD_LIBRARY_PATH`에 Android 라이브러리 경로를, `ADSP_LIBRARY_PATH`에 DSP 경로를
지정합니다. 두 환경 변수의 구분자는 다릅니다. 검증된 Android 설정은 다음 형태입니다.

```text
LD_LIBRARY_PATH=<nativeLibraryDir>:<runtimeDirectory>:/vendor/lib64
ADSP_LIBRARY_PATH=<runtimeDirectory>;/vendor/lib/rfsa/adsp;/vendor/dsp
```

## 3. Kotlin 연결 예제

[examples/android/RaInferProcess.kt](../examples/android/RaInferProcess.kt)는 UI가 없는 작은 프로세스
어댑터입니다. Android의 `org.json`과 Java 기본 API만 사용합니다. 자기 앱의 namespace에 맞게
복사·수정할 수 있으며 MIT 라이선스입니다. 호스트 Android SDK로 컴파일 검사한 예제이며,
이 예제 자체를 새 앱으로 설치·실측한 것은 아닙니다.

아래 코드는 **UI 스레드가 아닌 작업 스레드**에서 실행합니다.

```kotlin
val modelDirectory = File(context.filesDir, "models/gemma-e2b-dflash2/npu")
val worker = RaInferProcess.start(
    nativeLibraryDir = File(context.applicationInfo.nativeLibraryDir),
    runtimeDirectory = File(context.filesDir, "qnn-v69"),
    modelDirectory = modelDirectory,
    workDirectory = File(context.cacheDir, "rainfer"),
    targetFile = File(modelDirectory, "gemma4-e2b-dflash2-vision-verify-2k.pte"),
    contractFile = File(modelDirectory, "io-contract.vision.json")
)
try {
    // Token IDs produced by your exact Gemma tokenizer and chat template.
    val report = worker.generate(promptTokenIds, maxNewTokens = 96)
    val ids = report.getJSONArray("tokens")
    val generated = LongArray(ids.length()) { ids.getLong(it) }
    // Decode generated with the same tokenizer; then update your UI.
} finally {
    worker.close()
}
```

`start`는 모델 로딩 및 ready 응답을 기다리고, `generate`는 완성된 응답을 기다리는 블로킹 함수입니다.
어댑터의 기본 파일명은 기존 텍스트 전용 설치본과 호환되도록 남겨두었으므로, 공개 비전 PTE를
사용할 때는 위처럼 `targetFile`·`contractFile`을 명시합니다. 이 PTE로 텍스트 생성도 가능합니다.
코루틴을 사용하는 앱이라면 `Dispatchers.IO`, Java 앱이라면 작업용 Executor 등에 배치하세요.
예제에는 모든 앱에 공통으로 적용할 수 있는 타임아웃·서비스·권한 UI를 넣지 않았습니다.

stderr를 stdout과 합치지 않고 별도 스레드로 계속 비워, 진단 로그가 가득 차서 worker가 멈추는 것을
막습니다. 실제 앱에서는 오류와 진단 수집을 추가하되 프롬프트나 사용자 데이터를 불필요하게 남기지 마세요.

## 4. 토큰화·오류·취소 처리

- Gemma 모델과 정확히 같은 토크나이저·템플릿을 사용합니다. 문자열의 문자 코드를 토큰 ID로 전달하면 안 됩니다.
- BOS, 사용자/assistant turn 경계, generation prefix, thinking 설정을 일관되게 적용합니다.
- 결과의 `tokens`를 디코딩하며, 특수 토큰 표시 여부도 앱에서 결정합니다.
- 한 worker에는 요청을 순서대로 보냅니다. 병렬 대화는 앱이 큐로 관리하고 메모리 사용을 고려합니다.
- 생성 중 스트리밍 토큰 이벤트는 아직 없으므로, 앱은 대기 UI와 요청 취소 동작을 구현해야 합니다.
- 요청 처리 기한과 모델 로딩 기한을 앱에서 관리합니다. 기한을 넘기면 소유한 worker 프로세스를 종료하고 다시 시작합니다.
- 예제의 `cancel()`은 진행 중인 `generate()`를 끊을 때 사용합니다. 종료된 worker는 다시 생성해야 합니다.
- 정상 종료는 idle 상태에서 `close()`로 수행합니다. 활성 요청 중 `close()`는 직렬화 락을 기다리므로 취소 용도로 쓰지 않습니다.
- 초기 `start()`가 ready를 기다리는 동안의 취소까지 필요하다면, 어댑터의 프로세스 시작/ready 대기를 분리해 앱의 수명 관리자에 프로세스 핸들을 먼저 등록하세요.

현재 전체 결과를 한 번에 반환하고, 대화별 캐시를 유지하지 않으며, C API/JNI 패키지도 없다는 점은
연동 시 고려할 제약입니다. 다른 앱의 UI나 저장소 구조를 강제하지 않는 대신, 이 책임을 호스트가 갖습니다.
실제 메시지 형식과 통계 필드는 [PROTOCOL.md](PROTOCOL.md)를 참고하세요.
