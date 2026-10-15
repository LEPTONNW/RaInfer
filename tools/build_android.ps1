# SPDX-License-Identifier: MIT
param(
    [Parameter(Mandatory=$true)][string]$Install,
    [Parameter(Mandatory=$true)][string]$ExecuTorchSource,
    [Parameter(Mandatory=$true)][string]$QnnSdk,
    [Parameter(Mandatory=$true)][string]$Ndk,
    [string]$CMake = 'cmake',
    [string]$Ninja = 'ninja',
    [string]$Build = (Join-Path $PSScriptRoot '..\build\android-release'),
    [int]$Jobs = 4
)
$ErrorActionPreference = 'Stop'
$engineRoot = (Resolve-Path -LiteralPath (Join-Path $PSScriptRoot '..')).Path
$installRoot = (Resolve-Path -LiteralPath $Install).Path
$etRoot = (Resolve-Path -LiteralPath $ExecuTorchSource).Path
$sdkRoot = (Resolve-Path -LiteralPath $QnnSdk).Path
$ndkRoot = (Resolve-Path -LiteralPath $Ndk).Path
$buildRoot = [IO.Path]::GetFullPath($Build)
if (-not $buildRoot.StartsWith($engineRoot + '\', [StringComparison]::OrdinalIgnoreCase)) {
    throw 'Build directory must remain inside this project'
}
$cmakeExe = (Get-Command $CMake -ErrorAction Stop).Source
$ninjaExe = (Get-Command $Ninja -ErrorAction Stop).Source
foreach ($needed in @("$ndkRoot\build\cmake\android.toolchain.cmake", "$installRoot\lib\libexecutorch.a", "$sdkRoot\include\QNN\QnnInterface.h")) {
    if (-not (Test-Path -LiteralPath $needed)) { throw "Missing dependency: $needed" }
}
$ErrorActionPreference = 'Continue' # Windows PowerShell native warning stderr handling.
& $cmakeExe -S $engineRoot -B $buildRoot -G Ninja "-DCMAKE_MAKE_PROGRAM=$ninjaExe" `
    "-DCMAKE_TOOLCHAIN_FILE=$ndkRoot\build\cmake\android.toolchain.cmake" `
    '-DANDROID_ABI=arm64-v8a' '-DANDROID_PLATFORM=android-28' '-DCMAKE_BUILD_TYPE=Release' `
    '-DBUILD_TESTING=OFF' "-DEXECUTORCH_INSTALL=$installRoot" "-DEXECUTORCH_SOURCE=$etRoot" `
    "-DQNN_SDK_ROOT=$sdkRoot" '-DCMAKE_FIND_ROOT_PATH_MODE_PACKAGE=BOTH'
if ($LASTEXITCODE -ne 0) { throw 'Android configuration failed' }
& $cmakeExe --build $buildRoot --parallel $Jobs
if ($LASTEXITCODE -ne 0) { throw 'Android build failed' }
$ErrorActionPreference = 'Stop'
Write-Output "RaInfer Android executables: $buildRoot"
