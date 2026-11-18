# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

[CmdletBinding()]
param(
    [Parameter(Position = 0)]
    [ValidateSet("help", "menu", "plan", "bootstrap", "configure", "build", "test", "app", "verify", "python-wheel", "clean-bootstrap", "example", "examples", "demo", "benchmark", "format", "check", "lint", "bump-version", "docs")]
    [string]$Command = "help",
    [Parameter(Position = 1)]
    [string]$Lesson,
    [switch]$Help,
    [string]$Python,
    [string]$Venv,
    [string]$NativeBuild,
    [ValidateSet("windows-x86_64", "manylinux_2_35_x86_64")]
    [string]$Platform = "windows-x86_64",
    [ValidateSet("Auto", "Library", "Library-Cuda")]
    [string]$Scope = "Auto",
    [switch]$SkipNativeTests,
    [switch]$RequireUnavailableCuda,
    [switch]$Patch,
    [switch]$Minor,
    [switch]$Major,
    [ValidateSet("python", "native", "both")]
    [string]$Language = "python",
    [switch]$Headless,
    [int]$Frames = 3,
    [string]$Evidence,
    [switch]$RequireCuda,
    [string]$Suite = "quick",
    [switch]$PythonObservation,
    [switch]$Dev,
    [switch]$Measure,
    [string]$Cases,
    [string]$Backends,
    [switch]$List,
    [ValidateRange(0, 10000)]
    [int]$PairToleranceMilli = 0,
    [ValidateRange(0, 100000)]
    [int]$ActorHitCapacity = 32,
    [ValidateSet(0, 1)]
    [int]$ReusePlans = 1,
    [ValidateRange(0, 256)]
    [int]$RadiusGroups = 0,
    [ValidateRange(2, 224)]
    [int]$MeshSide = 32,
    [ValidateSet(0, 1)]
    [int]$UniqueMeshes = 0,
    [Parameter(ValueFromRemainingArguments = $true)]
    [string[]]$DemoArguments
)

Set-StrictMode -Version Latest
$ErrorActionPreference = "Stop"

if ($SkipNativeTests -and $Command -notin @("bootstrap", "python-wheel")) {
    throw "-SkipNativeTests is supported only for bootstrap and python-wheel"
}
if ($RequireUnavailableCuda -and $Command -ne "python-wheel") {
    throw "-RequireUnavailableCuda is supported only for python-wheel"
}

if ($Command -eq "examples") { $Command = "example" }
$menuSection = $null
if (-not $env:CI -and
    -not [Console]::IsInputRedirected -and -not [Console]::IsOutputRedirected) {
    if ($PSBoundParameters.Count -eq 0) {
        $Command = "menu"
    } elseif ($PSBoundParameters.Count -eq 1 -and $Command -in @("demo", "example")) {
        $menuSection = $Command
        $Command = "menu"
    }
}

if ($Command -eq "help" -or ($Help -and $Command -ne "demo")) {
    $entrypoint = if ($env:OVSPATIALQUERY_ENTRYPOINT) {
        $env:OVSPATIALQUERY_ENTRYPOINT
    } else {
        "dev.ps1"
    }
    @"
usage: $entrypoint [plan|bootstrap|configure|build|test|app|verify|python-wheel|clean-bootstrap|example|demo|benchmark|format|check|lint|bump-version|docs] [options]

Commands:
  (no arguments)   Open the development menu in an interactive terminal.
  demo / example   Open just that menu when no name is given (interactive only).
  help             Print this usage without initializing dependencies.
  plan             Print the locked dependency and build plan.
  bootstrap        Create the repository-local venv from the locked wheels,
                   install the Python/App packages editable, stage CMake and
                   PhysX with Packman, then configure, build, install, and verify.
  build            Reconfigure, build, and verify-install the native-stable tree.
  configure        Stage locked dependencies and generate build/test metadata without compiling.
  test             Run the native CTest lane and the Python test suites.
  app              Launch the viewer from the worktree venv.
  verify           Verify the venv contents, the native install, and run CTest.
  python-wheel     Build a platform wheel from that install and import it in a fresh venv.
  clean-bootstrap  Remove only the bootstrap venv and native build.
  example <NN>     Run one visual lesson (default language: python).
  demo <name>      Prepare and run one browser demo (warehouse); -Help lists its options.
  benchmark        Qualify installed-library correctness (-Suite; -Cases 'a,b' selects cases; -Backends 'a,b' selects backends; -List prints them as JSON). -Measure enables repeated timings; -Dev runs cheap exploratory timings.
  format           Format repository-owned C, C++, CUDA, and Python sources.
  check            Check native and Python formatting, run Python correctness lint, and check the source version against release tags and uv.lock.
  lint             Run Python correctness lint without modifying files.
  docs             Build the SDK guide into _build\docs\html; --serve [--port N] serves it
                   locally, --hub builds the whole internal hub instead.
  bump-version     Set the library source version to the next X.Y.Z.dev0 in
                   python/pyproject.toml and the App uv.lock: <X.Y.Z>, -Patch, -Minor or -Major.

Options:
  -Platform <windows-x86_64|manylinux_2_35_x86_64> (plan only; default: windows-x86_64)
  -Scope <Auto|Library|Library-Cuda> (plan/bootstrap/build/test/verify/python-wheel; default: Auto)
  -SkipNativeTests (bootstrap/python-wheel only; CI runs the separate test command)
  -RequireUnavailableCuda (python-wheel only; CPU builder consumption before GPU qualification)
  OVSPATIALQUERY_NATIVE_CONFIG=Release|Debug selects the native configuration.
  OVSPATIALQUERY_TRAIN=next selects isolated ovspatialqueryapp-next and native-next trees.
  -Python <path> -Venv <path> -NativeBuild <path>
  example: -Language <python|native|both> -Headless -Frames <N> -Evidence <dir>
           -RequireCuda (lesson 11 only: fail unless the CUDA lane runs on a device)
  demo:    further arguments are forwarded, for example: dev.cmd demo warehouse --sanity
"@
    exit 0
}

$projectRoot = $PSScriptRoot
$packmanPythonPackage = "python@3.12.13-nv1-windows-x86_64"
$packmanPythonVersion = $packmanPythonPackage.Split("@", 2)[1]
$packmanCache = if ($env:PM_PACKAGES_ROOT) {
    [System.IO.Path]::GetFullPath($env:PM_PACKAGES_ROOT)
} else {
    Join-Path ([System.IO.Path]::GetPathRoot($projectRoot)) "packman-repo"
}
$env:PM_PACKAGES_ROOT = $packmanCache
$managedPython = Join-Path $packmanCache "python\$packmanPythonVersion\python.exe"
$packmanPythonAdapter = Join-Path $projectRoot "tools\packman\python.bat"
$interpreterProbe = Join-Path $projectRoot "tools\bootstrap\interpreter_probe.py"

$train = if ($env:OVSPATIALQUERY_TRAIN) { $env:OVSPATIALQUERY_TRAIN } else { "stable" }
if ($train -notin @("stable", "next")) {
    throw "OVSPATIALQUERY_TRAIN must be stable or next; found $train"
}
$trainSuffix = if ($train -eq "next") { "-next" } else { "" }
if (-not $Venv) {
    $Venv = Join-Path $projectRoot "_build\venv\ovspatialqueryapp$trainSuffix"
}
if (-not $NativeBuild) {
    $NativeBuild = Join-Path $projectRoot "_build\bootstrap\native-$train"
}
# Unlike a shell subprocess, this script shares the caller's environment. Restore
# the train overrides on exit so a subsequent stable command cannot inherit them.
$savedTrainEnvironment = @{}
foreach ($name in @("OVSPATIALQUERY_LIBRARY", "OVSPATIALQUERY_APP_DLL", "OVSPATIALQUERY_NEXT_TRAIN_VERSIONS")) {
    $savedTrainEnvironment[$name] = [Environment]::GetEnvironmentVariable($name, "Process")
}
try {
if ($train -eq "next") {
    $nextLibrary = Join-Path $NativeBuild "install\bin\ovspatialquery.dll"
    if (-not $env:OVSPATIALQUERY_LIBRARY) { $env:OVSPATIALQUERY_LIBRARY = $nextLibrary }
    if (-not $env:OVSPATIALQUERY_APP_DLL) { $env:OVSPATIALQUERY_APP_DLL = $nextLibrary }
    $resolutionPath = Join-Path $projectRoot "_build\next-train\resolution.json"
    if (Test-Path $resolutionPath) {
        $resolution = Get-Content -Raw $resolutionPath | ConvertFrom-Json
        $env:OVSPATIALQUERY_NEXT_TRAIN_VERSIONS = ($resolution.wheels.PSObject.Properties |
            Sort-Object Name | ForEach-Object { "$($_.Name)=$($_.Value.version)" }) -join ","
    }
}
$targetDeps = Join-Path $projectRoot "_build\target-deps"
$driver = Join-Path $projectRoot "tools\bootstrap\main.py"
$exampleDriver = Join-Path $projectRoot "tools\examples\main.py"
$demoDriver = Join-Path $projectRoot "tools\demos\main.py"
$venvPython = Join-Path $Venv "Scripts\python.exe"

# Interpreter candidates, in order.  The first compatible 64-bit CPython 3.12
# wins; a "Required" candidate stops the search when it is not compatible.
$candidates = [System.Collections.Generic.List[object]]::new()
if ($Python) {
    $candidates.Add([pscustomobject]@{
        Source = "-Python override"
        Command = $Python
        Prefix = @()
        Required = $true
    })
} elseif ($env:OVSPATIAL_PYTHON) {
    $candidates.Add([pscustomobject]@{
        Source = "OVSPATIAL_PYTHON"
        Command = $env:OVSPATIAL_PYTHON
        Prefix = @()
        Required = $true
    })
} else {
    if ((Test-Path $venvPython) -and $Command -notin @("bootstrap", "configure", "clean-bootstrap")) {
        $candidates.Add([pscustomobject]@{
            Source = "existing worktree venv"
            Command = $venvPython
            Prefix = @()
            Required = $false
        })
    }
    if (Test-Path -LiteralPath $managedPython -PathType Leaf) {
        $candidates.Add([pscustomobject]@{
            Source = "Packman Python"
            Command = $managedPython
            Prefix = @()
            Required = $true
        })
    } elseif ($Command -in @("bootstrap", "configure")) {
        # tools\packman\python.bat initializes Packman and fetches its Python.
        $candidates.Add([pscustomobject]@{
            Source = "Packman-managed Python"
            Command = $packmanPythonAdapter
            Prefix = @()
            Required = $true
        })
    } else {
        $candidates.Add([pscustomobject]@{
            Source = "Python launcher"
            Command = "py"
            Prefix = @("-3.12")
            Required = $false
        })
        $candidates.Add([pscustomobject]@{
            Source = "PATH python3.12"
            Command = "python3.12"
            Prefix = @()
            Required = $false
        })
        $candidates.Add([pscustomobject]@{
            Source = "PATH python"
            Command = "python"
            Prefix = @()
            Required = $false
        })
    }
}

$selected = $null
$rejections = [System.Collections.Generic.List[string]]::new()
foreach ($candidate in $candidates) {
    try {
        $previousErrorActionPreference = $ErrorActionPreference
        $ErrorActionPreference = "Continue"
        try {
            $raw = & $candidate.Command @($candidate.Prefix) $interpreterProbe 2>$null
            $probeExitCode = $LASTEXITCODE
        } finally {
            $ErrorActionPreference = $previousErrorActionPreference
        }
        if ($probeExitCode -ne 0) {
            throw "probe exited with code $probeExitCode"
        }
        $jsonLine = @($raw) |
            Where-Object { $_ -is [string] -and $_.TrimStart().StartsWith("{") } |
            Select-Object -Last 1
        if (-not $jsonLine) {
            throw "probe did not emit interpreter metadata"
        }
        $details = $jsonLine | ConvertFrom-Json
        $compatible = (
            $details.implementation -eq "CPython" -and
            $details.version -match '^3\.12\.' -and
            $details.bits -eq 64 -and
            $details.machine -in @("AMD64", "x86_64")
        )
        if ($compatible) {
            $selected = [pscustomobject]@{
                Source = $candidate.Source
                Command = $details.executable
                Prefix = @()
                Details = $details
            }
            break
        }
        $reason = "$($candidate.Source): $($details.implementation) $($details.version), $($details.bits)-bit $($details.machine)"
    } catch {
        $reason = "$($candidate.Source): unavailable ($($_.Exception.Message))"
    }
    $rejections.Add($reason)
    if ($candidate.Required) {
        break
    }
}

if ($null -eq $selected) {
    $attempts = if ($rejections.Count) { $rejections -join "`n  - " } else { "no candidates" }
    throw @"
No compatible 64-bit CPython 3.12 interpreter was found.
Run bootstrap to stage $packmanPythonPackage with Packman, or set -Python <path> /
OVSPATIAL_PYTHON explicitly.
Rejected candidates:
  - $attempts
"@
}

Write-Host (
    "[bootstrap] Python: {0} ({1} {2}, {3}-bit {4}; source: {5})" -f
    $selected.Details.executable,
    $selected.Details.implementation,
    $selected.Details.version,
    $selected.Details.bits,
    $selected.Details.machine,
    $selected.Source
)

$arguments = [System.Collections.Generic.List[string]]::new()
if ($Command -eq "menu") {
    $arguments.Add((Join-Path $projectRoot "tools\launch.py"))
    if ($menuSection) { $arguments.Add($menuSection) }
    & $selected.Command @($selected.Prefix) @arguments
    exit $LASTEXITCODE
}
if ($Command -eq "bump-version") {
    $arguments.Add((Join-Path $projectRoot "tools\format\release_version.py"))
    if ($Lesson) { $arguments.Add($Lesson) }
    if ($Patch) { $arguments.Add("--patch") }
    if ($Minor) { $arguments.Add("--minor") }
    if ($Major) { $arguments.Add("--major") }
    & $selected.Command @($selected.Prefix) @arguments
    exit $LASTEXITCODE
}
if ($Command -eq "docs") {
    $arguments.Add((Join-Path $projectRoot "tools\docs\main.py"))
    if ($Python) {
        $arguments.Add("--python")
        $arguments.Add($Python)
    }
    if ($Lesson) { $arguments.Add($Lesson) }
    if ($DemoArguments) { $arguments.AddRange($DemoArguments) }
    & $selected.Command @($selected.Prefix) @arguments
    exit $LASTEXITCODE
}
if ($Command -in @("format", "check", "lint")) {
    $arguments.Add((Join-Path $projectRoot "tools\format\main.py"))
    $arguments.Add($Command)
    & $selected.Command @($selected.Prefix) @arguments
    exit $LASTEXITCODE
}
if ($Command -eq "demo") {
    $arguments.Add($demoDriver)
    if ($Lesson) {
        $arguments.Add($Lesson)
    }
    $arguments.Add("--python")
    $arguments.Add($venvPython)
    if ($Help) {
        $arguments.Add("--help")
    }
    if ($DemoArguments) {
        $arguments.AddRange($DemoArguments)
    }
    & $selected.Command @($selected.Prefix) @arguments
    exit $LASTEXITCODE
}
if ($Command -eq "example") {
    if (-not $Lesson) {
        throw "example requires a lesson number, for example: dev.cmd example 01"
    }
    $arguments.Add($exampleDriver)
    $arguments.Add($Lesson)
    $arguments.Add("--language")
    $arguments.Add($Language)
    $arguments.Add("--frames")
    $arguments.Add([string]$Frames)
    if ($Headless) {
        $arguments.Add("--headless")
    }
    if ($Evidence) {
        $arguments.Add("--evidence")
        $arguments.Add($Evidence)
    }
    if ($RequireCuda) {
        $arguments.Add("--require-cuda")
    }
    & $selected.Command @($selected.Prefix) @arguments
    exit $LASTEXITCODE
}
if ($Command -eq "benchmark") {
    $arguments.Add((Join-Path $projectRoot "benchmarks\run.py"))
    $arguments.Add("--suite")
    $arguments.Add($Suite)
    $arguments.Add("--native-build")
    $arguments.Add($NativeBuild)
    if ($Evidence) { $arguments.Add("--output"); $arguments.Add($Evidence) }
    if ($PythonObservation) { $arguments.Add("--python-observation") }
    if ($Dev) { $arguments.Add("--dev") }
    if ($Measure) { $arguments.Add("--measure") }
    if ($Cases) { $arguments.Add("--cases"); $arguments.Add($Cases) }
    if ($Backends) { $arguments.Add("--backends"); $arguments.Add($Backends) }
    if ($List) { $arguments.Add("--list") }
    $arguments.Add("--pair-tolerance-milli"); $arguments.Add([string]$PairToleranceMilli)
    $arguments.Add("--actor-hit-capacity"); $arguments.Add([string]$ActorHitCapacity)
    $arguments.Add("--reuse-plans"); $arguments.Add([string]$ReusePlans)
    $arguments.Add("--radius-groups"); $arguments.Add([string]$RadiusGroups)
    $arguments.Add("--mesh-side"); $arguments.Add([string]$MeshSide)
    $arguments.Add("--unique-meshes"); $arguments.Add([string]$UniqueMeshes)
    & $selected.Command @($selected.Prefix) @arguments
    exit $LASTEXITCODE
}
$arguments.Add($driver)
$arguments.Add($Command)
$arguments.Add("--venv")
$arguments.Add($Venv)
$arguments.Add("--native-build")
$arguments.Add($NativeBuild)
$arguments.Add("--target-deps")
$arguments.Add($targetDeps)
$arguments.Add("--platform")
$arguments.Add($Platform)
if ($Command -in @("plan", "bootstrap", "configure", "build", "test", "verify", "python-wheel")) {
    $arguments.Add("--scope")
    $arguments.Add($Scope.ToLowerInvariant())
}
if ($SkipNativeTests) {
    $arguments.Add("--skip-native-tests")
}
if ($RequireUnavailableCuda) {
    $arguments.Add("--require-unavailable-cuda")
}

& $selected.Command @($selected.Prefix) @arguments
exit $LASTEXITCODE
} finally {
    foreach ($name in $savedTrainEnvironment.Keys) {
        if ($null -eq $savedTrainEnvironment[$name]) {
            Remove-Item "Env:$name" -ErrorAction SilentlyContinue
        } else {
            [Environment]::SetEnvironmentVariable($name, $savedTrainEnvironment[$name], "Process")
        }
    }
}
