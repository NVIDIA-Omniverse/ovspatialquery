#!/usr/bin/env bash
# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

set -euo pipefail

print_usage()
{
    cat <<'EOF'
usage: dev.sh [plan|bootstrap|configure|build|cross-build|test|app|verify|python-wheel|clean-bootstrap|example|demo|benchmark|format|check|lint|bump-version|docs] [options]

Commands:
  (no arguments) Open the development menu in an interactive terminal.
  demo / example Open just that menu when no name is given (interactive only).
  plan       Print the locked dependency and native-build plan.
  bootstrap  Create the repository-local venv from the locked wheels, install
             the Python/App packages editable, stage CMake and PhysX with
             Packman, then configure, build, install, and verify.
  build      Reconfigure, build, and install the native-stable tree.
  cross-build Cross-build Linux ARM64 CPU/CUDA SDKs, wheels and tests on x64; never execute ARM binaries.
  configure  Stage locked dependencies and generate build/test metadata without compiling.
  test       Run the native CTest lane and the Python test suites.
  app        Run the viewer from the worktree venv.
  verify     Verify the venv contents, the native install, and run CTest.
  python-wheel
             Build a platform wheel from that install and import it in a fresh venv.
  clean-bootstrap
             Remove only the bootstrap venv and native build.
  example NN Run one visual lesson; Linux interactive requires X11 DISPLAY.
  demo <name>
             Prepare and run one browser demo (warehouse). Add --help after the
             name to list the driver options and the demo's own options.
  benchmark  Qualify installed-library correctness (--suite; --cases a,b selects cases; --backends a,b selects backends; --list prints them as JSON). --measure enables repeated timings; --dev runs cheap exploratory timings.
  format     Format repository-owned C, C++, CUDA, and Python sources.
  check      Check native and Python formatting, run Python correctness lint, and check the source version against release tags and uv.lock.
  lint       Run Python correctness lint without modifying files.
  docs       Build the SDK guide into _build/docs/html; --serve [--port N] serves it
             locally, --hub builds the whole internal hub instead.
  bump-version <X.Y.Z>|--patch|--minor|--major
             Set the library source version to the next X.Y.Z.dev0 in
             python/pyproject.toml and the App uv.lock.

Options:
  -h, --help, help     Print this help without initializing Packman.
  OVSPATIALQUERY_NATIVE_CONFIG=Release|Debug selects the native configuration.
  OVSPATIALQUERY_TRAIN=next uses the newest in-development Ovstage and ovRTX
             builds (deps/next-train.json) in _build/venv/ovspatialqueryapp-next
             and _build/bootstrap/native-next; bootstrap resolves them.

Command-specific options are forwarded to the selected driver.
EOF
}

if [[ $# -eq 0 && -t 0 && -t 1 && -z "${CI:-}" ]]; then
    set -- menu
fi
if [[ "${1:-}" == "examples" ]]; then
    shift
    set -- example "$@"
fi
if [[ $# -eq 1 && ( "$1" == "demo" || "$1" == "example" ) && -t 0 && -t 1 && -z "${CI:-}" ]]; then
    set -- menu "$1"
fi

case "${1:-}" in
    ""|-h|--help|help)
        print_usage
        exit 0
        ;;
esac

project_root="$(cd "$(dirname "${BASH_SOURCE[0]}")" && pwd)"
architecture="$(uname -m)"
# aarch64 builds with the host compiler: OVSPATIALQUERY_COMPILER=host.
if [[ "${architecture}" != "x86_64" && "${architecture}" != "aarch64" ]]; then
    echo "ovspatialquery Linux bootstrap supports x86_64 and aarch64; found ${architecture}" >&2
    exit 2
fi

export PM_PACKAGES_ROOT="${PM_PACKAGES_ROOT:-${XDG_CACHE_HOME:-${HOME}/.cache}/packman}"
export PYTHONNOUSERSITE=1

app_venv="${project_root}/_build/venv/ovspatialqueryapp"
case "${OVSPATIALQUERY_TRAIN:-stable}" in
    stable) ;;
    next)
        # A separate venv and native tree; the stable environment is never touched.
        app_venv="${project_root}/_build/venv/ovspatialqueryapp-next"
        next_library="${project_root}/_build/bootstrap/native-next/install/lib/libovspatialquery.so"
        # The library override, and the viewer's own name for it.
        export OVSPATIALQUERY_LIBRARY="${OVSPATIALQUERY_LIBRARY:-${next_library}}"
        export OVSPATIALQUERY_APP_DLL="${OVSPATIALQUERY_APP_DLL:-${next_library}}"
        resolution="${project_root}/_build/next-train/resolution.json"
        if [[ -f "${resolution}" && -x "${app_venv}/bin/python" ]]; then
            # The viewer accepts exactly these builds in place of its locked pins.
            OVSPATIALQUERY_NEXT_TRAIN_VERSIONS="$("${app_venv}/bin/python" -c 'import json, sys; wheels = json.load(open(sys.argv[1]))["wheels"]; print(",".join("{}={}".format(name, wheel["version"]) for name, wheel in sorted(wheels.items())))' "${resolution}")"
            export OVSPATIALQUERY_NEXT_TRAIN_VERSIONS
        fi
        ;;
    *)
        echo "OVSPATIALQUERY_TRAIN must be stable or next; found ${OVSPATIALQUERY_TRAIN}" >&2
        exit 2
        ;;
esac

command_name="bootstrap"
if [[ $# -gt 0 ]]; then
    case "${1}" in
        menu|plan|bootstrap|configure|build|cross-build|test|app|verify|python-wheel|clean-bootstrap|example|demo|benchmark|format|check|lint|bump-version|docs)
            command_name="${1}"
            shift
            ;;
    esac
fi

packman_python="${PM_PACKAGES_ROOT}/python/3.12.13-nv1-manylinux_2_35-${architecture}/python"

if [[ "${command_name}" == "format" || "${command_name}" == "check" || "${command_name}" == "lint" || "${command_name}" == "bump-version" || "${command_name}" == "docs" ]]; then
    quality_python="${OVSPATIAL_PYTHON:-}"
    if [[ -z "${quality_python}" && -x "${app_venv}/bin/python" ]]; then
        quality_python="${app_venv}/bin/python"
    fi
    if [[ -z "${quality_python}" && -x "${packman_python}" ]]; then
        quality_python="${packman_python}"
    fi
    if [[ -z "${quality_python}" ]] && command -v python3 >/dev/null 2>&1; then
        quality_python="$(command -v python3)"
    fi
    if [[ -z "${quality_python}" ]]; then
        echo "${command_name} requires Python 3 on PATH to create the isolated tooling environment" >&2
        exit 2
    fi
    if [[ "${command_name}" == "bump-version" ]]; then
        exec "${quality_python}" "${project_root}/tools/format/release_version.py" "${@}"
    fi
    if [[ "${command_name}" == "docs" ]]; then
        exec "${quality_python}" "${project_root}/tools/docs/main.py" "${@}"
    fi
    exec "${quality_python}" "${project_root}/tools/format/main.py" "${command_name}" "${@}"
fi

if [[ "${command_name}" == "benchmark" ]]; then
    exec "${app_venv}/bin/python" "${project_root}/benchmarks/run.py" "${@}"
fi

if [[ "${command_name}" == "example" ]]; then
    source_python="${app_venv}/bin/python"
    if [[ ! -x "${source_python}" ]]; then
        echo "example requires the bootstrapped environment at ${source_python}" >&2
        exit 2
    fi
    exec "${source_python}" "${project_root}/tools/examples/main.py" "${@}"
fi

if [[ "${command_name}" == "demo" || "${command_name}" == "menu" ]]; then
    demo_python="${app_venv}/bin/python"
    if [[ ! -x "${demo_python}" ]]; then
        # The driver needs only the standard library; it reports the missing environment itself.
        demo_python="$(command -v python3 || true)"
        if [[ -z "${demo_python}" ]]; then
            echo "demo requires the bootstrapped environment or python3 on PATH" >&2
            exit 2
        fi
    fi
    if [[ "${command_name}" == "menu" ]]; then
        exec "${demo_python}" "${project_root}/tools/launch.py" "${@}"
    fi
    exec "${demo_python}" "${project_root}/tools/demos/main.py" "${@}"
fi

if [[ "${command_name}" == "cross-build" ]]; then
    exec "${project_root}/tools/packman/python.sh" "${project_root}/tools/ci/arm64_cross_build.py" "${@}"
fi

driver_arguments=(
    "${project_root}/tools/bootstrap/main.py"
    "${command_name}"
    --platform
    "manylinux_2_35_${architecture}"
)

# Prefer the Packman Python already in the cache; otherwise let the Packman
# adapter initialize the cache and fetch it.
if [[ -x "${packman_python}" ]]; then
    exec "${packman_python}" "${driver_arguments[@]}" "${@}"
fi
exec "${project_root}/tools/packman/python.sh" "${driver_arguments[@]}" "${@}"
