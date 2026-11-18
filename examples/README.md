# ovspatialquery Examples

The numbered examples are small rendered lessons for the ovspatialquery C API and Python package.
Each lesson teaches one new concept. Lessons 01 to 10 each have a Python program and a C++
program that do the same work. The C++ program calls the C API directly; it is not a second
wrapper. Lesson 11 combines one C API producer with a Python presenter. Every lesson keeps its
ovstage inputs, spatial descriptors, publication, queries, source ordinals, result checks, and
cleanup in the lesson file, where you can read them.

The lessons run on Linux x86-64 and Windows x86-64; the Linux AArch64 SDK includes their sources
only. They render their scenes with the ovRTX renderer, so they need an NVIDIA GPU even when their
queries run on the CPU. They also need ovstage and NVIDIA Warp (`warp-lang`); Warp installs NumPy,
which the lessons do not import. These packages are not dependencies of the library, and the SDK
and wheel do not bundle them. ovstage loads the lessons' inline USD scenes; the lessons need no
Python USD bindings.

The lessons support these package versions:

| Package | Supported Versions |
| --- | --- |
| ovstage | 0.4 |
| ovRTX | 0.7 |
| NVIDIA Warp (`warp-lang`) | 1.18 |

ovRTX 0.7 works only with ovstage 0.4, so the lessons need ovstage 0.4 even though the library
also supports 0.3.

## Curriculum

Each lesson builds on the lessons listed in its last column:

| No. | Lesson | New Concept | Principal Public Surface | Depends On |
| ---: | --- | --- | --- | --- |
| 01 | `static_candidate_raycast` | Synchronous bounding-box raycast | Core instance, ovstage population, `RaycastQuery`, closest hit | None |
| 02 | `sphere_sweep` | Swept sphere | `SphereSweepQuery`, closest hit | 01 |
| 03 | `shape_overlap` | Stateless caller-shape overlap | `OverlapQuery` with a sphere, any hit | 01 |
| 04 | `filtered_view` | Immutable target subset | ovstage view filter, view target | 01 |
| 05 | `two_set_pair_snapshot` | Object-pair bounding-box contacts | Pair snapshot and both source ordinals | 04 |
| 06 | `precise_triangle` | Exact triangle ray hit | Triangle population, `EXACT`, source face and barycentrics | 01 |
| 07 | `time_sampled_update` | Publish a new USD time sample | ovstage time update and new source ordinal | 01 |
| 08 | `reusable_cpu_batch` | Reuse one batch plan, on the GPU when there is one | Raycast batch plan, two launches, result tensors | 01 |
| 09 | `immutable_version_overlap` | A query pins the version it accepted | In-flight batch operation across the next publication | 07, 08 |
| 10 | `pair_reporter_deltas` | Stateful pair baseline and differences | Pair reporter, entered and exited deltas | 05, 07 |
| 11 | `capability_showcase` | Combined installed capabilities | Core, ovstage participation, explicit CPU and CUDA plans | 01 to 10 |

The Python programs are under [`python/`](python/) and the C++ programs under
[`native/`](native/), with the same file name and number.

## What the Lessons Show

Lessons 08 to 10 run their plan or pair reporter on the first CUDA device when there is one, and
on the CPU otherwise. Before they create it, they ask the library whether it supports that work on
the device, so a library without its CUDA backend runs them on the CPU. After the library accepts
CUDA work, the lessons never move it to the CPU. The Python programs pass Warp arrays to the
library and read the DLPack result columns with Warp, on the same device. On CUDA, the copy of the
columns waits on the `result_ready` event, and one aggregate `consumer_done` event retires the
operation. The lessons copy to the host only the values they record, after the device copy. The
file name of lesson 08 still says `cpu`, so that its commands and output directories stay the same.
Every Python lesson also checks its rendered frame with Warp kernels, on the GPU when there is one.

Lesson 11 sets a participation map on an ovstage population and checks which objects the
published version accepts and rejects. It also publishes explicit geometry and queries it on the
CPU and, when you request it, on a CUDA device. It does not run CUDA queries against an
ovstage-backed target.

The curriculum pairs Python with C++ because a C copy of every lesson would add syntax rather than
a new concept. [`native_consumer_c.c`](../tests/integration/installed/native_consumer_c.c) is a
pure C program that uses the installed SDK.

## Set Up the Lessons

Build the library with its CUDA backend from the repository root. On Linux, run this command:

```bash
./dev.sh bootstrap --scope library-cuda
```

On Windows, run `dev.cmd bootstrap -Scope Library-Cuda`. The SDK installs under
`_build/bootstrap/native-stable/install`, where the lesson runner finds it.

You can run a lesson in two ways:

| Command | When to Use It |
| --- | --- |
| `./dev.sh example 01` (`dev.cmd example 01` on Windows) | The bootstrap environment contains the lesson packages, including ovRTX; a `library` or `library-cuda` scope does not |
| `python tools/examples/main.py 01` | Any Python 3.12 environment that contains the lesson packages and the ovspatialquery package, such as a separate lesson environment |

A bootstrap with `--scope library` or `--scope library-cuda` leaves the lesson packages out of its
environment. In that case, create a separate CPython 3.12 lesson environment. Source releases
include hash-pinned `examples/requirements-<platform>.txt` files, derived from the same pins as the
library. On Linux x86-64, run these commands:

```bash
python3.12 -m venv _build/venv/lessons
_build/venv/lessons/bin/python -m pip install --require-hashes --no-deps -r examples/requirements-manylinux_2_35_x86_64.txt
_build/venv/lessons/bin/python -m pip install --no-deps -e ./python
_build/venv/lessons/bin/python tools/examples/main.py 01 --python _build/venv/lessons/bin/python --headless
```

On Windows, use `py -3.12 -m venv _build/venv/lessons`, the environment's `Scripts/python.exe`,
and `examples/requirements-windows-x86_64.txt` instead. The editable install makes the Python
package from this source tree available to the lesson processes, and the runner selects the built
SDK. The lesson packages do not become requirements of the installed library or of the bootstrap
environment.

## Run a Lesson

Both commands run the Python program by default and write their output to a new directory under
`_build/evidence/examples/manual/`. They accept these options:

| Option | Effect |
| --- | --- |
| `--language python`, `native` or `both` | Run the Python program, the C++ program, or both and compare their results |
| `--headless` | Render a fixed number of frames and exit, without opening a window |
| `--frames N` | Number of headless frames to render (default 3) |
| `--evidence DIR` | Write the output to `DIR` |
| `--require-cuda` | Lesson 11 only: fail unless its CUDA lane runs on a device |

For example, `./dev.sh example 01 --language both --headless` runs both programs of lesson 01 and
compares their results. On Windows, `dev.cmd` and `dev.ps1` take the same options in PowerShell
form: `-Language both -Headless`.

Each run renders a fixed 1280×720 scene with a deterministic camera, lighting, and render
ordinals, validates the frame, and writes `semantic.json`. The JSON file records the inputs, the
stable USD paths, the exact source ordinals, the query results, the identities of the installed
headers and native library, errors, and cleanup. The C++ lessons also write the frame to
`frame.png`. The Python lessons need no image-writing package and write no image file.

Without `--headless`, the lesson writes the same output first, then opens a viewport that orbits the
scene until you press Escape or close the window. The orbit moves only the camera: it does not
republish a population, rerun a query, or change the recorded results. On Linux, the viewport needs
a graphical session with `DISPLAY` set; use `--headless` in a terminal without one.

The locked ovstage Python package can print a startup notice when its optional physics-schema
bundle is not registered. The lessons author no physics-schema properties, so the notice does not
affect their results.

## Report Format Versions

The two kinds of lesson write different versions of the `semantic.json` report:

| Version | Written By | Frame Evidence |
| --- | --- | --- |
| 2 | Python lessons | Frame width, height, nonblack-pixel count, and unique-color count, with no image path or digest |
| 1 | C++ lessons | `frame.png`, with its path and SHA-256 digest in the `png` and `sha256` fields of the `evidence` object |

The runner reads both versions, verifies the PNG path and digest of a version 1 report, and rejects
any other version. When it compares the two programs of a lesson, it compares their inputs and
query results and ignores this difference in frame evidence. Its `index.json` uses version 2 and
indexes the reports without image entries. A reader of Python reports must not expect the PNG
fields. These report formats belong to the lessons; they are not part of the library API.

## Build the C++ Lessons

The C++ programs build against the installed SDK and the ovstage and ovRTX SDK directories that
those packages supply. On Linux, configuration also needs the host's X11 development headers and
`libX11`. Configure and build them into the directory that the runner reads by default:

```bash
cmake -S examples -B _build/examples/native-vs \
  -DOVSPATIALQUERY_INSTALL_DIR=<sdk-install> \
  -DOVSTAGE_DIR=<ovstage-root> \
  -DOVRTX_DIR=<ovrtx-root>
cmake --build _build/examples/native-vs --config Release
```

Then pass `--language native` or `--language both`. To use another build directory, pass it with
`--native-dir`.

The shared helpers handle only argument parsing, evidence files, ovstage and ovRTX startup, scene
loading, rendering, and cleanup. The Python helpers are [`_runtime.py`](python/_runtime.py) and
[`_evidence.py`](python/_evidence.py). The C++ helpers are
[`example_runtime.h`](native/example_runtime.h),
[`example_evidence.h`](native/example_evidence.h), and
[`example_ovstage.h`](native/example_ovstage.h), which holds the ovstage 0.3 and 0.4 spellings.
The helpers never build a spatial descriptor, publish a population, choose a precision, issue a
query, or interpret a hit. Those steps stay in each lesson.

## Lesson 11: Installed Capability Showcase

Lesson 11 runs all its stations in one scene. Run it with `./dev.sh example 11`
(`dev.cmd example 11` on Windows) or `python tools/examples/main.py 11`. It runs on Linux x86-64
and Windows x86-64, and it needs CMake, CTest, and a C++17 compiler in addition to the lesson
packages.

Lesson 11 is not a Python and C++ pair. One native C API program, in
[`native/11_capability_showcase.cpp`](native/11_capability_showcase.cpp), produces every query
result. The Python program, [`python/11_capability_showcase.py`](python/11_capability_showcase.py),
configures and builds it with [`11_capability_showcase/CMakeLists.txt`](11_capability_showcase/CMakeLists.txt),
runs its CTest check, validates its JSON output, and renders the scene. Each station shows what the
native program reported:

| Station | What It Shows |
| --- | --- |
| ovstage | The participation bits that the published version accepted and rejected |
| Explicit | The published objects, a bounding-box hit, and a pair |
| CUDA | Typed raycast, sphere-sweep, and overlap plans run on a device |

By default, the lesson records the core, ovstage participation, and explicit CPU results. It also
runs the `capability_showcase_cpu_headless_contract` CTest check. The CUDA station reports
`NOT_AVAILABLE` and shows an unavailable marker instead of results.

With `--require-cuda`, the lesson also configures the producer with
`-DOVSPATIALQUERY_ENABLE_CUDA_CONTRACT=ON` and runs `capability_showcase_cuda_headless_contract`,
which fails unless the CUDA lane runs on a device. The CUDA lane passes device DLPack inputs to the
plans, waits on the `result_ready` events, validates the device result columns, and retires each
operation with one aggregate `consumer_done` event. It never falls back to the CPU or copies
results to the host in place of a device query. The CUDA lane includes a shared driver header from
[`support/native/`](../support/native/), so build lesson 11 from the source tree, not from a copy of
`examples/` alone.

## Other Example Programs

The complete programs under [`docs/site/snippets/`](../docs/site/snippets/) need no renderer.
After you build the wheel, check them with `./dev.sh docs --check-snippets`, and add
`--require-cuda` on a CUDA host. Source releases also include the built user guide under
`docs/html/`; its Examples page describes these lessons.
