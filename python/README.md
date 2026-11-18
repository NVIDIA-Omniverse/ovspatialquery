# NVIDIA ovspatialquery

[![Documentation](https://img.shields.io/badge/docs-ovspatialquery-blue)](https://nvidia-omniverse.github.io/ovspatialquery/)
[![Repository](https://img.shields.io/badge/github-NVIDIA--Omniverse%2Fovspatialquery-181717)](https://github.com/NVIDIA-Omniverse/ovspatialquery)
[![License](https://img.shields.io/badge/license-Apache%202.0-blue)](https://github.com/NVIDIA-Omniverse/ovspatialquery/blob/main/LICENSE)

**ovspatialquery** is a C and Python library for spatial queries: raycasts, sphere and box
sweeps, overlaps, and bounding-box pairs over geometry that a simulation, an animation system,
or a tool publishes. It runs on the CPU with no GPU required, and on CUDA devices when you ask
for it. It performs no dynamics simulation.

It integrates with the NVIDIA Omniverse [ovstage](https://github.com/NVIDIA-Omniverse/ovstage)
library: it publishes the prims of an ovstage stage, with their shapes and world transforms, at
the ordinals ovstage seals. Geometry that does not live in a stage can be published from arrays
your program supplies. Inputs and results are any array with `__dlpack__`, such as NVIDIA Warp,
PyTorch, or CuPy arrays, on the CPU or on a CUDA device.

> **Pre-release:** while the version is 0.x, a minor release may change any declaration.
> Experimental declarations, including pair discovery, may change in any minor release.

## Install

Python 3.11, 3.12, and 3.13 are supported on Linux x86-64 and AArch64 (glibc 2.35 or newer) and
on Windows x86-64. Each wheel contains the native library and has no Python dependencies:

```bash
pip install ovspatialquery
```

Install ovstage 0.3 or 0.4 only to publish geometry from an ovstage stage:

```bash
pip install "ovstage==0.4.*"
```

Queries run on the CPU without a GPU, a CUDA driver, or a CUDA Toolkit. CUDA work needs a GPU of
compute capability 7.5 (Turing) or newer and an NVIDIA driver for CUDA 12.8 (R570) or newer:
570.26 on Linux, 570.65 on Windows.

Check the installation by printing the native library version:

```bash
python -c "from ovspatialquery import Library; print(Library.open().version)"
```

## Documentation

The [user guide](https://nvidia-omniverse.github.io/ovspatialquery/) covers
[installation](https://nvidia-omniverse.github.io/ovspatialquery/install.html),
[a first query](https://nvidia-omniverse.github.io/ovspatialquery/first-query.html),
[the concepts](https://nvidia-omniverse.github.io/ovspatialquery/concepts/index.html), and
[the Python API](https://nvidia-omniverse.github.io/ovspatialquery/reference/python.html).
The source, the C SDK, rendered lessons, and skills for AI coding agents are at
<https://github.com/NVIDIA-Omniverse/ovspatialquery>, and
[CHANGELOG.md](https://github.com/NVIDIA-Omniverse/ovspatialquery/blob/main/CHANGELOG.md) lists
what each release changes.

## Support

Report documentation issues, installation problems, and runtime issues through the
[NVIDIA Omniverse developer forum](https://forums.developer.nvidia.com/c/omniverse/300).

## License

ovspatialquery is licensed under the
[Apache License 2.0](https://github.com/NVIDIA-Omniverse/ovspatialquery/blob/main/LICENSE). The
wheel also contains the third-party notices listed in
[NOTICE](https://github.com/NVIDIA-Omniverse/ovspatialquery/blob/main/NOTICE).
