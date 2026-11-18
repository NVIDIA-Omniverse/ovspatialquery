# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Compare the compiler's installed-header ABI witness with ctypes."""

from __future__ import annotations
import argparse
import ctypes
from enum import IntEnum, IntFlag
from pathlib import Path
import subprocess
import sys
from ovspatialquery import raw

ENUM_TYPES = {
    name: value
    for name, value in vars(raw).items()
    if isinstance(value, type)
    and issubclass(value, (IntEnum, IntFlag))
    and value.__module__ == raw.__name__
}


def verify(output: str, expect_ovstage: bool) -> None:
    observed_types: set[str] = set()
    observed_fields: set[tuple[str, str]] = set()
    observed_enums: set[tuple[str, str]] = set()
    for line in output.splitlines():
        columns = line.split("\t")
        if columns[0] == "T":
            _, name, size, alignment = columns
            projected = raw.TRANSPARENT_TYPES[name]
            if ctypes.sizeof(projected) != int(size):
                raise AssertionError(f"{name} size: ctypes={ctypes.sizeof(projected)} C={size}")
            if ctypes.alignment(projected) != int(alignment):
                raise AssertionError(
                    f"{name} alignment: ctypes={ctypes.alignment(projected)} C={alignment}"
                )
            observed_types.add(name)
        elif columns[0] == "F":
            _, name, field, offset = columns
            projected = raw.TRANSPARENT_TYPES[name]
            actual = getattr(projected, field).offset
            if actual != int(offset):
                raise AssertionError(f"{name}.{field} offset: ctypes={actual} C={offset}")
            observed_fields.add((name, field))
        elif columns[0] == "E":
            _, enum_name, member, value = columns
            actual = int(ENUM_TYPES[enum_name][member])
            if actual != int(value):
                raise AssertionError(f"{enum_name}.{member}: ctypes={actual} C={value}")
            observed_enums.add((enum_name, member))
        else:
            raise AssertionError(f"Unknown witness row: {line!r}")
    expected_types = {
        name for name in raw.TRANSPARENT_TYPES if expect_ovstage or not name.startswith("Ovstage")
    }
    if observed_types != expected_types:
        raise AssertionError(
            f"ABI type inventory mismatch: missing={sorted(expected_types - observed_types)} extra={sorted(observed_types - expected_types)}"
        )
    expected_fields = {
        (name, field_name)
        for name in expected_types
        for field_name, _ in raw.TRANSPARENT_TYPES[name]._fields_
    }
    if observed_fields != expected_fields:
        raise AssertionError(
            f"ABI field inventory mismatch: missing={sorted(expected_fields - observed_fields)} extra={sorted(observed_fields - expected_fields)}"
        )
    expected_enums = {
        (enum_name, member)
        for enum_name, enum_type in ENUM_TYPES.items()
        for member in enum_type.__members__
        if expect_ovstage or enum_name != "OvstageExclusionReason"
    }
    if observed_enums != expected_enums:
        raise AssertionError(
            f"ABI enum inventory mismatch: missing={sorted(expected_enums - observed_enums)} extra={sorted(observed_enums - expected_enums)}"
        )


def main() -> int:
    parser = argparse.ArgumentParser()
    parser.add_argument("--witness", type=Path, required=True)
    parser.add_argument("--expect-ovstage", action="store_true")
    options = parser.parse_args()
    completed = subprocess.run(
        [str(options.witness.resolve())], check=True, capture_output=True, text=True
    )
    verify(completed.stdout, options.expect_ovstage)
    print(
        f"verified ctypes ABI against {options.witness.resolve()} (ovstage={options.expect_ovstage})"
    )
    return 0


if __name__ == "__main__":
    sys.exit(main())
