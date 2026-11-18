# SPDX-FileCopyrightText: Copyright (c) 2026 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
# SPDX-License-Identifier: Apache-2.0

"""Split Linux debug information with the selected compiler's objcopy."""

from __future__ import annotations

import argparse
from pathlib import Path
import struct
import subprocess
import tempfile
import zlib


def verify(library: bytes, symbols: bytes, filename: str) -> None:
    """Check a transported ELF64 companion without requiring target binutils."""

    def section(elf: bytes, wanted: str) -> bytes:
        if elf[:6] != b"\x7fELF\x02\x01":
            raise RuntimeError("Expected a little-endian ELF64 symbol pair")
        offset = struct.unpack_from("<Q", elf, 40)[0]
        size, count, names_index = struct.unpack_from("<HHH", elf, 58)
        headers = [struct.unpack_from("<IIQQQQIIQQ", elf, offset + i * size) for i in range(count)]
        names_header = headers[names_index]
        names = elf[names_header[4] : names_header[4] + names_header[5]]
        for header in headers:
            name = names[header[0] :].split(b"\0", 1)[0].decode()
            if name == wanted:
                return elf[header[4] : header[4] + header[5]]
        raise RuntimeError(f"Missing ELF section {wanted}")

    for name in (".debug_info", ".debug_line"):
        if not section(symbols, name):
            raise RuntimeError(f"Empty DWARF section {name}")
    link = section(library, ".gnu_debuglink")
    if (
        link.split(b"\0", 1)[0].decode() != filename
        or int.from_bytes(link[-4:], "little") != zlib.crc32(symbols)
        or section(library, ".note.gnu.build-id") != section(symbols, ".note.gnu.build-id")
        or library[18:20] != symbols[18:20]
    ):
        raise RuntimeError("Debug symbols do not match the runtime identity")


def split(objcopy: Path, library: Path, symbols: Path) -> None:
    if not objcopy.is_file():
        raise RuntimeError("Selected compiler must provide objcopy")
    symbols.parent.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=library.parent) as directory:
        temporary = Path(directory)
        runtime = temporary / library.name

        def run(*args):
            subprocess.run([str(objcopy), *map(str, args)], check=True)

        def section(binary: Path, name: str) -> bytes:
            target = temporary / "section"
            target.unlink(missing_ok=True)
            run("--dump-section", f"{name}={target}", binary)
            return target.read_bytes()

        # Verify the executable text, dynamic exports, unwind tables and build ID
        # are unchanged, not merely that objcopy returned success.
        retained = {
            name: section(library, name)
            for name in (".text", ".dynsym", ".eh_frame", ".note.gnu.build-id")
        }
        run("--only-keep-debug", library, symbols)
        run("--strip-debug", library, runtime)
        run(f"--add-gnu-debuglink={symbols}", runtime)
        for name, original in retained.items():
            if section(runtime, name) != original:
                raise RuntimeError(f"Symbol split changed runtime section {name}")
        verify(runtime.read_bytes(), symbols.read_bytes(), symbols.name)
        runtime.replace(library)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument("--objcopy", type=Path, required=True)
    parser.add_argument("--library", type=Path, required=True)
    parser.add_argument("--symbols", type=Path, required=True)
    arguments = parser.parse_args()
    split(arguments.objcopy, arguments.library, arguments.symbols)
