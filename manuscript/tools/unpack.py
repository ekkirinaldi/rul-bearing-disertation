#!/usr/bin/env python3
"""Unpack a .docx (ZIP) into a directory, pretty-printing XML parts for editing.

Usage: python3 unpack.py document.docx unpacked_dir/

Pretty-printing uses lxml, which honors xml:space="preserve" so meaningful
whitespace inside <w:t> runs survives the round trip (see pack.py).
"""
import sys
import zipfile
from pathlib import Path

from lxml import etree


def unpack(docx_path: str, out_dir: str) -> None:
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    with zipfile.ZipFile(docx_path) as zf:
        for name in zf.namelist():
            target = out / name
            target.parent.mkdir(parents=True, exist_ok=True)
            data = zf.read(name)
            if name.endswith((".xml", ".rels")):
                try:
                    root = etree.fromstring(data)
                    data = etree.tostring(
                        root, pretty_print=True, xml_declaration=True,
                        encoding="UTF-8", standalone=True)
                except etree.XMLSyntaxError:
                    pass  # leave non-parseable parts as-is
            target.write_bytes(data)
    print(f"Unpacked {docx_path} -> {out_dir}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    unpack(sys.argv[1], sys.argv[2])
