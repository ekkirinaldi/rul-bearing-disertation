#!/usr/bin/env python3
"""Repack an unpacked docx directory (see unpack.py) into a .docx file.

Usage: python3 pack.py unpacked_dir/ output.docx

XML parts are re-condensed: the indentation whitespace added by unpack.py is
stripped (lxml remove_blank_text honors xml:space="preserve", so meaningful
run whitespace is kept). [Content_Types].xml is written first per OPC spec.
"""
import sys
import zipfile
from pathlib import Path

from lxml import etree

PARSER = etree.XMLParser(remove_blank_text=True)


def pack(src_dir: str, docx_path: str) -> None:
    src = Path(src_dir)
    files = sorted(p for p in src.rglob("*") if p.is_file())
    content_types = src / "[Content_Types].xml"
    ordered = [content_types] + [p for p in files if p != content_types]
    with zipfile.ZipFile(docx_path, "w", zipfile.ZIP_DEFLATED) as zf:
        for path in ordered:
            arcname = path.relative_to(src).as_posix()
            data = path.read_bytes()
            if arcname.endswith((".xml", ".rels")):
                root = etree.fromstring(data, PARSER)
                data = etree.tostring(
                    root, xml_declaration=True, encoding="UTF-8",
                    standalone=True)
            zf.writestr(arcname, data)
    print(f"Packed {src_dir} -> {docx_path}")


if __name__ == "__main__":
    if len(sys.argv) != 3:
        sys.exit(__doc__)
    pack(sys.argv[1], sys.argv[2])
