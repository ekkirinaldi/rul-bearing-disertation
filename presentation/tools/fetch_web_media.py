"""Fetch the openly licensed photographs the deck borrows from the web.

Every entry is a Wikimedia Commons file, chosen because Commons records the
author and licence in machine-readable form: the tool downloads a 1.600 px
rendition into ``assets/web/`` and writes ``assets/web/CREDITS.md`` from the
API metadata, so the credit line on the slide can be checked against it.

Usage: python3 tools/fetch_web_media.py  (or ``make web-assets``)
"""

from __future__ import annotations

import json
import re
import sys
import urllib.parse
import urllib.request
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
OUT = ROOT / "assets" / "web"
API = "https://commons.wikimedia.org/w/api.php"
HEADERS = {"User-Agent": "sidang-deck-builder/1.0 (ekki@vektor.international)"}
WIDTH = 1600

# local name -> Commons file title (slide that uses it in the comment)
MAP: dict[str, str] = {
    # slide "Mesin Rotasi Ada di Setiap Industri"
    "motor_listrik_cutaway.jpg": "File:Cut-away version of an electric motor (1).JPG",
    "fan_sentrifugal_industri.jpg": "File:High Pressure Centrifugal Fan.jpg",
    "pompa_cutaway.jpg": "File:Pump (Cut-Away).JPG",
    "mesin_grinding_spindle.jpg": "File:Cylindrical grinder.jpg",
}


def _api(**params) -> dict:
    params["format"] = "json"
    url = API + "?" + urllib.parse.urlencode(params)
    with urllib.request.urlopen(urllib.request.Request(url, headers=HEADERS), timeout=60) as r:
        return json.load(r)


def _strip(html: str) -> str:
    return re.sub(r"<[^>]+>", "", html).strip()


def main() -> int:
    OUT.mkdir(parents=True, exist_ok=True)
    titles = list(MAP.values())
    data = _api(action="query", titles="|".join(titles), prop="imageinfo",
                iiprop="url|size|extmetadata", iiurlwidth=WIDTH)
    by_title = {}
    for page in data["query"]["pages"].values():
        by_title[page["title"]] = page["imageinfo"][0]
    lines = ["# Credits for assets/web/", "",
             "Fetched by `make web-assets` from Wikimedia Commons; the slide that shows a",
             "file credits its author and licence in its `source` line.", ""]
    for name, title in MAP.items():
        info = by_title.get(title)
        if info is None:
            # Commons normalises some titles (quotes, spacing); fall back to a loose match.
            info = next((v for k, v in by_title.items()
                         if k.casefold().replace('"', '') == title.casefold().replace('"', '')), None)
        if info is None:
            print(f"not found on Commons: {title}", file=sys.stderr)
            return 1
        meta = info["extmetadata"]
        author = _strip(meta.get("Artist", {}).get("value", "unknown"))
        licence = meta.get("LicenseShortName", {}).get("value", "unknown")
        # The date the work is dated on Commons; the dissertation bibliography
        # cites a photograph by its year, so it has to be recorded here.
        dated = _strip(meta.get("DateTimeOriginal", {}).get("value", "")
                       or meta.get("DateTime", {}).get("value", ""))[:10] or "tanpa tahun"
        page_url = info["descriptionurl"]
        with urllib.request.urlopen(urllib.request.Request(info["thumburl"], headers=HEADERS),
                                    timeout=120) as r:
            (OUT / name).write_bytes(r.read())
        lines.append(f"- `{name}` — {title[5:]} — {author} — {licence} — {dated} — {page_url}")
        print(f"{name:32s} {author[:24]:24s} {licence:14s} {info['width']}x{info['height']} -> {WIDTH}px")
    (OUT / "CREDITS.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
