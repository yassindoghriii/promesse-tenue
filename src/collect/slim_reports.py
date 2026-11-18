"""
Allège les rapports XHTML en retirant les images intégrées (encodées en base64).

Pourquoi : un rapport annuel pèse 10 à 160 Mo, surtout à cause des images (graphiques,
photos, logos) intégrées directement dans le fichier. Le projet n'exploite que le texte :
retirer ces images divise la taille par 5 à 20, sans rien perdre d'utile.
Choix documenté : la zone bronze contient le rapport « allégé » (texte et balises XBRL
intacts), pas une copie octet par octet. L'original reste téléchargeable via report_url.

Usage :
    python -m src.collect.slim_reports            # traite data/bronze/reports/*.xhtml.gz
"""
from __future__ import annotations

import argparse
import gzip
import logging
import re
from pathlib import Path

log = logging.getLogger("slim")

# src="data:image/png;base64,iVBORw0..."  ou  url(data:image/svg+xml;base64,...)
BASE64_IMAGE = re.compile(rb"data:image/[\w.+-]+;base64,[A-Za-z0-9+/=\s]+")
MARKER = b"<!-- allege:images-retirees -->"


def slim(content: bytes) -> bytes:
    """Remplace chaque image intégrée par une image vide ; marque le fichier comme allégé."""
    if MARKER in content[:200]:
        return content
    return MARKER + BASE64_IMAGE.sub(b"data:,", content)


def slim_file(path: Path) -> tuple[int, int]:
    raw = gzip.decompress(path.read_bytes())
    if MARKER in raw[:200]:
        return len(raw), len(raw)
    light = slim(raw)
    tmp = path.with_suffix(".tmp")
    tmp.write_bytes(gzip.compress(light, compresslevel=6))
    tmp.replace(path)                    # remplacement atomique : pas de fichier à moitié écrit
    return len(raw), len(light)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--dir", type=Path, default=Path("data/bronze/reports"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    before = after = 0
    for f in sorted(args.dir.glob("*.xhtml.gz")):
        size_before = f.stat().st_size
        slim_file(f)
        before += size_before
        after += f.stat().st_size
        log.info("%s : %.1f Mo -> %.1f Mo", f.name, size_before / 1e6, f.stat().st_size / 1e6)
    log.info("total : %.0f Mo -> %.0f Mo", before / 1e6, after / 1e6)


if __name__ == "__main__":
    main()