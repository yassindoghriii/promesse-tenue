"""
Étape 3 — Collecte du texte des rapports annuels (données NON structurées).

Pour chaque dépôt du périmètre, on télécharge le rapport complet au format XHTML
(report_url) : c'est le document lisible par un humain, celui où les dirigeants écrivent
leurs objectifs. Il est rangé en zone bronze, COMPRESSÉ (gzip, 5 à 10 fois plus petit) :
data/bronze/reports/<filing_id>.xhtml.gz

Idempotent : un rapport déjà téléchargé n'est pas re-téléchargé.

Usage :
    python -m src.collect.reports
    python -m src.collect.reports --limit 2      # pour tester sur 2 rapports
"""
from __future__ import annotations

import argparse
import gzip
import logging
import time
from pathlib import Path

import pandas as pd
import requests

from src.collect.esef_filings import PAUSE_S, USER_AGENT, get_with_retry
from src.collect.slim_reports import slim

log = logging.getLogger("reports")
OUT_DIR = Path("data/bronze/reports")


def download_reports(session: requests.Session, meta: pd.DataFrame, out_dir: Path = OUT_DIR,
                     sleep=time.sleep) -> tuple[int, int]:
    """Retourne (téléchargés, déjà présents)."""
    out_dir.mkdir(parents=True, exist_ok=True)
    done = skipped = 0
    for m in meta.itertuples():
        if not isinstance(m.report_url, str):
            log.warning("pas de rapport XHTML pour %s (%s)", m.entity_name, m.filing_id)
            continue
        target = out_dir / f"{m.filing_id}.xhtml.gz"
        if target.exists() and target.stat().st_size > 0:
            skipped += 1
            continue
        resp = get_with_retry(session, m.report_url, timeout=300)
        if resp.status_code != 200:
            log.warning("échec %s : HTTP %s", m.report_url, resp.status_code)
            continue
        target.write_bytes(gzip.compress(slim(resp.content), compresslevel=6))  # sans images
        done += 1
        log.info("rapport téléchargé : %s %s (%.1f Mo, %.1f Mo compressé)", m.entity_name,
                 m.period_end, len(resp.content) / 1e6, target.stat().st_size / 1e6)
        sleep(PAUSE_S)
    return done, skipped


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=Path("data/bronze/esef/filings_FR_perimetre.jsonl"))
    ap.add_argument("--limit", type=int, help="nombre maximum de rapports (tests)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta = pd.read_json(args.meta, lines=True)
    # Quand une entreprise a republié un rapport corrigé, on ne garde que la dernière version
    meta = meta.sort_values("date_added").drop_duplicates(["lei", "period_end"], keep="last")
    if args.limit:
        meta = meta.head(args.limit)
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT
    done, skipped = download_reports(session, meta)
    log.info("%s rapport(s) téléchargé(s), %s déjà présent(s)", done, skipped)


if __name__ == "__main__":
    main()