"""
Outil d'exploration : quels concepts une entreprise utilise-t-elle pour un indicateur ?

Sert quand un indicateur est MANQUANT dans le tableau de couverture : l'entreprise a sans
doute utilisé sa propre étiquette (« extension », ex. sny:NetSales) au lieu d'un concept IFRS.

Usage :
    python -m src.parse.explore_concepts --entreprise SANOFI --mot "revenue|sales"
    python -m src.parse.explore_concepts --entreprise LVMH --mot "operat|recurring"
"""
from __future__ import annotations

import argparse
import json
import re
from pathlib import Path

import pandas as pd

from src.parse.xbrl_facts import BASE_DIMENSIONS, fiscal_period


def explore(doc: dict, pattern: str) -> pd.DataFrame:
    rx = re.compile(pattern, re.I)
    rows = []
    for fact in (doc.get("facts") or {}).values():
        dims = fact.get("dimensions", {})
        concept = dims.get("concept", "")
        if not rx.search(concept) or set(dims) - BASE_DIMENSIONS:
            continue
        fp = fiscal_period(dims.get("period", ""))
        if not fp or not 350 <= fp[2] <= 380:
            continue
        try:
            value = float(fact.get("value"))
        except (TypeError, ValueError):
            continue
        rows.append({"concept": concept, "fin": fp[1], "valeur_Md": round(value / 1e9, 3),
                     "unite": dims.get("unit", "").split(":")[-1]})
    if not rows:
        return pd.DataFrame()
    return (pd.DataFrame(rows).drop_duplicates()
              .sort_values(["concept", "fin"]).reset_index(drop=True))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--entreprise", required=True, help="morceau du nom (ex. SANOFI)")
    ap.add_argument("--mot", default="revenue|sales|operat|recurring",
                    help="expression cherchée dans le nom du concept")
    ap.add_argument("--meta", type=Path, default=Path("data/bronze/esef/filings_FR_perimetre.jsonl"))
    ap.add_argument("--json-dir", type=Path, default=Path("data/bronze/esef/json"))
    args = ap.parse_args()

    meta = pd.read_json(args.meta, lines=True)
    meta = meta[meta["entity_name"].str.contains(args.entreprise, case=False, na=False)]
    if meta.empty:
        raise SystemExit("aucune entreprise ne correspond")
    last = meta.sort_values("period_end").iloc[-1]  # rapport le plus récent
    f = args.json_dir / f"{last.filing_id}.json"
    print(f"{last.entity_name} — rapport {last.period_end} ({f})\n")
    df = explore(json.loads(f.read_text(encoding="utf-8")), args.mot)
    pd.set_option("display.width", 200)
    print(df.to_string() if not df.empty else "aucun concept trouvé")


if __name__ == "__main__":
    main()
