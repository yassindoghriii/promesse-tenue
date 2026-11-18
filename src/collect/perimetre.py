"""
Construit config/perimetre.csv (nom, lei) à partir de noms d'entreprises.

Pourquoi : l'API identifie les entreprises par leur LEI (Legal Entity Identifier, code
international à 20 caractères). Personne ne connaît ces codes par cœur : on part donc des noms
(config/perimetre_noms.csv) et on les retrouve dans la liste collectée.

Entrée  : config/perimetre_noms.csv  (colonnes nom, mot_cle)
          data/bronze/esef/filings_FR.jsonl (liste complète des dépôts)
Sortie  : config/perimetre.csv (colonnes nom, lei, nom_officiel, nb_depots)

Usage :
    python -m src.collect.perimetre
"""
from __future__ import annotations

import argparse
import csv
import json
import re
import unicodedata
from collections import defaultdict
from pathlib import Path


def normalize(s: str | None) -> str:
    """'L'Oréal S.A.' -> 'L OREAL S A' : majuscules, sans accents ni ponctuation."""
    s = unicodedata.normalize("NFKD", s or "").encode("ascii", "ignore").decode()
    return re.sub(r"[^A-Z0-9]+", " ", s.upper()).strip()


def match(noms: list[dict], filings: list[dict]) -> tuple[list[dict], list[str]]:
    """Associe chaque mot-clé aux entreprises dont le nom officiel le contient."""
    entities: dict[str, dict] = defaultdict(lambda: {"nom_officiel": None, "nb_depots": 0})
    for f in filings:
        if f.get("lei"):
            e = entities[f["lei"]]
            e["nom_officiel"] = f.get("entity_name")
            e["nb_depots"] += 1
    found, missing = [], []
    for n in noms:
        key = f" {normalize(n['mot_cle'])} "
        hits = [(lei, e) for lei, e in entities.items()
                if key in f" {normalize(e['nom_officiel'])} "]
        if not hits:
            missing.append(n["nom"])
        for lei, e in hits:
            found.append({"nom": n["nom"], "lei": lei, **e})
    return found, missing


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--noms", type=Path, default=Path("config/perimetre_noms.csv"))
    ap.add_argument("--filings", type=Path, default=Path("data/bronze/esef/filings_FR.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("config/perimetre.csv"))
    args = ap.parse_args()

    filings = [json.loads(l) for l in args.filings.read_text(encoding="utf-8").splitlines() if l.strip()]
    if not filings:
        raise SystemExit(f"{args.filings} est vide : relance d'abord la collecte complète.")
    with args.noms.open(encoding="utf-8") as f:
        noms = list(csv.DictReader(f))

    found, missing = match(noms, filings)
    with args.out.open("w", newline="", encoding="utf-8") as f:
        w = csv.DictWriter(f, fieldnames=["nom", "lei", "nom_officiel", "nb_depots"])
        w.writeheader()
        w.writerows(found)

    print(f"\n{len(found)} entité(s) retenue(s) -> {args.out}\n")
    for r in found:
        print(f"  {r['nom']:<20} {r['lei']}  {r['nom_officiel']}  ({r['nb_depots']} dépôts)")
    dup = {r["nom"] for r in found if sum(x["nom"] == r["nom"] for x in found) > 1}
    if dup:
        print(f"\n⚠ Plusieurs entités pour : {', '.join(sorted(dup))} -> "
              "garde la bonne ligne dans perimetre.csv et supprime les autres.")
    if missing:
        print(f"\n⚠ Introuvables : {', '.join(missing)} -> change le mot_cle dans {args.noms}.")


if __name__ == "__main__":
    main()
