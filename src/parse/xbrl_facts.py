"""
Étape 1 (suite) — Extraire les chiffres clés d'un fichier xBRL-JSON.

Un fichier xBRL-JSON contient des milliers de « faits » : un fait = un chiffre + son
étiquette standard (concept), sa période, son unité et éventuellement des « dimensions »
(ex. ventilation par pays ou par activité).

On ne garde ici que :
  - les concepts qui nous intéressent (chiffre d'affaires, résultat, etc.) ;
  - les faits « consolidés » (sans dimension supplémentaire = total du groupe) ;
  - les périodes d'environ 12 mois (exercice annuel).

Sortie : data/silver/esef_indicateurs.csv (une ligne = entreprise × indicateur × exercice).

Usage :
    python -m src.parse.xbrl_facts --meta data/bronze/esef/filings_FR.jsonl
"""
from __future__ import annotations

import argparse
import json
import logging
from datetime import datetime, timedelta
from pathlib import Path

import pandas as pd

log = logging.getLogger("xbrl")

# Indicateur métier -> concepts IFRS possibles (par ordre de préférence).
# Attention : beaucoup d'entreprises utilisent leurs propres concepts (« extensions ») pour
# des notions comme le résultat opérationnel courant ou la dette nette. C'est la
# « situation de recherche » à documenter dans le dossier.
CONCEPTS = {
    # Chaque indicateur -> groupes de concepts, par ordre de préférence.
    # Dans un même groupe, les concepts sont équivalents (on peut les mélanger d'une année à
    # l'autre). Deux groupes différents = deux notions différentes : on ne les mélange JAMAIS
    # pour une même entreprise (ex. capitaux propres « part du groupe » vs « totaux »).
    "chiffre_affaires": [["ifrs-full:Revenue", "ifrs-full:RevenueFromContractsWithCustomers"],
                         # Secours : Sanofi (ventes de biens = son chiffre d'affaires publié)
                         ["ifrs-full:RevenueFromSaleOfGoods"]],
    "resultat_operationnel": [["ifrs-full:ProfitLossFromOperatingActivities",
                               # LVMH (même valeur que l'IFRS en 2022 : 21,001 Md€)
                               "LVM:ProfitLossFromOperatingActivitiesAfterShareOfProfitLossOfAssociatesAndJointVentures"]],
    # TotalEnergies ne publie pas de résultat opérationnel : case laissée vide (choix assumé).
    "resultat_net": [["ifrs-full:ProfitLossAttributableToOwnersOfParent"], ["ifrs-full:ProfitLoss"]],
    "capitaux_propres": [["ifrs-full:EquityAttributableToOwnersOfParent"], ["ifrs-full:Equity"]],
    "tresorerie": [["ifrs-full:CashAndCashEquivalents"]],
}
# Rang du groupe de chaque concept (0 = groupe préféré)
GROUP_RANK = {c: i for groups in CONCEPTS.values() for i, g in enumerate(groups) for c in g}
MIN_EXERCICE = 2019  # avant 2019 : seulement quelques chiffres comparatifs isolés
BASE_DIMENSIONS = {"concept", "entity", "period", "unit", "language"}


def _parse_dt(s: str) -> datetime:
    return datetime.fromisoformat(s.replace("Z", ""))


def fiscal_period(period: str) -> tuple[str, str, int] | None:
    """
    '2023-01-01T00:00:00/2024-01-01T00:00:00' -> ('2023-01-01', '2023-12-31', 365)
    En xBRL-JSON, la fin d'une durée est le début du jour suivant : on retire 1 jour.
    Retourne None pour un instant (bilan) -> traité à part.
    """
    if "/" not in period:
        return None
    start, end = (_parse_dt(p) for p in period.split("/"))
    end_incl = end - timedelta(days=1)
    return start.date().isoformat(), end_incl.date().isoformat(), (end - start).days


def instant_date(period: str) -> str:
    """Un instant '2024-01-01T00:00:00' désigne la fin du 31/12/2023."""
    return (_parse_dt(period) - timedelta(days=1)).date().isoformat()


def extract_facts(doc: dict, filing_id: str) -> list[dict]:
    """Extrait les faits utiles d'un document xBRL-JSON."""
    wanted = {c: ind for ind, groups in CONCEPTS.items() for g in groups for c in g}
    rows = []
    for fact in (doc.get("facts") or {}).values():
        dims = fact.get("dimensions", {})
        concept = dims.get("concept")
        if concept not in wanted:
            continue
        if set(dims) - BASE_DIMENSIONS:  # fait ventilé (segment) -> on l'ignore
            continue
        try:
            value = float(fact.get("value"))
        except (TypeError, ValueError):
            continue
        period = dims.get("period", "")
        fp = fiscal_period(period)
        if fp:
            start, end, days = fp
            if not 350 <= days <= 380:  # on ne garde que les exercices annuels
                continue
        else:
            start, end = None, instant_date(period)
        rows.append({
            "filing_id": filing_id,
            "lei": dims.get("entity", "").split(":")[-1],
            "indicateur": wanted[concept],
            "concept": concept,
            "debut": start,
            "fin": end,
            "exercice": int(end[:4]) if end[5:7] >= "06" else int(end[:4]) - 1,
            "valeur": value,
            "unite": dims.get("unit", "").split(":")[-1],
        })
    return rows


def deduplicate(df: pd.DataFrame) -> pd.DataFrame:
    """
    Un même chiffre peut apparaître plusieurs fois (rapport N et rapport N+1 qui reprend N
    en comparatif) et sous plusieurs concepts. Règles :
      1. on préfère le concept le plus haut dans la liste CONCEPTS ;
      2. on garde le chiffre du rapport le plus récent (il peut avoir été retraité).
    """
    if df.empty:
        return df
    df = df.assign(_rank=df["concept"].map(GROUP_RANK))
    df = df.sort_values(["lei", "indicateur", "exercice", "_rank", "filing_period_end"],
                        ascending=[True, True, True, True, False])
    return df.drop_duplicates(["lei", "indicateur", "exercice"]).drop(columns="_rank")


def harmonize(df: pd.DataFrame, min_exercice: int = MIN_EXERCICE) -> pd.DataFrame:
    """
    Cohérence dans le temps : pour une entreprise et un indicateur, on garde un seul groupe de
    concepts (le meilleur disponible), sinon la série mélange deux notions et les variations
    d'une année à l'autre n'ont plus de sens. On retire aussi les exercices trop anciens.
    """
    if df.empty:
        return df
    df = df[df["exercice"] >= min_exercice]
    rank = df["concept"].map(GROUP_RANK)
    best = rank.groupby([df["lei"], df["indicateur"]]).transform("min")
    return df[rank == best].reset_index(drop=True)


def coverage(df: pd.DataFrame) -> pd.DataFrame:
    """Tableau de couverture : nombre d'exercices disponibles par entreprise et indicateur."""
    return (df.pivot_table(index="entreprise", columns="indicateur", values="exercice",
                           aggfunc=lambda s: f"{s.min()}-{s.max()} ({s.nunique()})")
              .fillna("MANQUANT"))


def run(meta_path: Path, json_dir: Path, out_path: Path) -> pd.DataFrame:
    meta = pd.read_json(meta_path, lines=True)
    all_rows = []
    for m in meta.itertuples():
        f = json_dir / f"{m.filing_id}.json"
        if not f.exists():
            continue
        doc = json.loads(f.read_text(encoding="utf-8"))
        for r in extract_facts(doc, m.filing_id):
            r["entreprise"] = m.entity_name
            r["filing_period_end"] = m.period_end
            all_rows.append(r)
    df = harmonize(deduplicate(pd.DataFrame(all_rows)))
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_csv(out_path, index=False)
    log.info("%s lignes écrites dans %s", len(df), out_path)
    if not df.empty:
        cov = coverage(df)
        cov.to_csv(out_path.parent / "couverture.csv")
        print("\nCouverture (exercices disponibles) :\n")
        print(cov.to_string())
    return df


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=Path("data/bronze/esef/filings_FR_perimetre.jsonl"))
    ap.add_argument("--json-dir", type=Path, default=Path("data/bronze/esef/json"))
    ap.add_argument("--out", type=Path, default=Path("data/silver/esef_indicateurs.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    run(args.meta, args.json_dir, args.out)


if __name__ == "__main__":
    main()