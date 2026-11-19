"""
Étape 4 (suite) — Évaluer la qualité de l'extraction des promesses par le LLM.

Principe : un humain annote à la main un échantillon de phrases (la « vérité terrain »),
puis on compare avec ce que le LLM a répondu sur ces mêmes phrases.

Deux commandes :
  1. python -m src.ai.evaluation echantillon
     -> crée data/eval/annotations.csv : 60 phrases tirées au hasard (6 par entreprise,
        tirage reproductible), avec des colonnes VIDES à remplir à la main (Excel ou Numbers).
  2. python -m src.ai.evaluation score
     -> compare tes annotations aux réponses du LLM et calcule :
        - précision : parmi les promesses trouvées par le LLM, combien sont vraies ?
        - rappel    : parmi les vraies promesses, combien le LLM a-t-il trouvées ?
        - F1        : moyenne harmonique des deux (objectif du cahier des charges : >= 0,80)
        - exactitude de l'échéance et de la valeur sur les promesses bien détectées.
"""
from __future__ import annotations

import argparse
import hashlib
import json
from pathlib import Path

import pandas as pd

EVAL_DIR = Path("data/eval")
ANNOT = EVAL_DIR / "annotations.csv"
COLS_A_REMPLIR = ["est_promesse", "annee_echeance", "valeur_min", "valeur_max"]


def phrase_id(filing_id, phrase: str) -> str:
    """Identifiant stable d'une phrase (même valeur dans les annotations et la sortie du LLM)."""
    return hashlib.sha1(f"{filing_id}|{phrase}".encode()).hexdigest()[:10]


def echantillon(passages: pd.DataFrame, par_entreprise: int = 6, seed: int = 42) -> pd.DataFrame:
    df = passages.copy()
    df["id"] = [phrase_id(f, p) for f, p in zip(df["filing_id"], df["phrase"])]
    sample = pd.concat([g.sample(min(par_entreprise, len(g)), random_state=seed)
                        for _, g in df.groupby("entreprise")])
    out = sample[["id", "entreprise", "exercice_rapport", "phrase"]].reset_index(drop=True)
    for c in COLS_A_REMPLIR:
        out[c] = ""
    return out


def _oui(v) -> bool | None:
    s = str(v).strip().lower()
    if s in {"oui", "o", "1", "true", "vrai", "yes", "x"}:
        return True
    if s in {"non", "n", "0", "false", "faux", "no"}:
        return False
    return None


def _num(v) -> float | None:
    try:
        s = str(v).replace(",", ".").strip()
        return float(s) if s and s.lower() != "nan" else None
    except ValueError:
        return None


def score(annotations: pd.DataFrame, predictions: pd.DataFrame) -> dict:
    a = annotations.copy()
    if "valide_par_humain" in a and (a["valide_par_humain"] == "oui").any():
        a = a[a["valide_par_humain"] == "oui"]       # annotation hybride : seules les lignes relues comptent
    a["verite"] = a["est_promesse"].map(_oui)
    a = a[a["verite"].notna()]                       # lignes non annotées ignorées
    p = predictions.copy()
    p["id"] = [phrase_id(f, ph) for f, ph in zip(p["filing_id"], p["phrase"])]
    m = a.merge(p[["id", "est_promesse", "annee_echeance", "valeur_min", "valeur_max"]],
                on="id", how="left", suffixes=("_humain", "_llm"))
    non_traitees = int(m["est_promesse_llm"].isna().sum())
    m = m[m["est_promesse_llm"].notna()]             # phrases pas encore analysées par le LLM : exclues
    m["llm"] = m["est_promesse_llm"].astype(bool)

    tp = int((m["verite"] & m["llm"]).sum())
    fp = int((~m["verite"].astype(bool) & m["llm"]).sum())
    fn = int((m["verite"].astype(bool) & ~m["llm"]).sum())
    tn = int((~m["verite"].astype(bool) & ~m["llm"]).sum())
    precision = tp / (tp + fp) if tp + fp else 0.0
    rappel = tp / (tp + fn) if tp + fn else 0.0
    f1 = 2 * precision * rappel / (precision + rappel) if precision + rappel else 0.0

    # Exactitude des champs, sur les vraies promesses bien détectées et annotées
    ok = m[m["verite"].astype(bool) & m["llm"]]
    def exact(col_h, col_l, tol=1e-6):
        pairs = [(_num(h), _num(l)) for h, l in zip(ok[col_h], ok[col_l]) if _num(h) is not None]
        return (sum(l is not None and abs(h - l) <= tol for h, l in pairs) / len(pairs)) if pairs else None

    erreurs = m[m["verite"].astype(bool) != m["llm"]][["entreprise", "phrase", "verite", "llm"]]
    return {
        "phrases_annotees": len(m), "non_traitees_par_llm": non_traitees, "vrais_positifs": tp, "faux_positifs": fp,
        "faux_negatifs": fn, "vrais_negatifs": tn,
        "precision": round(precision, 3), "rappel": round(rappel, 3), "f1": round(f1, 3),
        "exactitude_echeance": exact("annee_echeance_humain", "annee_echeance_llm"),
        "exactitude_valeur_min": exact("valeur_min_humain", "valeur_min_llm"),
        "erreurs": erreurs.to_dict(orient="records"),
    }


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("commande", choices=["echantillon", "score"])
    ap.add_argument("--passages", type=Path, default=Path("data/silver/passages.jsonl"))
    ap.add_argument("--promesses", type=Path, default=Path("data/silver/promesses.jsonl"))
    args = ap.parse_args()
    EVAL_DIR.mkdir(parents=True, exist_ok=True)

    if args.commande == "echantillon":
        if ANNOT.exists():
            raise SystemExit(f"{ANNOT} existe déjà : on ne l'écrase pas (tes annotations seraient perdues).")
        passages = pd.read_json(args.passages, lines=True, dtype={"filing_id": str})
        out = echantillon(passages)
        # séparateur « ; » et BOM : s'ouvre directement dans Excel / Numbers en français
        out.to_csv(ANNOT, sep=";", index=False, encoding="utf-8-sig")
        print(f"{len(out)} phrases à annoter dans {ANNOT}")
        return

    annotations = pd.read_csv(ANNOT, sep=";", dtype=str, encoding="utf-8-sig").fillna("")
    predictions = pd.read_json(args.promesses, lines=True, dtype={"filing_id": str})
    res = score(annotations, predictions)
    (EVAL_DIR / "rapport_evaluation.json").write_text(json.dumps(res, ensure_ascii=False, indent=2,
                                                                 default=str), encoding="utf-8")
    print(f"\nPhrases annotées et analysées par le LLM : {res['phrases_annotees']}"
          f"  (annotées mais pas encore analysées : {res['non_traitees_par_llm']})")
    print(f"Vrais positifs {res['vrais_positifs']} | faux positifs {res['faux_positifs']} | "
          f"faux négatifs {res['faux_negatifs']} | vrais négatifs {res['vrais_negatifs']}")
    print(f"\nPrécision : {res['precision']:.2f}   Rappel : {res['rappel']:.2f}   F1 : {res['f1']:.2f}"
          f"   (objectif F1 >= 0,80)")
    for k in ("exactitude_echeance", "exactitude_valeur_min"):
        if res[k] is not None:
            print(f"{k.replace('_', ' ').capitalize()} : {res[k]:.0%}")
    if res["erreurs"]:
        print("\nDésaccords humain / LLM (à analyser) :")
        for e in res["erreurs"]:
            qui = "faux positif" if e["llm"] else "faux négatif"
            print(f"  [{qui}] {e['entreprise']} : {e['phrase'][:140]}")


if __name__ == "__main__":
    main()
