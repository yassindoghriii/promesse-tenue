"""
Relecture humaine d'une pré-annotation faite par un second modèle (annotation « hybride »).

Pour limiter le biais d'ancrage (se laisser influencer par la réponse proposée) :
  1. la phrase s'affiche SEULE et tu donnes TON avis (o/n) avant de voir la proposition ;
  2. la pré-annotation est ensuite révélée ;
  3. tu valides (Entrée) ou tu corriges. Ta décision finale fait foi.

Ton avis initial est conservé : il permet de calculer l'ACCORD entre toi et le second modèle,
un indicateur classique de fiabilité d'une annotation.

Usage :
    python -m src.ai.relire
"""
from __future__ import annotations

import textwrap

import pandas as pd

from src.ai.evaluation import ANNOT


def save(df):
    df.to_csv(ANNOT, sep=";", index=False, encoding="utf-8-sig")


def ask(q, default=""):
    rep = input(f"{q} [{default}] : " if default else f"{q} : ").strip().replace(",", ".")
    return rep or default


def main():
    df = pd.read_csv(ANNOT, sep=";", dtype=str, encoding="utf-8-sig").fillna("")
    for col in ("avis_humain_initial", "valide_par_humain", "note_preannotation"):
        if col not in df:
            df[col] = ""
    todo = df.index[df["valide_par_humain"] != "oui"].tolist()
    print(f"\n{len(df) - len(todo)} phrases déjà relues, {len(todo)} restantes. (q = quitter)\n")
    for i in todo:
        r = df.loc[i]
        print("─" * 80)
        print(f"{r['entreprise']} — rapport {r['exercice_rapport']}\n")
        print(textwrap.fill(r["phrase"], width=80))
        avis = ""
        while avis not in {"o", "n", "q"}:
            avis = input("\nTON avis d'abord : promesse ? (o/n/q) : ").strip().lower()
        if avis == "q":
            break
        df.at[i, "avis_humain_initial"] = "oui" if avis == "o" else "non"
        prop = r["est_promesse"]
        detail = (f" | échéance {r['annee_echeance'] or '-'} | min {r['valeur_min'] or '-'}"
                  f" | max {r['valeur_max'] or '-'}") if prop == "oui" else ""
        print(f"\n  Pré-annotation : {prop.upper()}{detail}")
        if r["note_preannotation"]:
            print(f"  Note : {r['note_preannotation']}")
        final = "oui" if avis == "o" else "non"
        if final != prop:
            choix = ""
            while choix not in {"o", "n"}:
                choix = input("  Désaccord. Ta décision FINALE : promesse ? (o/n) : ").strip().lower()
            final = "oui" if choix == "o" else "non"
        df.at[i, "est_promesse"] = final
        if final == "oui":
            df.at[i, "annee_echeance"] = ask("  Année visée (Entrée = garder)", r["annee_echeance"])
            df.at[i, "valeur_min"] = ask("  Valeur min (Entrée = garder)", r["valeur_min"])
            df.at[i, "valeur_max"] = ask("  Valeur max (Entrée = garder, - = vider)", r["valeur_max"])
            if df.at[i, "valeur_max"] == "-":
                df.at[i, "valeur_max"] = ""
        else:
            df.loc[i, ["annee_echeance", "valeur_min", "valeur_max"]] = ""
        df.at[i, "valide_par_humain"] = "oui"
        save(df)

    relues = df[df["valide_par_humain"] == "oui"]
    if len(relues):
        accord = (relues["avis_humain_initial"] == relues["est_promesse"]).mean()
        print(f"\n{len(relues)}/{len(df)} relues. Ta décision finale = ton avis initial dans {accord:.0%} des cas.")
    if len(relues) == len(df):
        print("Tout est relu : lance  python -m src.ai.evaluation score")


if __name__ == "__main__":
    main()
