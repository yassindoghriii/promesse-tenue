"""
Outil d'annotation dans le terminal pour le jeu de test (data/eval/annotations.csv).

Pour chaque phrase pas encore annotée : affiche la phrase, demande si c'est une promesse,
puis (si oui) l'échéance et la valeur. Enregistre après CHAQUE réponse : on peut quitter
(touche q) et reprendre plus tard là où on s'était arrêté.

Usage :
    python -m src.ai.annoter
"""
from __future__ import annotations

import textwrap

import pandas as pd

from src.ai.evaluation import ANNOT

AIDE = """
Règle : « oui » si L'ENTREPRISE s'engage sur un RÉSULTAT FUTUR CHIFFRÉ (financier ou climat).
« non » : résultat passé, hypothèse comptable, échéance de dette/contrat, rémunération,
prévision d'un tiers.   Touches : o = oui, n = non, q = quitter (tout est déjà enregistré).
"""


def save(df: pd.DataFrame) -> None:
    df.to_csv(ANNOT, sep=";", index=False, encoding="utf-8-sig")


def ask(question: str) -> str:
    return input(question).strip().replace(",", ".")


def main() -> None:
    df = pd.read_csv(ANNOT, sep=";", dtype=str, encoding="utf-8-sig").fillna("")
    todo = df.index[df["est_promesse"] == ""].tolist()
    print(AIDE)
    print(f"{len(df) - len(todo)} déjà annotées, {len(todo)} restantes.\n")
    for k, i in enumerate(todo, start=1):
        r = df.loc[i]
        print("─" * 80)
        print(f"[{len(df) - len(todo) + k}/{len(df)}] {r['entreprise']} — rapport {r['exercice_rapport']}\n")
        print(textwrap.fill(r["phrase"], width=80))
        rep = ""
        while rep not in {"o", "n", "q"}:
            rep = ask("\nPromesse ? (o/n/q) : ").lower()
        if rep == "q":
            break
        df.at[i, "est_promesse"] = "oui" if rep == "o" else "non"
        if rep == "o":
            df.at[i, "annee_echeance"] = ask("  Année visée (Entrée si aucune) : ")
            df.at[i, "valeur_min"] = ask("  Valeur visée, ou la plus basse d'une fourchette (Entrée si aucune) : ")
            df.at[i, "valeur_max"] = ask("  Valeur la plus haute d'une fourchette (Entrée si ce n'est pas une fourchette) : ")
        save(df)
    restant = int((df["est_promesse"] == "").sum())
    print(f"\nEnregistré. Restantes : {restant}." + ("" if restant else
          "  Tout est annoté : lance  python -m src.ai.evaluation score"))


if __name__ == "__main__":
    main()