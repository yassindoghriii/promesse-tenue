import pandas as pd

from src.ai.evaluation import echantillon, phrase_id, score


def _passages():
    rows = []
    for e in ("A", "B"):
        for i in range(10):
            rows.append({"filing_id": f"{e}1", "entreprise": e, "exercice_rapport": 2024,
                         "phrase": f"phrase {e}{i}", "lei": "L"})
    return pd.DataFrame(rows)


def test_echantillon_reproductible_et_equilibre():
    s1, s2 = echantillon(_passages(), par_entreprise=3), echantillon(_passages(), par_entreprise=3)
    assert list(s1["id"]) == list(s2["id"])
    assert s1["entreprise"].value_counts().to_dict() == {"A": 3, "B": 3}
    assert (s1["est_promesse"] == "").all()


def test_score_precision_rappel_f1():
    p = _passages()
    ids = [phrase_id(f, ph) for f, ph in zip(p["filing_id"], p["phrase"])]
    annot = pd.DataFrame({"id": ids[:4], "entreprise": "A", "phrase": p["phrase"][:4],
                          "est_promesse": ["oui", "oui", "non", "non"],
                          "annee_echeance": ["2026", "2027", "", ""],
                          "valeur_min": ["5", "3,5", "", ""], "valeur_max": ""})
    pred = p.iloc[:4].assign(est_promesse=[True, False, True, False],
                             annee_echeance=[2026, None, 2026, None],
                             valeur_min=[5.0, None, 1.0, None], valeur_max=None)
    r = score(annot, pred)
    assert (r["vrais_positifs"], r["faux_positifs"], r["faux_negatifs"], r["vrais_negatifs"]) == (1, 1, 1, 1)
    assert r["precision"] == 0.5 and r["rappel"] == 0.5 and r["f1"] == 0.5
    assert r["exactitude_echeance"] == 1.0 and r["exactitude_valeur_min"] == 1.0