import json
from pathlib import Path

import pandas as pd

from src.collect.esef_filings import build_params, parse_page
from src.parse.xbrl_facts import extract_facts, fiscal_period, run

FIX = Path(__file__).parent / "fixtures"


def test_build_params_filtre_pays_et_periode():
    p = build_params("FR", "2020-01-01", "2025-12-31")
    filters = json.loads(p["filter"])
    assert {"name": "country", "op": "eq", "val": "FR"} in filters
    assert p["include"] == "entity"


def test_parse_page_aplatit_et_rend_les_urls_absolues():
    rows = parse_page(json.loads((FIX / "api_page.json").read_text()))
    assert rows[0]["entity_name"] == "Alpha SA"
    assert rows[0]["lei"] == "969500ALPHA"
    assert rows[0]["json_url"].startswith("https://filings.xbrl.org/")


def test_fiscal_period_retire_un_jour():
    assert fiscal_period("2023-01-01T00:00:00/2024-01-01T00:00:00") == ("2023-01-01", "2023-12-31", 365)


def test_extract_facts_garde_le_consolide_annuel():
    doc = json.loads((FIX / "alpha_2023.json").read_text())
    rows = extract_facts(doc, "4242")
    ca = sorted((r["exercice"], r["valeur"]) for r in rows if r["indicateur"] == "chiffre_affaires")
    # segment France (f3) et semestre (f4) exclus ; N et N-1 gardés
    assert ca == [(2022, 1.0e9), (2023, 1.05e9)]
    treso = [r for r in rows if r["indicateur"] == "tresorerie"][0]
    assert treso["fin"] == "2023-12-31" and treso["exercice"] == 2023


def test_run_bout_en_bout(tmp_path):
    meta = tmp_path / "filings.jsonl"
    rows = parse_page(json.loads((FIX / "api_page.json").read_text()))
    meta.write_text("\n".join(json.dumps(r) for r in rows))
    jdir = tmp_path / "json"
    jdir.mkdir()
    (jdir / "4242.json").write_text((FIX / "alpha_2023.json").read_text())
    df = run(meta, jdir, tmp_path / "out.csv")
    assert len(df) == 4  # CA 2022, CA 2023, résultat op. 2023, trésorerie 2023 (tous >= 2019)
    assert set(df["entreprise"]) == {"Alpha SA"}


class _Resp:
    def __init__(self, code, payload=None, headers=None):
        self.status_code, self._p, self.headers = code, payload or {}, headers or {}

    def json(self):
        return self._p

    def raise_for_status(self):
        if self.status_code >= 400:
            raise RuntimeError(self.status_code)


class _Session:
    def __init__(self, responses):
        self.responses, self.calls = list(responses), 0

    def get(self, *a, **k):
        self.calls += 1
        return self.responses.pop(0)


def test_fetch_reessaie_apres_429():
    from src.collect.esef_filings import fetch_filings
    page = json.loads((FIX / "api_page.json").read_text())
    s = _Session([_Resp(429, headers={"Retry-After": "3"}), _Resp(429), _Resp(200, page)])
    waits = []
    rows = fetch_filings(s, {}, sleep=waits.append)
    assert len(rows) == 1 and s.calls == 3
    assert waits == [3.0, 20]


def test_perimetre_retrouve_les_lei_par_nom():
    from src.collect.perimetre import match, normalize
    assert normalize("L'Oréal S.A.") == "L OREAL S A"
    filings = [
        {"lei": "LEI1", "entity_name": "L'Oréal"},
        {"lei": "LEI1", "entity_name": "L'Oréal"},
        {"lei": "LEI2", "entity_name": "Compagnie Générale des Établissements Michelin"},
        {"lei": "LEI3", "entity_name": "Orange"},
    ]
    found, missing = match([{"nom": "L'Oréal", "mot_cle": "OREAL"},
                            {"nom": "Michelin", "mot_cle": "MICHELIN"},
                            {"nom": "Danone", "mot_cle": "DANONE"}], filings)
    assert {(r["nom"], r["lei"]) for r in found} == {("L'Oréal", "LEI1"), ("Michelin", "LEI2")}
    assert [r["nb_depots"] for r in found if r["lei"] == "LEI1"] == [2]
    assert missing == ["Danone"]


def test_harmonize_ne_melange_pas_deux_notions():
    from src.parse.xbrl_facts import harmonize
    df = pd.DataFrame([
        {"lei": "A", "indicateur": "capitaux_propres", "concept": "ifrs-full:Equity", "exercice": 2019},
        {"lei": "A", "indicateur": "capitaux_propres", "concept": "ifrs-full:EquityAttributableToOwnersOfParent", "exercice": 2020},
        {"lei": "A", "indicateur": "chiffre_affaires", "concept": "ifrs-full:Revenue", "exercice": 2020},
        {"lei": "A", "indicateur": "chiffre_affaires", "concept": "ifrs-full:RevenueFromContractsWithCustomers", "exercice": 2021},
        {"lei": "A", "indicateur": "tresorerie", "concept": "ifrs-full:CashAndCashEquivalents", "exercice": 2017},
    ])
    h = harmonize(df)
    assert set(zip(h["indicateur"], h["exercice"])) == {
        ("capitaux_propres", 2020), ("chiffre_affaires", 2020), ("chiffre_affaires", 2021)}


def test_explore_trouve_les_extensions():
    from src.parse.explore_concepts import explore
    doc = {"facts": {"x": {"value": "43000000000", "dimensions": {
        "concept": "sny:NetSales", "entity": "s:L", "unit": "iso4217:EUR",
        "period": "2024-01-01T00:00:00/2025-01-01T00:00:00"}}}}
    df = explore(doc, "sales")
    assert df.iloc[0]["concept"] == "sny:NetSales" and df.iloc[0]["valeur_Md"] == 43.0


def test_fetch_reessaie_apres_coupure_reseau():
    import requests
    from src.collect.esef_filings import get_with_retry

    class S:
        calls = 0

        def get(self, *a, **k):
            S.calls += 1
            if S.calls == 1:
                raise requests.ConnectionError("Remote end closed connection")
            return _Resp(200)

    waits = []
    assert get_with_retry(S(), "https://x", sleep=waits.append).status_code == 200
    assert waits == [10]