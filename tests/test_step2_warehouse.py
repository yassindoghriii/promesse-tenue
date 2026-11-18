"""
Test d'intégration de l'entrepôt : nécessite un PostgreSQL accessible.
Il est ignoré automatiquement si la base n'est pas joignable (ex. Docker arrêté).
"""
import os

import pandas as pd
import psycopg
import pytest

from src.warehouse.load_silver import get_conninfo, run


def _conninfo():
    try:
        info = os.environ.get("PG_TEST_DSN") or get_conninfo()
        psycopg.connect(info, connect_timeout=2).close()
        return info
    except Exception:
        return None


CONNINFO = _conninfo()
pytestmark = pytest.mark.skipif(CONNINFO is None, reason="PostgreSQL non joignable")


def test_chargement_idempotent():
    perimetre = pd.DataFrame([{"lei": "TESTLEI0000000000001", "nom": "Alpha", "nom_officiel": "ALPHA SA"}])
    df = pd.DataFrame([
        {"lei": "TESTLEI0000000000001", "indicateur": "chiffre_affaires", "exercice": 2023,
         "valeur": 1.05e9, "unite": "EUR", "concept": "ifrs-full:Revenue", "filing_id": 42},
        {"lei": "AUTRE_LEI_HORS_PERIM", "indicateur": "chiffre_affaires", "exercice": 2023,
         "valeur": 1.0, "unite": "EUR", "concept": "ifrs-full:Revenue", "filing_id": 43},
    ])
    try:
        run(CONNINFO, df, perimetre)
        res = run(CONNINFO, df, perimetre)          # 2e passage : pas de doublon
        alpha = res[res["entreprise"] == "Alpha"].iloc[0]
        assert alpha["lignes"] == 1
        with psycopg.connect(CONNINFO) as conn:
            v = conn.execute("SELECT valeur_md FROM dw.v_resultats WHERE entreprise = 'Alpha'").fetchone()
            assert float(v[0]) == 1.05
    finally:
        with psycopg.connect(CONNINFO) as conn:     # nettoyage
            conn.execute("DELETE FROM dw.fait_resultat WHERE lei = 'TESTLEI0000000000001'")
            conn.execute("DELETE FROM dw.dim_entreprise WHERE lei = 'TESTLEI0000000000001'")
