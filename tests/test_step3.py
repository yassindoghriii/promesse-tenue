from pathlib import Path

import pandas as pd

from src.collect.reports import download_reports
from src.parse.passages import extract_text, find_passages, is_candidate, run

FIX = Path(__file__).parent / "fixtures"


def test_extract_text_ignore_le_xbrl_cache_et_recolle_les_mots():
    text = extract_text(FIX / "rapport_alpha_2025.xhtml")
    assert "secret caché" not in text
    assert "croissance organique" in text


def test_is_candidate():
    assert is_candidate("Le Groupe prévoit une croissance de 4 % à 6 % en 2026.", 2025) == (True, [2026])
    # passé : pas une promesse
    assert is_candidate("En 2025, le chiffre d'affaires a progressé de 5 % par rapport à 2024.", 2025)[0] is False
    # pas de chiffre
    assert is_candidate("Nos objectifs restent ambitieux pour 2027 et au-delà, comme prévu.", 2025)[0] is False


def test_find_passages_sur_un_rapport():
    phrases = [p["phrase"] for p in find_passages(extract_text(FIX / "rapport_alpha_2025.xhtml"), 2025)]
    assert any("entre 4 % et 6 %" in p for p in phrases)
    assert any("12,5 %" in p for p in phrases)
    assert any("l'an prochain" in p for p in phrases)
    assert sum("CO2" in p for p in phrases) == 1          # phrase répétée gardée une seule fois
    assert not any("progressé" in p for p in phrases)


def test_run_et_telechargement(tmp_path):
    meta = pd.DataFrame([{"filing_id": 7, "lei": "L", "entity_name": "ALPHA", "period_end": "2025-12-31",
                          "report_url": "https://x/r.xhtml"}])
    reports = tmp_path / "reports"
    reports.mkdir()
    import gzip
    (reports / "7.xhtml.gz").write_bytes(gzip.compress((FIX / "rapport_alpha_2025.xhtml").read_bytes()))
    df = run(meta, reports, tmp_path / "passages.jsonl")
    assert len(df) == 4 and set(df["exercice_rapport"]) == {2025}
    # déjà présent -> pas de téléchargement
    assert download_reports(None, meta, reports, sleep=lambda s: None) == (0, 1)


def test_exclut_la_remuneration_des_dirigeants():
    s = ("Les objectifs individuels définis pour la rémunération variable au titre de 2021 "
         "représentent 50 % du bonus.")
    assert is_candidate(s, 2020)[0] is False
    assert is_candidate("Sanofi porte son objectif d'économies à 2,5 milliards d'euros d'ici à 2022.", 2020)[0]


def test_caracteres_invisibles_supprimes():
    from src.parse.passages import clean
    texte = clean("croissance comprise entre + 3 et + 5% en 2022 ; marge de 13 %")
    assert "5%" in texte and "" not in texte and " " not in texte
    assert is_candidate("Danone vise une croissance comprise entre + 3 et + 5% en 2022.".replace("", ""), 2021)[0]


def test_slim_retire_les_images_et_garde_le_texte(tmp_path):
    import base64
    import gzip
    from src.collect.slim_reports import slim_file
    img = base64.b64encode(b"\x89PNG" + b"\x00" * 50000).decode()
    page = (f'<html><body><img src="data:image/png;base64,{img}"/>'
            f'<div>Le Groupe prévoit une croissance de 5 % en 2026.</div></body></html>').encode()
    f = tmp_path / "1.xhtml.gz"
    f.write_bytes(gzip.compress(page))
    avant, apres = slim_file(f)
    assert apres < avant / 10
    assert slim_file(f) == (apres, apres)               # 2e passage : rien à faire
    assert "croissance de 5 % en 2026" in extract_text(f)