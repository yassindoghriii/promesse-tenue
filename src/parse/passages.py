"""
Étape 3 (suite) — Repérer dans les rapports les phrases candidates à être des « promesses ».

Un rapport fait des centaines de pages ; seules quelques dizaines de phrases parlent
d'objectifs futurs. On les présélectionne ici avec des règles simples (pas d'IA) :
  1. on extrait le texte du rapport XHTML (en ignorant les données XBRL cachées) ;
  2. on le découpe en phrases ;
  3. on garde une phrase si elle contient À LA FOIS :
       - un mot d'intention (objectif, prévoit, vise, s'attend, perspectives…, ou en anglais) ;
       - un chiffre (%, montant, multiple) ;
       - une année FUTURE par rapport à l'exercice du rapport (ex. 2026 dans le rapport 2025),
         ou une formule comme « l'an prochain ».
Pourquoi ce filtre ? Il réduit fortement le volume de texte envoyé au LLM à l'étape 4
(donc le coût), tout en gardant les phrases utiles. Sa qualité sera mesurée à l'étape 4.

Sortie : data/silver/passages.jsonl (une ligne = une phrase candidate + son contexte)

Usage :
    python -m src.parse.passages
"""
from __future__ import annotations

import argparse
import gzip
import json
import logging
import re
from pathlib import Path

import pandas as pd
from lxml import etree

log = logging.getLogger("passages")

# Éléments dont le texte n'est pas du contenu lisible
SKIP_TAGS = {"script", "style", "header", "hidden", "references", "resources"}

INTENT = re.compile(
    r"\b(objectifs?|ambitions?|perspectives?|pr[ée]vo(?:it|ient|yons|ir)|pr[ée]visions?|"
    r"anticip\w*|s.attend\w*|attendons|vis(?:e|ons|ent|er)|cibles?|guidance|confiants?|"
    r"devrai(?:t|ent)|esp[ée]r\w*|engag\w*|"
    r"expects?|expected|targets?|outlook|aims?|anticipates?|forecasts?|projects?)\b", re.I)
NUMBER = re.compile(r"\d+(?:[.,]\d+)?\s*(?:%|pts?|points?|x\b|milliards?|millions?|bn|m€|md€|€)|"
                    r"\d+(?:[.,]\d+)?\s*(?:à|-|–)\s*\d+(?:[.,]\d+)?\s*%", re.I)
YEAR = re.compile(r"\b(20[2-4]\d)\b")
NEXT_YEAR = re.compile(r"\b(l.an prochain|l.ann[ée]e prochaine|prochain exercice|next year|"
                       r"moyen terme|medium[- ]term)\b", re.I)
# Bruit fréquent : les critères de rémunération des dirigeants ressemblent à des objectifs
# (« objectifs fixés pour 2026 », « conditions de performance ») mais ne sont pas des
# engagements de l'entreprise envers les investisseurs.
EXCLUDE = re.compile(r"r[ée]mun[ée]ration|unit[ée]s de performance|actions de performance|"
                     r"conditions? de performance|mandataires? sociaux|stock[- ]options?|"
                     r"part variable|rémunération variable|plan d.incitation|"
                     r"remuneration|performance shares|incentive plan", re.I)
SENTENCE_END = re.compile(r"(?<=[.!?;])\s+(?=[A-ZÀÂÉÈÊÎÔÛÇ«\"(•–-])")


def extract_text(path: Path) -> str:
    """Texte lisible du rapport XHTML, sans les données XBRL cachées (ix:header)."""
    parser = etree.XMLParser(huge_tree=True, recover=True, remove_comments=True)
    opener = gzip.open if path.suffix == ".gz" else open   # rapport compressé ou non
    with opener(path, "rb") as fh:
        root = etree.parse(fh, parser).getroot()
    parts: list[str] = []

    def walk(el):
        tag = etree.QName(el).localname.lower() if isinstance(el.tag, str) else ""
        if tag in SKIP_TAGS or "display:none" in (el.get("style") or "").replace(" ", ""):
            if el.tail:
                parts.append(el.tail)
            return
        if el.text:
            parts.append(el.text)
        for child in el:
            walk(child)
        if tag in {"div", "p", "br", "tr", "li", "h1", "h2", "h3", "h4", "td"}:
            parts.append("\n")
        if el.tail:
            parts.append(el.tail)

    walk(root)
    return clean("".join(parts))


def clean(text: str) -> str:
    # Caractères invisibles : zone « à usage privé » des polices (ex. \ue61f chez Danone, glissé
    # entre « 5 » et « % »), espaces de largeur nulle, traits d'union conditionnels -> supprimés
    text = re.sub(r"[\ue000-\uf8ff\u200b-\u200d\u2060\ufeff\u00ad]", "", text)
    # Espaces spéciaux (insécable, fine, etc.) -> espace normal
    text = re.sub(r"[\xa0\u2000-\u200a\u202f\u205f\u3000]", " ", text)
    text = re.sub(r"(\w)-\s*\n\s*(\w)", r"\1\2", text)  # mots coupés en fin de ligne
    text = re.sub(r"[ \t]*\n[ \t]*", "\n", text)
    text = re.sub(r"\n{2,}", "\n\n", text)                # paragraphes
    text = re.sub(r"(?<!\n)\n(?!\n)", " ", text)           # retours à la ligne simples -> espace
    return re.sub(r"[ \t]{2,}", " ", text).strip()


def split_sentences(text: str) -> list[str]:
    out = []
    for para in text.split("\n\n"):
        out.extend(s.strip() for s in SENTENCE_END.split(para) if s.strip())
    return out


def is_candidate(sentence: str, exercice: int) -> tuple[bool, list[int]]:
    """Retourne (candidate ?, années futures citées)."""
    if not 40 <= len(sentence) <= 800:      # trop court = titre, trop long = tableau mal découpé
        return False, []
    if not INTENT.search(sentence) or not NUMBER.search(sentence):
        return False, []
    if EXCLUDE.search(sentence):
        return False, []
    futures = sorted({int(y) for y in YEAR.findall(sentence) if int(y) > exercice})
    if futures or NEXT_YEAR.search(sentence):
        return True, futures
    return False, []


def find_passages(text: str, exercice: int) -> list[dict]:
    sentences = split_sentences(text)
    found, seen = [], set()
    for i, s in enumerate(sentences):
        ok, years = is_candidate(s, exercice)
        key = s.lower()
        if not ok or key in seen:           # les DEU répètent souvent les mêmes phrases
            continue
        seen.add(key)
        found.append({
            "phrase": s,
            "contexte": " ".join(sentences[max(0, i - 1): i + 2]),
            "annees_cibles": years,
            "mots_intention": sorted({m.lower() for m in INTENT.findall(s)}),
            "position": round(i / max(1, len(sentences)), 3),  # 0 = début du rapport, 1 = fin
        })
    return found


def run(meta: pd.DataFrame, reports_dir: Path, out_path: Path) -> pd.DataFrame:
    rows = []
    for m in meta.itertuples():
        f = reports_dir / f"{m.filing_id}.xhtml.gz"
        if not f.exists():
            f = reports_dir / f"{m.filing_id}.xhtml"     # ancien format, non compressé
        if not f.exists():
            continue
        exercice = int(str(m.period_end)[:4])
        passages = find_passages(extract_text(f), exercice)
        log.info("%s %s : %s phrase(s) candidate(s)", m.entity_name, exercice, len(passages))
        for p in passages:
            rows.append({"filing_id": str(m.filing_id), "lei": m.lei, "entreprise": m.entity_name,
                         "exercice_rapport": exercice, **p})
    out_path.parent.mkdir(parents=True, exist_ok=True)
    with out_path.open("w", encoding="utf-8") as fh:
        for r in rows:
            fh.write(json.dumps(r, ensure_ascii=False) + "\n")
    log.info("%s phrases candidates écrites dans %s", len(rows), out_path)
    return pd.DataFrame(rows)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--meta", type=Path, default=Path("data/bronze/esef/filings_FR_perimetre.jsonl"))
    ap.add_argument("--reports-dir", type=Path, default=Path("data/bronze/reports"))
    ap.add_argument("--out", type=Path, default=Path("data/silver/passages.jsonl"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    meta = pd.read_json(args.meta, lines=True)
    meta = meta.sort_values("date_added").drop_duplicates(["lei", "period_end"], keep="last")
    df = run(meta, args.reports_dir, args.out)
    if not df.empty:
        print("\nPhrases candidates par entreprise et par rapport :\n")
        print(df.pivot_table(index="entreprise", columns="exercice_rapport", values="phrase",
                             aggfunc="count", fill_value=0).to_string())


if __name__ == "__main__":
    main()