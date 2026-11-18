"""
Étape 1 — Collecte des comptes officiels (ESEF) depuis filings.xbrl.org.

Ce script :
  1. interroge l'API publique filings.xbrl.org (format JSON:API) ;
  2. récupère la liste des dépôts ESEF d'un pays (France par défaut) sur une période ;
  3. enregistre la liste brute (métadonnées) dans data/bronze/esef/filings_FR.jsonl ;
  4. télécharge, pour chaque dépôt, le fichier xBRL-JSON (tous les chiffres des comptes)
     dans data/bronze/esef/json/<id>.json.

Couche "bronze" = données brutes, telles que reçues, jamais modifiées.

Usage :
    python -m src.collect.esef_filings --country FR --from 2020-01-01 --to 2025-12-31
    python -m src.collect.esef_filings --country FR --lei-file config/perimetre.csv --download
"""
from __future__ import annotations

import argparse
import csv
import json
import logging
import os
import time
from pathlib import Path
from urllib.parse import urljoin

import requests

BASE_URL = "https://filings.xbrl.org"
API_URL = f"{BASE_URL}/api/filings"
# L'API demande qu'on s'identifie poliment (sinon risque de blocage).


def _read_env(path: Path = Path(".env")) -> dict:
    """Lit un fichier .env simple (CLE=valeur). Ce fichier n'est pas envoyé sur Git."""
    if not path.exists():
        return {}
    pairs = (l.split("=", 1) for l in path.read_text(encoding="utf-8").splitlines()
             if "=" in l and not l.lstrip().startswith("#"))
    return {k.strip(): v.strip().strip('"') for k, v in pairs}


CONTACT_EMAIL = os.environ.get("CONTACT_EMAIL") or _read_env().get("CONTACT_EMAIL", "a-remplacer@exemple.fr")
USER_AGENT = f"promesse-tenue/0.1 (projet etudiant Data Engineer; contact: {CONTACT_EMAIL})"

BRONZE_DIR = Path("data/bronze/esef")

log = logging.getLogger("esef")


def build_params(country: str, date_from: str, date_to: str, page_size: int = 100) -> dict:
    """Construit les paramètres de requête (filtre pays + période de clôture des comptes)."""
    filters = [
        {"name": "country", "op": "eq", "val": country},
        {"name": "period_end", "op": "ge", "val": date_from},
        {"name": "period_end", "op": "le", "val": date_to},
    ]
    return {
        "filter": json.dumps(filters),
        "include": "entity",          # on veut aussi le nom et le LEI de l'entreprise
        "page[size]": page_size,
        "sort": "period_end",
    }


def parse_page(payload: dict) -> list[dict]:
    """Transforme une page JSON:API en liste de dépôts « à plat » (une ligne = un dépôt)."""
    entities = {
        e["id"]: e.get("attributes", {})
        for e in payload.get("included", [])
        if e.get("type") == "entity"
    }
    rows = []
    for item in payload.get("data", []):
        attrs = item.get("attributes", {})
        ent_ref = (item.get("relationships", {}).get("entity", {}) or {}).get("data") or {}
        ent = entities.get(ent_ref.get("id"), {})
        rows.append({
            "filing_id": item.get("id"),
            "fxo_id": attrs.get("fxo_id"),
            "country": attrs.get("country"),
            "period_end": attrs.get("period_end"),
            "date_added": attrs.get("date_added"),
            "entity_name": ent.get("name"),
            "lei": ent.get("identifier"),
            "json_url": _abs(attrs.get("json_url")),
            "report_url": _abs(attrs.get("report_url")),
            "package_url": _abs(attrs.get("package_url")),
            "error_count": attrs.get("error_count"),
        })
    return rows


def _abs(url: str | None) -> str | None:
    """L'API renvoie souvent des chemins relatifs : on les rend absolus."""
    return urljoin(BASE_URL, url) if url else None


PAUSE_S = 2.0          # pause entre deux requêtes
MAX_RETRIES = 6


def get_with_retry(session: requests.Session, url: str, params: dict | None = None,
                   timeout: int = 60, sleep=time.sleep) -> requests.Response:
    """
    GET avec nouvelles tentatives.
    Le serveur limite le nombre de requêtes : s'il répond 429 (« trop de requêtes ») ou une
    erreur 5xx, on attend (durée indiquée par l'en-tête Retry-After, sinon 10 s, 20 s, 40 s…)
    puis on réessaie.
    """
    for attempt in range(MAX_RETRIES):
        resp = session.get(url, params=params, timeout=timeout)
        if resp.status_code != 429 and resp.status_code < 500:
            return resp
        retry_after = resp.headers.get("Retry-After")
        wait = float(retry_after) if retry_after and retry_after.isdigit() else 10 * 2 ** attempt
        log.warning("HTTP %s, nouvelle tentative dans %.0f s (%s/%s)",
                    resp.status_code, wait, attempt + 1, MAX_RETRIES)
        sleep(wait)
    resp.raise_for_status()
    return resp


def fetch_filings(session: requests.Session, params: dict, max_pages: int | None = None,
                  sleep=time.sleep) -> list[dict]:
    """Parcourt toutes les pages de résultats (pagination via links.next)."""
    rows: list[dict] = []
    url, query = API_URL, params
    page = 0
    while url:
        resp = get_with_retry(session, url, query, sleep=sleep)
        resp.raise_for_status()
        payload = resp.json()
        rows.extend(parse_page(payload))
        page += 1
        log.info("page %s : %s dépôts cumulés", page, len(rows))
        nxt = (payload.get("links") or {}).get("next")
        url, query = (_abs(nxt), None) if nxt else (None, None)
        if max_pages and page >= max_pages:
            break
        if url:
            sleep(PAUSE_S)  # on reste courtois avec le serveur
    return rows


def load_lei_filter(path: Path | None) -> set[str] | None:
    """Lit la liste des entreprises du périmètre (colonne 'lei'). None = pas de filtre."""
    if not path:
        return None
    with path.open(encoding="utf-8") as f:
        return {r["lei"].strip() for r in csv.DictReader(f) if r.get("lei")}


def save_jsonl(rows: list[dict], path: Path) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open("w", encoding="utf-8") as f:
        for r in rows:
            f.write(json.dumps(r, ensure_ascii=False) + "\n")


def download_json(session: requests.Session, rows: list[dict], out_dir: Path) -> int:
    """Télécharge les xBRL-JSON. Idempotent : un fichier déjà présent n'est pas re-téléchargé."""
    out_dir.mkdir(parents=True, exist_ok=True)
    n = 0
    for r in rows:
        if not r["json_url"]:
            log.warning("pas de xBRL-JSON pour %s (%s)", r["entity_name"], r["filing_id"])
            continue
        target = out_dir / f"{r['filing_id']}.json"
        if target.exists():
            continue
        resp = get_with_retry(session, r["json_url"], timeout=120)
        if resp.status_code != 200:
            log.warning("échec %s : HTTP %s", r["json_url"], resp.status_code)
            continue
        target.write_bytes(resp.content)
        n += 1
        log.info("téléchargé : %s (%s)", r["entity_name"], r["period_end"])
        time.sleep(PAUSE_S)
    return n


def main() -> None:
    ap = argparse.ArgumentParser(description="Collecte des dépôts ESEF")
    ap.add_argument("--country", default="FR")
    ap.add_argument("--from", dest="date_from", default="2020-01-01")
    ap.add_argument("--to", dest="date_to", default="2025-12-31")
    ap.add_argument("--lei-file", type=Path, help="CSV du périmètre (colonne lei)")
    ap.add_argument("--download", action="store_true", help="télécharger aussi les xBRL-JSON")
    ap.add_argument("--max-pages", type=int, help="limiter le nombre de pages (tests)")
    ap.add_argument("--use-cache", action="store_true",
                    help="réutiliser data/bronze/esef/filings_<pays>.jsonl déjà collecté")
    args = ap.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")
    session = requests.Session()
    session.headers["User-Agent"] = USER_AGENT

    out = BRONZE_DIR / f"filings_{args.country}.jsonl"
    if args.use_cache and out.exists():
        # On réutilise la liste déjà collectée au lieu d'interroger à nouveau l'API.
        rows = [json.loads(l) for l in out.read_text(encoding="utf-8").splitlines() if l.strip()]
        log.info("liste existante réutilisée : %s dépôts", len(rows))
    else:
        rows = fetch_filings(session, build_params(args.country, args.date_from, args.date_to),
                             args.max_pages)
        save_jsonl(rows, out)  # liste complète (bronze), jamais filtrée
        log.info("métadonnées enregistrées : %s (%s lignes)", out, len(rows))

    leis = load_lei_filter(args.lei_file)
    if leis is not None:
        if not leis:
            raise SystemExit(f"{args.lei_file} est vide : lance d'abord "
                             "`python -m src.collect.perimetre` pour le remplir.")
        rows = [r for r in rows if r["lei"] in leis]
        log.info("%s dépôts après filtre sur le périmètre", len(rows))
        sub = BRONZE_DIR / f"filings_{args.country}_perimetre.jsonl"
        save_jsonl(rows, sub)
        log.info("dépôts du périmètre enregistrés : %s", sub)

    if args.download:
        n = download_json(session, rows, BRONZE_DIR / "json")
        log.info("%s fichiers xBRL-JSON téléchargés", n)


if __name__ == "__main__":
    main()
