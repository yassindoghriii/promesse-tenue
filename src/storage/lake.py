"""
Étape 2 — Data lake (stockage S3 : RustFS en local, Amazon S3 ou autre en production).

Le data lake est organisé en 3 zones (architecture « médaillon ») :
  - bronze : fichiers bruts, tels que téléchargés (xBRL-JSON, listes de dépôts) ;
  - silver : données nettoyées (indicateurs harmonisés) ;
  - gold   : tables prêtes pour l'analyse (verdicts, scores) — étapes suivantes.

Ce module :
  1. crée les 3 zones (« buckets ») si elles n'existent pas ;
  2. envoie le contenu des dossiers locaux data/bronze et data/silver dans les zones du même nom.
     Idempotent : un fichier déjà présent avec la même taille n'est pas renvoyé.

Usage :
    python -m src.storage.lake            # crée les zones et synchronise
    python -m src.storage.lake --list     # affiche le contenu du data lake
"""
from __future__ import annotations

import argparse
import logging
import os
from pathlib import Path

import boto3
from botocore.exceptions import ClientError

from src.collect.esef_filings import _read_env

ZONES = ("bronze", "silver", "gold")
log = logging.getLogger("lake")


def get_client():
    """Client S3 configuré depuis le .env (ou les variables d'environnement)."""
    env = {**_read_env(), **os.environ}
    return boto3.client(
        "s3",
        endpoint_url=env.get("S3_ENDPOINT", "http://localhost:9000"),
        aws_access_key_id=env["S3_ACCESS_KEY"],
        aws_secret_access_key=env["S3_SECRET_KEY"],
        region_name=env.get("S3_REGION", "us-east-1"),
    )


def ensure_zones(s3) -> list[str]:
    """Crée les buckets manquants. Retourne ceux qui ont été créés."""
    existing = {b["Name"] for b in s3.list_buckets().get("Buckets", [])}
    created = []
    for zone in ZONES:
        if zone not in existing:
            s3.create_bucket(Bucket=zone)
            created.append(zone)
    return created


def _remote_size(s3, bucket: str, key: str) -> int | None:
    try:
        return s3.head_object(Bucket=bucket, Key=key)["ContentLength"]
    except ClientError:
        return None


def sync_dir(s3, local_dir: Path, bucket: str) -> tuple[int, int]:
    """Envoie local_dir dans le bucket (même arborescence). Retourne (envoyés, ignorés)."""
    sent = skipped = 0
    for f in sorted(p for p in local_dir.rglob("*") if p.is_file() and not p.name.startswith(".")):
        key = f.relative_to(local_dir).as_posix()
        if _remote_size(s3, bucket, key) == f.stat().st_size:
            skipped += 1
            continue
        s3.upload_file(str(f), bucket, key)
        sent += 1
    return sent, skipped


def list_lake(s3) -> dict[str, tuple[int, int]]:
    """Nombre de fichiers et taille totale (octets) par zone."""
    out = {}
    for zone in ZONES:
        n = size = 0
        for page in s3.get_paginator("list_objects_v2").paginate(Bucket=zone):
            for obj in page.get("Contents", []):
                n += 1
                size += obj["Size"]
        out[zone] = (n, size)
    return out


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--data-dir", type=Path, default=Path("data"))
    ap.add_argument("--list", action="store_true", help="afficher seulement le contenu")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    s3 = get_client()
    if not args.list:
        created = ensure_zones(s3)
        log.info("zones créées : %s", ", ".join(created) or "aucune (déjà présentes)")
        for zone in ("bronze", "silver"):
            d = args.data_dir / zone
            if d.exists():
                sent, skipped = sync_dir(s3, d, zone)
                log.info("%s : %s fichier(s) envoyé(s), %s déjà à jour", zone, sent, skipped)

    print("\nContenu du data lake :")
    for zone, (n, size) in list_lake(s3).items():
        print(f"  {zone:<7} {n:>5} fichier(s)  {size / 1e6:>8.1f} Mo")


if __name__ == "__main__":
    main()