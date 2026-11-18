"""
Étape 2 (suite) — Chargement de l'entrepôt PostgreSQL depuis la zone silver du data lake.

Flux : data lake (RustFS, zone silver)  ->  entrepôt (PostgreSQL, schéma dw)

  1. applique le schéma SQL (sql/*.sql, dans l'ordre) — idempotent ;
  2. charge les entreprises du périmètre (config/perimetre.csv) dans dw.dim_entreprise ;
  3. lit silver/esef_indicateurs.csv dans le data lake et l'insère dans dw.fait_resultat.
     « Upsert » : une ligne déjà présente est mise à jour, jamais dupliquée.

Usage :
    python -m src.warehouse.load_silver
    python -m src.warehouse.load_silver --source local   # lire data/silver au lieu du data lake
"""
from __future__ import annotations

import argparse
import io
import logging
import os
from pathlib import Path

import pandas as pd
import psycopg

from src.collect.esef_filings import _read_env

log = logging.getLogger("warehouse")
# Chemin absolu : fonctionne quel que soit le dossier depuis lequel on lance la commande
SQL_DIR = Path(__file__).resolve().parents[2] / "sql"
SILVER_KEY = "esef_indicateurs.csv"


def get_conninfo() -> str:
    env = {**_read_env(), **os.environ}
    return (f"host={env.get('POSTGRES_HOST', 'localhost')} port={env.get('POSTGRES_PORT', '5433')} "
            f"dbname={env['POSTGRES_DB']} user={env['POSTGRES_USER']} "
            f"password={env.get('POSTGRES_PASSWORD', '')}")


def apply_schema(conn: psycopg.Connection, sql_dir: Path = SQL_DIR) -> list[str]:
    files = sorted(sql_dir.glob("*.sql"))
    if not files:  # on s'arrête net plutôt que de continuer sans tables
        raise FileNotFoundError(f"aucun script SQL trouvé dans {sql_dir}")
    applied = []
    for f in files:
        conn.execute(f.read_text(encoding="utf-8"))
        applied.append(f.name)
    return applied


def read_silver(source: str) -> pd.DataFrame:
    if source == "local":
        return pd.read_csv(Path("data/silver") / SILVER_KEY)
    from src.storage.lake import get_client  # import ici : pas besoin de boto3 en mode local
    obj = get_client().get_object(Bucket="silver", Key=SILVER_KEY)
    return pd.read_csv(io.BytesIO(obj["Body"].read()))


def load_entreprises(conn: psycopg.Connection, perimetre: pd.DataFrame) -> int:
    rows = perimetre[["lei", "nom", "nom_officiel"]].itertuples(index=False, name=None)
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO dw.dim_entreprise (lei, nom, nom_officiel) VALUES (%s, %s, %s)
               ON CONFLICT (lei) DO UPDATE SET nom = EXCLUDED.nom, nom_officiel = EXCLUDED.nom_officiel""",
            list(rows))
    return len(perimetre)


def load_resultats(conn: psycopg.Connection, df: pd.DataFrame) -> int:
    cols = ["lei", "indicateur", "exercice", "valeur", "unite", "concept", "filing_id"]
    data = df[cols].astype({"exercice": int, "filing_id": str}).itertuples(index=False, name=None)
    with conn.cursor() as cur:
        cur.executemany(
            """INSERT INTO dw.fait_resultat (lei, indicateur, exercice, valeur, unite, concept, filing_id)
               VALUES (%s, %s, %s, %s, %s, %s, %s)
               ON CONFLICT (lei, indicateur, exercice) DO UPDATE
               SET valeur = EXCLUDED.valeur, unite = EXCLUDED.unite, concept = EXCLUDED.concept,
                   filing_id = EXCLUDED.filing_id, charge_le = now()""",
            list(data))
    return len(df)


def summary(conn: psycopg.Connection) -> pd.DataFrame:
    rows = conn.execute(
        """SELECT e.nom, COUNT(*) AS lignes, MIN(f.exercice) AS de, MAX(f.exercice) AS a
           FROM dw.fait_resultat f JOIN dw.dim_entreprise e USING (lei)
           GROUP BY e.nom ORDER BY e.nom""").fetchall()
    return pd.DataFrame(rows, columns=["entreprise", "lignes", "de", "a"])


def run(conninfo: str, df: pd.DataFrame, perimetre: pd.DataFrame) -> pd.DataFrame:
    # Une seule transaction : soit tout est chargé, soit rien (pas d'état à moitié chargé)
    with psycopg.connect(conninfo) as conn:
        log.info("schéma appliqué : %s", ", ".join(apply_schema(conn)))
        log.info("%s entreprises chargées", load_entreprises(conn, perimetre))
        df = df[df["lei"].isin(perimetre["lei"])]
        log.info("%s lignes de résultats chargées", load_resultats(conn, df))
        return summary(conn)


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--source", choices=["lake", "local"], default="lake")
    ap.add_argument("--perimetre", type=Path, default=Path("config/perimetre.csv"))
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    res = run(get_conninfo(), read_silver(args.source), pd.read_csv(args.perimetre))
    print("\nContenu de l'entrepôt (dw.fait_resultat) :\n")
    print(res.to_string(index=False))


if __name__ == "__main__":
    main()