-- Étape 2 — Entrepôt de données « Promesse tenue ? » (PostgreSQL)
-- Modèle en étoile : des tables de dimensions (qui ? quoi ?) autour de tables de faits (les mesures).
-- Script idempotent : on peut le relancer sans erreur ni perte de données.

CREATE SCHEMA IF NOT EXISTS dw;

-- Dimension : les entreprises du périmètre
CREATE TABLE IF NOT EXISTS dw.dim_entreprise (
    lei            CHAR(20) PRIMARY KEY,          -- identifiant international (Legal Entity Identifier)
    nom            TEXT NOT NULL,                 -- nom court (ex. L'Oréal)
    nom_officiel   TEXT,                          -- nom légal tel que publié
    pays           CHAR(2) NOT NULL DEFAULT 'FR'
);

-- Dimension : les indicateurs financiers suivis
CREATE TABLE IF NOT EXISTS dw.dim_indicateur (
    code     TEXT PRIMARY KEY,                    -- ex. chiffre_affaires
    libelle  TEXT NOT NULL,                       -- ex. Chiffre d'affaires
    nature   TEXT NOT NULL CHECK (nature IN ('flux', 'stock'))  -- flux = sur l'année, stock = au 31/12
);

INSERT INTO dw.dim_indicateur (code, libelle, nature) VALUES
    ('chiffre_affaires',      'Chiffre d''affaires',                 'flux'),
    ('resultat_operationnel', 'Résultat opérationnel',               'flux'),
    ('resultat_net',          'Résultat net part du groupe',         'flux'),
    ('capitaux_propres',      'Capitaux propres',                    'stock'),
    ('tresorerie',            'Trésorerie et équivalents',           'stock')
ON CONFLICT (code) DO UPDATE SET libelle = EXCLUDED.libelle, nature = EXCLUDED.nature;

-- Faits : les chiffres réels publiés (une ligne = entreprise × indicateur × exercice)
CREATE TABLE IF NOT EXISTS dw.fait_resultat (
    lei         CHAR(20) NOT NULL REFERENCES dw.dim_entreprise (lei),
    indicateur  TEXT     NOT NULL REFERENCES dw.dim_indicateur (code),
    exercice    SMALLINT NOT NULL CHECK (exercice BETWEEN 2000 AND 2100),
    valeur      NUMERIC(20, 2) NOT NULL,
    unite       CHAR(3)  NOT NULL,                -- EUR, USD…
    concept     TEXT     NOT NULL,                -- étiquette XBRL d'origine (traçabilité)
    filing_id   TEXT     NOT NULL,                -- rapport source sur filings.xbrl.org
    charge_le   TIMESTAMPTZ NOT NULL DEFAULT now(),
    PRIMARY KEY (lei, indicateur, exercice)
);

-- Vue lisible pour l'analyse : noms en clair, montants en milliards
CREATE OR REPLACE VIEW dw.v_resultats AS
SELECT e.nom          AS entreprise,
       i.libelle      AS indicateur,
       f.exercice,
       ROUND(f.valeur / 1e9, 3) AS valeur_md,
       f.unite,
       f.concept,
       f.filing_id
FROM dw.fait_resultat f
JOIN dw.dim_entreprise e ON e.lei = f.lei
JOIN dw.dim_indicateur i ON i.code = f.indicateur;
