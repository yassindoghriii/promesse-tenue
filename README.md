# Promesse tenue ? — Projet fil rouge Data Engineer

Plateforme qui vérifie si les dirigeants des sociétés cotées tiennent les objectifs chiffrés qu'ils annoncent.

## Feuille de route

| Étape | Contenu | Compétences | État |
|---|---|---|---|
| 1 | Structure du projet + collecte des comptes officiels (ESEF) + extraction des chiffres clés | C4, C6, C8 | **en cours** |
| 2 | Infrastructure locale : Docker, data lake MinIO, PostgreSQL | C7, C14 | à faire |
| 3 | Collecte du texte des rapports + découpage en passages | C6, C8 | à faire |
| 4 | Extraction des promesses par LLM (Gemini + LangChain + Langfuse) + jeu de test | C12 | à faire |
| 5 | Rapprochement promesse ↔ chiffre réel, verdicts, score (dbt) | C9, C10 | à faire |
| 6 | Cours de bourse + analyses | C6, C10 | à faire |
| 7 | Modèle prédictif XGBoost + MLflow | C12, C15 | à faire |
| 8 | API FastAPI + tableau de bord Streamlit | C13 | à faire |
| 9 | Orchestration Airflow | C11, C16 | à faire |
| 10 | Kubernetes, CI/CD GitHub Actions, monitoring | C16, C17, C18 | à faire |

## Étape 1 — Collecte des comptes officiels

### Ce qu'on fait et pourquoi

Depuis 2021, les sociétés cotées européennes publient leur rapport annuel au format **ESEF** :
un document lisible par un humain (HTML) dans lequel chaque chiffre porte une étiquette lisible
par une machine (XBRL). Exemple : le chiffre d'affaires porte l'étiquette `ifrs-full:Revenue`.

Le site **filings.xbrl.org** rassemble ces rapports et propose, pour chacun :
- `json_url` : tous les chiffres étiquetés, en JSON → nos **données structurées** (chiffres réels) ;
- `report_url` : le rapport complet en HTML → nos **données non structurées** (les promesses, étape 3).

Bonne nouvelle : la même source nous donne les deux types de données.

### Organisation des données (architecture « médaillon »)

- `data/bronze/` : données brutes, telles que téléchargées, jamais modifiées.
- `data/silver/` : données nettoyées et filtrées.
- `data/gold/` (plus tard) : tables prêtes pour l'analyse (verdicts, scores).

### Lancer l'étape 1

```bash
# 1. Créer l'environnement Python
python -m venv .venv
source .venv/bin/activate          # Windows : .venv\Scripts\activate
pip install -r requirements.txt

# 2. Vérifier que les tests passent
python -m pytest -q

# 3. Mettre son e-mail dans un fichier .env (jamais envoyé sur Git)
cp .env.example .env               # puis ouvrir .env et remplacer l'adresse

# 4. Récupérer la liste complète des dépôts ESEF français 2020-2025 (métadonnées)
python -m src.collect.esef_filings --country FR --from 2020-01-01 --to 2025-12-31

# 5. Retrouver les identifiants (LEI) des entreprises du périmètre à partir de leur nom
python -m src.collect.perimetre

# 6. Télécharger les chiffres des seules entreprises du périmètre (sans réinterroger la liste)
python -m src.collect.esef_filings --country FR --lei-file config/perimetre.csv --use-cache --download

# 7. Extraire les chiffres clés
python -m src.parse.xbrl_facts
```

### Choisir le périmètre

Le périmètre se décrit par des noms dans `config/perimetre_noms.csv` (colonnes `nom,mot_cle`).
Le `mot_cle` est cherché dans le nom officiel, sans tenir compte des accents ni des majuscules.
La commande 5 écrit `config/perimetre.csv` et affiche les entreprises trouvées et introuvables.

Départ : 10 grandes entreprises industrielles et de services du CAC 40. Les banques et assurances
sont exclues pour l'instant : elles n'ont pas de « chiffre d'affaires » au sens classique
(on parle de produit net bancaire), ce qui compliquerait le rapprochement.

### Ce qu'on obtient

`data/silver/esef_indicateurs.csv` : une ligne par entreprise × indicateur × exercice.

| Colonne | Sens |
|---|---|
| lei | Identifiant international unique de l'entreprise |
| indicateur | chiffre_affaires, resultat_operationnel, resultat_net, capitaux_propres, tresorerie |
| exercice | Année comptable |
| valeur | Montant (dans l'unité indiquée, en général EUR) |

### Règles de nettoyage appliquées (à expliquer au jury)

1. On garde les faits **consolidés** (total du groupe), pas ceux ventilés par pays ou activité.
2. On garde les périodes d'**environ 12 mois** (pas les semestres).
3. Un même chiffre apparaît dans le rapport N et en comparatif dans le rapport N+1 :
   on garde **le plus récent** (il peut avoir été retraité).
4. Plusieurs étiquettes possibles pour un même indicateur : on suit un ordre de préférence.

### Limite connue (= ta « situation de recherche » pour le dossier)

Certaines notions (résultat opérationnel courant, dette nette, EBITDA) ne sont pas des étiquettes IFRS
standard : chaque entreprise crée la sienne (« extension »). À l'étape 5, il faudra décider comment les relier
aux promesses. Note dès maintenant les cas rencontrés.
