"""
Étape 4 — Extraction des promesses par un LLM (Gemini), en sortie structurée.

Entrée  : data/silver/passages.jsonl  (2 210 phrases candidates, pré-filtrées à l'étape 3)
Sortie  : data/silver/promesses.jsonl (une ligne = une phrase analysée par le LLM)

Principes :
  - LOTS : on envoie ~15 phrases par appel (au lieu d'une) -> 15 fois moins d'appels, ce qui
    respecte le quota gratuit et réduit le coût.
  - SORTIE STRUCTURÉE : le LLM doit répondre au format JSON imposé par un schéma (Pydantic),
    vérifié automatiquement. Pas de texte libre à « deviner ».
  - CACHE : chaque lot déjà traité est enregistré (data/silver/llm_cache/). Relancer le script ne
    repaie jamais un appel déjà fait -> idempotent, et reprise possible après une coupure.
  - TEMPÉRATURE 0 : réponses les plus stables possible d'un lancement à l'autre.

Usage :
    python -m src.ai.extract_promises --limit 2      # test sur 2 lots
    python -m src.ai.extract_promises                # tout
"""
from __future__ import annotations

import argparse
import hashlib
import json
import re
import logging
import os
import time
from enum import Enum
from pathlib import Path

import pandas as pd
from pydantic import BaseModel, Field

from src.collect.esef_filings import _read_env

log = logging.getLogger("llm")
PROMPT_VERSION = "v2"          # à changer quand on modifie le prompt (invalide le cache)
BATCH_SIZE = 15
CACHE_DIR = Path("data/silver/llm_cache")


def http_status(exc: Exception) -> int | None:
    """
    Code HTTP réel d'une erreur d'API. On ne cherche PAS « 403 » n'importe où dans le message :
    « Requested 4036 tokens » contient « 403 » et serait pris à tort pour un accès refusé.
    """
    for attr in ("status_code", "code"):
        v = getattr(exc, attr, None)
        if isinstance(v, int) and 100 <= v <= 599:
            return v
    m = re.match(r"\s*(?:Error code:\s*)?(\d{3})\b", str(exc))
    return int(m.group(1)) if m else None


class LLMDefinitiveError(RuntimeError):
    """Erreur qui ne se résoudra pas en réessayant (clé invalide, accès refusé)."""


# ---------- Schéma de sortie imposé au LLM ----------
class Indicateur(str, Enum):
    chiffre_affaires = "chiffre_affaires"            # niveau ou croissance du CA
    marge_operationnelle = "marge_operationnelle"
    resultat_operationnel = "resultat_operationnel"
    resultat_net = "resultat_net"
    bpa = "bpa"                                      # bénéfice par action
    flux_tresorerie = "flux_tresorerie"              # free cash flow, cash flow opérationnel
    dette = "dette"                                  # dette nette, levier
    dividende = "dividende"                          # dividende, rachats d'actions
    economies_couts = "economies_couts"
    investissements = "investissements"              # capex, acquisitions
    environnement = "environnement"                  # CO2, énergie, eau…
    autre = "autre"


class TypeCible(str, Enum):
    valeur = "valeur"            # « atteindre 12 % »
    minimum = "minimum"          # « au moins 12 % », « supérieure à »
    maximum = "maximum"          # « inférieur à 3x »
    fourchette = "fourchette"    # « entre 4 % et 6 % »
    qualitative = "qualitative"  # « en hausse », sans chiffre exploitable


class Analyse(BaseModel):
    numero: int = Field(description="numéro de la phrase dans la liste fournie")
    est_promesse: bool = Field(description="vrai si l'entreprise s'engage sur un résultat futur chiffré")
    indicateur: Indicateur | None = None
    mesure: str | None = Field(None, description="ce qui est mesuré, en quelques mots (ex. croissance organique du CA)")
    type_cible: TypeCible | None = None
    valeur_min: float | None = None
    valeur_max: float | None = None
    unite: str | None = Field(None, description="%, points de base, EUR, Md EUR, x (multiple)…")
    annee_echeance: int | None = None
    prudence: str | None = Field(None, description="'ferme' (s'engage, vise, atteindra) ou 'prudent' (devrait, anticipe, espère)")
    justification: str | None = Field(None, description="une phrase courte expliquant la décision")


class Lot(BaseModel):
    analyses: list[Analyse]


PROMPT = """Tu es analyste financier. Voici des phrases extraites du rapport annuel {exercice} de {entreprise}.
Pour CHAQUE phrase, décide si c'est une PROMESSE : un engagement ou un objectif de l'entreprise
sur un résultat FUTUR CHIFFRÉ (après {exercice}), vérifiable plus tard dans ses comptes ou rapports.

N'est PAS une promesse :
- un résultat passé ou un constat (« le CA a progressé de 5 % ») ;
- une hypothèse comptable, un taux d'actualisation, une sensibilité, un test de dépréciation ;
- une échéance de dette, d'emprunt, de contrat ou de plan d'actions ;
- un objectif de rémunération des dirigeants ;
- une prévision de marché ou d'un tiers, pas de l'entreprise.

Si c'est une promesse, renseigne les champs. Pourcentages : 5 % -> 5 (pas 0.05).
Montants : garde l'unité du texte (ex. 2,5 milliards d'euros -> valeur 2.5, unite « Md EUR »).
Si ce n'est pas une promesse, laisse les autres champs vides ; « justification » en 15 mots maximum.
Si une phrase contient plusieurs promesses (ex. 30 % en 2022 puis 32 % en 2025), garde la plus
proche dans le temps et signale l'autre dans « justification ».
Sois concis : pas de texte hors du JSON.

Phrases :
{phrases}"""


def batches(df: pd.DataFrame, size: int = BATCH_SIZE):
    """Lots homogènes : même rapport (même entreprise, même exercice)."""
    for _, grp in df.groupby("filing_id", sort=False):
        for i in range(0, len(grp), size):
            yield grp.iloc[i:i + size]


def build_prompt(lot: pd.DataFrame) -> str:
    phrases = "\n".join(f"{n}. {p}" for n, p in enumerate(lot["phrase"], start=1))
    first = lot.iloc[0]
    return PROMPT.format(exercice=first["exercice_rapport"], entreprise=first["entreprise"], phrases=phrases)


def cache_key(prompt: str) -> str:
    return hashlib.sha256(f"{PROMPT_VERSION}|{prompt}".encode()).hexdigest()[:24]


class GeminiExtractor:
    def __init__(self, client, models: str | list[str], cache_dir: Path = CACHE_DIR,
                 pause_s: float = 4.0, sleep=time.sleep):
        # Plusieurs modèles possibles, par ordre de préférence : si l'un est surchargé (503),
        # on bascule sur le suivant -> le traitement continue au lieu d'attendre.
        self.models = [m.strip() for m in models.split(",")] if isinstance(models, str) else list(models)
        self.current = 0
        self.client, self.cache_dir = client, cache_dir
        self.used: dict[str, int] = {}
        self.saturations = 0
        self.failed = 0
        self.offline = False          # True = n'utiliser que le cache (aucun appel au LLM)
        self.skipped = 0
        self.pause_s, self.sleep = pause_s, sleep
        self.calls = self.cached = 0
        self.tokens_in = self.tokens_out = 0
        cache_dir.mkdir(parents=True, exist_ok=True)

    def analyse(self, lot: pd.DataFrame) -> list[dict]:
        prompt = build_prompt(lot)
        f = self.cache_dir / f"{cache_key(prompt)}.json"
        if f.exists():
            self.cached += 1
            data = json.loads(f.read_text(encoding="utf-8"))
        elif self.offline:
            self.skipped += 1           # lot pas encore traité : laissé vide, sans appel
            data = {"analyses": [], "modele": None}
        else:
            try:
                data = self._call(prompt)
                f.write_text(json.dumps(data, ensure_ascii=False), encoding="utf-8")
            except LLMDefinitiveError:
                raise
            except RuntimeError as exc:
                # Lot en échec après tous les essais : on le met de côté (non mis en cache,
                # il sera retenté au prochain lancement) et on continue avec les autres.
                self.failed += 1
                log.error("lot mis de côté (%s phrases) : %s", len(lot), exc)
                data = {"analyses": [], "modele": None}
        by_num = {a["numero"]: {**a, "modele": data.get("modele")} for a in data["analyses"]}
        rows = []
        for n, (_, p) in enumerate(lot.iterrows(), start=1):
            a = by_num.get(n, {"numero": n, "est_promesse": None, "modele": data.get("modele"),
                               "justification": "échec LLM ou phrase absente de la réponse"})
            rows.append({**p.drop(labels=["contexte"], errors="ignore").to_dict(), **a})
        return rows

    @property
    def model(self) -> str:
        return self.models[self.current]

    def _request(self, prompt: str) -> tuple[str, int, int]:
        """Un appel Gemini. Retourne (texte JSON, tokens entrée, tokens sortie)."""
        from google.genai import types
        config = types.GenerateContentConfig(response_mime_type="application/json",
                                             response_schema=Lot, temperature=0)
        resp = self.client.models.generate_content(model=self.model, contents=prompt, config=config)
        usage = getattr(resp, "usage_metadata", None)
        return (resp.text, getattr(usage, "prompt_token_count", 0) or 0,
                getattr(usage, "candidates_token_count", 0) or 0)

    def _call(self, prompt: str, retries: int = 40) -> dict:
        for attempt in range(retries):
            try:
                text, t_in, t_out = self._request(prompt)
                self.calls += 1
                self.tokens_in += t_in
                self.tokens_out += t_out
                lot = Lot.model_validate_json(text)
                self.used[self.model] = self.used.get(self.model, 0) + 1
                self.saturations = 0              # le service répond de nouveau
                self.sleep(self.pause_s)          # rester sous la limite d'appels par minute
                data = lot.model_dump(mode="json")
                data["modele"] = self.model       # traçabilité : quel modèle a répondu
                return data
            except Exception as exc:
                msg = str(exc)
                status = http_status(exc)
                if ("JSON" in msg or "json" in msg or "ValidationError" in type(exc).__name__
                        or "JSONDecodeError" in type(exc).__name__):
                    # Le modèle a produit un JSON invalide : erreur passagère, on essaie le suivant
                    old = self.model
                    self.current = (self.current + 1) % len(self.models)
                    log.warning("%s : JSON invalide -> nouvel essai avec %s", old, self.model)
                    self.sleep(2)
                    continue
                if "Request too large" in msg and len(self.models) > 1:
                    # Requête trop grosse pour ce modèle (limite fixe) : inutile d'insister avec lui
                    gone = self.models.pop(self.current)
                    self.current %= len(self.models)
                    log.warning("modèle %s : requête trop grosse -> retiré, suite avec %s", gone, self.model)
                    continue
                if (status == 404 or "NOT_FOUND" in msg) and len(self.models) > 1:
                    # Modèle retiré ou inconnu : on le sort de la liste et on passe au suivant
                    gone = self.models.pop(self.current)
                    self.current %= len(self.models)
                    log.warning("modèle %s indisponible (404) -> retiré, suite avec %s", gone, self.model)
                    continue
                if status in (401, 403) or "API key not valid" in msg:
                    # Clé invalide, accès refusé : réessayer ne changera rien
                    raise LLMDefinitiveError(f"erreur définitive du LLM, vérifier .env : {msg[:200]}") from exc
                if status == 404 and len(self.models) == 1:
                    raise LLMDefinitiveError(f"modèle introuvable, vérifier .env : {msg[:200]}") from exc
                if status in (429, 503) or "UNAVAILABLE" in msg or "RESOURCE_EXHAUSTED" in msg:
                    old = self.model
                    self.current = (self.current + 1) % len(self.models)
                    if self.current == 0:
                        # Tous les modèles sont saturés : pause croissante (1, 2, 4, 8, 10 min max)
                        self.saturations += 1
                        wait = min(60 * 2 ** (self.saturations - 1), 600)
                        log.warning("tous les modèles sont saturés, pause de %s min avant un nouveau tour",
                                    wait // 60)
                        self.sleep(wait)
                    else:
                        log.warning("%s indisponible (%s) -> bascule sur %s", old, msg[:40], self.model)
                        self.sleep(5)
                    continue
                wait = 15 * 2 ** min(attempt, 4)
                log.warning("appel LLM en échec (%s), nouvel essai dans %s s (%s/%s)",
                            msg[:120], wait, attempt + 1, retries)
                self.sleep(wait)
        raise RuntimeError("appel LLM impossible après plusieurs essais")


class OpenAICompatExtractor(GeminiExtractor):
    """
    Même logique (lots, cache, bascule, patience) mais pour toute API au format OpenAI :
    Groq, xAI (Grok), Mistral, OpenAI… Seuls l'adresse, la clé et le nom du modèle changent.
    Le format JSON est imposé par le mode « json_object », puis vérifié avec le schéma Pydantic.
    """

    def _request(self, prompt: str) -> tuple[str, int, int]:
        schema = json.dumps(Lot.model_json_schema(), ensure_ascii=False)
        extra = {}
        if "gpt-oss" in self.model:
            # Modèle « à raisonnement » : réflexion courte = moins de tokens consommés
            extra["reasoning_effort"] = "low"
        resp = self.client.chat.completions.create(
            model=self.model,
            temperature=0,
            max_tokens=3000,          # plafond de réponse : évite « Request too large » chez Groq
            response_format={"type": "json_object"},
            **extra,
            messages=[
                {"role": "system", "content": "Réponds uniquement par un objet JSON conforme à ce schéma : "
                                              + schema},
                {"role": "user", "content": prompt},
            ],
        )
        usage = getattr(resp, "usage", None)
        return (resp.choices[0].message.content,
                getattr(usage, "prompt_tokens", 0) or 0, getattr(usage, "completion_tokens", 0) or 0)


def run(extractor: GeminiExtractor, passages: pd.DataFrame, out_path: Path,
        limit: int | None = None) -> pd.DataFrame:
    rows = []
    for k, lot in enumerate(batches(passages)):
        if limit is not None and k >= limit:
            break
        rows.extend(extractor.analyse(lot))
        if (k + 1) % 10 == 0:
            log.info("%s lots traités (%s appels, %s depuis le cache)", k + 1, extractor.calls, extractor.cached)
    df = pd.DataFrame(rows)
    out_path.parent.mkdir(parents=True, exist_ok=True)
    df.to_json(out_path, orient="records", lines=True, force_ascii=False)
    return df


PROVIDERS = {   # adresse de l'API au format OpenAI
    "groq": "https://api.groq.com/openai/v1",
    "xai": "https://api.x.ai/v1",
    "mistral": "https://api.mistral.ai/v1",
    "openai": "https://api.openai.com/v1",
}


def make_extractor(env: dict) -> GeminiExtractor:
    """LLM_PROVIDER dans .env : gemini (défaut), groq, xai, mistral ou openai."""
    provider = env.get("LLM_PROVIDER", "gemini").lower()
    if provider == "gemini":
        from google import genai
        return GeminiExtractor(genai.Client(api_key=env["GEMINI_API_KEY"]), env["GEMINI_MODEL"])
    from openai import OpenAI
    client = OpenAI(api_key=env["LLM_API_KEY"], base_url=env.get("LLM_BASE_URL", PROVIDERS[provider]),
                    max_retries=0)
    # Offres gratuites : limite de tokens par minute -> pause plus longue entre deux appels
    return OpenAICompatExtractor(client, env["LLM_MODEL"], pause_s=float(env.get("LLM_PAUSE_S", 10)))


def main() -> None:
    ap = argparse.ArgumentParser()
    ap.add_argument("--passages", type=Path, default=Path("data/silver/passages.jsonl"))
    ap.add_argument("--out", type=Path, default=Path("data/silver/promesses.jsonl"))
    ap.add_argument("--limit", type=int, help="nombre maximum de lots (tests)")
    ap.add_argument("--cache-only", action="store_true",
                    help="n'utiliser que les lots déjà traités (aucun appel, aucun quota consommé)")
    args = ap.parse_args()
    logging.basicConfig(level=logging.INFO, format="%(asctime)s %(levelname)s %(message)s")

    extractor = make_extractor({**_read_env(), **os.environ})
    extractor.offline = args.cache_only
    log.info("fournisseur %s, modèles (par ordre de préférence) : %s",
             type(extractor).__name__, ", ".join(extractor.models))
    passages = pd.read_json(args.passages, lines=True, dtype={"filing_id": str})
    df = run(extractor, passages, args.out, args.limit)

    n_prom = int(df["est_promesse"].fillna(False).sum())
    if extractor.skipped:
        print(f"\nℹ {extractor.skipped} lot(s) pas encore traités (mode --cache-only) : "
              "relancer sans --cache-only pour les compléter.")
    if extractor.failed:
        print(f"\n⚠ {extractor.failed} lot(s) mis de côté : relancer la commande pour les retenter.")
    print(f"\nModèles utilisés : {extractor.used or 'aucun appel (tout depuis le cache)'}")
    print(f"Phrases analysées : {len(df)}  |  promesses : {n_prom} ({n_prom / max(1, len(df)):.0%})")
    print(f"Appels LLM : {extractor.calls} (+ {extractor.cached} depuis le cache)  |  "
          f"tokens entrée/sortie : {extractor.tokens_in} / {extractor.tokens_out}")
    if n_prom:
        cols = ["entreprise", "exercice_rapport", "indicateur", "type_cible", "valeur_min",
                "valeur_max", "unite", "annee_echeance", "prudence"]
        print("\nExemples de promesses :\n")
        print(df[df["est_promesse"] == True][cols].head(10).to_string(index=False))  # noqa: E712


if __name__ == "__main__":
    main()