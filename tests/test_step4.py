import json

import pandas as pd

from src.ai.extract_promises import GeminiExtractor, Lot, OpenAICompatExtractor, batches, build_prompt, run


class _Resp:
    def __init__(self, lot):
        self.parsed, self.text, self.usage_metadata = lot, lot.model_dump_json(), None


class _FakeModels:
    def __init__(self, fail_first=False):
        self.calls, self.fail_first = 0, fail_first

    def generate_content(self, model, contents, config):
        self.calls += 1
        if self.fail_first and self.calls == 1:
            raise RuntimeError("429 RESOURCE_EXHAUSTED")
        n = contents.count("\n") - contents.split("Phrases :")[0].count("\n")
        analyses = []
        for i in range(1, n + 1):
            promise = "prévoit" in contents.split(f"\n{i}. ")[1].split("\n")[0]
            analyses.append({"numero": i, "est_promesse": promise,
                             "indicateur": "chiffre_affaires" if promise else None,
                             "type_cible": "fourchette" if promise else None,
                             "valeur_min": 4 if promise else None, "valeur_max": 6 if promise else None,
                             "unite": "%" if promise else None, "annee_echeance": 2026 if promise else None})
        return _Resp(Lot.model_validate({"analyses": analyses}))


class _FakeClient:
    def __init__(self, **kw):
        self.models = _FakeModels(**kw)


def _passages(n=20):
    return pd.DataFrame([{"filing_id": "7", "lei": "L", "entreprise": "ALPHA", "exercice_rapport": 2025,
                          "phrase": (f"Le Groupe prévoit une croissance de 4 % à 6 % en 2026 ({i})." if i % 2
                                     else f"Le taux d'actualisation retenu est de 8 % jusqu'en 2030 ({i})."),
                          "contexte": "..."} for i in range(n)])


def test_lots_de_15_par_rapport():
    assert [len(b) for b in batches(_passages(20))] == [15, 5]
    assert "rapport annuel 2025 de ALPHA" in build_prompt(_passages(3))


def test_extraction_cache_et_reprise(tmp_path):
    client = _FakeClient(fail_first=True)
    ex = GeminiExtractor(client, "m", cache_dir=tmp_path / "cache", pause_s=0, sleep=lambda s: None)
    df = run(ex, _passages(20), tmp_path / "out.jsonl")
    assert len(df) == 20 and df["est_promesse"].sum() == 10
    assert client.models.calls == 3                   # 1 échec + 2 lots
    assert df.loc[df["est_promesse"], "annee_echeance"].eq(2026).all()

    ex2 = GeminiExtractor(client, "m", cache_dir=tmp_path / "cache", pause_s=0, sleep=lambda s: None)
    run(ex2, _passages(20), tmp_path / "out.jsonl")
    assert client.models.calls == 3 and ex2.cached == 2   # 2e passage : tout vient du cache
    assert json.loads((tmp_path / "out.jsonl").read_text().splitlines()[0])["entreprise"] == "ALPHA"


def test_cle_invalide_arrete_tout_de_suite(tmp_path):
    import pytest

    class Bad:
        class models:
            calls = 0

            @classmethod
            def generate_content(cls, **kw):
                cls.calls += 1
                raise RuntimeError("401 UNAUTHENTICATED. API key not valid")

    ex = GeminiExtractor(Bad(), "m", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    with pytest.raises(RuntimeError, match="vérifier .env"):  # LLMDefinitiveError hérite de RuntimeError
        run(ex, _passages(3), tmp_path / "o.jsonl")
    assert Bad.models.calls == 1


def test_bascule_sur_un_autre_modele_si_surcharge(tmp_path):
    client = _FakeClient()
    real = client.models.generate_content

    def overloaded(model, contents, config):
        if model == "a":
            raise RuntimeError("503 UNAVAILABLE high demand")
        return real(model=model, contents=contents, config=config)

    client.models.generate_content = overloaded
    ex = GeminiExtractor(client, "a,b", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    df = run(ex, _passages(5), tmp_path / "o.jsonl")
    assert ex.used == {"b": 1} and set(df["modele"]) == {"b"}


def test_modele_retire_404_on_passe_au_suivant(tmp_path):
    client = _FakeClient()
    real = client.models.generate_content

    def gone(model, contents, config):
        if model == "vieux":
            raise RuntimeError("404 NOT_FOUND. model is no longer available")
        return real(model=model, contents=contents, config=config)

    client.models.generate_content = gone
    ex = GeminiExtractor(client, "vieux,neuf", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    run(ex, _passages(5), tmp_path / "o.jsonl")
    assert ex.models == ["neuf"] and ex.used == {"neuf": 1}


def test_patience_quand_tous_les_modeles_sont_satures(tmp_path):
    client = _FakeClient()
    real = client.models.generate_content
    state = {"n": 0}

    def busy(model, contents, config):
        state["n"] += 1
        if state["n"] <= 4:                      # 2 tours complets de saturation
            raise RuntimeError("503 UNAVAILABLE")
        return real(model=model, contents=contents, config=config)

    client.models.generate_content = busy
    waits = []
    ex = GeminiExtractor(client, "a,b", cache_dir=tmp_path, pause_s=0, sleep=waits.append)
    run(ex, _passages(3), tmp_path / "o.jsonl")
    assert [w for w in waits if w >= 60] == [60, 120]      # pauses croissantes


def test_fournisseur_compatible_openai(tmp_path):
    fake = _FakeModels()

    class Chat:
        class completions:
            @staticmethod
            def create(model, messages, **kw):
                assert kw["response_format"] == {"type": "json_object"}
                lot = fake.generate_content(model, messages[1]["content"], None).parsed

                class R:
                    choices = [type("C", (), {"message": type("M", (), {"content": lot.model_dump_json()})})]
                    usage = type("U", (), {"prompt_tokens": 100, "completion_tokens": 50})
                return R

    client = type("Client", (), {"chat": Chat})()
    ex = OpenAICompatExtractor(client, "llama", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    df = run(ex, _passages(4), tmp_path / "o.jsonl")
    assert df["est_promesse"].sum() == 2 and ex.tokens_in == 100 and ex.used == {"llama": 1}


def test_json_invalide_et_requete_trop_grosse(tmp_path):
    client = _FakeClient()
    real = client.models.generate_content
    state = {"n": 0}

    def flaky(model, contents, config):
        state["n"] += 1
        if model == "gros":
            raise RuntimeError("Error code: 429 - Request too large for model")
        if state["n"] == 1:
            raise RuntimeError("Error code: 400 - Failed to validate JSON")
        return real(model=model, contents=contents, config=config)

    client.models.generate_content = flaky
    ex = GeminiExtractor(client, "a,gros,b", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    df = run(ex, _passages(3), tmp_path / "o.jsonl")
    assert len(df) == 3 and "gros" not in ex.models


def test_lot_en_echec_mis_de_cote_sans_tout_arreter(tmp_path):
    client = _FakeClient()
    real = client.models.generate_content

    def json_ko(model, contents, config):
        if "(0)" in contents:                       # le 1er lot échoue toujours
            raise RuntimeError("Error code: 400 - Failed to generate JSON")
        return real(model=model, contents=contents, config=config)

    client.models.generate_content = json_ko
    ex = GeminiExtractor(client, "a", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    df = run(ex, _passages(20), tmp_path / "o.jsonl")
    assert ex.failed == 1 and len(df) == 20
    assert df["est_promesse"].isna().sum() == 15 and df["est_promesse"].sum() == 3
    assert len(list(tmp_path.glob("*.json"))) == 1     # seul le lot réussi est en cache


def test_mode_cache_seulement(tmp_path):
    client = _FakeClient()
    ex = GeminiExtractor(client, "a", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    run(ex, _passages(15), tmp_path / "o.jsonl")          # 1 lot traité et mis en cache
    ex2 = GeminiExtractor(client, "a", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    ex2.offline = True
    df = run(ex2, _passages(30), tmp_path / "o.jsonl")
    assert client.models.calls == 1 and ex2.skipped == 1 and df["est_promesse"].notna().sum() == 15


def test_429_avec_4036_tokens_n_est_pas_un_403(tmp_path):
    from src.ai.extract_promises import http_status
    msg = ("Error code: 429 - Rate limit reached for model `openai/gpt-oss-120b` on tokens per minute "
           "(TPM): Limit 8000, Used 4962, Requested 4036. Please try again in 7.485s.")
    assert http_status(RuntimeError(msg)) == 429
    assert http_status(RuntimeError("401 UNAUTHENTICATED. API key not valid")) == 401

    client = _FakeClient()
    real = client.models.generate_content
    state = {"n": 0}

    def limited(model, contents, config):
        state["n"] += 1
        if state["n"] == 1:
            raise RuntimeError(msg)
        return real(model=model, contents=contents, config=config)

    client.models.generate_content = limited
    ex = GeminiExtractor(client, "a,b", cache_dir=tmp_path, pause_s=0, sleep=lambda s: None)
    assert len(run(ex, _passages(3), tmp_path / "o.jsonl")) == 3