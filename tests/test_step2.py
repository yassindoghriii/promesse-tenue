import boto3
from moto import mock_aws

from src.storage.lake import ensure_zones, list_lake, sync_dir


@mock_aws
def test_zones_et_synchronisation_idempotente(tmp_path):
    s3 = boto3.client("s3", region_name="us-east-1")
    assert ensure_zones(s3) == ["bronze", "silver", "gold"]
    assert ensure_zones(s3) == []                      # 2e appel : rien à créer

    (tmp_path / "esef" / "json").mkdir(parents=True)
    (tmp_path / "esef" / "json" / "1.json").write_text('{"a": 1}')
    (tmp_path / "esef" / "filings_FR.jsonl").write_text("{}\n")
    (tmp_path / ".gitkeep").write_text("")

    assert sync_dir(s3, tmp_path, "bronze") == (2, 0)  # .gitkeep ignoré
    assert sync_dir(s3, tmp_path, "bronze") == (0, 2)  # déjà à jour
    keys = {o["Key"] for o in s3.list_objects_v2(Bucket="bronze")["Contents"]}
    assert keys == {"esef/json/1.json", "esef/filings_FR.jsonl"}
    assert list_lake(s3)["bronze"][0] == 2