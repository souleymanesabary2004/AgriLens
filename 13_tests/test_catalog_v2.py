"""Tests E5 (corrections) : noms uniques, cle par chemin, familles de colonnes."""
import pandas as pd

from agrilens import catalog as cat


def _two_inventories(tmp_path):
    (tmp_path / "bronze").mkdir()
    (tmp_path / "metadata").mkdir()
    pd.DataFrame({"a": [1, 2]}).to_parquet(tmp_path / "bronze" / "inventory.parquet")
    pd.DataFrame({"a": [1, 2, 3]}).to_csv(tmp_path / "metadata" / "inventory.csv", index=False)


def test_duplicate_stems_get_unique_names(tmp_path):
    _two_inventories(tmp_path)
    c = cat.build_catalog([tmp_path])
    names = sorted(d["name"] for d in c["datasets"])
    assert names == ["bronze.inventory", "metadata.inventory"]
    assert cat.find_dataset(c, "metadata.inventory")["technical"]["n_rows"] == 3


def test_unique_stem_keeps_plain_name(tmp_path):
    pd.DataFrame({"a": [1]}).to_csv(tmp_path / "solo.csv", index=False)
    assert [d["name"] for d in cat.build_catalog([tmp_path])["datasets"]] == ["solo"]


def test_description_key_by_path_has_priority_over_stem(tmp_path):
    _two_inventories(tmp_path)
    key = (tmp_path / "metadata" / "inventory.csv").as_posix()
    c = cat.build_catalog([tmp_path], {key: {"description": "csv", "layer": "gold"}, "inventory": {"description": "autre"}})
    d = {x["name"]: x for x in c["datasets"]}
    assert d["metadata.inventory"]["functional"]["description"] == "csv"
    assert d["metadata.inventory"]["functional"]["layer"] == "gold"
    assert d["bronze.inventory"]["functional"]["description"] == "autre"


def test_column_groups_document_families_but_exact_wins(tmp_path):
    df = pd.DataFrame({"filepath_rel": ["x"], "e_0": [0.1], "e_1": [0.2], "other": [1]})
    df.to_parquet(tmp_path / "emb.parquet")
    meta = {"emb": {"columns": {"e_1": "exacte"}, "column_groups": [{"pattern": r"e_\d+", "description": "groupe"}]}}
    c = cat.build_catalog([tmp_path], meta)
    cols = {x["name"]: x["description"] for x in c["datasets"][0]["technical"]["columns"]}
    assert cols["e_0"] == "groupe" and cols["e_1"] == "exacte" and cols["other"] is None
    assert c["datasets"][0]["functional"]["documentation_coverage"] == 0.5
