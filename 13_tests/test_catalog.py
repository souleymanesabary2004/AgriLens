"""Tests E5 : catalogue, lineage SQL, routes REST. Donnees synthetiques (aucun fichier reel requis)."""
import pandas as pd
import pytest
from fastapi.testclient import TestClient
from main import create_app

from agrilens import catalog as cat
from agrilens import lineage as lin


@pytest.fixture()
def lake(tmp_path):
    (tmp_path / "gold").mkdir()
    (tmp_path / "bronze").mkdir()
    df = pd.DataFrame({"filepath_rel": ["a.jpg", "b.jpg", "c.jpg"], "split": ["train", "val", None],
                       "class_id": [0, 1, 1]})
    df.to_parquet(tmp_path / "gold" / "manifest.parquet")
    df.to_csv(tmp_path / "bronze" / "raw.csv", index=False)
    df.to_excel(tmp_path / "bronze" / "sheet.xlsx", index=False)
    (tmp_path / "bronze" / "bad.csv").write_bytes(b"\x00\x01\x02")
    return tmp_path


def test_catalog_three_formats_and_layers(lake):
    c = cat.build_catalog([lake], {"manifest": {"description": "d", "owner": "o", "columns": {"split": "partition"}}})
    names = {d["name"]: d for d in c["datasets"]}
    assert {"manifest", "raw", "sheet"} <= set(names)
    assert names["manifest"]["functional"]["layer"] == "gold"
    assert names["raw"]["functional"]["layer"] == "bronze"
    m = names["manifest"]
    assert m["technical"]["n_rows"] == 3 and m["technical"]["n_columns"] == 3
    assert next(x for x in m["technical"]["columns"] if x["name"] == "split")["null_count"] == 1
    assert m["functional"]["documentation_coverage"] == round(1 / 3, 3)
    assert len(m["operational"]["sha256"]) == 64


def test_unreadable_file_does_not_break_catalog(lake):
    c = cat.build_catalog([lake])
    assert any(e["path"].endswith("bad.csv") for e in c["errors"]) or any(d["name"] == "bad" for d in c["datasets"])
    assert c["summary"]["n_datasets"] >= 3


def test_sha256_changes_with_content(lake):
    p = lake / "bronze" / "raw.csv"
    h1 = cat.sha256_file(p)
    p.write_text("x\n1\n")
    assert cat.sha256_file(p) != h1


def test_sql_lineage_tables_and_columns():
    sql = """CREATE TABLE out_t AS SELECT m.display_fr AS classe, COUNT(*) AS n
             FROM manifest AS m WHERE m.split = 'train' GROUP BY m.display_fr"""
    info = lin.sql_tables(sql)
    assert info == {"reads": ["manifest"], "target": "out_t"}
    cols = {c["column"]: c for c in lin.column_lineage(sql, schema={"manifest": {"display_fr": "TEXT", "split": "TEXT"}})}
    assert cols["classe"]["sources"] == ["manifest.display_fr"]


def test_sql_join_and_cte():
    sql = """CREATE TABLE r AS WITH t AS (SELECT a.id, a.v FROM A a) SELECT t.id, b.w FROM t JOIN B b ON t.id = b.id"""
    info = lin.sql_tables(sql)
    assert set(info["reads"]) == {"A", "B"} and info["target"] == "r"
    cols = {c["column"]: c["sources"] for c in lin.column_lineage(
        sql, schema={"A": {"id": "INT", "v": "INT"}, "B": {"id": "INT", "w": "INT"}})}
    assert cols["id"] == ["a.id"] and cols["w"] == ["b.w"]


def test_graph_traverse_and_mermaid(tmp_path):
    f = tmp_path / "s.sql"
    f.write_text("CREATE TABLE c AS SELECT x FROM b")
    g = lin.build_graph([{"from": "a", "to": "b", "step": "s1"}], [f], {"b": {"x": "INT"}})
    assert lin.traverse(g, "a") == ["b", "c"]
    assert lin.traverse(g, "c", "upstream") == ["b", "a"]
    kinds = {(e["from"], e["to"]): e["kind"] for e in g["edges"]}
    assert kinds[("a", "b")] == "declared" and kinds[("b", "c")] == "sql_parsed"
    assert "-.->" in lin.to_mermaid(g) and "-->" in lin.to_mermaid(g)
    with pytest.raises(ValueError):
        lin.traverse(g, "a", "sideways")


@pytest.fixture()
def client(lake, tmp_path):
    c = cat.build_catalog([lake])
    g = lin.build_graph([{"from": "raw", "to": "manifest", "step": "x"}])
    state = {"catalog": c, "graph": g, "path": tmp_path / "catalog.json"}
    with TestClient(create_app(predictor=object(), retriever=None, catalog_state=state)) as c:
        yield c


def test_api_catalog_routes(client):
    r = client.get("/api/v1/catalog")
    assert r.status_code == 200 and r.json()["count"] >= 3
    assert client.get("/api/v1/catalog?layer=gold").json()["count"] == 1
    assert client.get("/api/v1/catalog/manifest").json()["technical"]["n_rows"] == 3
    assert client.get("/api/v1/catalog/inconnu").status_code == 404
    assert "documentation_coverage" in client.get("/api/v1/catalog/summary").json()


def test_api_registration_validation(client):
    body = {"name": "ext", "layer": "silver", "columns": [{"name": "id", "dtype": "int"}]}
    assert client.post("/api/v1/catalog/datasets", json=body).status_code == 201
    assert client.post("/api/v1/catalog/datasets", json=body).status_code == 409
    assert client.get("/api/v1/catalog/ext").status_code == 200
    assert client.post("/api/v1/catalog/datasets", json={"name": "z", "columns": []}).status_code == 422
    dup = {"name": "d", "columns": [{"name": "a"}, {"name": "a"}]}
    assert client.post("/api/v1/catalog/datasets", json=dup).status_code == 422
    assert client.post("/api/v1/catalog/datasets", json={"columns": [{"name": "a"}]}).status_code == 422


def test_api_lineage_routes(client):
    assert client.get("/api/v1/lineage").json()["edges"][0]["from"] == "raw"
    assert client.get("/api/v1/lineage?format=mermaid").text.startswith("graph LR")
    assert client.get("/api/v1/lineage/raw").json()["nodes"] == ["manifest"]
    assert client.get("/api/v1/lineage/manifest?direction=upstream").json()["nodes"] == ["raw"]
    assert client.get("/api/v1/lineage/nope").status_code == 404
    assert client.get("/api/v1/lineage/raw?direction=x").status_code == 422


def test_api_without_catalog_returns_503(lake):
    with TestClient(create_app(predictor=object(), retriever=None, catalog_state=None)) as c:
        assert c.get("/api/v1/catalog").status_code == 503
