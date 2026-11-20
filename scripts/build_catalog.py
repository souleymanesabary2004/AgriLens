"""build_catalog.py - E5 : genere 11_catalog/catalog.json et 11_catalog/lineage.json.

Usage (depuis la racine du projet) :
    python scripts/build_catalog.py --scan 01_data
    python scripts/build_catalog.py --scan 01_data --scan 05_exports/excel
Ne modifie JAMAIS les donnees : lecture seule. Les images ne sont pas cataloguees (seulement tabulaire).
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agrilens import catalog as cat  # noqa: E402
from agrilens import lineage as lin  # noqa: E402


def main():
    ap = argparse.ArgumentParser(description=__doc__)
    ap.add_argument("--scan", action="append", default=None, help="dossier a parcourir (repetable)")
    ap.add_argument("--descriptions", default=str(ROOT / "11_catalog/descriptions.json"))
    ap.add_argument("--declared", default=str(ROOT / "11_catalog/lineage_declared.json"))
    ap.add_argument("--sql-dir", default=str(ROOT / "11_catalog/sql"))
    ap.add_argument("--out-dir", default=str(ROOT / "11_catalog"))
    a = ap.parse_args()
    scan = a.scan or [str(ROOT / "01_data")]

    desc = json.loads(Path(a.descriptions).read_text(encoding="utf-8")) if Path(a.descriptions).exists() else {}
    catalog = cat.build_catalog(scan, desc, root=ROOT)
    cat.save_catalog(catalog, Path(a.out_dir) / "catalog.json")

    # Schema pour resoudre les colonnes SQL : issu du catalogue lui-meme (pas ecrit a la main)
    schema = {d["name"]: {c["name"]: "TEXT" for c in d["technical"]["columns"]} for d in catalog["datasets"]}
    declared = json.loads(Path(a.declared).read_text(encoding="utf-8")) if Path(a.declared).exists() else []
    sql_files = sorted(Path(a.sql_dir).glob("*.sql")) if Path(a.sql_dir).exists() else []
    graph = lin.build_graph(declared, sql_files, schema)
    (Path(a.out_dir) / "lineage.json").write_text(json.dumps(graph, ensure_ascii=False, indent=2), encoding="utf-8")
    (Path(a.out_dir) / "lineage.mmd").write_text(lin.to_mermaid(graph), encoding="utf-8")

    s = catalog["summary"]
    print(f"Catalogue : {s['n_datasets']} jeux, {s['n_columns']} colonnes, "
          f"couverture documentaire {s['documentation_coverage']:.0%}")
    for e in catalog["errors"]:
        print("  ILLISIBLE :", e["path"], "->", e["error"])
    print(f"Lineage : {len(graph['nodes'])} noeuds, {len(graph['edges'])} aretes "
          f"({sum(e['kind']=='sql_parsed' for e in graph['edges'])} SQL, "
          f"{sum(e['kind']=='declared' for e in graph['edges'])} declarees)")
    print("Ecrit dans", a.out_dir)


if __name__ == "__main__":
    main()
