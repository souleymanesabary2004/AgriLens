"""catalog.py - E5 : mini catalogue de donnees (metadonnees techniques, fonctionnelles, operationnelles).

Ce que fait ce module (et RIEN de plus) :
  - il LIT des fichiers de donnees (Parquet, CSV, Excel) et en extrait des metadonnees TECHNIQUES
    (format, taille, lignes, colonnes, types, valeurs manquantes) ;
  - il y ajoute des metadonnees FONCTIONNELLES ecrites par un humain (description, proprietaire,
    description de chaque colonne) lues dans un fichier de configuration ;
  - il calcule des metadonnees OPERATIONNELLES (empreinte SHA-256, date de modification, date de
    catalogage, commit git si disponible) ;
  - il mesure la couverture documentaire (colonnes decrites / colonnes totales), utile a un Data Steward.

Ce que ce module NE fait PAS : pas de base de donnees, pas de connecteur Kafka / Schema Registry,
pas d'ordonnancement (TWS), pas d'authentification. C'est un prototype local, pas Collibra/DataHub.
"""
from __future__ import annotations

import hashlib
import json
import re
import subprocess
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

import pandas as pd

SUPPORTED = {".parquet", ".csv", ".xlsx", ".xls"}
LAYER_HINTS = ("bronze", "silver", "gold")
SCHEMA_VERSION = "1.0"


def _now() -> str:
    return datetime.now(timezone.utc).isoformat(timespec="seconds")


def sha256_file(path: Path, chunk: int = 1 << 20) -> str:
    h = hashlib.sha256()
    with open(path, "rb") as f:
        for block in iter(lambda: f.read(chunk), b""):
            h.update(block)
    return h.hexdigest()


def git_commit(cwd: Path) -> str | None:
    """Commit courant si on est dans un depot git, sinon None (jamais d'erreur)."""
    try:
        out = subprocess.run(["git", "rev-parse", "--short", "HEAD"], cwd=cwd, capture_output=True,
                             text=True, timeout=5)
        return out.stdout.strip() or None if out.returncode == 0 else None
    except (OSError, subprocess.SubprocessError):
        return None


def infer_layer(path: Path) -> str:
    """Couche medaillon deduite du chemin (bronze/silver/gold), sinon 'unclassified'."""
    for part in (p.lower() for p in path.parts):
        for hint in LAYER_HINTS:
            if hint in part:
                return hint
    return "unclassified"


def read_table(path: Path) -> pd.DataFrame:
    ext = path.suffix.lower()
    if ext == ".parquet":
        return pd.read_parquet(path)
    if ext == ".csv":
        return pd.read_csv(path)
    if ext in {".xlsx", ".xls"}:
        return pd.read_excel(path)
    raise ValueError(f"Format non supporte : {ext} (attendu : {sorted(SUPPORTED)})")


def describe_columns(df: pd.DataFrame, column_docs: dict[str, str] | None = None,
                     column_groups: list[dict[str, str]] | None = None) -> list[dict[str, Any]]:
    """`column_docs` : description colonne par colonne. `column_groups` : [{"pattern": regex, "description": texte}]
    pour documenter d'un coup des familles de colonnes (ex. e_0 ... e_1279). La description exacte prime."""
    column_docs = column_docs or {}
    groups = [(re.compile(g["pattern"]), g["description"]) for g in (column_groups or [])]

    def doc_for(col: str) -> str | None:
        if col in column_docs:
            return column_docs[col]
        return next((d for rx, d in groups if rx.fullmatch(col)), None)

    cols = []
    for name in df.columns:
        s = df[name]
        cols.append({
            "name": str(name),
            "dtype": str(s.dtype),
            "null_count": int(s.isna().sum()),
            "n_unique": int(s.nunique(dropna=True)) if s.dtype != "object" or len(s) < 200_000 else None,
            "description": doc_for(str(name)),   # None = non documentee
        })
    return cols


def build_entry(path: str | Path, name: str | None = None, layer: str | None = None,
                meta: dict[str, Any] | None = None, root: Path | None = None) -> dict[str, Any]:
    """Construit la fiche catalogue d'UN fichier. `meta` = metadonnees fonctionnelles saisies par un humain."""
    path = Path(path)
    meta = meta or {}
    root = root or Path.cwd()
    df = read_table(path)
    columns = describe_columns(df, meta.get("columns"), meta.get("column_groups"))
    documented = sum(1 for c in columns if c["description"])
    st = path.stat()
    return {
        "name": name or path.stem,
        "technical": {
            "path": str(path.as_posix()),
            "format": path.suffix.lower().lstrip("."),
            "size_bytes": st.st_size,
            "n_rows": int(len(df)),
            "n_columns": int(df.shape[1]),
            "columns": columns,
        },
        "functional": {
            "layer": layer or meta.get("layer") or infer_layer(path),
            "description": meta.get("description"),
            "owner": meta.get("owner"),
            "steward": meta.get("steward"),
            "tags": meta.get("tags", []),
            "documentation_coverage": round(documented / len(columns), 3) if columns else 0.0,
        },
        "operational": {
            "sha256": sha256_file(path),
            "modified_utc": datetime.fromtimestamp(st.st_mtime, timezone.utc).isoformat(timespec="seconds"),
            "cataloged_utc": _now(),
            "git_commit": git_commit(root),
            "source_system": meta.get("source_system", "fichier local"),
        },
    }


def discover(scan_dir: str | Path) -> list[Path]:
    """Trouve tous les fichiers Parquet/CSV/Excel sous un dossier (ignore .git, caches, .dvc)."""
    skip = {".git", "__pycache__", ".dvc", ".venv", "venv", "node_modules"}
    found = []
    for p in sorted(Path(scan_dir).rglob("*")):
        if p.is_file() and p.suffix.lower() in SUPPORTED and not (set(p.parts) & skip):
            found.append(p)
    return found


def unique_names(paths: list[Path]) -> dict[Path, str]:
    """Nom du jeu = nom du fichier sans extension. Si deux fichiers ont le meme nom, on prefixe par le dossier
    parent (ex. bronze.inventory et metadata.inventory) pour que chaque nom designe UN seul jeu (cle d'API)."""
    count: dict[str, int] = {}
    for p in paths:
        count[p.stem] = count.get(p.stem, 0) + 1
    return {p: (p.stem if count[p.stem] == 1 else f"{p.parent.name}.{p.stem}") for p in paths}


def build_catalog(scan_dirs: list[str | Path], descriptions: dict[str, Any] | None = None,
                  root: Path | None = None) -> dict[str, Any]:
    """Catalogue complet.

    `descriptions` = {cle: {description, owner, columns:{col: texte}, column_groups:[...], layer, ...}}.
    La cle est le chemin du fichier tel que scanne (ex. "01_data/metadata/inventory.csv") ou, a defaut,
    son nom sans extension (ex. "inventory_clean"). Le chemin exact a priorite : il leve les ambiguites.
    """
    descriptions = descriptions or {}
    datasets, errors = [], []
    paths, seen = [], set()
    for d in scan_dirs:
        for p in discover(d):
            if p not in seen:
                seen.add(p)
                paths.append(p)
    names = unique_names(paths)
    for p in paths:
        meta = descriptions.get(p.as_posix()) or descriptions.get(p.stem, {})
        try:
            datasets.append(build_entry(p, name=names[p], meta=meta, root=root))
        except Exception as exc:  # un fichier illisible ne doit pas casser tout le catalogue
            errors.append({"path": str(p.as_posix()), "error": f"{type(exc).__name__}: {exc}"})
    total_cols = sum(x["technical"]["n_columns"] for x in datasets)
    doc_cols = sum(sum(1 for c in x["technical"]["columns"] if c["description"]) for x in datasets)
    return {
        "schema_version": SCHEMA_VERSION,
        "generated_utc": _now(),
        "datasets": datasets,
        "errors": errors,
        "summary": {
            "n_datasets": len(datasets),
            "n_columns": total_cols,
            "documented_columns": doc_cols,
            "documentation_coverage": round(doc_cols / total_cols, 3) if total_cols else 0.0,
        },
    }


def save_catalog(cat: dict[str, Any], out: str | Path) -> None:
    out = Path(out)
    out.parent.mkdir(parents=True, exist_ok=True)
    out.write_text(json.dumps(cat, ensure_ascii=False, indent=2), encoding="utf-8")


def load_catalog(path: str | Path) -> dict[str, Any]:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def find_dataset(cat: dict[str, Any], name: str) -> dict[str, Any] | None:
    return next((d for d in cat["datasets"] if d["name"] == name), None)


def validate_registration(entry: dict[str, Any]) -> list[str]:
    """Controles minimaux pour un enregistrement de metadonnees recu par l'API (ingestion REST)."""
    problems = []
    if not str(entry.get("name", "")).strip():
        problems.append("name obligatoire")
    cols = entry.get("columns") or []
    if not cols:
        problems.append("au moins une colonne attendue")
    names = [c.get("name") for c in cols]
    if len(names) != len(set(names)):
        problems.append("noms de colonnes dupliques")
    if any(not n for n in names):
        problems.append("nom de colonne vide")
    return problems