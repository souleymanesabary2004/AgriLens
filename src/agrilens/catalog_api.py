"""catalog_api.py - E5 : routes REST du catalogue et du lineage (prefixe /api/v1).

  GET  /api/v1/catalog                  liste (filtre ?layer=gold)
  GET  /api/v1/catalog/summary          nombre de jeux, couverture documentaire
  GET  /api/v1/catalog/{name}           fiche complete d'un jeu de donnees
  POST /api/v1/catalog/datasets         enregistre des metadonnees recues (ingestion REST, validation)
  GET  /api/v1/lineage                  graphe complet (?format=mermaid pour le dessin)
  GET  /api/v1/lineage/{node}           impact ?direction=downstream|upstream
Prototype local : pas d'authentification (a ajouter avant toute exposition reelle).
"""
from __future__ import annotations

import json
import threading
from pathlib import Path
from typing import Any, Dict, List, Optional

from fastapi import APIRouter, HTTPException, Request
from fastapi.responses import PlainTextResponse
from pydantic import BaseModel, Field

from . import catalog as cat
from . import lineage as lin

router = APIRouter(prefix="/api/v1", tags=["catalog"])
_lock = threading.Lock()


class ColumnIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    dtype: str = Field(default="unknown", max_length=64)
    description: Optional[str] = Field(default=None, max_length=1000)


class DatasetIn(BaseModel):
    name: str = Field(min_length=1, max_length=128)
    layer: str = "unclassified"
    description: Optional[str] = Field(default=None, max_length=2000)
    owner: Optional[str] = None
    source_system: str = "API REST"
    columns: List[ColumnIn]


def _state(request: Request) -> Dict[str, Any]:
    st = getattr(request.app.state, "catalog_state", None)
    if st is None:
        raise HTTPException(503, "Catalogue non charge : lance scripts/build_catalog.py.")
    return st


@router.get("/catalog")
def list_catalog(request: Request, layer: Optional[str] = None):
    c = _state(request)["catalog"]
    items = [{"name": d["name"], "layer": d["functional"]["layer"], "n_rows": d["technical"]["n_rows"],
              "n_columns": d["technical"]["n_columns"],
              "documentation_coverage": d["functional"]["documentation_coverage"]}
             for d in c["datasets"] if layer is None or d["functional"]["layer"] == layer]
    return {"count": len(items), "items": items}


@router.get("/catalog/summary")
def summary(request: Request):
    c = _state(request)["catalog"]
    return {**c["summary"], "generated_utc": c["generated_utc"], "errors": len(c.get("errors", []))}


@router.get("/catalog/{name}")
def get_dataset(request: Request, name: str):
    d = cat.find_dataset(_state(request)["catalog"], name)
    if d is None:
        raise HTTPException(404, f"Jeu de donnees inconnu : {name}")
    return d


@router.post("/catalog/datasets", status_code=201)
def register(request: Request, body: DatasetIn):
    st = _state(request)
    entry = body.model_dump()
    problems = cat.validate_registration(entry)
    if problems:
        raise HTTPException(422, "; ".join(problems))
    with _lock:
        if cat.find_dataset(st["catalog"], body.name):
            raise HTTPException(409, f"Jeu de donnees deja catalogue : {body.name}")
        cols = [{"name": c.name, "dtype": c.dtype, "null_count": None, "n_unique": None,
                 "description": c.description} for c in body.columns]
        documented = sum(1 for c in cols if c["description"])
        st["catalog"]["datasets"].append({
            "name": body.name,
            "technical": {"path": None, "format": "api", "size_bytes": None, "n_rows": None,
                          "n_columns": len(cols), "columns": cols},
            "functional": {"layer": body.layer, "description": body.description, "owner": body.owner,
                           "steward": None, "tags": ["ingere-par-api"],
                           "documentation_coverage": round(documented / len(cols), 3)},
            "operational": {"sha256": None, "modified_utc": None, "cataloged_utc": cat._now(),
                            "git_commit": None, "source_system": body.source_system},
        })
        if st.get("path"):
            cat.save_catalog(st["catalog"], st["path"])
    return {"registered": body.name, "n_columns": len(cols)}


@router.get("/lineage")
def get_lineage(request: Request, format: str = "json"):
    g = _state(request)["graph"]
    if format == "mermaid":
        return PlainTextResponse(lin.to_mermaid(g))
    if format != "json":
        raise HTTPException(422, "format doit etre json ou mermaid")
    return g


@router.get("/lineage/{node}")
def impact(request: Request, node: str, direction: str = "downstream"):
    g = _state(request)["graph"]
    if node not in {n["id"] for n in g["nodes"]}:
        raise HTTPException(404, f"Noeud inconnu : {node}")
    if direction not in {"downstream", "upstream"}:
        raise HTTPException(422, "direction doit etre downstream ou upstream")
    return {"node": node, "direction": direction, "nodes": lin.traverse(g, node, direction)}


def load_state(catalog_path: Path, lineage_path: Path) -> Optional[Dict[str, Any]]:
    """Charge catalog.json et lineage.json s'ils existent, sinon None (l'API reste utilisable sans)."""
    if not catalog_path.exists():
        return None
    graph = json.loads(lineage_path.read_text(encoding="utf-8")) if lineage_path.exists() else {"nodes": [], "edges": []}
    return {"catalog": cat.load_catalog(catalog_path), "graph": graph, "path": catalog_path}
