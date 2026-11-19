"""lineage.py - E5 : lineage de donnees (table -> table, colonne -> colonne).

Deux sources d'aretes, TOUJOURS distinguees dans le graphe :
  - "sql_parsed" : detectee automatiquement dans une requete SQL avec sqlglot (fiable, verifiable) ;
  - "declared"   : declaree a la main dans la config (ex. etapes pandas du pipeline Bronze -> Silver -> Gold).
Le lineage des etapes pandas/PySpark n'est PAS detecte automatiquement : il est declare.
"""
from __future__ import annotations

from collections import defaultdict, deque
from pathlib import Path
from typing import Any

import sqlglot
from sqlglot import exp
from sqlglot.lineage import lineage as sg_lineage


def sql_tables(sql: str, dialect: str | None = None) -> dict[str, Any]:
    """Tables lues et table(s) ecrite(s) par une instruction SQL."""
    tree = sqlglot.parse_one(sql, dialect=dialect)
    cte_names = {c.alias_or_name for c in tree.find_all(exp.CTE)}
    target = None
    if isinstance(tree, exp.Create) and isinstance(tree.this, (exp.Table, exp.Schema)):
        t = tree.this if isinstance(tree.this, exp.Table) else tree.this.this
        target = t.name
    elif isinstance(tree, exp.Insert):
        t = tree.this if isinstance(tree.this, exp.Table) else tree.this.this
        target = t.name
    reads = []
    for t in tree.find_all(exp.Table):
        if t.name and t.name != target and t.name not in cte_names and t.name not in reads:
            reads.append(t.name)
    return {"reads": reads, "target": target}


def _select_of(tree: exp.Expression) -> exp.Expression:
    if isinstance(tree, (exp.Create, exp.Insert)):
        return tree.expression
    return tree


def column_lineage(sql: str, schema: dict[str, dict[str, str]] | None = None,
                   dialect: str | None = None) -> list[dict[str, Any]]:
    """Pour chaque colonne de sortie : de quelles colonnes sources elle provient.

    `schema` = {table: {colonne: type}} : indispensable pour resoudre `SELECT *` et les colonnes ambigues.
    Convention : sqlglot normalise les identifiants non quotes en minuscules ; les sources sont donc
    renvoyees en minuscules (nommer les tables du projet en minuscules, ce qui est deja le cas).
    """
    tree = sqlglot.parse_one(sql, dialect=dialect)
    select = _select_of(tree)
    select_sql = select.sql(dialect=dialect)
    out_cols = [c for c in select.named_selects]
    result = []
    for col in out_cols:
        node = sg_lineage(col, select_sql, schema=schema, dialect=dialect)
        sources = []
        for leaf in node.walk():
            if not leaf.downstream:                     # feuille = colonne d'une table source
                tbl = leaf.source.name if isinstance(leaf.source, exp.Table) else None
                name = leaf.name.split(".")[-1]
                if tbl:
                    sources.append(f"{tbl}.{name}")
        result.append({"column": col, "sources": sorted(set(sources)),
                       "expression": node.expression.sql(dialect=dialect)})
    return result


def build_graph(declared: list[dict[str, Any]] | None = None,
                sql_files: list[str | Path] | None = None,
                schema: dict[str, dict[str, str]] | None = None) -> dict[str, Any]:
    """Graphe {nodes, edges}. Chaque arete porte `kind` (sql_parsed | declared) et sa preuve."""
    nodes: dict[str, dict[str, Any]] = {}
    edges: list[dict[str, Any]] = []

    def add_node(n: str):
        nodes.setdefault(n, {"id": n})

    for e in declared or []:
        add_node(e["from"]); add_node(e["to"])
        edges.append({"from": e["from"], "to": e["to"], "kind": "declared",
                      "step": e.get("step"), "evidence": e.get("evidence")})

    for f in sql_files or []:
        sql = Path(f).read_text(encoding="utf-8")
        info = sql_tables(sql)
        if not info["target"]:
            continue
        cols = column_lineage(sql, schema=schema)
        add_node(info["target"])
        for src in info["reads"]:
            add_node(src)
            edges.append({"from": src, "to": info["target"], "kind": "sql_parsed",
                          "step": Path(f).name, "evidence": None,
                          "columns": [c for c in cols if any(s.lower().startswith(src.lower() + ".") for s in c["sources"])]})
    return {"nodes": sorted(nodes.values(), key=lambda n: n["id"]), "edges": edges}


def _adj(graph: dict[str, Any], reverse: bool) -> dict[str, list[str]]:
    adj = defaultdict(list)
    for e in graph["edges"]:
        a, b = (e["to"], e["from"]) if reverse else (e["from"], e["to"])
        adj[a].append(b)
    return adj


def traverse(graph: dict[str, Any], node: str, direction: str = "downstream") -> list[str]:
    """Analyse d'impact : tout ce qui depend de `node` (downstream) ou dont `node` depend (upstream)."""
    if direction not in {"downstream", "upstream"}:
        raise ValueError("direction doit etre 'downstream' ou 'upstream'")
    adj = _adj(graph, reverse=(direction == "upstream"))
    seen, q, out = {node}, deque([node]), []
    while q:
        cur = q.popleft()
        for nxt in adj.get(cur, []):
            if nxt not in seen:
                seen.add(nxt); out.append(nxt); q.append(nxt)
    return out


def to_mermaid(graph: dict[str, Any]) -> str:
    """Export Mermaid : fleche pleine = detectee en SQL, pointillee = declaree."""
    safe = lambda s: "".join(ch if ch.isalnum() else "_" for ch in s)
    lines = ["graph LR"]
    for n in graph["nodes"]:
        lines.append(f'  {safe(n["id"])}["{n["id"]}"]')
    for e in graph["edges"]:
        arrow = "-->" if e["kind"] == "sql_parsed" else "-.->"
        lines.append(f'  {safe(e["from"])} {arrow} {safe(e["to"])}')
    return "\n".join(lines)
