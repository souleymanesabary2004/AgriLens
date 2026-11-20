"""
eval_rag.py - E4 : evalue la recherche sur 07_rag_genai/eval_questions.json et choisit le seuil de refus.

Regle (fixee AVANT de regarder les resultats, comme pour l'abstention du modele d'image) :
  - le seuil est choisi sur le sous-ensemble "dev" : le plus HAUT seuil qui verifie TROIS conditions :
      (1) au moins 90 % des questions du corpus repondues (couverture),
      (2) le bon document dans le top-3 pour au moins 90 % d'entre elles (qualite),
      (3) au moins 90 % des questions hors sujet refusees (garde-fou) ;
  - il est ensuite mesure UNE fois sur le sous-ensemble "test".
Limite assumee : questions ecrites par nous, 30 au total : c'est un controle de sante, pas un benchmark.
Usage : python scripts/eval_rag.py [--write-threshold]
"""
import argparse
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT / "src"))
from agrilens.rag import Retriever  # noqa: E402


def evaluate(r: Retriever, qs, threshold):
    r.threshold = threshold
    ins = [q for q in qs if q["attendu"]]
    outs = [q for q in qs if not q["attendu"]]
    hit1 = hit3 = answered_in = 0
    misses = []
    for q in ins:
        res = r.search(q["q"], top_k=3)
        ids = [x["doc_id"] for x in res["results"]]
        answered_in += res["answered"]
        h1 = bool(ids) and ids[0] in q["attendu"]
        h3 = any(i in q["attendu"] for i in ids)
        hit1 += h1; hit3 += h3
        if not h3:
            misses.append((q["id"], q["q"], res["best_score"], ids[:1]))
    refused = 0
    wrong = []
    for q in outs:
        res = r.search(q["q"], top_k=3)
        if not res["answered"]:
            refused += 1
        else:
            wrong.append((q["id"], q["q"], res["best_score"], res["results"][0]["doc_id"]))
    n_in, n_out = max(len(ins), 1), max(len(outs), 1)
    return dict(threshold=threshold, n_in=len(ins), n_out=len(outs), recall_at_1=hit1 / n_in, recall_at_3=hit3 / n_in,
                answered_in_scope=answered_in / n_in, refused_out_of_scope=refused / n_out,
                misses=misses, out_of_scope_wrongly_answered=wrong)


if __name__ == "__main__":
    ap = argparse.ArgumentParser()
    ap.add_argument("--write-threshold", action="store_true")
    a = ap.parse_args()
    qs = json.loads((ROOT / "07_rag_genai/eval_questions.json").read_text(encoding="utf-8"))
    r = Retriever(ROOT / "07_rag_genai/corpus")
    dev = [q for q in qs if q["split"] == "dev"]
    test = [q for q in qs if q["split"] == "test"]
    grid = [round(x, 2) for x in [i * 0.01 for i in range(0, 61)]]
    best = None
    print("seuil | dev: rappel@3  repondues(corpus)  refus(hors-corpus)")
    for t in grid:
        e = evaluate(r, dev, t)
        ok = e["answered_in_scope"] >= 0.90 and e["recall_at_3"] >= 0.90 and e["refused_out_of_scope"] >= 0.90
        if t % 0.05 < 1e-9 or ok and (best is None or t > best):
            print(f"{t:5.2f} | {e['recall_at_3']:.2f}  {e['answered_in_scope']:.2f}  {e['refused_out_of_scope']:.2f}  {'OK' if ok else ''}")
        if ok:
            best = t
    if best is None:
        sys.exit("Aucun seuil ne satisfait la regle : corpus ou questions a revoir.")
    print(f"\nSeuil retenu sur dev : {best}")
    tr = evaluate(r, test, best)
    print("\n=== TEST (une seule mesure) ===")
    for k in ["n_in", "n_out", "recall_at_1", "recall_at_3", "answered_in_scope", "refused_out_of_scope"]:
        print(f"{k}: {tr[k]:.3f}" if isinstance(tr[k], float) else f"{k}: {tr[k]}")
    print("Rates (corpus, bon doc absent du top-3):", tr["misses"])
    print("Hors-corpus repondus a tort:", tr["out_of_scope_wrongly_answered"])
    if a.write_threshold:
        out = ROOT / "07_rag_genai/rag_threshold.json"
        out.write_text(json.dumps({"threshold": best, "rule": "plus haut seuil sur dev avec >=90% repondu, recall@3 >=90% et >=90% des hors-sujet refuses",
                                   "test": {k: v for k, v in tr.items() if k not in ("misses", "out_of_scope_wrongly_answered")}},
                                  indent=2, ensure_ascii=False), encoding="utf-8")
        print("ecrit:", out)
