# Runbook - Catalogue de donnees et lineage (AgriLens E5)

Prototype local. Destine a un Data Steward ou a un developpeur qui reprend le projet.

## Perimetre reel
| Fait | Pas fait |
|---|---|
| Metadonnees techniques / fonctionnelles / operationnelles de fichiers Parquet, CSV, Excel | Kafka, Schema Registry |
| Couverture documentaire (colonnes decrites / colonnes totales) | Ordonnancement (TWS, Airflow) |
| Lineage SQL table + colonne (sqlglot) | Lineage PySpark / pandas detecte automatiquement (il est DECLARE) |
| API REST /api/v1 : lecture, enregistrement valide de metadonnees, lineage, analyse d'impact | Authentification, base de donnees, haute disponibilite |
| 10 tests automatises (donnees synthetiques) | Execution validee sur Databricks / environnement entreprise |

## Installation
    pip install sqlglot pyarrow openpyxl

## Regenerer le catalogue (a chaque changement de donnees)
    python scripts/build_catalog.py --scan 01_data
Produit `11_catalog/catalog.json`, `lineage.json`, `lineage.mmd` (schema Mermaid).
Lecture seule sur les donnees. Un fichier illisible est liste dans `errors`, il ne bloque pas le reste.

## Documenter un jeu de donnees (role du Data Steward)
Editer `11_catalog/descriptions.json` : cle = nom du fichier sans extension ; champs `description`, `owner`,
`steward`, `columns` {colonne: texte}. Relancer le script : la couverture documentaire monte.

## Declarer une etape de lineage non-SQL
Ajouter une ligne dans `11_catalog/lineage_declared.json` : {"from", "to", "step", "evidence"}.
Ajouter une requete SQL dans `11_catalog/sql/*.sql` (CREATE TABLE x AS SELECT ...) : elle est lue automatiquement,
y compris les colonnes de sortie. Nommer tables et colonnes en minuscules.

## Lancer l'API
    uvicorn main:app --app-dir 08_api --port 8000
    http://127.0.0.1:8000/docs   (section "catalog")
Routes : GET /api/v1/catalog, /catalog/summary, /catalog/{name}, POST /catalog/datasets,
GET /api/v1/lineage (?format=mermaid), /lineage/{node}?direction=downstream|upstream.

## Depannage
- 503 sur /api/v1/... : `catalog.json` absent -> lancer `scripts/build_catalog.py`, puis redemarrer l'API.
- Fichier dans `errors` : le format ou l'encodage est illisible ; ouvrir le fichier a la main.
- Une colonne SQL ressort sans source : verifier que la table source existe dans le catalogue (le schema en est tire).
- 409 a l'enregistrement : le nom existe deja ; 422 : metadonnees invalides (colonnes vides ou dupliquees).
