# ADR 0001 : Modèle d'embedding du RAG v2 (BGE-M3)

- Statut : accepté, **provisoire jusqu'à la mesure de P1.11**
- Date : 24 novembre 2026
- Décideur : Souleymane Soumahoro
- Étape du projet : P1.3 (embeddings), décidé à la fin de P1.2

## 1. Contexte

AgriLens v2 contient un RAG qui retrouve des passages dans un corpus de 17 fiches (14 maladies ou ravageurs, 3 plantes saines) pour conseiller avec des sources. Le corpus est écrit en français. Les questions peuvent être posées dans une autre langue.

Mesures faites en P1.2 sur le corpus :

- 82 sections hors « Sources », de 220 à 989 caractères (moyenne 426).
- 2 sections dépassent 800 caractères.
- Après découpage : 84 chunks, de 220 à 799 caractères (moyenne 400).

La v1 utilisait `paraphrase-multilingual-MiniLM-L12-v2`. Ce modèle a une longueur maximale de 128 tokens : un chunk de 800 caractères peut être tronqué avant même d'être transformé en vecteur, donc une partie du texte ne compte pas dans la recherche.

Contraintes de la machine : 16 Go de RAM, processeur i5-10310U, pas de GPU CUDA, Python 3.11.

## 2. Critères de décision

1. Longueur de contexte compatible avec nos chunks.
2. Multilingue (français au minimum, autres langues possibles).
3. Compatible avec la recherche hybride prévue (dense + sparse + BM25 + RRF).
4. Licence utilisable en entreprise.
5. Exécutable sur CPU, sans service payant.
6. Outil reconnu par les entreprises.

## 3. Options comparées

| Option | Contexte | Multilingue | Sparse | Licence | Remarque |
|---|---|---|---|---|---|
| paraphrase-multilingual-MiniLM-L12-v2 (v1) | 128 tokens | Oui | Non | Open source | Très léger, mais tronque nos chunks |
| multilingual-e5-small | 512 tokens | Oui | Non | Open source | Léger, contexte suffisant |
| **BGE-M3** | **8192 tokens** | **Oui** | **Oui** | **MIT** | Plus lourd : environ 568 M de paramètres, environ 2 Go à télécharger |
| Qwen3-Embedding | 32 000 tokens | Oui | Non | Open source | Plus récent, plus gros, moins de recul |
| OpenAI text-embedding-3 | 8191 tokens | Oui | Non | Service payant | Données envoyées à un tiers, coût, dépendance réseau |

Les scores de benchmark publiés (blogs, classements) diffèrent selon les sources. Je les utilise comme ordre de grandeur, pas comme preuve : ils ne portent pas sur notre domaine ni sur notre corpus.

## 4. Décision

Utiliser **BGE-M3** (1024 dimensions) comme modèle d'embedding principal du RAG v2. Garder MiniLM-L12-v2 comme **baseline** de comparaison en P1.11.

Raisons :

- Contexte de 8192 tokens : aucun de nos chunks (maximum 799 caractères) n'est tronqué.
- Multilingue : adapté à un corpus français et à des questions dans d'autres langues.
- Produit des vecteurs denses et des poids sparse : cohérent avec la recherche hybride prévue.
- Licence MIT : réutilisable en entreprise.
- Fonctionne en local : pas de donnée envoyée à un service externe, pas de coût par requête.

## 5. Conséquences

Positives :

- Plus de troncature des chunks.
- Un seul modèle peut alimenter les recherches dense et sparse.

Négatives et coûts :

- Téléchargement d'environ 2 Go et temps de calcul plus long sur CPU : à mesurer en P1.3 (temps d'indexation des 84 chunks).
- La collection ChromaDB doit être créée en 1024 dimensions (P1.4) ; changer de modèle plus tard impose de réindexer.
- La sortie sparse demande une bibliothèque dédiée : vérifier l'API actuelle avant P1.3 (règle R5).

## 6. Ce qui reste à prouver

Ce choix repose sur des caractéristiques techniques, pas sur une mesure sur nos données. Il sera confirmé ou remis en cause en P1.11 :

- Comparer BGE-M3 et MiniLM-L12-v2 sur un jeu de validation v2 couvrant les 17 classes.
- Métriques : recall@5 (cible C4 ≥ 0,85) et MRR, plus le temps d'indexation et la latence d'une requête.
- Le jeu de test reste verrouillé jusqu'à P3.9 (il a déjà été consulté une fois par la v1).
- Si BGE-M3 n'apporte pas de gain mesurable face à son coût, une option plus légère sera réexaminée et ce document sera mis à jour (nouveau statut ou ADR de remplacement).

## 7. Références internes

- Découpage : `src/agrilens/genai/chunking.py`, décisions D9 à D14 du cahier d'avancement.
- Mesures de sections : `scripts/measure_sections.py`.
- Étapes suivantes : P1.3 (embeddings), P1.4 (ChromaDB, 1024 dimensions), P1.11 (évaluation).
