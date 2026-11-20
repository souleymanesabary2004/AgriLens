# Changelog

Toutes les modifications notables d'AgriLens sont notées ici.
Format : Keep a Changelog. Versions : tags git (v1.0, puis v2-p0 à v2-p8, puis v2.0.0).

## [Non publié] - v2 en cours (branche develop)

### Ajouté
- P0.1 : mesure de la machine (16 Go de RAM, CPU 4 coeurs, pas de GPU CUDA).
- P0.1b : nettoyage du disque (de 16 Go à 69 Go libres).
- P0.3 : branche main renommée depuis github-clean ; branche develop créée.
- P0.4 : CHANGELOG.
- P0.5 : gel des dependances de la v1 (requirements-v1.txt, 275 paquets) et requirements-v2.txt.
- P0.6 : ruff, pre-commit, detect-private-key, gitleaks ; 23 corrections de style, 62 tests verts.
- P0.7 : CI GitHub Actions (ruff et pytest sur Ubuntu, Python 3.11), requirements-ci.txt, marqueur pytest data.
- P0.8 : structure v2 (paquets genai, agents, vision, monitoring, scale) et .gitignore complete.
- P0.9 : modele v1 (104 Ko) versionne par exception dans .gitignore.
- P0.10 : regle du jeu de test v2 (second verrou prevu en P3.9).

### Modifié
- P0.2 : .gitignore durci (secrets, poids de modèles, données personnelles).

## [1.0] - 2026-11-19

### Ajouté
- Classification de 17 classes de maladies de feuilles (MobileNetV2 gelé + tête linéaire).
- Abstention au seuil 0,90 (couverture 90,4 %, précision 98,07 %).
- RAG v1 par TF-IDF sur 14 fiches.
- Catalogue de données (15 jeux, 2 857 colonnes) et API /api/v1.
- 62 tests pytest.
- Tag v1.0 et branche release/v1 (commit a9e35d7).
