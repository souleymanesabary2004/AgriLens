# Changelog

Toutes les modifications notables d'AgriLens sont notées ici.
Format : Keep a Changelog. Versions : tags git (v1.0, puis v2-p0 à v2-p8, puis v2.0.0).

## [Non publié] - v2 en cours (branche develop)

### Ajouté
- P0.1 : mesure de la machine (16 Go de RAM, CPU 4 coeurs, pas de GPU CUDA).
- P0.1b : nettoyage du disque (de 16 Go à 69 Go libres).
- P0.3 : branche main renommée depuis github-clean ; branche develop créée.

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
