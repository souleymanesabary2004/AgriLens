# AgriLens

**Assistant de diagnostic des maladies de plantes à partir d'une photo** (tomate, maïs, cacao).

Projet de bout en bout : données auditées et versionnées, modèle comparé à des références, évaluation rigoureuse, abstention (renvoi des cas incertains vers un humain), API, interface et assistant documentaire sourcé.

| | |
|---|---|
| **Statut** | Prototype local, évalué et documenté (étapes E1 à E4 terminées) |
| **Classes** | 17 (14 PlantVillage + 3 cacao) |
| **Données** | 26 397 images après audit qualité |
| **Stack** | Python, PyTorch, scikit-learn, FastAPI, Streamlit, JupyterLab, DVC, MLflow, pytest (voir [Technologies](#2-technologies)) |
| **Auteur** | Souleymane Soumahoro, élève ingénieur Data Science & IA, EFREI Paris (M1) |

## Sommaire

1. [Principes du projet](#1-principes-du-projet)
2. [Technologies](#2-technologies)
3. [Résultats](#3-résultats)
4. [Données et pipeline](#4-données-et-pipeline)
5. [Modèle et abstention](#5-modèle-et-abstention)
6. [Service et assistant documentaire](#6-service-et-assistant-documentaire)
7. [Périmètre et limites documentées](#7-périmètre-et-limites-documentées)
8. [Feuille de route](#8-feuille-de-route)
9. [Structure du dépôt](#9-structure-du-dépôt)
10. [Reproduire](#10-reproduire)
11. [Autres projets](#11-autres-projets)
12. [Sources et licences](#12-sources-et-licences)

## 1. Principes du projet

- **La donnée d'abord.** Chaque image est inventoriée, contrôlée et rattachée à un découpage reproductible avant tout entraînement.
- **Des mesures comparées.** Le modèle final est situé par rapport à des références : modèle naïf, régression logistique, Random Forest.
- **Une évaluation sans fuite.** Le jeu de test n'est mesuré qu'une fois. Les doublons entre découpages sont détectés, puis corrigés.
- **Savoir s'abstenir.** Le système répond seulement quand sa confiance dépasse un seuil choisi sur la validation. Les autres cas sont renvoyés à un humain.
- **Aucune réponse sans source.** L'assistant cite ses sources et refuse les questions hors périmètre ou de dosage.
- **Des limites écrites.** Chaque limite connue est documentée dans la [fiche modèle](15_docs/model_card.md).

## 2. Technologies

| Domaine | Outils |
|---|---|
| Langage et analyse | Python 3.11, JupyterLab et notebooks, pandas |
| Données et stockage | PyArrow (Parquet, architecture Medallion), PySpark (benchmark), DVC |
| Modèle | PyTorch (MobileNetV2), scikit-learn (régression logistique, Random Forest) |
| Assistant documentaire | Recherche TF-IDF avec citations et refus (partie recherche d'une approche RAG, sans génération) |
| Service | FastAPI (API), Streamlit (interface) |
| Qualité et suivi | pytest, Git (tags d'étape), MLflow |

## 3. Résultats

### Modèles (macro-F1 sur la validation)

| Modèle | macro-F1 |
|---|---:|
| Classe majoritaire (référence naïve) | 0,02 |
| Régression logistique | 0,834 |
| Random Forest | 0,859 |
| **MobileNetV2 gelé + tête linéaire** | **0,921** |

### Test (mesuré une seule fois)

| Mesure | Valeur |
|---|---:|
| macro-F1 | 0,933 |
| Balanced accuracy | 0,933 |
| Accuracy | 0,949 |
| F1 PlantVillage | 0,952 |
| F1 cacao | 0,843 |

### Abstention (seuil de confiance 0,90, sur le test)

| Mesure | Valeur |
|---|---:|
| Couverture (images traitées automatiquement) | 90,4 % |
| Précision sur les images acceptées | 98,07 % |
| Images renvoyées à un humain | 254 |
| Erreurs évitées | 90 sur 136 |

### Assistant documentaire (15 questions de test)

| Mesure | Valeur |
|---|---:|
| recall@1 | 80 % |
| recall@3 | 90 % |
| Refus des questions hors périmètre | 5 sur 5 |

## 4. Données et pipeline

| Étape | Réalisation |
|---|---|
| Inventaire | 26 402 images, 17 classes, 2 sources, métadonnées techniques par image (format, dimensions, mode, canaux) |
| Audit qualité | Intégrité, formats, géométrie, couleurs, luminosité, doublons exacts (MD5), quasi-doublons (pHash) |
| Nettoyage | 5 images retirées, 26 397 conservées ; indice de santé des données : 99,98 % |
| Découpage | Stratifié par classe, seed 2026 : 21 117 / 2 640 / 2 640 (manifeste `split_manifest_v2.csv`) |
| Contrôle des fuites | Détection de groupes de doublons répartis sur plusieurs découpages, corrigés dans le manifeste v2 |
| Data lake | Architecture Medallion en Parquet : Bronze (inventaire brut), Silver (nettoyé), Gold (26 397 lignes, 23 colonnes) |
| Versionnement | Git avec tags d'étape, DVC pour les données (`01_data/lake.dvc`), MLflow pour les expériences |
| Benchmark | Pandas 0,0126 s contre PySpark 8,35 s (démarrage de la session : 15 s) sur 26 000 lignes. À cette taille, Pandas est le bon outil. |

## 5. Modèle et abstention

- **Modèle :** MobileNetV2 pré-entraîné sur ImageNet, gelé, avec une tête linéaire 1280 → 17 (21 777 paramètres entraînés). Choix adapté à l'absence de GPU.
- **Métrique principale :** macro-F1, adaptée aux classes déséquilibrées (de 103 à 5 357 images par classe).
- **Abstention :** seuil de 0,90 sur la confiance maximale, choisi sur la validation comme le plus petit seuil atteignant une précision d'au moins 98 % avec une couverture d'au moins 80 %. Le jeu de test n'intervient pas dans ce choix.

## 6. Service et assistant documentaire

**API (FastAPI)**

| Route | Rôle |
|---|---|
| `GET /health` | État du service |
| `POST /predict` | Diagnostic d'une image, avec décision d'abstention |
| `GET /fiche` | Fiche sourcée associée à une classe |
| `POST /ask` | Question sur le corpus, réponse avec citations ou refus |

Les entrées sont validées (type de fichier, taille maximale de 10 MiB, 50 millions de pixels, longueur de la question) et renvoient des codes d'erreur explicites.

**Interface (Streamlit).** Analyse automatique à l'envoi d'une photo. La fiche associée s'affiche lorsque la décision est « accepté ».

**Assistant documentaire.** Recherche TF-IDF sur 14 fiches en français (2 à 3 sources par fiche, sans produit ni dose), découpées par section. Un score minimal (0,12) déclenche le refus, ainsi que des règles dédiées aux questions de dosage et aux cultures hors périmètre. L'assistant restitue des extraits sourcés et ne génère pas de texte libre.

**Qualité logicielle.** 48 tests automatisés (API, inférence, recherche documentaire), exécutés avec pytest.

## 7. Périmètre et limites documentées

| Sujet | Constat |
|---|---|
| Hors domaine | Le modèle choisit parmi 17 classes, y compris pour une image sans rapport. Un cas hors domaine a été classé avec 0,988 de confiance. |
| Calibration | Le seuil de 0,90 porte sur un softmax non calibré : il ne correspond pas à une probabilité de réussite. |
| Données | PlantVillage est photographié en laboratoire sur fond uniforme. Aucun test sur photos de terrain n'a été réalisé. |
| Classes peu représentées | `black_pod_rot` : F1 0,731, rappel 0,681. `pod_borer` : seulement 11 images de test. |
| Statistiques | Un seed, un jeu de test, pas d'intervalle de confiance. |
| Assistant | Questions de test rédigées par l'auteur, lemmatisation française simple, pas de génération de texte. |
| Licences | Celles des jeux d'images restent à vérifier (voir [`SOURCES.md`](SOURCES.md)). |

Détails et justifications : [`15_docs/model_card.md`](15_docs/model_card.md).

## 8. Feuille de route

| Priorité | Évolution |
|---|---|
| 1 | Catalogue de métadonnées généré automatiquement (techniques, fonctionnelles, opérationnelles), graphe de lignage du brut jusqu'au modèle, analyse de lignage SQL |
| 2 | Conteneurisation (Docker), intégration continue (GitHub Actions), environnement reproductible exporté |
| 3 | Orchestration des contrôles qualité et du catalogue avec Airflow |
| 4 | Génération contrôlée : modèle de langage, sortie validée par schéma, citations vérifiées, relecture humaine |
| 5 | Embeddings pour la recherche, détection hors domaine, calibration, fine-tuning, évaluation sur photos de terrain |

Les fichiers `docker-compose.yml` et `environment.yml` sont des ébauches, non validées à ce jour.

## 9. Structure du dépôt

```
01_data/            métadonnées, manifestes, correspondance des classes, pointeur DVC
02_eda/             analyse exploratoire
notebooks/          inventaire, qualité, découpage, Medallion, benchmark Spark, couche Gold
04_ml_baselines/    références classiques et rapports
05_deep_learning/   résultats MobileNetV2 et abstention
07_rag_genai/       corpus de 14 fiches, questions d'évaluation, seuil
08_api/             API FastAPI
09_ui/              interface Streamlit
13_tests/           tests pytest
15_docs/            fiche modèle
scripts/            entraînement, évaluation, contrôles
src/agrilens/       bibliothèque (données, caractéristiques, métriques, inférence, recherche)
```

## 10. Reproduire

Les images (26 397) et les poids du modèle **ne sont pas versionnés dans ce dépôt** : les données sont suivies par DVC et aucun stockage distant n'est configuré à ce jour. Le dépôt seul ne permet donc pas de réentraîner le modèle.

Ce qui s'exécute sans les données :

```
python -m pytest -q
python scripts/check_class_mapping.py
```

## 11. Autres projets

| Projet | Contenu |
|---|---|
| [phm-copilot](https://github.com/souleymanesabary2004/phm-copilot) | Maintenance prédictive sur NASA C-MAPSS (GRU PyTorch), copilote IA avec outils, citations vérifiées, refus sans preuve et validation humaine ; 29 tests, GitHub Actions et GitLab CI |
| [movie-recommendation-system](https://github.com/souleymanesabary2004/movie-recommendation-system) | Système de recommandation avec Docker, MySQL et Spark |
| [france-renewable-energy-dashboard](https://github.com/souleymanesabary2004/france-renewable-energy-dashboard) | Analyse et tableau de bord de la transition énergétique en France |

## 12. Sources et licences

Voir [`SOURCES.md`](SOURCES.md) : jeux d'images (PlantVillage et Cacao Diseases, sur Kaggle) et sources du corpus pour chaque maladie. Les licences des jeux d'images restent à vérifier avant toute redistribution.
