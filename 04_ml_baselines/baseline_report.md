# AgriLens - Baseline Report

## 1. Contexte

AgriLens est un projet de classification d'images visant à identifier différentes maladies et états sanitaires de cultures agricoles à partir de photographies de feuilles, fruits et cabosses.

Avant l'entraînement de modèles de Deep Learning, il est nécessaire d'établir plusieurs niveaux de référence afin de disposer d'un point de comparaison objectif. Ces références permettent d'évaluer la difficulté réelle du problème et de vérifier qu'un modèle plus complexe apporte effectivement une amélioration mesurable.

L'ensemble des expérimentations présentées dans ce rapport repose sur le protocole d'évaluation défini dans le document `evaluation_protocol.md`.

---

## 2. Objectifs des baselines

Les baselines ont pour objectifs :

- établir un niveau de performance minimal à dépasser ;
- comprendre le comportement du jeu de données avant l'utilisation de modèles complexes ;
- analyser l'impact du déséquilibre des classes sur les métriques ;
- comparer différentes familles d'approches (triviales, Machine Learning classique et Deep Learning) ;
- fournir des points de référence reproductibles pour les expérimentations futures.

Toutes les évaluations sont réalisées exclusivement sur l'ensemble de validation. Le jeu de test reste verrouillé et n'est jamais utilisé pour prendre des décisions de conception ou de réglage des modèles.

---

## 3. Jeu de données utilisé

Version des données :

- Dataset : AgriLens
- Manifest : split_manifest_v2.csv
- Nombre total d'images : 26 397
- Nombre de classes : 17
- Répartition :
  - Train : 21 117 images
  - Validation : 2 640 images
  - Test : 2 640 images

Le dataset présente un déséquilibre important entre certaines classes, ce qui rend l'Accuracy insuffisante comme unique critère d'évaluation.

---

## 4. Métriques retenues

### Métrique principale

**Macro-F1**

Le Macro-F1 accorde la même importance à chaque classe et constitue donc la métrique principale du projet.

### Métriques secondaires

- Balanced Accuracy
- Accuracy

Ces métriques sont utilisées pour compléter l'analyse et faciliter les comparaisons entre approches.

---

## 5. Baselines triviales (DummyClassifier)

# 3. Baselines triviales (DummyClassifier)

Afin de disposer d'un point de comparaison avant l'entraînement de modèles plus complexes, deux baselines ont été mises en place à l'aide de DummyClassifier de scikit-learn.

Le premier modèle utilise la stratégie "most_frequent". Il prédit systématiquement la classe majoritaire observée dans les données d'entraînement. Cette approche obtient une Accuracy de 20,30 %, mais un Macro-F1 de seulement 1,99 %. Ce résultat montre qu'une Accuracy relativement élevée peut être obtenue sans apprendre à reconnaître correctement la majorité des classes du dataset.

Le second modèle utilise la stratégie "stratified". Les prédictions sont générées aléatoirement selon la distribution observée des classes dans le jeu d'entraînement. L'Accuracy obtenue est plus faible (10,04 %), mais le Macro-F1 est supérieur (6,31 %), ce qui traduit une meilleure prise en compte de l'ensemble des classes.

## Résultats obtenus sur l'ensemble de validation :

### DummyClassifier (most_frequent)
- Accuracy : 0.2030
- Balanced Accuracy : 0.0588
- Macro-F1 : 0.0199

### DummyClassifier (stratified)
- Accuracy : 0.1004
- Balanced Accuracy : 0.0632
- Macro-F1 : 0.0631

Cette expérimentation confirme que l'Accuracy seule est insuffisante pour évaluer correctement AgriLens. Le fort déséquilibre du dataset favorise artificiellement les modèles qui prédisent majoritairement certaines classes. Pour cette raison, le Macro-F1 est retenu comme métrique principale du projet pour les expérimentations futures.

Les deux baselines ont été enregistrées dans MLflow afin d'assurer la traçabilité des paramètres, métriques et métadonnées associées aux données utilisées.

---

## 6. Features manuelles couleur et texture

À compléter en S2.3.

---

## 7. Régression Logistique

À compléter en S2.4.

---

## 8. Random Forest

À compléter en S2.4.

---

## 9. Comparaison globale des approches

À compléter en S2.10.

---

## 10. Conclusions

À compléter en fin de S2.