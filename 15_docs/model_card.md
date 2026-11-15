# Fiche modèle – AgriLens v2 (MobileNetV2 gelé + couche linéaire)

Date : 15 novembre 2026 · Tags Git : v2.0-baselines, v2.1-threshold-frozen, v2.2-test-evaluated, v2.4-inference, v2.5-api, v2.6-e3-complete, v2.7-e4, v2.8-e4-lien
Statut : prototype de recherche et d'apprentissage. Ne pas utiliser pour une décision agronomique réelle.

## 1. Usage prévu
Classer une photo de feuille (tomate, maïs) ou de cabosse de cacao parmi 17 classes
(14 de PlantVillage, 3 de cacao), avec un score de confiance. Si la confiance est
inférieure à 0,90, le modèle s'abstient et renvoie « à vérifier par un humain ».

## 2. Hors périmètre
- Autres cultures, autres maladies, autres conditions de prise de vue.
- Diagnostic ou traitement : le modèle ne remplace pas un agronome.
- Photos de terrain pour les classes PlantVillage (photos de laboratoire).
- Images qui ne sont pas des photos de feuille ou de cabosse (documents, captures d'écran, objets) : le modèle a 17 classes fermées et ne sait pas dire « aucune des 17 ».
- Détection automatique du hors-sujet : non fournie (voir section 7 et registre R-31).

## 3. Données
- 26 397 images (26 402 d'origine, 5 retirées), 17 classes.
- Sources : PlantVillage (14 classes) et un jeu de maladies du cacao (3 classes). Licences : voir SOURCES.md.
- Nettoyage : doublons exacts (MD5), quasi-doublons (pHash), fuites train/test corrigées.
- Découpage stratifié, graine 2026 : train 21 117 / validation 2 640 / test 2 640.
  Le découpage est logique : c'est le manifeste qui assigne chaque image à un jeu, les fichiers ne sont pas déplacés
  (certains chemins contiennent « train » car ils viennent du jeu d'origine, sans rapport avec le découpage).
- Manifeste : 01_data/metadata/split_manifest_v2.csv (SHA-256 enregistré dans les résultats).

## 4. Modèle
- Backbone : MobileNetV2 pré-entraîné sur ImageNet, gelé (mode eval). Sortie : 1 280 nombres par image.
- Tête : une couche linéaire 1 280 -> 17 (21 777 paramètres), seule partie entraînée.
- Entraînement : perte cross-entropy pondérée par classe, AdamW (lr 2e-3, weight decay 1e-4),
  lots de 256, arrêt anticipé (patience 12), meilleur epoch choisi sur la validation (macro-F1).
- Standardisation des embeddings : moyenne et écart-type du train uniquement.
- Aucune feature technique (largeur, hauteur, taille de fichier) pour limiter le raccourci de source.
- Graine unique : 2026 (variabilité entre graines non mesurée).

## 5. Abstention
- Confiance = plus grande probabilité parmi les 17.
- Objectif fixé avant les résultats : précision >= 98 % sur les images traitées, couverture >= 80 %.
- Seuil choisi sur la validation : 0,90 (le plus petit seuil qui atteint l'objectif).

## 6. Résultats
Mesure unique sur le test (2 640 images), seuil figé avant mesure.

| Mesure | Validation | Test |
|---|---|---|
| macro-F1 (sans abstention) | 0,921 | 0,933 |
| Balanced accuracy | 0,922 | 0,933 |
| Accuracy | 0,946 | 0,949 |
| Couverture au seuil 0,90 | 89,3 % | 90,4 % |
| Précision sur images traitées | 98,2 % | 98,1 % |
| Erreurs évitées par l'abstention | 70 % | 66 % (90 sur 136) |

- macro-F1 par source (test) : PlantVillage 0,952 ; cacao 0,843.
- Sur le test : 254 images envoyées à un humain, 46 erreurs restantes sur 2 386 images traitées.
- Comparaison (validation) : plancher naïf 0,019 ; régression logistique 0,834 ; Random Forest 0,859 ; ce modèle 0,921.
- Le gain sur les features à la main vient du cacao (F1 0,47 -> 0,85). PlantVillage était déjà à 0,94.

Classes les plus faibles (test) :

| Classe | F1 | Rappel | Images |
|---|---|---|---|
| cacao__black_pod_rot | 0,731 | 0,681 | 94 |
| cacao__pod_borer | 0,870 | 0,909 | 11 (score instable) |
| corn__cercospora_leaf_spot_gray_leaf_spot | 0,885 | 0,885 | 52 |
| tomato__early_blight | 0,899 | 0,890 | 100 |
| tomato__target_spot | 0,901 | 0,907 | 140 |

## 6 bis. Service et assistant documentaire (E3, E4)
- API FastAPI : GET /health, POST /predict, GET /fiche, POST /ask. Le modèle est chargé une fois au démarrage.
- Contrôle d'équivalence : l'inférence redonne les mêmes prédictions que l'évaluation (30 images sur 30).
- Interface Streamlit : la photo déclenche la prédiction ; si la décision est « acceptée », la fiche de la maladie s'affiche avec ses sources ; sinon aucune fiche n'est affichée. Les classes saines reçoivent un message fixe sans source.
- Corpus : 14 fiches en français (12 maladies de la tomate et du maïs, 2 du cacao), reformulées à partir de services d'extension agricole, avec sources et date de consultation (15 novembre 2026). Aucun produit ni dosage.
- Recherche : TF-IDF par section, similarité cosinus, top 3, refus sous le seuil 0,12 ; refus explicite pour les questions de dosage et les cultures hors périmètre. Aucune génération de texte : seuls des passages cités sont renvoyés.
- Évaluation de la recherche (15 questions de test écrites par nous, mesure unique) : recall@1 80 %, recall@3 90 %, 100 % des questions du corpus répondues, 5 hors-sujet sur 5 refusés. Contrôle de bon fonctionnement, pas un benchmark.
- Tests : 48 tests automatiques (dossier 13_tests), tous réussis.

## 7. Limites connues
- black_pod_rot est reconnue seulement à 68 % : environ un tiers des cas est manqué.
- pod_borer : 11 images de test, 103 au total ; conclusion statistiquement fragile.
- Les photos PlantVillage sont prises en laboratoire ; la performance sur le terrain n'est pas mesurée.
- Raccourci de source possible : un classifieur devine la source à 99,9 % avec les features à la main ;
  le modèle peut en partie reconnaître le style du jeu de données.
- Confiance non calibrée : 0,90 ne signifie pas 90 % de chances d'avoir raison ; la précision observée
  est vérifiée empiriquement sur le test uniquement pour ce seuil (calibration : registre R-28).
- Une seule graine, un seul découpage : intervalles de confiance non calculés.
- Validation utilisée pour choisir l'epoch et le seuil : ses scores sont légèrement optimistes.
- Images hors sujet (R-31) : le modèle est un classifieur à ensemble fermé (17 classes, probabilités qui totalisent 100 %). Une image sans rapport avec les cultures peut être classée avec une confiance élevée. Constaté le 15 novembre 2026 lors d'un test de l'interface : un document photographié a été classé « cacao__black_pod_rot » avec une confiance de 0,988 et la décision « accepte ». Le seuil d'abstention filtre les cas ambigus, pas les images hors sujet. Ne soumettre que des photos de feuilles de tomate ou de maïs, ou de cabosses de cacao. Pistes (non implémentées) : détection par distance dans l'espace des embeddings (Mahalanobis aux centres de classes, seuil réglé sur la validation), classe « autre » entraînée avec des images négatives, filtre préalable « est-ce une plante ? ».
- Service : l'API vérifie le type, la taille (10 Mo), le nombre de pixels et la lisibilité de l'image, mais pas que l'image montre une plante.
- Assistant documentaire : la recherche TF-IDF ne gère que les pluriels ; une question naturelle peut ramener la bonne fiche en 2e ou 3e position (R-41). Les sources du corpus sont surtout américaines et non relues par un agronome ; les conditions de réutilisation des pages restent à vérifier (R-42).

## 8. Reproductibilité
- scripts : build_features.py, train_baselines.py, extract_embeddings.py, train_head.py, evaluate_test.py, check_inference.py, eval_rag.py, check_class_mapping.py
- Tests : python -m pytest -q (attendu : 48 passed)
- Corpus : 07_rag_genai/corpus ; questions d'évaluation : 07_rag_genai/eval_questions.json ; seuil : 07_rag_genai/rag_threshold.json
- Lancement : uvicorn main:app --app-dir 08_api --port 8000, puis streamlit run 09_ui/app.py
- Suivi MLflow : 10_mlops/mlflow.db (expérience agrilens-s2-final)
- Poids : models/head_mobilenetv2_v1.pt (SHA-256 dans abstention_threshold.json)
- Environnement : torch 2.15.0 (CPU), torchvision 0.29.2, Python/conda « agrilens ».
  environment.yml à figer (registre R-19).
