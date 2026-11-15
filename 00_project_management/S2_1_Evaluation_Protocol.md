====================================================
S2.1 - EVALUATION PROTOCOL
====================================================

Dataset

Train :
21117

Validation :
2640

Test :
2640

----------------------------------------------------

Metric principale :

Macro-F1

Justification :

Dataset déséquilibré
17 classes

----------------------------------------------------

Metrics secondaires :

Accuracy
Precision Macro
Recall Macro

----------------------------------------------------

Visualisation :

Confusion Matrix

Version :

Normalisée par ligne

----------------------------------------------------

Règles :

Le jeu de test est interdit
pendant le développement.

Tous les modèles devront être comparés
sur le jeu de validation.

----------------------------------------------------

Modules implémentés :

evaluate_classification()

print_metrics()

plot_confusion_matrix()

----------------------------------------------------

Statut :

VALIDÉ