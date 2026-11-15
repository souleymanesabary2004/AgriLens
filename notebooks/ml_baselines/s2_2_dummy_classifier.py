"""
S2.2 - DummyClassifier Baseline

Objectif :
Construire le baseline officiel du projet AgriLens
à l'aide de DummyClassifier.
"""

from agrilens.data import load_manifest
from agrilens.metrics import (
    evaluate_classification,
    print_metrics,
)

from sklearn.dummy import DummyClassifier

import numpy as np
import mlflow


mlflow.set_tracking_uri(
    "sqlite:///10_mlops/mlflow.db"
)

mlflow.set_experiment(
    "agrilens-s2-final"
)


print("MLflow configuré.")


# Chargement des données

train_df = load_manifest("train")
val_df = load_manifest("validation")

print(f"Train samples      : {len(train_df)}")
print(f"Validation samples : {len(val_df)}")


# Labels

y_train = train_df["class_id"]
y_val = val_df["class_id"]

print(f"Train labels      : {len(y_train)}")
print(f"Validation labels : {len(y_val)}")


# Features factices

X_train = np.zeros((len(train_df), 1))
X_val = np.zeros((len(val_df), 1))

print(f"X_train shape : {X_train.shape}")
print(f"X_val shape   : {X_val.shape}")


# DummyClassifier : most_frequent

dummy = DummyClassifier(
    strategy="most_frequent"
)

print("DummyClassifier créé.")

dummy.fit(
    X_train,
    y_train
)

print("DummyClassifier entraîné.")

print(
    f"Classe majoritaire mémorisée : "
    f"{dummy.classes_[dummy.class_prior_.argmax()]}"
)


# Prédictions

y_pred = dummy.predict(X_val)

print("Premières prédictions :")
print(y_pred[:20])


# Évaluation

scores = evaluate_classification(
    y_val,
    y_pred
)

print()
print("DummyClassifier Results")
print_metrics(scores)

print()
print()


# DummyClassifier : stratified

dummy_stratified = DummyClassifier(
    strategy="stratified",
    random_state=2026
)

dummy_stratified.fit(
    X_train,
    y_train
)

y_pred_stratified = dummy_stratified.predict(
    X_val
)

print("Premières prédictions stratified :")
print(y_pred_stratified[:20])


# Évaluation

scores_stratified = evaluate_classification(
    y_val,
    y_pred_stratified
)

print()
print("DummyClassifier Stratified Results")
print_metrics(scores_stratified)


# MLflow : run most_frequent

with mlflow.start_run(
    run_name="dummy_most_frequent"
) as run:

    print(
        "Run ID :",
        run.info.run_id
    )

    mlflow.log_param(
        "strategy",
        "most_frequent"
    )

    mlflow.log_metric(
        "accuracy",
        scores["accuracy"]
    )

    mlflow.log_metric(
        "balanced_accuracy",
        scores["balanced_accuracy"]
    )

    mlflow.log_metric(
        "f1_macro",
        scores["f1_macro"]
    )
    mlflow.set_tag(
        "dataset",
        "AgriLens"
    )

    mlflow.set_tag(
        "data_version",
        "split_manifest_v2"
    )

    mlflow.set_tag(
        "manifest_sha256",
        "9b48bec835bcdd86d6ba41d39a4ed5e854856fabd3084ac5fd027032cd745730"
    )

print("Run MLflow most_frequent créé.")

with mlflow.start_run(
    run_name="dummy_stratified"
) as run:

    print(
        "Run ID :",
        run.info.run_id
    )

    mlflow.log_param(
        "strategy",
        "stratified"
    )

    mlflow.set_tag(
        "dataset",
        "AgriLens"
    )

    mlflow.set_tag(
        "data_version",
        "split_manifest_v2"
    )

    mlflow.set_tag(
        "manifest_sha256",
        "9b48bec835bcdd86d6ba41d39a4ed5e854856fabd3084ac5fd027032cd745730"
    )

    mlflow.log_metric(
        "accuracy",
        scores_stratified["accuracy"]
    )

    mlflow.log_metric(
        "balanced_accuracy",
        scores_stratified["balanced_accuracy"]
    )

    mlflow.log_metric(
        "f1_macro",
        scores_stratified["f1_macro"]
    )

print("Run MLflow stratified créé.")