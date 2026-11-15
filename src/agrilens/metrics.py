from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    confusion_matrix,
    balanced_accuracy_score,
)

import matplotlib.pyplot as plt


def evaluate_classification(y_true, y_pred):
    """
    Compute classification metrics.
    """

    return {
        "accuracy": accuracy_score(y_true, y_pred),
        "precision_macro": precision_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "recall_macro": recall_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "f1_macro": f1_score(
            y_true,
            y_pred,
            average="macro",
            zero_division=0,
        ),
        "balanced_accuracy": balanced_accuracy_score(
    y_true,
    y_pred
),
    }
def print_metrics(scores):
    print(f"Accuracy         : {scores['accuracy']:.4f}")
    print(f"Precision Macro  : {scores['precision_macro']:.4f}")
    print(f"Recall Macro     : {scores['recall_macro']:.4f}")
    print(f"F1 Macro         : {scores['f1_macro']:.4f}")
    print(
    f"Balanced Accuracy : "
    f"{scores['balanced_accuracy']:.4f}"
)
# matrice de confusion
def plot_confusion_matrix(y_true, y_pred):
    cm = confusion_matrix(
        y_true,
        y_pred
    )

    plt.figure(figsize=(8, 6))

    plt.imshow(
        cm,
        cmap="Blues"
    )

    plt.colorbar()

    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.show()

    return cm

