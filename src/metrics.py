"""Single agreed metric definition for script, notebook and Streamlit app."""
from __future__ import annotations

import numpy as np
from sklearn.metrics import (
    accuracy_score, classification_report, confusion_matrix, f1_score,
    matthews_corrcoef, precision_score, recall_score, roc_auc_score
)
from .experiment import CLASSES

METRICS = ["Accuracy", "AUC", "Precision", "Recall", "F1", "MCC"]


def evaluate(model, X, y):
    """Ten-class macro precision/recall/F1; OvR macro AUC; multiclass MCC.

    AUC requires at least two classes and all ten classes with non-zero support for
    this ten-way OvR macro convention. Subset uploads correctly return NaN for AUC.
    """
    y_true = np.asarray(y, dtype=int)
    predictions = np.asarray(model.predict(X), dtype=int)
    trained_labels = np.asarray(model.classes_, dtype=int)
    if not np.array_equal(trained_labels, np.asarray(CLASSES)):
        raise ValueError("Saved classifier's class mapping differs from this project.")
    prob = np.asarray(model.predict_proba(X), dtype=float)
    if prob.shape != (len(y_true), len(CLASSES)):
        raise ValueError("Invalid class probability matrix shape.")
    if not np.all(np.isfinite(prob)):
        raise ValueError("A saved classifier returned non-finite probabilities.")
    if np.all(np.isin(CLASSES, np.unique(y_true))):
        auc = float(roc_auc_score(
            y_true, prob, labels=CLASSES, average="macro", multi_class="ovr"
        ))
    else:
        auc = float("nan")
    scores = {
        "Accuracy": float(accuracy_score(y_true, predictions)),
        "AUC": auc,
        "Precision": float(precision_score(y_true, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "Recall": float(recall_score(y_true, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "F1": float(f1_score(y_true, predictions, labels=CLASSES, average="macro", zero_division=0)),
        "MCC": float(matthews_corrcoef(y_true, predictions)),
    }
    return scores, confusion_matrix(y_true, predictions, labels=CLASSES), predictions, prob


def per_class_report(y, predictions):
    return classification_report(y, predictions, labels=CLASSES,
                                target_names=[str(i) for i in CLASSES],
                                output_dict=True, zero_division=0)


def winner_by_mcc(results_frame):
    """Explicit, reproducible descriptive winner rule; not a tuning mechanism."""
    return results_frame.sort_values(["MCC", "F1"], ascending=[False, False]).iloc[0]["ML Model Name"]
