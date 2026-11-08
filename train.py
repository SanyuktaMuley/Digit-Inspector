"""Rebuild trained models and all export artifacts from the bundled public data."""
from __future__ import annotations

import hashlib
import json
from pathlib import Path

import joblib
import numpy as np
import pandas as pd
import sklearn
from sklearn.model_selection import train_test_split

from src.data_io import validate_labeled_csv
from src.experiment import SEED, TEST_FRACTION, TARGET, PIXELS, MODEL_FILES, make_models
from src.metrics import METRICS, evaluate, winner_by_mcc

ROOT = Path(__file__).resolve().parent


def run():
    dataset_path = ROOT / "data" / "digits_uci_subset.csv"
    raw = pd.read_csv(dataset_path)
    X, y = validate_labeled_csv(raw)
    assert raw.shape == (1797, 65), raw.shape
    assert int(raw.isna().sum().sum()) == 0
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=TEST_FRACTION, random_state=SEED, stratify=y
    )
    (ROOT / "model").mkdir(exist_ok=True)
    (ROOT / "results").mkdir(exist_ok=True)
    rows, matrices = [], {}
    for name, model in make_models(PIXELS).items():
        model.fit(X_train, y_train)
        scores, cm, _, _ = evaluate(model, X_test, y_test)
        rows.append({"ML Model Name": name, **scores})
        matrices[name] = cm.tolist()
        joblib.dump(model, ROOT / "model" / MODEL_FILES[name], compress=3)
        print(f"Saved {name:23} accuracy={scores['Accuracy']:.4f} MCC={scores['MCC']:.4f}")

    test_data = X_test.copy()
    test_data[TARGET] = y_test.to_numpy()
    test_data.to_csv(ROOT / "test_data.csv", index=False)
    comparison = pd.DataFrame(rows)[["ML Model Name", *METRICS]]
    comparison.to_csv(ROOT / "results" / "model_metrics.csv", index=False)
    payload = {
        "dataset": "Optical Recognition of Handwritten Digits - UCI copy via scikit-learn",
        "rows_full": len(raw), "n_features": len(PIXELS),
        "train_rows": len(X_train), "test_rows": len(X_test),
        "split_seed": SEED, "test_fraction": TEST_FRACTION,
        "target": TARGET, "features": PIXELS,
        "labels": list(range(10)), "models": MODEL_FILES,
        "metric_definitions": "Macro precision/recall/F1 over ten classes; OvR macro multiclass AUC; multiclass MCC",
        "positive_class": None,
        "winner_rule": "MCC first, F1 as a tie-breaker",
        "winner": winner_by_mcc(comparison),
        "versions": {
            "python": __import__("sys").version.split()[0],
            "sklearn": sklearn.__version__, "numpy": np.__version__,
            "pandas": pd.__version__, "joblib": joblib.__version__,
        },
        "dataset_sha256": hashlib.sha256(dataset_path.read_bytes()).hexdigest(),
        "confusion_matrices": matrices,
    }
    (ROOT / "results" / "experiment.json").write_text(json.dumps(payload, indent=2), encoding="utf-8")
    print("\nSix-metric comparison (unrounded values saved in CSV):")
    print(comparison.to_string(index=False, float_format=lambda v: f"{v:.4f}"))
    print("\nMCC-selected winner:", payload["winner"])
    return comparison


if __name__ == "__main__":
    run()
