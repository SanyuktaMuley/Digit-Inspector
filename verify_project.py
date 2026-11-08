"""Verify that the five saved models reproduce notebook/test-CSV metrics."""
from __future__ import annotations
import json
from pathlib import Path
import joblib
import numpy as np
import pandas as pd
from src.data_io import validate_labeled_csv
from src.metrics import METRICS, evaluate

ROOT = Path(__file__).resolve().parent

def verify(verbose=True):
    metadata = json.loads((ROOT / "results" / "experiment.json").read_text())
    test = pd.read_csv(ROOT / "test_data.csv")
    X, y = validate_labeled_csv(test)
    comparison = pd.read_csv(ROOT / "results" / "model_metrics.csv").set_index("ML Model Name")
    assert len(test) == metadata["test_rows"]
    for name, filename in metadata["models"].items():
        model = joblib.load(ROOT / "model" / filename)
        scores, cm, y_pred, proba = evaluate(model, X, y)
        assert cm.shape == (10, 10) and cm.sum() == len(test)
        assert proba.shape == (len(test), 10)
        assert np.allclose(proba.sum(axis=1), 1.0, atol=1e-8)
        for metric in METRICS:
            assert np.isclose(scores[metric], comparison.loc[name, metric], atol=1e-12), (name, metric)
        if verbose:
            print(f"PASS {name:24} matching all six saved metrics")
    if verbose:
        print(f"PASS: {len(test)} labelled rows; five saved pipelines; all metrics reproduced without fitting.")
    return True

if __name__ == "__main__":
    verify()
