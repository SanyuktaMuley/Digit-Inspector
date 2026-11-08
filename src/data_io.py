"""Strict test-file contract used identically in the notebook's verification and app."""
from __future__ import annotations

import numpy as np
import pandas as pd
from sklearn.datasets import load_digits

from .experiment import CLASSES, PIXELS, TARGET


def canonical_dataset() -> pd.DataFrame:
    """Create the CSV from scikit-learn's copy of the UCI handwritten-digits test subset."""
    digits = load_digits()
    frame = pd.DataFrame(digits.data, columns=PIXELS)
    frame[TARGET] = digits.target.astype(int)
    return frame


def validate_labeled_csv(frame: pd.DataFrame, max_rows: int = 6000):
    """Return (X, y); accept reordered columns / extras, reject malformed cases.

    A dataset with labels is mandatory to calculate the assignment's metrics.
    Only bundled trusted local model files are ever deserialized.
    """
    if not isinstance(frame, pd.DataFrame):
        raise ValueError("Expected a CSV table.")
    if frame.empty:
        raise ValueError("The CSV is empty; upload at least one labelled row.")
    if len(frame) > max_rows:
        raise ValueError(f"Upload at most {max_rows:,} labelled rows on the free-tier demo.")
    if not frame.columns.is_unique:
        raise ValueError("Duplicate CSV column names are not permitted.")
    missing = sorted(set(PIXELS + [TARGET]) - set(frame.columns))
    if missing:
        preview = ", ".join(missing[:8])
        suffix = f" (and {len(missing) - 8} more)" if len(missing) > 8 else ""
        raise ValueError("Required columns missing: " + preview + suffix)

    selected = frame.loc[:, PIXELS + [TARGET]].copy()
    try:
        X = selected[PIXELS].apply(pd.to_numeric, errors="raise")
        y_numeric = pd.to_numeric(selected[TARGET], errors="raise")
    except (TypeError, ValueError) as ex:
        raise ValueError("Pixel columns and Digit labels must contain numeric values.") from ex
    if X.isna().to_numpy().any() or y_numeric.isna().any():
        raise ValueError("Missing pixel intensities / class labels are not supported.")
    if not np.isfinite(X.to_numpy(dtype=float)).all() or not np.isfinite(y_numeric.to_numpy(dtype=float)).all():
        raise ValueError("Missing, infinite or nonnumeric values are not supported.")
    arr = X.to_numpy(dtype=float)
    if np.any((arr < 0) | (arr > 16)):
        raise ValueError("Pixel intensities must lie between 0 and 16 (inclusive).")
    if not np.allclose(arr, np.rint(arr), atol=1e-10):
        raise ValueError("Pixel intensities must be integer counts from 0 to 16.")
    raw_y = y_numeric.to_numpy(dtype=float)
    if not np.all(np.isin(raw_y, CLASSES)):
        raise ValueError("Digit labels must be integers from 0 to 9.")
    return X.astype(float), pd.Series(raw_y.astype(int), index=X.index, name=TARGET)
