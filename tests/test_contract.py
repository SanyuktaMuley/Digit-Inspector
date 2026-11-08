"""Offline-positive and intentionally malformed-CSV regression cases."""
import ast
import unittest
from pathlib import Path
import numpy as np
import pandas as pd

from src.data_io import canonical_dataset, validate_labeled_csv
from src.experiment import PIXELS, TARGET
from src.metrics import evaluate
from verify_project import verify
import joblib

ROOT = Path(__file__).resolve().parents[1]


class ContractTests(unittest.TestCase):
    @classmethod
    def setUpClass(cls):
        cls.raw = pd.read_csv(ROOT / "test_data.csv")

    def test_original_public_subset(self):
        source = pd.read_csv(ROOT / "data" / "digits_uci_subset.csv")
        self.assertEqual(source.shape, (1797, 65))
        self.assertEqual(len(PIXELS), 64)
        self.assertTrue(source.equals(canonical_dataset().astype(source.dtypes.to_dict())))

    def test_same_test_set_and_metrics_after_reload(self):
        self.assertTrue(verify(verbose=False))

    def test_accepts_reordered_columns_and_extras(self):
        data = self.raw.copy()
        data['Unneeded'] = "ignored"
        data = data[data.columns[::-1]]
        X, y = validate_labeled_csv(data)
        self.assertEqual(list(X.columns), PIXELS)
        self.assertEqual(y.name, TARGET)
        self.assertEqual(len(y), len(data))

    def test_rejects_missing_feature(self):
        with self.assertRaisesRegex(ValueError, "Required columns missing"):
            validate_labeled_csv(self.raw.drop(columns=[PIXELS[0]]))

    def test_rejects_bad_target(self):
        data = self.raw.copy()
        data.loc[data.index[0], TARGET] = 11
        with self.assertRaisesRegex(ValueError, "Digit labels"):
            validate_labeled_csv(data)

    def test_rejects_out_of_range_pixels(self):
        data = self.raw.copy()
        data.loc[data.index[0], PIXELS[10]] = 42
        with self.assertRaisesRegex(ValueError, "Pixel intensities"):
            validate_labeled_csv(data)

    def test_rejects_nulls(self):
        data = self.raw.copy()
        data.loc[data.index[2], PIXELS[8]] = np.nan
        with self.assertRaisesRegex(ValueError, "Missing pixel"):
            validate_labeled_csv(data)

    def test_auc_na_if_a_class_missing(self):
        only_digit_zero = self.raw[self.raw[TARGET] == 0]
        X,y = validate_labeled_csv(only_digit_zero)
        model = joblib.load(ROOT / "model" / "svm.joblib")
        scores, cm, _, _ = evaluate(model, X, y)
        self.assertTrue(np.isnan(scores["AUC"]))
        self.assertEqual(cm.sum(), len(X))

    def test_app_has_no_fit_calls(self):
        # Static contract only; deployment UI still needs testing via Community Cloud.
        tree = ast.parse((ROOT / "app.py").read_text())
        calls = [node.func.attr for node in ast.walk(tree)
                 if isinstance(node, ast.Call) and isinstance(node.func, ast.Attribute)]
        self.assertNotIn("fit", calls)
        self.assertNotIn("fit_transform", calls)

    def test_all_five_model_files_exist(self):
        from src.experiment import MODEL_FILES
        self.assertEqual(len(MODEL_FILES), 5)
        for name in MODEL_FILES.values():
            self.assertTrue((ROOT / 'model' / name).is_file())


if __name__ == "__main__":
    unittest.main()
