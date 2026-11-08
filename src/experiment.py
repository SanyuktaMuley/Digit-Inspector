"""Fixed, interpretable five-model experiment on the UCI-derived digit dataset."""
from __future__ import annotations

from sklearn.base import clone
from sklearn.compose import ColumnTransformer
from sklearn.linear_model import LogisticRegression
from sklearn.naive_bayes import GaussianNB
from sklearn.neighbors import KNeighborsClassifier
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

SEED = 37
TEST_FRACTION = 0.20
TARGET = "Digit"
PIXELS = [f"pixel_{row}_{col}" for row in range(8) for col in range(8)]
CLASSES = list(range(10))

MODEL_FILES = {
    "Logistic Regression": "logistic_regression.joblib",
    "Decision Tree": "decision_tree.joblib",
    "K-Nearest Neighbors": "knn.joblib",
    "Gaussian Naive Bayes": "naive_bayes.joblib",
    "Support Vector Machine": "svm.joblib",
}

MODEL_NOTES = {
    "Logistic Regression": "Linear multiclass baseline; its standardized coefficients represent a globally linear decision surface.",
    "Decision Tree": "A depth-limited series of pixel-intensity rules, readable but sensitive to local variation.",
    "K-Nearest Neighbors": "Classifies an image using nearby examples in standardized 64-pixel space.",
    "Gaussian Naive Bayes": "Fast probabilistic baseline that assumes conditional independence between pixel features.",
    "Support Vector Machine": "RBF-kernel decision surface captures nonlinear handwriting patterns; calibrated probabilities enable AUC.",
}


def make_models(feature_columns=PIXELS):
    """New independent leakage-safe preprocessing+estimator pipelines."""
    # The pixel grid is entirely numerical. ColumnTransformer ensures a stable schema
    # and learns per-feature scale only from the training split, inside each pipeline.
    transformer = ColumnTransformer(
        [("pixel_scale", StandardScaler(), list(feature_columns))], remainder="drop"
    )
    classifiers = {
        "Logistic Regression": LogisticRegression(
            C=0.7, max_iter=2500, solver="lbfgs", random_state=SEED
        ),
        "Decision Tree": DecisionTreeClassifier(
            max_depth=12, min_samples_leaf=2, random_state=SEED
        ),
        "K-Nearest Neighbors": KNeighborsClassifier(
            n_neighbors=5, weights="distance"
        ),
        "Gaussian Naive Bayes": GaussianNB(var_smoothing=1e-8),
        "Support Vector Machine": SVC(
            kernel="rbf", C=4.0, gamma="scale", probability=True, random_state=SEED
        ),
    }
    return {
        name: Pipeline([("preprocess", clone(transformer)), ("classifier", classifier)])
        for name, classifier in classifiers.items()
    }
