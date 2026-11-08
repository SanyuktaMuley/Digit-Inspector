"""INK / Digit Inspector: inference-only, five-model multiclass evaluation dashboard.

Deployment: Streamlit Community Cloud -> main -> app.py; Python 3.13.
The app only reads trusted .joblib files committed with this project. No training code
is imported or executed on uploaded data.
"""
from __future__ import annotations

import json
from pathlib import Path

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import streamlit as st
from sklearn.metrics import roc_curve, auc

from src.data_io import validate_labeled_csv
from src.experiment import CLASSES, MODEL_NOTES, PIXELS, TARGET
from src.metrics import METRICS, evaluate, per_class_report, winner_by_mcc

ROOT = Path(__file__).resolve().parent

st.set_page_config(page_title="INK | Digit Inspector", page_icon="✍️", layout="wide", initial_sidebar_state="expanded")

st.markdown("""
<style>
.block-container {max-width:1210px; padding-top:1.4rem; padding-bottom:3rem}
h1, h2, h3 {letter-spacing:-.025em}
div[data-testid="stMetric"] {background:#F3F8FB; border:1px solid #DBE8EB;
    border-radius:13px; padding:14px 16px; min-height:102px}
div[data-testid="stMetricLabel"] {color:#476279}
.hero {background:linear-gradient(110deg,#13263e 0%,#145e69 100%); color:white;
    border-radius:19px; padding:27px 32px; margin-bottom:20px}
.hero h1 {color:white; margin:0; font-size:2.1rem}
.hero p {color:#deecf1; margin:.55rem 0 0; font-size:1rem}
.tag {display:inline-block; color:#a5ede5; font-size:.76rem; font-weight:750;
    text-transform:uppercase; letter-spacing:.12em; margin-bottom:10px}
.note {background:#f7fafb; border-left:4px solid #047b83; padding:12px 18px;
    border-radius:5px; margin:14px 0; color:#3f596a}
</style>
""", unsafe_allow_html=True)

@st.cache_data(show_spinner=False)
def read_metadata():
    return json.loads((ROOT / "results" / "experiment.json").read_text(encoding="utf-8"))


@st.cache_data(show_spinner=False)
def read_example():
    return pd.read_csv(ROOT / "test_data.csv")


@st.cache_resource(show_spinner="Opening trained models...")
def open_models(files):
    # Whitelisted model locations; the user cannot supply a pickle or select a path.
    return {name: joblib.load(ROOT / "model" / file) for name, file in files}


def plot_confusion(matrix, normalize=False):
    fig, ax = plt.subplots(figsize=(7.2, 5.7))
    matrix = np.asarray(matrix)
    if normalize:
        sums = matrix.sum(axis=1, keepdims=True)
        values = np.divide(matrix, sums, out=np.zeros_like(matrix, dtype=float), where=sums!=0) * 100
        fmt = lambda val: f"{val:.0f}%"
        label = "Percent within true digit"
    else:
        values = matrix
        fmt = lambda val: str(int(val))
        label = "Number of samples"
    image = ax.imshow(values, cmap="YlGnBu", aspect="auto", vmin=0)
    ax.set_xticks(CLASSES, labels=CLASSES)
    ax.set_yticks(CLASSES, labels=CLASSES)
    ax.set_xlabel("Predicted digit")
    ax.set_ylabel("True digit")
    ax.set_title("Confusion matrix", weight="bold", pad=15)
    cutoff = (values.max() * 0.55) if values.size else 0
    for r in CLASSES:
        for c in CLASSES:
            val = values[r, c]
            ax.text(c, r, fmt(val), ha="center", va="center", fontsize=8.5,
                    color="white" if val > cutoff else "#17334a")
    fig.colorbar(image, ax=ax, shrink=.83, label=label)
    fig.tight_layout()
    return fig


def plot_digit(row, title="Original 8 x 8 bitmap"):
    arr = row.loc[PIXELS].to_numpy(dtype=float).reshape(8, 8)
    fig, ax = plt.subplots(figsize=(3.2, 3.2))
    ax.imshow(arr, cmap="binary", vmin=0, vmax=16, interpolation="nearest")
    ax.set_xticks([]); ax.set_yticks([])
    ax.set_title(title, fontsize=12, pad=12, weight="bold")
    fig.tight_layout()
    return fig


meta = read_metadata()
model_names = list(meta["models"])
file_manifest = tuple(meta["models"].items())
models = open_models(file_manifest)

st.markdown("""
<div class="hero"><span class="tag">ML ASSIGNMENT 02 · MODEL EXPLORER</span>
<h1>INK / Digit Inspector</h1>
<p>Five trained classifiers. One held-out dataset. Ten handwriting classes.</p>
</div>
""", unsafe_allow_html=True)

with st.sidebar:
    st.header("Evaluation controls")
    selected_model = st.selectbox("Saved classifier", model_names, index=0)
    st.caption(MODEL_NOTES[selected_model])
    st.divider()
    st.subheader("Test dataset")
    upload = st.file_uploader("Upload a labelled test CSV", type=["csv"], help="The file must contain pixel_0_0 ... pixel_7_7 and Digit.")
    demo = st.checkbox("Use bundled held-out test CSV", value=True, help="The uploaded file takes priority when one is provided.")
    st.download_button("Download sample test_data.csv", data=(ROOT / "test_data.csv").read_bytes(),
                       file_name="test_data.csv", mime="text/csv", use_container_width=True)
    st.caption(f"Original source: UCI (via scikit-learn). {meta['rows_full']:,} images · {meta['n_features']} features.")
    st.caption(f"Training: {meta['train_rows']:,} | held-out test: {meta['test_rows']:,}")
    st.divider()
    st.caption("The models are already trained. Uploaded test data is used for evaluation only; no retraining occurs.")

with st.expander("CSV format, labels and scoring definitions", expanded=False):
    st.markdown("A valid CSV needs **64 numeric pixel columns** (`pixel_0_0` through `pixel_7_7`) plus an integer **`Digit`** column (0–9). Intensities are integers from 0 to 16. Extra columns and any column order are accepted.")
    st.code(",".join(PIXELS + [TARGET]), language="text")
    st.markdown("**Precision, Recall and F1**: macro averaged over all ten digits. **AUC**: macro one-vs-rest, from class probabilities (N/A if any digit is absent). **MCC**: multiclass Matthews correlation coefficient. Accuracy is the overall correct-prediction proportion.")

if upload is not None:
    if len(upload.getvalue()) > 6_000_000:
        st.error("Please upload a CSV smaller than 6 MB.")
        st.stop()
    try:
        raw = pd.read_csv(upload)
    except (ValueError, pd.errors.ParserError, UnicodeDecodeError) as ex:
        st.error(f"Could not read the CSV: {ex}")
        st.stop()
    source_name = upload.name
elif demo:
    raw = read_example()
    source_name = "Bundled held-out test_data.csv"
else:
    st.info("Upload your labelled test_data.csv using the sidebar. You can also tick the bundled test CSV to explore the application immediately.")
    st.stop()

try:
    X_test, y_test = validate_labeled_csv(raw)
except (ValueError, TypeError) as ex:
    st.error(f"CSV validation failed: {ex}")
    st.stop()

st.markdown(f"**Evaluation dataset:** `{source_name}` · **{len(X_test):,} labelled rows** · 64 predictors · {y_test.nunique()}/10 true digit classes")
if y_test.nunique() != 10:
    st.warning("This upload does not include all ten digit classes. Macro OvR AUC is undefined under the fixed ten-class definition and will display N/A. Other metrics still use all ten classes.")

results = {}
for name, estimator in models.items():
    scores, cm, predictions, prob = evaluate(estimator, X_test, y_test)
    results[name] = (scores, cm, predictions, prob)

selected_scores, selected_cm, selected_predictions, selected_probs = results[selected_model]
comparison = pd.DataFrame([{"ML Model Name": key, **result[0]} for key, result in results.items()])

first, second, third, fourth = st.tabs(["01 · Evaluation", "02 · Compare models", "03 · Digit explorer", "04 · Data & export"])

with first:
    st.subheader(selected_model)
    st.caption("Six required evaluation scores, calculated from the current labelled test CSV.")
    for offset in (0, 3):
        columns = st.columns(3)
        for i, column in enumerate(columns):
            label = METRICS[offset + i]
            val = selected_scores[label]
            column.metric(label, f"{val:.4f}" if np.isfinite(val) else "N/A")
    st.markdown("<div class='note'>Every value above is computed from saved model predictions. Model selection does not refit or retune the classifier.</div>", unsafe_allow_html=True)
    left, right = st.columns([1.07, .93], gap="large")
    with left:
        normalized = st.toggle("Show row-normalized percentages", value=False)
        fig = plot_confusion(selected_cm, normalize=normalized)
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)
    with right:
        st.markdown("**Detailed classification report**")
        report = pd.DataFrame(per_class_report(y_test, selected_predictions)).T.drop(index="accuracy", errors="ignore")
        st.dataframe(report.style.format({"precision":"{:.3f}", "recall":"{:.3f}", "f1-score":"{:.3f}", "support":"{:.0f}"}),
                     use_container_width=True, height=425)
        st.caption("Each digit's support is its actual number of examples. Macro average gives equal weight to each digit.")
    cm_errors = int(len(y_test) - np.trace(selected_cm))
    st.caption(f"Total errors for {selected_model}: {cm_errors} out of {len(y_test)} examples.")

with second:
    st.subheader("All five models · same test rows")
    pretty = comparison.set_index("ML Model Name")[METRICS]
    st.dataframe(pretty.style.format("{:.4f}", na_rep="N/A").highlight_max(axis=0, color="#D6F0ED"),
                 use_container_width=True)
    st.markdown(f"**Highest MCC on these rows:** {winner_by_mcc(comparison)}")
    st.caption("Selection rule: largest MCC, then F1 for ties. The best model may vary across different test files and scoring priorities.")
    chosen_metric = st.selectbox("Comparison chart metric", METRICS, index=5)
    st.bar_chart(pretty[chosen_metric].sort_values(ascending=False), horizontal=True)
    st.markdown("**Across-class ROC inspection**")
    class_for_roc = st.selectbox("Digit for one-vs-rest ROC", CLASSES, index=3)
    binary_truth = (y_test.to_numpy() == class_for_roc).astype(int)
    if len(np.unique(binary_truth)) != 2:
        st.info("ROC is undefined because this CSV does not contain both positive and negative examples for the selected digit.")
    else:
        fig, ax = plt.subplots(figsize=(7.4, 3.5))
        for name, (_, _, _, probabilities) in results.items():
            model_class_index = list(models[name].classes_).index(class_for_roc)
            false_pos, true_pos, _ = roc_curve(binary_truth, probabilities[:, model_class_index])
            ax.plot(false_pos, true_pos, label=f"{name} (AUC {auc(false_pos, true_pos):.3f})", lw=1.7)
        ax.plot([0, 1], [0, 1], "--", color="#a2a9ae", lw=1)
        ax.set(xlabel="False positive rate", ylabel="True positive rate", xlim=(0, 1), ylim=(0, 1))
        ax.set_title(f"Digit {class_for_roc} versus other digits")
        ax.legend(fontsize=8, loc="lower right", frameon=False)
        fig.tight_layout()
        st.pyplot(fig, use_container_width=True)
        plt.close(fig)

with third:
    st.subheader("Explore individual handwritten digits")
    st.caption("Each image is reconstructed from its 64 input pixel values, making the CSV features interpretable.")
    incorrect = np.flatnonzero(selected_predictions != y_test.to_numpy())
    if len(incorrect):
        show_wrong_only = st.checkbox(f"Show only the {len(incorrect)} misclassifications", value=False)
    else:
        show_wrong_only = False
        st.success("No errors on this CSV for the selected model.")
    available = incorrect if show_wrong_only else np.arange(len(X_test))
    if len(available):
        position = st.select_slider("Select example by test-row index", options=available.tolist(), value=int(available[0]))
        row = X_test.iloc[position]
        true_digit = int(y_test.iloc[position])
        predicted_digit = int(selected_predictions[position])
        model_class_index = list(models[selected_model].classes_).index(predicted_digit)
        probability = float(selected_probs[position, model_class_index])
        left, right = st.columns([.5, .5], gap="large")
        with left:
            fig = plot_digit(row, "64 original pixel features")
            st.pyplot(fig, use_container_width=False)
            plt.close(fig)
        with right:
            st.metric("True digit", str(true_digit))
            st.metric("Predicted digit", str(predicted_digit))
            st.metric("Estimated class probability", f"{probability:.1%}")
            st.success("Correct prediction") if predicted_digit == true_digit else st.warning("Misclassified example")
            st.caption("Probability is a model estimate, not a guarantee of correctness.")

with fourth:
    st.subheader("Your evaluation set")
    l, r = st.columns([1, 1], gap="large")
    with l:
        st.markdown("**True-label counts**")
        counts = y_test.value_counts().reindex(CLASSES, fill_value=0)
        st.bar_chart(counts.rename("Examples"))
    with r:
        st.markdown("**File validation summary**")
        st.write({"Rows": int(len(X_test)), "Pixel features": len(PIXELS),
                  "Present classes": int(y_test.nunique()), "Missing values": 0,
                  "Extra columns ignored": max(0, len(raw.columns) - 65)})
        st.caption("For reliable evaluation use the pre-saved held-out test data, not the training dataset.")
    predictions_df = X_test.copy()
    predictions_df[TARGET] = y_test.to_numpy()
    predictions_df["PredictedDigit"] = selected_predictions
    predictions_df["Correct"] = predictions_df[TARGET] == predictions_df["PredictedDigit"]
    predictions_df["PredictedProbability"] = selected_probs.max(axis=1)
    st.dataframe(predictions_df[[TARGET, "PredictedDigit", "Correct", "PredictedProbability"]].head(40),
                 use_container_width=True)
    st.download_button("Download predictions and labels", predictions_df.to_csv(index=False).encode("utf-8"),
                       file_name="digit_prediction_results.csv", mime="text/csv")

st.divider()
st.caption("Source: UCI Optical Recognition of Handwritten Digits (subset distributed with scikit-learn). Educational comparison only. No uploaded row is used for training.")

