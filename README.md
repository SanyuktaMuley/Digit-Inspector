# INK / Digit Inspector — ML Assignment 2

*A reproducible ten-class classification experiment with an interactive, inference-only Streamlit model evaluator.*

> **Course:** Machine Learning, M.Tech (Artificial Intelligence & Machine Learning), BITS Pilani WILP.  
> **Student:** [YOUR NAME] · **BITS ID:** [YOUR BITS ID]  
> **Repository:** [REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL]  
> **Deployed application:** [REPLACE_WITH_YOUR_STREAMLIT_APP_URL]

## a. Problem statement

The task is to predict which of the **ten handwritten digits (0–9)** is represented by an 8 × 8 intensity grid. Five required classification algorithms—**Logistic Regression, Decision Tree, K-Nearest Neighbors, Gaussian Naive Bayes, and Support Vector Machine**—are trained on the **same data split** and evaluated using **Accuracy, AUC, Precision, Recall, F1, and Matthews Correlation Coefficient (MCC)**. The trained models are exported with their preprocessing and exposed through a Streamlit app. Users provide a **labelled test CSV**, select a saved classifier and view reproducible metrics, confusion matrices and class reports **without retraining**.

The goal is not to reach an artificially perfect accuracy; it is to compare consistent predictions, explain model trade-offs, and demonstrate the real deployment workflow from a notebook to a web application.

## b. Dataset description [1 mark]

| Attribute | Description |
|---|---|
| Original public dataset | **Optical Recognition of Handwritten Digits**, UCI Machine Learning Repository (dataset ID 80) |
| Source URLs | https://archive.ics.uci.edu/dataset/80/optical+recognition+of+handwritten+digits and https://scikit-learn.org/stable/modules/generated/sklearn.datasets.load_digits.html |
| Exact subset used | **1,797** digit images: scikit-learn's published copy of the **UCI test-set subset**, **not the entire 5,620-row UCI dataset** |
| Predictors | **64** numerical pixel-intensity counts: `pixel_0_0` to `pixel_7_7`, each in **0–16** |
| Target | **`Digit`**, categorical code represented as integers **0–9** |
| Data quality | **0 missing values**, **0 duplicate full rows** in the bundled subset |
| Class balance | Roughly **174–183 examples per digit** in the 1,797-row subset |
| Split | **1,437 training** and **360 held-out test** samples; stratified **80/20**, `random_state=37` |
| Licence | **CC BY 4.0** as recorded on the UCI dataset page |

The original UCI collection contains **5,620** rows. The specific scikit-learn **1,797-row UCI-derived test subset** is independently reproducible with `sklearn.datasets.load_digits()`. This distinction avoids incorrectly claiming to use all 5,620 original rows. For offline lab access, the precise selected dataset is included as `data/digits_uci_subset.csv` and can be reconstructed by `src.data_io.canonical_dataset()`.

**Attribution:** Alpaydin, E. and Kaynak, C. (1998), *Optical Recognition of Handwritten Digits*, UCI Machine Learning Repository, DOI: https://doi.org/10.24432/C50P49. Original pixel extraction traces to NIST image preprocessing. The scikit-learn distribution identifies its digit data as a copy of the test set from the UCI original.

## c. GitHub repository link [1 mark]

**Repository:** [REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL]  
**Live Streamlit Community Cloud application:** [REPLACE_WITH_YOUR_STREAMLIT_APP_URL]

Both placeholders must be updated with **real, clickable, tested links** after the manual GitHub upload and deployment. The source code, executed notebook, dependency lock, complete `.joblib` pipelines, public dataset, labelled `test_data.csv`, experiment metadata and original observations are committed in the repository.

## d. Models used and evaluation [5 marks]

### Common preprocessing and reproducibility

Every method uses the **same** `X_train`, `X_test`, `y_train`, `y_test` and pixel schema. **Stratified sampling precedes all preprocessing**. A new independent `Pipeline` for each model includes a numeric `ColumnTransformer` with `StandardScaler` and then the classifier. Training-only statistics are learned during `pipeline.fit(X_train, y_train)`; no held-out rows are involved in fitting. Constant pixel columns are safely handled by `StandardScaler`.

| ML model | Fixed settings (training only) |
|---|---|
| Logistic Regression | `C=0.7`, `max_iter=2500`, `solver='lbfgs'`, seed 37 |
| Decision Tree | `max_depth=12`, `min_samples_leaf=2`, seed 37 |
| K-Nearest Neighbors | `n_neighbors=5`, `weights='distance'` |
| Gaussian Naive Bayes | `var_smoothing=1e-8` |
| Support Vector Machine | RBF kernel, `C=4.0`, `gamma='scale'`, `probability=True`, seed 37 |

No tuning is performed using the held-out labels. Exactly **five** classifiers are trained on the **same 1,437 training rows** and evaluated on the **same 360 test rows**.

### Definition of all six metrics

- **Accuracy**: correct predictions divided by total predictions.
- **AUC**: multiclass **one-versus-rest (OvR), macro averaged** over all **10** digits, computed from the ten-column predicted **probabilities**, never from hard labels. It is undefined on an uploaded CSV missing any class, and the app displays **N/A**.
- **Precision, Recall and F1**: unweighted **macro averages across all ten classes** (`zero_division=0`). Each digit contributes equally regardless of frequency.
- **MCC**: scikit-learn's **multiclass** Matthews correlation coefficient, taking all confusion-matrix cells into account.

### Comparison table — measured holdout results

| ML Model Name | Accuracy | AUC | Precision | Recall | F1 | MCC |
|---|---:|---:|---:|---:|---:|---:|
| Logistic Regression | 0.9917 | 0.9997 | 0.9919 | 0.9918 | 0.9917 | 0.9908 |
| Decision Tree | 0.8500 | 0.9289 | 0.8580 | 0.8492 | 0.8507 | 0.8340 |
| K-Nearest Neighbors | 0.9833 | 0.9953 | 0.9839 | 0.9831 | 0.9833 | 0.9815 |
| Gaussian Naive Bayes | 0.8278 | 0.9732 | 0.8462 | 0.8275 | 0.8263 | 0.8111 |
| Support Vector Machine | 0.9889 | **0.9999** | 0.9890 | 0.9889 | 0.9888 | 0.9877 |

*Rounding is to four decimal places for readability. The precise unrounded numbers appear in `results/model_metrics.csv`; the notebook and `verify_project.py` regenerate them from bundled `.joblib` files.*

### Error-count cross-check

| Classifier | Misclassified test images | Correct images |
|---|---:|---:|
| Logistic Regression | **3** | **357** |
| Decision Tree | 54 | 306 |
| K-Nearest Neighbors | 6 | 354 |
| Gaussian Naive Bayes | 62 | 298 |
| Support Vector Machine | 4 | 356 |

### Observations for all five models [3 marks]

| ML Model Name | Observation about actual performance on this dataset |
|---|---|
| **Logistic Regression** | **Best overall MCC (0.9908), accuracy (0.9917), macro F1 (0.9917)** with only **3 mistakes**. It mistakes digit **3 for 8** once and **3 for 9** once, and **6 for 8** once. Linear separation works surprisingly well on this normalized 64-dimensional representation. |
| **Decision Tree** | Accuracy **0.8500**, MCC **0.8340** and **54 mistakes**. A single tree's axis-aligned thresholds lose subtle combinations of pixels; among other confusions, it predicts **2 as 1** four times. Limiting depth helps constrain complexity but does not give the flexibility of a distance or kernel method here. |
| **K-Nearest Neighbors** | Accuracy **0.9833**, MCC **0.9815**, and **6 mistakes**. Similar pixel patterns tend to be nearby after standardization; prediction, however, must retain and consult training examples. No digit pair dominates its relatively small number of mistakes. |
| **Gaussian Naive Bayes** | Lowest MCC **0.8111** and **62 mistakes** despite respectable AUC **0.9732**. The 64 pixel intensities are not conditionally independent—neighboring strokes covary. It misclassifies **4 as 7** nine times; its probability ranking can still outperform its fixed argmax decisions on some classes. |
| **Support Vector Machine** | Accuracy **0.9889**, MCC **0.9877**, and **4 mistakes**. It gives the **highest macro OvR AUC (0.9999)**, suggesting excellent probability ranking, but trails Logistic Regression on MCC/F1 for this split. The RBF boundary captures non-linear writing shapes at the cost of lower transparency. |
| **Overall winner for this dataset** | **Logistic Regression** using the predefined **MCC** ranking, with **F1** as a tie-breaker; it also leads on Accuracy, Recall and F1. **SVM leads on AUC**, so the winner depends on the metric and deployment priority. |

**Caveats:** This is **one** pre-specified holdout split from a curated UCI-derived subset, not a writer-independent prospective test. Different splits, handwriting, hyperparameters, or costs of confusing particular digits can change the order. No clinical or production claim is made.

## e. Streamlit application [4 marks]

The custom **INK / Digit Inspector** interface supports every faculty-mandated feature:

1. **CSV upload** of labelled test samples (`Digit` and all 64 pixel columns), or the bundled held-out dataset for instant demonstration. Missing columns, invalid values, empty files and bad labels receive a useful error. Extra columns and reordered columns are accepted.
2. **Model dropdown** for the five classifiers, with short explanations of each.
3. **Six selected-model score cards** and a comparison table covering all **five** saved models on the uploaded rows.
4. **Ten-by-ten confusion matrix**, with a count/row-percentage switch, and **per-class classification report**.

Extra UI features are a separate **comparison tab** with metric bar charts and a per-digit one-vs-rest ROC display; a **digit explorer** reconstructing 8×8 pixel grids and highlighting mistakes; and a **data/predictions tab** with class counts and downloadable predictions. All of these work on **saved model inference**. The app has **no training step**.

**Test CSV contract:** `test_data.csv` includes all **64 pixel columns** and `Digit`, with 360 samples, so evaluation is possible. Predictions without true labels could not produce the mandatory metrics, which is why a target is required.

## f. Project directory and execution

```text
ML_Assignment2_Final_GitHub_Ready/
├── app.py                       # Custom Streamlit evaluation interface
├── ML_Assignment2.ipynb         # Executed, stage-by-stage primary experiment
├── train.py                     # Deterministic model/CSV export from scratch
├── verify_project.py            # Independent saved-model metric verification
├── requirements.txt             # Pinned Cloud dependencies
├── requirements-notebook.txt    # Optional Jupyter dependencies
├── requirements-report.txt      # Optional local PDF-builder dependencies
├── README.md                    # Faculty report content and full instructions
├── QUICKSTART.md                # Installation, manual GitHub and deployment
├── RUBRIC_AUDIT.md              # Every mark and required evidence tracked
├── make_submission_pdf.py       # Final PDF: links, real screenshot, README content
├── data/
│   └── digits_uci_subset.csv    # Exact 1,797-row subset from UCI via scikit-learn
├── model/
│   ├── logistic_regression.joblib
│   ├── decision_tree.joblib
│   ├── knn.joblib
│   ├── naive_bayes.joblib
│   └── svm.joblib
├── results/
│   ├── model_metrics.csv        # All 30 exact test metrics
│   └── experiment.json          # Saved split, versions, model names, matrices
├── src/                          # Shared split, dataset-validation and metric logic
├── tests/                        # Positive and negative integrity checks
├── .streamlit/config.toml        # Customized app theme
├── evidence/README.md            # Genuine BITS screenshot instructions
└── test_data.csv                 # 360 test rows + 10-class target
```

### Reproduce the results

From the project root with **Python 3.13**:

```bash
python -m pip install -r requirements.txt
python train.py
python verify_project.py
python -m unittest discover -s tests -v
streamlit run app.py
```

The `ML_Assignment2.ipynb` notebook should be run **top to bottom from the project root** after installing `requirements-notebook.txt` if Jupyter is unavailable. It recreates the five pipelines and verifies reload consistency with no hidden state. The optional `train.py` script can recreate the same results from the committed dataset without notebook tooling. **Do not retrain in Streamlit; do not upload training rows to the demo.**

**Deployment:** Sign in to Streamlit Community Cloud using GitHub, choose your repository, **`main`**, **`app.py`**, and **Python 3.13** (to match the saved serialization environment). After deploying, upload `test_data.csv` and confirm all five models match `results/model_metrics.csv`. If deployment fails due to package compatibility, compare the Cloud build log with pinned versions—do not randomly regenerate models.

## g. Limitations, provenance and integrity

- **UCI citation and licensing:** Alpaydin & Kaynak (1998), DOI https://doi.org/10.24432/C50P49; UCI dataset is published under CC BY 4.0. This project explicitly identifies the 1,797-row scikit-learn subset, not the full 5,620-row UCI release.
- **Reproducibility:** Both the notebook and script use a fixed stratified split, the exact same pipeline constructors and the exact same metric implementation. JSON records the raw CSV's SHA-256 fingerprint and the fitted environment versions.
- **Test-data separation:** Test rows are not used to scale, fit or tune models. Reporting a descriptive winner on test data is not hyperparameter tuning.
- **Serialization:** Only trusted model files stored in this repository are loaded. Untrusted uploaded `.pkl` / `.joblib` files are never accepted.
- **Academic integrity:** The original assignment explicitly states that AI assistance is **for learning**, not a direct copy-paste submission. Review and independently execute the project, adjust and explain your own modeling and UI decisions, write observations in your own words, and build a genuine commit history. Only your **own BITS Virtual Lab screenshot and real live links** may appear in the final PDF.

