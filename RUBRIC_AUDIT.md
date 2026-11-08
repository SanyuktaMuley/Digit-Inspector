# Complete faculty rubric and guide compliance audit

**Scope:** Original 6-page faculty Assignment 2 PDF + 8-page supplementary implementation guide + comparison with the example student's PDF and the earlier Wisconsin notebook.

**Important precedence:** The faculty Assignment 2 PDF specifies what is **mandatory**. The supplemental guide recommends good workflow; it does not increase the original 15-point marking scheme. The example student's already-submitted PDF is *a reference for page ordering and proof style*, not a source to copy.

| Requirement from the official PDF | Marks | Implementation location | Status / action |
|---|---:|---|---|
| Public Kaggle or UCI dataset: ≥500 rows and ≥12 features | — | `data/digits_uci_subset.csv`, README b | **DONE**: UCI-derived documented 1,797 rows, 64 features |
| One dataset for all five models | — | Notebook, `src/experiment.py`, `train.py` | **DONE**: same stratified 80/20 partition |
| Logistic Regression + all six metrics | 1 | `model/logistic_regression.joblib`, comparison CSV | **DONE** |
| Decision Tree + all six metrics | 1 | `model/decision_tree.joblib`, comparison CSV | **DONE** |
| KNN + all six metrics | 1 | `model/knn.joblib`, comparison CSV | **DONE** |
| Gaussian/Multinomial Naive Bayes + all six metrics | 1 | `model/naive_bayes.joblib`, comparison CSV | **DONE**: GaussianNB |
| SVM + all six metrics | 1 | `model/svm.joblib`, comparison CSV | **DONE** |
| Dataset description (target, attributes, source) | 1 | README b, notebook, report PDF | **DONE** |
| GitHub link and complete repo with files | 1 | README c, full project directory | **USER ACTION**: upload and paste live link |
| Model-wise observations and winner | 3 | README d, notebook concluding table | **DONE as reference**, personalize explanation |
| Streamlit CSV test dataset upload | 1 | `app.py` sidebar & `validate_labeled_csv` | **IMPLEMENTED; CLOUD TEST PENDING** |
| Streamlit model selection dropdown | 1 | `app.py` sidebar | **IMPLEMENTED; CLOUD TEST PENDING** |
| Streamlit six metrics display | 1 | `app.py` Evaluation and Compare tabs | **IMPLEMENTED; CLOUD TEST PENDING** |
| Confusion matrix OR classification report | 1 | `app.py` shows **both** | **IMPLEMENTED; CLOUD TEST PENDING** |
| BITS Virtual Lab execution proof | 1 | `evidence/bits_lab_execution.png` | **USER ACTION**: only user can capture genuine evidence |
| Submit **single PDF** in exact mandated order: repo, app, screenshot, README content | — | `make_submission_pdf.py` | **USER ACTION** for live URL + evidence, then generator |
| Cloud app loads saved models, **never retrains** with test CSV | — | `app.py` + five saved pipelines | **IMPLEMENTED; CLOUD TEST PENDING** |
| App shows results of different models on test data | — | comparison tab with complete 5×6 table | **IMPLEMENTED; CLOUD TEST PENDING** |
| requirements.txt covers dependencies | — | version-pinned `requirements.txt` | **DONE**, Cloud installation must be validated |
| Source code (.py / .ipynb), preprocessing artifacts | — | `ML_Assignment2.ipynb`, `src/`, `model/` | **DONE** |
| README.md must appear in final PDF | — | Generator reads the **actual README** | **DONE in template; real PDF pending** |
| Plagiarism prevention: unique code, UI, metadata and commits | Crucial | Unique digit explorer UI and data; personal customization needed | **USER ACTION**: understand and genuinely adapt; do not copy example |

### Additional good practices from the eight-page guide

- Executed Jupyter notebook with meaningful Markdown and figures. **DONE**.
- EDA: shape, info, missing values, duplicates, target distribution, visualization. **DONE**.
- No leakage: split before scaling; full `Pipeline`. **DONE**.
- Consistent multiclass score definitions using `predict_proba`. **DONE**.
- Complete `.joblib` pipelines plus label-containing CSV. **DONE**.
- Standalone saved-model reload test. **DONE**.
- CSV errors, sample download, explanations, 6 metric cards, count and percentage confusion matrices. **IMPLEMENTED**.
- Version pinning for joblib compatibility. **DONE**.
- Extra UI: per-class ROC, mistakes explorer, true/predicted labels and export. **IMPLEMENTED**.
- Real, not synthetic, BITS Lab proof. **USER ACTION**.

### Distinguishing the original UCI archive from the chosen sub-dataset

UCI lists 5,620 total rows, but scikit-learn's canonical `load_digits()` contains a 1,797-row copy of an original UCI test subset. The exact 1,797-row subset is included in the project. Claiming that the project uses all 5,620 original rows would be inaccurate; we do **not** make that claim.

### Risks that cannot be eliminated by a ZIP

- The Cloud deployment is not live or confirmed until the user deploys and tests it.
- GitHub repo and public links cannot be valid until the user creates them.
- The 1-mark lab screenshot cannot be fabricated; it must show real Virtual Lab execution.
- Faculty can still award fewer than 15 based on plagiarism rules, originality, portal submission, or rubric interpretation. No score guarantee is possible.
- The submission deadline is **8 November 2026 at 23:59 IST**; faculty explicitly says no extension and no drafts.
