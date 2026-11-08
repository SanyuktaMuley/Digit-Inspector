# Quickstart: before GitHub, before Streamlit, before final PDF

**Your deliverable is the ZIP's extracted contents, not the ZIP nested inside GitHub.** Upload all files/folders at the repository root and preserve the five files in `model/`.

## 1. Run and validate locally or on BITS Virtual Lab

Open a terminal in the extracted project folder, ideally using the **BITS Prayogshala / Virtual Lab** where required:

```bash
python --version
python -m pip install -r requirements.txt
python train.py
python verify_project.py
python -m unittest discover -s tests -v
streamlit run app.py
```

The app starts at `http://localhost:8501`. Its sidebar already selects the bundled held-out CSV; also use the real CSV uploader with `test_data.csv`. Select **each of the five models** and check that all six metrics appear and the confusion matrix is visible. To execute notebook cells in the Virtual Lab, run Jupyter with `requirements-notebook.txt` if allowed; the fallback script (`train.py`) suffices to demonstrate your model execution.

Expected bundle: **1,797** rows, **64** features, stratified **1,437/360** train/test split, **five** joblib pipelines and **30 metrics**. The app MUST NOT train upon CSV upload.

## 2. Manual GitHub upload: you handle this

1. Create an empty public GitHub repo named meaningfully (example: `ink-digit-classification-bits-ml2`). Do not copy the example student's repo name.
2. Either upload extracted project files through GitHub's **Add file → Upload files** or use git from your laptop (if installed):

   ```bash
   git init
   git add .
   git commit -m "Add independently validated digit classification project"
   git branch -M main
   git remote add origin https://github.com/YOUR_USERNAME/YOUR_REPO.git
   git push -u origin main
   ```

3. Confirm files including `app.py`, `README.md`, `requirements.txt`, `test_data.csv`, `data/`, `ML_Assignment2.ipynb`, `src/`, `results/`, and **all five** `model/*.joblib` are displayed on GitHub. GitHub may show image outputs embedded in the executed notebook.
4. **Do not fake or backdate commits.** The guide expects genuine development history; make new commits only for real future modifications and document your own changes.

## 3. Deploy the app (essential for 4 marks)

- Visit https://share.streamlit.io/ or https://streamlit.io/cloud and sign in with GitHub.
- Choose **New app** → your public repo → branch **main** → entry **app.py**.
- Choose Python **3.13** to match the pinned model-building stack. Deploy.
- Confirm your actual `https://...streamlit.app` URL loads. Test sample and upload workflows **for all five models**; check the 6 metrics, 10×10 confusion matrix and classification report.
- If a package install fails, use the build log to diagnose dependency/Cloud Python version, do **not** randomly drop pinned versions.

## 4. BITS Virtual Lab screenshot (required 1 mark)

- In your *own* BITS lab session open the project, run the training or verification terminal and inspect the local or deployed Streamlit app in its browser.
- Capture an **authentic screenshot** showing identifiable lab context, selected model, **six scores**, confusion matrix and app URL when possible. Save as `evidence/bits_lab_execution.png`.
- **Do not** use the sample PDF's screenshots, a mockup or any rendered UI illustration as actual lab proof. Hide passwords/tokens.

## 5. Update repository README and create the single PDF

- Replace **all** `[YOUR NAME]`, `[YOUR BITS ID]`, `[REPLACE_WITH_YOUR_GITHUB_REPOSITORY_URL]`, and `[REPLACE_WITH_YOUR_STREAMLIT_APP_URL]` placeholders in README.
- Commit the real updated README to GitHub.
- Install local PDF-builder dependencies: `python -m pip install -r requirements-report.txt`.
- From the project root, run with your **real** details and screenshot:

  ```bash
  python make_submission_pdf.py \
    --name "YOUR FULL NAME" --bits-id "YOUR_BITS_ID" \
    --github "https://github.com/YOUR_USERNAME/YOUR_REPO" \
    --app "https://YOUR_LIVE_APP.streamlit.app" \
    --screenshot "evidence/bits_lab_execution.png" \
    --output "ML_Assignment2_FINAL.pdf"
  ```

- Open the PDF and ensure the **clickable** GitHub and Streamlit URLs appear first; **your screenshot** is next; **full README content** follows. Confirm no placeholder remains in the rendered PDF before uploading.
- Only submit the resulting **single final PDF** and click the faculty portal **SUBMIT** button; a draft save is not submission.

## Deadline

The original faculty PDF specifies **Sunday, November 8, 2026 at 23:59 IST**, with **no extension** and **no draft submissions accepted**.

## Academic-integrity reminder

Faculty permits AI use only as a learning aid. This package is a reproducible reference implementation. Adapt and understand it, author your own explanation/observations and real GitHub history before submission. No grade can be guaranteed by a generated package.
