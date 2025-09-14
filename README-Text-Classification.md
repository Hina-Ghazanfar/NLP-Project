# Text Classification — Classic NLP Pipeline (Vectorizers + Linear Models)

![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![scikit--learn](https://img.shields.io/badge/scikit--learn-1.x-orange)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09f)
![Transformers](https://img.shields.io/badge/HF-Transformers-ff69b4)

This repository contains **`Text classification.ipynb`**, a practical, end‑to‑end text‑classification workflow using **TF‑IDF/Bag‑of‑Words** features and strong **linear baselines** (e.g., `LinearSVC`, `LogisticRegression`, `SGDClassifier`). It also includes optional hooks for **spaCy** (linguistic features) and **Hugging Face Transformers** (embeddings / future extensions).

---

## What it covers

- **Data loading & cleaning** (CSV/JSON or inline text)
- **Vectorization:** `CountVectorizer` and `TfidfVectorizer`
- **Models:** `LinearSVC`, `LogisticRegression`, `SGDClassifier`, `PassiveAggressiveClassifier`, and `RandomForestClassifier` (baseline)
- **Evaluation:** accuracy, `classification_report`, `confusion_matrix` (+ heatmap)
- **Utilities:** progress bars (`tqdm`), plotting (`seaborn`/`matplotlib`)

> Designed to be **dataset‑agnostic**. By default, it expects columns like `text` and `label`. Update the data‑loading cell if your schema differs.

---

## Repository structure

```
.
├── Text classification.ipynb
├── data/                          # (optional) place your datasets here
├── requirements.txt
├── .gitignore
└── README.md
```

*`data/` is Git‑ignored by default to keep private datasets out of the repo.*

---

## Quickstart

### 1) Clone and create a virtual environment
```bash
git clone <YOUR_REPO_URL>.git
cd <YOUR_REPO_NAME>

python -m venv .venv
# macOS/Linux
source .venv/bin/activate
# Windows (PowerShell)
# .venv\Scripts\Activate.ps1

pip install -r requirements.txt
```

### 2) (Optional) Install a spaCy model
```bash
python -m spacy download en_core_web_sm
```
If you process larger corpora or need vectors, consider `en_core_web_md`/`lg`.

### 3) Add data
Place your dataset under `data/` (e.g., `data/train.csv`) and adjust the path in the notebook. A minimal schema is:
```
text,label
"This movie was great!",positive
"...",...
```

### 4) Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```
Open **`Text classification.ipynb`** and run cells top‑to‑bottom (Kernel → Restart & Run All).

---

## Notebook outline

1. **Setup & Imports** — set plotting style, seed, and paths
2. **Data Loading** — read CSV/JSON or construct a small demo dataset
3. **Preprocessing** — lowercase, punctuation/number handling (customizable); optional spaCy tokenization
4. **Vectorization** — `CountVectorizer` and/or `TfidfVectorizer` (n‑grams, stop‑words, min_df/max_df)
5. **Modeling** — fit `LinearSVC` / `LogisticRegression` / `SGDClassifier` / `PassiveAggressiveClassifier` (plus a tree baseline)
6. **Evaluation** — compute **accuracy**, print `classification_report`, and plot **confusion matrix**
7. **(Optional) Extensions** — hooks for HF **Transformers** (tokenizers/embeddings) and `gensim` utilities

---

## Reproducibility & ML hygiene

- Use **stratified** splits for classification (`train_test_split(..., stratify=y)`).
- Keep text transforms **inside a Pipeline** during CV to avoid leakage:
  ```python
  from sklearn.pipeline import make_pipeline
  from sklearn.feature_extraction.text import TfidfVectorizer
  from sklearn.svm import LinearSVC

  pipe = make_pipeline(TfidfVectorizer(ngram_range=(1,2)), LinearSVC())
  pipe.fit(X_train, y_train)
  ```
- Fix seeds: `random_state=42` (split/models where applicable).
- For **imbalanced** data, prefer **macro‑/weighted‑F1** and **PR‑AUC**; try `class_weight='balanced'`.
- Export results (CSV/PNG) to a subfolder (e.g., `outputs/`) for auditability.

---

## Extending the notebook

- **Hyperparameters:** Wrap in `GridSearchCV` / `RandomizedSearchCV` with a `Pipeline`.
- **Calibration:** If you need probabilities (e.g., thresholds/PR curves), use `CalibratedClassifierCV` with `LinearSVC` or switch to `LogisticRegression`.
- **Transformers baseline:** Add embeddings via Hugging Face (`transformers`) and compare to TF‑IDF.
- **Explainability:** Show top features per class using linear model coefficients.

---

## Requirements

Install with:
```bash
pip install -r requirements.txt
```

**Core dependencies**
- `pandas`, `numpy`
- `scikit-learn`
- `seaborn`, `matplotlib`
- `tqdm`
- `spacy` *(optional linguistic features)*
- `transformers`, `torch` *(optional embeddings / transformer baselines)*
- `gensim` *(optional utilities)*
- `jupyter`

> If you use spaCy, download a language model (see Quickstart). For Transformers on CPU, keep batch sizes small; on Apple Silicon, PyTorch + MPS can accelerate inference.

---

## License
Choose a license (MIT/Apache‑2.0/BSD‑3‑Clause) and include a `LICENSE` file.

## Acknowledgements
- scikit‑learn, pandas, numpy communities
- spaCy and Hugging Face ecosystems for NLP tooling

---

**Maintainer tips**  
Clear notebook outputs before committing to keep diffs tidy:
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "Text classification.ipynb"
```
Pin versions in `requirements.txt` for stable builds.
