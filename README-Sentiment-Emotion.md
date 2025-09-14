# Sentiment & Emotion Analysis — Rules, Transformers, and Diagnostics

![Jupyter](https://img.shields.io/badge/Notebook-Jupyter-blue)
![Python](https://img.shields.io/badge/Python-3.9%2B-green)
![NLTK](https://img.shields.io/badge/NLP-NLTK-yellow)
![spaCy](https://img.shields.io/badge/NLP-spaCy-09f)
![Transformers](https://img.shields.io/badge/HuggingFace-Transformers-ff69b4)

This repository contains **`Sentiment and Emotion-Analysis.ipynb`**, a compact, hands-on notebook for **sentiment and emotion analysis**. It blends **rule‑based** methods (NLTK **VADER**) with **Transformer** models (Hugging Face pipelines; optionally **CardiffNLP** and **GoEmotions**) and provides simple evaluation/diagnostics (accuracy, confusion matrix, classification report).

---

## What’s inside

- **Preprocessing & Linguistics**
  - Basic text loading/cleaning
  - Tokenization / inspection with **spaCy** (optional)

- **Sentiment Analysis**
  - **VADER** (lexicon‑ and rule‑based, great for social text)
  - **Transformers** pipeline for sentiment (e.g., SST‑2/roberta variants)
  - (Optional) **CardiffNLP** Twitter‑tuned models

- **Emotion Classification**
  - (Optional) **GoEmotions**‑style label set for fine‑grained emotions

- **Evaluation & Visualization**
  - `accuracy_score`, `classification_report`, `confusion_matrix`
  - Heatmaps and quick plots via **seaborn**/**matplotlib**

> The notebook is **dataset‑agnostic**. Bring your own CSV/JSON/TXT—define `text` and (optionally) `label` columns for supervised evaluation.

---

## Repository structure

```
.
├── Sentiment and Emotion-Analysis.ipynb
├── data/                           # (optional) place your datasets here
├── requirements.txt
├── .gitignore
└── README.md
```

`data/` is Git‑ignored by default to keep private datasets out of the repo.

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

### 2) Download runtime assets

**NLTK (VADER lexicon & tokenizers)**
```bash
python -m nltk.downloader vader_lexicon punkt stopwords
```

**spaCy (optional linguistic features)**
```bash
python -m spacy download en_core_web_sm
```

> For larger projects, consider `en_core_web_md`/`lg` (word vectors).

### 3) (Optional) Add data
Place your file under `data/` (e.g., `data/tweets.csv`) and point the data‑loading cell to its path. A minimal supervised schema:
```
text,label
"I love this!",positive
"...",...
```

### 4) Launch Jupyter
```bash
jupyter lab
# or
jupyter notebook
```
Open **`Sentiment and Emotion-Analysis.ipynb`** and run cells top‑to‑bottom (Kernel → Restart & Run All).

---

## Model notes

- **VADER** is fast, domain‑robust for social media, and requires no training.
- **Transformers** (via `transformers.pipeline("sentiment-analysis")`) provide stronger accuracy; internet access is not required after the first model download.
- **CardiffNLP** Twitter models (`cardiffnlp/twitter-roberta-base-sentiment-*`) are tuned for short, informal text.
- **GoEmotions** provides multi‑label emotion categories; the notebook demonstrates single‑/multi‑label mapping where applicable.

> Tip: If you enable GPU, PyTorch will accelerate transformer inference. On Apple Silicon, recent PyTorch builds support **MPS** acceleration.

---

## Evaluation

When labels are available, the notebook computes:
- **Accuracy**
- **Confusion matrix** (heatmap)
- **Classification report** (per‑class precision/recall/F1)

For imbalanced data, prefer **macro/weighted F1** and consider PR‑AUC.

---

## Extending the notebook

- **Domain lexicons:** Blend VADER with domain‑specific sentiment dictionaries.
- **Thresholding:** Tune decision thresholds for precision/recall trade‑offs.
- **Calibration:** Use `CalibratedClassifierCV` for probability estimates (if you add linear models).
- **Error analysis:** Log top false positives/negatives for iterative improvements.
- **Explainability:** Extract top indicative tokens/phrases; use SHAP for transformer logits.

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
- `nltk`
- `spacy`
- `transformers`, `torch`
- `tqdm`
- `jupyter`

> If you add datasets in Excel, include `openpyxl`. For Twitter models, no extra package is needed—Hugging Face downloads weights/tokenizers.

---

## License
Choose a license (MIT/Apache‑2.0/BSD‑3‑Clause) and add a `LICENSE` file.

## Acknowledgements
- NLTK (VADER) maintainers
- Hugging Face Transformers community
- spaCy team
- scikit‑learn, pandas, numpy, seaborn/matplotlib communities

---

**Maintainer tips**  
Clear outputs before committing to keep diffs tidy:
```bash
jupyter nbconvert --ClearOutputPreprocessor.enabled=True --inplace "Sentiment and Emotion-Analysis.ipynb"
```
Pin versions in `requirements.txt` for stable builds.
