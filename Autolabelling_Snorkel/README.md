# Big Data Analysis — Lab: Autolabelling Techniques (Snorkel + Label Studio)

This repo contains an end-to-end mini-pipeline for **automatic / weak-supervision labeling** on Amazon-style product reviews:
1) dataset preparation (cleaning + binary sentiment target),
2) tasks generation for **Label Studio**,
3) weak supervision experiment with **Snorkel** + a simple **TF-IDF + Logistic Regression** end model.

(See the lab report for theory + discussion.) :contentReference[oaicite:0]{index=0}

---

## Repository Contents

- `prepare_dataset.py` — creates (or loads) `amazon_reviews.csv`, cleans it, creates binary sentiment target, saves `processed_amazon_reviews.csv`. :contentReference[oaicite:1]{index=1}  
- `make_label_studio_tasks.py` — samples 1000 rows and exports `label_studio_tasks.json` for Label Studio import. :contentReference[oaicite:2]{index=2}  
- `snorkel_experiment.py` — defines labeling functions, trains Snorkel LabelModel, then trains a TF-IDF + Logistic Regression classifier, prints metrics and LF coverage. :contentReference[oaicite:3]{index=3}  
- `Lab_Report.md` — written report (mindmap + methodology + results). :contentReference[oaicite:4]{index=4}  

> Filenames above reflect the scripts in this lab (dataset prep, Label Studio JSON export, Snorkel pipeline). 

---

## Requirements

- Python **3.9+**
- Core:
  - `pandas`
  - `numpy`
  - `scikit-learn`
- Weak supervision:
  - `snorkel`
- Labeling UI:
  - `label-studio` (optional, only if you want the web UI)

---

## Setup

```bash
python -m venv .venv
# Windows:
# .venv\Scripts\activate
# macOS/Linux:
source .venv/bin/activate

pip install -U pip
pip install pandas numpy scikit-learn snorkel label-studio
