# Kaggle Learn

This repository contains my work across the Kaggle Learn curriculum.  
The material spans a broad range of data skills — from modelling, data handling, and visualisation to advanced areas such as feature engineering, deep learning, computer vision, time series, geospatial analysis, explainability, reinforcement learning, and complex SQL analytics.

---

## Structure

kaggle_learn/
notebooks/ (one notebook per course, ignored in git)
src/ (paired scripts for each notebook)
data/ (local datasets, ignored)
pyproject.toml
uv.lock
jupytext.toml
.gitignore
README.md


- **notebooks/** — interactive `.ipynb` files (excluded from version control)  
- **src/** — paired `.py` scripts generated via Jupytext (tracked)  
- **data/** — local datasets used for exercises (ignored)

---

## Working With the Project

From the repository root:

uv sync
uv run jupyter lab

## Notebook Pairing
Automatic notebook-script pairing as configured:

notebooks//ipynb  ↔  src//py:percent

This keeps the repository clean, maintains readable diffs, and ensures each course has a single, version-controlled script.
