# Tweeter Disaster — NLP Binary Classification

Predict whether a tweet describes a real disaster (`target = 1`) or a non-disaster
tweet (`target = 0`).

Built on the well-known Twitter disaster classification dataset.

## Get the data

The dataset is not included in this repo. Download `train.csv` from Kaggle:
[Real or Not? NLP with Disaster Tweets](https://www.kaggle.com/c/nlp-getting-started/data)
(login required), then place it next to the notebook.

## Setup

Create the virtual environment and install dependencies:

```bash
python3 -m venv .venv
source .venv/bin/activate
pip install -r requirements.txt
```

## Run the notebook

```bash
source .venv/bin/activate
jupyter notebook TweeterDisaster.ipynb
```

Select the **Python 3 (.venv)** kernel if Jupyter asks, then run all cells
(`Cell` → `Run All`).

## Workflow

1. Data loading and inspection
2. Text preprocessing
3. Train/validation split (stratified)
4. TF-IDF feature extraction
5. Logistic Regression classification
6. Model evaluation
7. Hyperparameter tuning (`GridSearchCV` on `C`)
8. Error analysis
9. Final interpretation and limitations

## Dependencies

| Package | Purpose |
| --- | --- |
| `numpy` | Numerical operations |
| `pandas` | CSV loading and data frames |
| `matplotlib` | Plots and confusion matrix |
| `scikit-learn` | TF-IDF, Logistic Regression, metrics, CV |
| `jupyter` / `notebook` | Notebook environment |

## Notes

- `train.csv` must sit in the same directory as the notebook.
- The venv is built on Python 3.13. Python 3.9 works too, but restricts you to
  `notebook 7.1.x`.
- `GridSearchCV` uses `n_jobs=-1`, so the tuning cell uses all CPU cores.
