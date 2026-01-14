# YouTube Comment Sentiment Analysis 🚀

End-to-end NLP pipeline for sentiment classification of YouTube comments, built using **TF-IDF, MLflow, Optuna, and LightGBM**, following a **production-oriented ML workflow**.

---

## 📌 Problem Statement

Analyze YouTube comments and classify them into:

- **Positive (1)**
- **Neutral (0)**
- **Negative (-1)**

The project focuses on **model performance, class imbalance handling, experiment tracking, and reproducibility**.

---

## 📂 Dataset

- **Source:** `reddit.csv`
- **Columns:**
  - `clean_comment` → comment text
  - `category` → sentiment label

---

## ⚙️ Pipeline Architecture

Project Organization
------------

    ├── LICENSE
    ├── Makefile           <- Makefile with commands like `make data` or `make train`
    ├── README.md          <- The top-level README for developers using this project.
    ├── data
    │   ├── external       <- Data from third party sources.
    │   ├── interim        <- Intermediate data that has been transformed.
    │   ├── processed      <- The final, canonical data sets for modeling.
    │   └── raw            <- The original, immutable data dump.
    │
    ├── docs               <- A default Sphinx project; see sphinx-doc.org for details
    │
    ├── models             <- Trained and serialized models, model predictions, or model summaries
    │
    ├── notebooks          <- Jupyter notebooks. Naming convention is a number (for ordering),
    │                         the creator's initials, and a short `-` delimited description, e.g.
    │                         `1.0-jqp-initial-data-exploration`.
    │
    ├── references         <- Data dictionaries, manuals, and all other explanatory materials.
    │
    ├── reports            <- Generated analysis as HTML, PDF, LaTeX, etc.
    │   └── figures        <- Generated graphics and figures to be used in reporting
    │
    ├── requirements.txt   <- The requirements file for reproducing the analysis environment, e.g.
    │                         generated with `pip freeze > requirements.txt`
    │
    ├── setup.py           <- makes project pip installable (pip install -e .) so src can be imported
    ├── src                <- Source code for use in this project.
    │   ├── __init__.py    <- Makes src a Python module
    │   │
    │   ├── data           <- Scripts to download or generate data
    │   │   └── make_dataset.py
    │   │
    │   ├── features       <- Scripts to turn raw data into features for modeling
    │   │   └── build_features.py
    │   │
    │   ├── models         <- Scripts to train models and then use trained models to make
    │   │   │                 predictions
    │   │   ├── predict_model.py
    │   │   └── train_model.py
    │   │
    │   └── visualization  <- Scripts to create exploratory and results oriented visualizations
    │       └── visualize.py
    │
    └── tox.ini            <- tox file with settings for running tox; see tox.readthedocs.io


--------

<p><small>Project based on the <a target="_blank" href="https://drivendata.github.io/cookiecutter-data-science/">cookiecutter data science project template</a>. #cookiecutterdatascience</small></p>
## 🧪 Notebook Walkthrough

### Notebook 1: Preprocessing & EDA
**File:** `yt-comment-analyzer-preprocessing.ipynb`

- Drops **100 NaN** values
- Removes **350 duplicate** entries
- Cleans empty strings, newlines, and extra whitespaces
- Converts text to lowercase

**EDA Includes:**
- Class distribution (imbalanced)
- Word count & character count analysis
- Stopword frequency
- Top 25 bigrams & trigrams
- WordCloud visualization

**Class Distribution:**
- Positive: **42.86%**
- Neutral: **34.71%**
- Negative: **22.42%**

---

### Notebook 2: Baseline Model
**File:** `experiment-1-baseline-model.ipynb`

- **Vectorization:** Bag of Words (10,000 features)
- **Model:** Random Forest
- **Tracking:** MLflow

**Purpose:** Establish a fast baseline for comparison.

---

### Notebook 3: BoW vs TF-IDF
**File:** `experiment-2-bow-tfidf.ipynb`

- **Models Tested:**
  - BoW → (1,1), (1,2), (1,3)
  - TF-IDF → (1,1), (1,2), (1,3)
- **Model:** Random Forest

**Result:** **TF-IDF outperformed BoW**

---

### Notebook 4: TF-IDF Feature Optimization
**File:** `experiment-3-tfidf-(1,3)-max_features.ipynb`

- Best n-gram range: **(1,3)**
- Feature sizes tested: **1000 → 9000**
- Focus: **Negative (-1) recall**

**Best trade-off:** `max_features = 1000`

---

### Notebook 5: Handling Class Imbalance
**File:** `experiment-4-handling-imbalanced-data.ipynb`

**Techniques Tested:**
- Class Weights
- Oversampling
- Undersampling
- ADASYN
- SMOTE-ENN

**Winner:** **SMOTE**

---

### Notebook 6–9: Model Comparison with HPT
**Experiment 5**

| Model          | Accuracy |
|----------------|----------|
| Random Forest  | ~0.71    |
| SVM            | Logged   |
| XGBoost        | Logged   |
| **LightGBM**   | **~0.909 ✅** |

- **Hyperparameter Tuning:** Optuna (30 trials)
- **Tracking:** MLflow

---

### Notebook 10: Final LightGBM Optimization
**File:** `experiment-6-lightgbm-detailed-hpt.ipynb`

- **Pipeline:** TF-IDF (1,2), 1000 features, SMOTE
- **Optuna Trials:** 100

**Tuned Parameters:**
- `num_leaves`
- `learning_rate`
- `reg_alpha`, `reg_lambda`
- `subsample`, `colsample_bytree`

**Result:** Final optimized model saved and logged.

---

## 🏗️ Production Pipeline (Python Scripts)

### Stage 1: Data Ingestion
**File:** `data_ingestion.py`

- Fetches dataset from URL
- Cleans NaNs & duplicates
- Performs train-test split
- Saves data to `data/raw/`

---

### Stage 2: Data Preprocessing
**File:** `data_preprocessing.py`

- Text normalization
- Stopword removal
- Lemmatization
- Saves data to `data/interim/`

---

### Stage 3: Model Building
**File:** `model_building.py`

- TF-IDF vectorization
- LightGBM model training

**Artifacts Saved:**
- `tfidf_vectorizer.pkl`
- `lgbm_model.pkl`

---

### Stage 4: Model Evaluation
**File:** `model_evaluation.py`

- Loads model and vectorizer
- Logs metrics & artifacts to **MLflow (DagsHub)**
- Generates classification report and confusion matrix

---

### Stage 5: Model Registration
**File:** `register_model.py`

Registers the model as:

Automatically transitions the model to **Staging**.

---

## 📊 Tools & Tech Stack

- Python
- Scikit-learn
- LightGBM
- MLflow
- Optuna
- NLTK / SpaCy
- DagsHub

---
<img width="1033" height="546" alt="image" src="https://github.com/user-attachments/assets/5baec4f5-4f62-43c1-b9ec-a855794e262f" />
<img width="1021" height="688" alt="image" src="https://github.com/user-attachments/assets/6ba9bd09-e13c-443e-aebb-a8c58f244b47" />


## ✅ Final Outcome

- **Best Model:** LightGBM  
- **Accuracy:** ~**90.9%**
- **Production-ready ML pipeline**
- **Fully tracked experiments**
- **Model registry enabled**
