"""
Hybrid Healthcare Knowledge+ML System
===================================
Single-file runnable example that:
 1. Loads and cleans a noisy CSV (attempts robust parsing)
 2. Extracts important features
 3. Builds a small rule-based symptom->condition knowledge base
 4. Trains a simple ML classifier to predict condition from a "symptom text"
 5. Combines rule-based candidates and ML probabilities to produce final responses

Notes
-----
- Adjust DATA_PATH to point to your CSV file.
- This script makes reasonable defaults for messy CSVs: normalizes columns, strips quotes,
  fixes common capitalization issues, coerces numeric columns, parses dates.
- The dataset in your screenshot contains a `Condition`-like field (e.g., Cancer, Obesity, Diabetes, Asthma).
  If your dataset contains a `Symptoms` column, the script will use it. If not, the script synthesizes
  simple symptom descriptions from conditions to bootstrap training for the proof-of-concept.

Requirements
------------
pip install pandas scikit-learn python-dateutil

Run
---
python hybrid_healthcare_system.py

"""

import re
import ast
from collections import defaultdict
import random
import json

import pandas as pd
import numpy as np
from dateutil.parser import parse as dateparse

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report

# -----------------------------
# CONFIG
# -----------------------------
DATA_PATH = "healthcare_dataset.csv"  # <- change to your file path
RANDOM_SEED = 42
SYNTHETIC_SYMPTOMS_PER_CONDITION = 6  # when dataset has no symptoms column

random.seed(RANDOM_SEED)
np.random.seed(RANDOM_SEED)

# -----------------------------
# UTILS: robust CSV loader & cleaner
# -----------------------------

def robust_read_csv(path):
    """Try multiple strategies to read a messy CSV file."""
    # Strategy 1: normal pandas read_csv
    try:
        df = pd.read_csv(path, dtype=str, encoding='utf-8')
        print(f"Loaded with pandas read_csv — shape={df.shape}")
        return df
    except Exception as e:
        print(f"pandas.read_csv failed: {e}")

    # Strategy 2: try python's csv sniffer via pandas (let pandas guess delimiter)
    try:
        df = pd.read_csv(path, sep=None, engine='python', dtype=str)
        print(f"Loaded with pandas read_csv (engine=python, sep=None) — shape={df.shape}")
        return df
    except Exception as e:
        print(f"read_csv(engine=python) failed: {e}")

    raise RuntimeError("Unable to read CSV — try cleaning file or providing a different path.")


def normalize_colname(c):
    if not isinstance(c, str):
        return c
    c = c.strip()
    c = c.replace('\ufeff', '')
    c = re.sub(r"\s+", "_", c)
    c = c.lower()
    return c


def clean_dataframe(df):
    # Normalize column names
    df.columns = [normalize_colname(c) for c in df.columns]

    # Trim whitespace from string values and remove stray quotes
    for col in df.columns:
        if df[col].dtype == object:
            df[col] = df[col].astype(str).str.strip()
            df[col] = df[col].str.strip('\"\' )
            df[col] = df[col].replace({'nan': None})

    # Try to coerce certain columns
    # date_of_admission | billing_amount | age
    if 'date_of_admission' in df.columns:
        def try_parse_date(x):
            try:
                return dateparse(x).date()
            except Exception:
                return pd.NaT
        df['date_of_admission'] = df['date_of_admission'].apply(lambda x: try_parse_date(x) if pd.notna(x) else pd.NaT)

    for numeric_col in ['billing_amount', 'age']:
        if numeric_col in df.columns:
            df[numeric_col] = df[numeric_col].astype(str).str.replace('[^0-9.-]', '', regex=True)
            df[numeric_col] = pd.to_numeric(df[numeric_col], errors='coerce')

    # Normalize gender strings
    if 'gender' in df.columns:
        df['gender'] = df['gender'].str.lower().str.replace('[^a-z]', '', regex=True)
        df['gender'] = df['gender'].replace({'m': 'male', 'f': 'female'})

    # Normalize condition column name variations
    cond_col = None
    for candidate in ['condition', 'disease', 'diagnosis']:
        if candidate in df.columns:
            cond_col = candidate
            break
    if cond_col is None:
        # attempt to guess by finding a column with many repeated disease-like tokens
        for col in df.columns:
            sample = ' '.join(df[col].dropna().astype(str).head(50).tolist()).lower()
            if any(word in sample for word in ['cancer', 'diabetes', 'asthma', 'obesity']):
                cond_col = col
                print(f"Guessed condition column as '{col}' based on content.")
                break

    if cond_col is not None:
        df = df.rename(columns={cond_col: 'condition'})
        df['condition'] = df['condition'].astype(str).str.strip().str.title()

    # If there's a symptoms column, normalize it
    if 'symptoms' in df.columns:
        df['symptoms'] = df['symptoms'].astype(str).str.strip().str.lower()

    return df

# -----------------------------
# Build a tiny knowledge base of symptoms for common conditions
# (This is an example — you should expand/replace these by domain knowledge or clinical sources)
# -----------------------------

DEFAULT_KB = {
    'Asthma': [
        'shortness of breath', 'wheezing', 'chest tightness', 'coughing', 'difficulty breathing'
    ],
    'Diabetes': [
        'increased thirst', 'frequent urination', 'fatigue', 'blurred vision', 'slow healing'
    ],
    'Cancer': [
        'lump or mass', 'unexplained weight loss', 'fatigue', 'pain', 'skin changes'
    ],
    'Obesity': [
        'weight gain', 'fatigue', 'shortness of breath', 'joint pain'
    ],
    'Hypertension': [
        'headache', 'dizziness', 'blurred vision', 'shortness of breath'
    ],
    'Heart Disease': [
        'chest pain', 'shortness of breath', 'fatigue', 'palpitations'
    ]
}


def kb_find_candidates(symptom_text, kb=DEFAULT_KB, top_k=3):
    """Return candidate conditions from rule-based KB by simple token overlap scoring."""
    text = symptom_text.lower()
    scores = []
    for cond, syns in kb.items():
        score = 0
        for s in syns:
            # count token overlap
            for token in s.split():
                if token in text:
                    score += 1
        # also increase score if full symptom phrase appears
        for phrase in syns:
            if phrase in text:
                score += 2
        scores.append((cond, score))
    scores.sort(key=lambda x: x[1], reverse=True)
    return [c for c, s in scores if s > 0][:top_k]

# -----------------------------
# Prepare labeled data for ML (predict condition from symptom text)
# If 'symptoms' column exists -> use it. Otherwise synthesize symptom examples from KB
# -----------------------------

def prepare_training_data(df, kb=DEFAULT_KB):
    rows = []
    if 'symptoms' in df.columns and df['symptoms'].notna().sum() > 20:
        print('Using existing symptoms column for training...')
        subset = df[df['symptoms'].notna() & df['condition'].notna()]
        for _, r in subset.iterrows():
            rows.append((r['symptoms'], r['condition']))
    else:
        print('No usable symptoms column found — synthesizing examples from KB for training...')
        # create multiple variants per condition (paraphrase-ish by mixing phrases)
        for cond, syns in kb.items():
            for i in range(SYNTHETIC_SYMPTOMS_PER_CONDITION):
                sample = ' and '.join(random.sample(syns, min(2, len(syns))))
                # add small noise
                if random.random() < 0.4:
                    sample += ' with ' + random.choice(['fatigue', 'cough', 'fever', 'pain'])
                rows.append((sample, cond))

    df_train = pd.DataFrame(rows, columns=['symptom_text', 'condition'])
    return df_train

# -----------------------------
# Train an ML classifier
# -----------------------------

def train_text_classifier(df_train):
    X = df_train['symptom_text'].astype(str).values
    y = df_train['condition'].astype(str).values

    vect = TfidfVectorizer(ngram_range=(1,2), max_features=2000)
    Xv = vect.fit_transform(X)

    Xtr, Xte, ytr, yte = train_test_split(Xv, y, test_size=0.2, random_state=RANDOM_SEED, stratify=y)

    clf = LogisticRegression(max_iter=1000)
    clf.fit(Xtr, ytr)
    ypred = clf.predict(Xte)
    print('\n=== Classification report on held-out set ===')
    print(classification_report(yte, ypred))

    return vect, clf

# -----------------------------
# Hybrid inference: combine KB rules + ML probabilities
# -----------------------------

def hybrid_infer(symptom_text, vect, clf, kb=DEFAULT_KB, alpha=0.6, top_k=3):
    """
    Hybrid scoring: final_score = alpha * rule_score_norm + (1-alpha) * ml_prob_norm
    - rule_score: token/phrase overlap from KB
    - ml_prob: classifier predicted probability for each label
    Returns ranked list of (condition, score, explanation)
    """
    # Rule candidates and raw rule scores
    # We'll compute raw rule scores for all KB labels
    rule_scores = {}
    for cond, syns in kb.items():
        score = 0
        t = symptom_text.lower()
        for phrase in syns:
            if phrase in t:
                score += 3
            # count token overlap
            for token in phrase.split():
                if token in t:
                    score += 1
        rule_scores[cond] = score

    max_rule = max(rule_scores.values()) if rule_scores else 1
    # normalize
    rule_norm = {k: v / max_rule for k, v in rule_scores.items()} if max_rule > 0 else {k: 0.0 for k in rule_scores}

    # ML probs
    Xv = vect.transform([symptom_text])
    if hasattr(clf, 'predict_proba'):
        labels = clf.classes_
        probs = clf.predict_proba(Xv)[0]
        ml_probs = dict(zip(labels, probs))
    else:
        # fallback: get decision_function scores
        labels = clf.classes_
        scores = clf.decision_function(Xv)[0]
        # convert to pseudo-probs
        exps = np.exp(scores - np.max(scores))
        probs = exps / exps.sum()
        ml_probs = dict(zip(labels, probs))

    # normalize ml_probs to include all KB labels (zeros for missing)
    all_labels = sorted(set(list(kb.keys()) + list(ml_probs.keys())))
    ml_norm = {lab: ml_probs.get(lab, 0.0) for lab in all_labels}

    # combine
    combined = []
    for lab in all_labels:
        r = rule_norm.get(lab, 0.0)
        m = ml_norm.get(lab, 0.0)
        final = alpha * r + (1 - alpha) * m
        explanation = f"rule={r:.3f}, ml={m:.3f}"
        combined.append((lab, final, explanation))

    combined.sort(key=lambda x: x[1], reverse=True)
    return combined[:top_k]

# -----------------------------
# Generate textual response for the user (knowledge-style)
# -----------------------------

def generate_response(symptom_text, candidates):
    if not candidates:
        return "I couldn't find a likely condition based on the symptoms provided. Please give more detail or contact healthcare professional."
    top = candidates[0]
    cond, score, explain = top
    lines = []
    lines.append(f"Most likely condition: {cond} (score={score:.3f})")
    lines.append(f"Explanation: {explain}")
    # add simple KB-backed advice if available
    if cond in DEFAULT_KB:
        advice = DEFAULT_KB[cond][:3]
        lines.append("Common related symptoms or signs: " + ', '.join(advice))
        lines.append("Suggested next steps: review symptoms with a clinician; consider relevant tests (bloodwork/imaging) depending on clinical judgement.")
    return '\n'.join(lines)

# -----------------------------
# MAIN flow
# -----------------------------

def main():
    df_raw = robust_read_csv(DATA_PATH)
    df = clean_dataframe(df_raw)

    print('\nColumns after cleaning:', df.columns.tolist())
    print('Sample rows:')
    print(df.head(3).to_string(index=False))

    df_train = prepare_training_data(df)
    print(f"Training examples: {len(df_train)}")

    vect, clf = train_text_classifier(df_train)

    # Example interactive loop (simple demo)
    while True:
        print('\nEnter patient symptoms (or type "quit" to exit):')
        symptom_text = input('> ').strip()
        if not symptom_text or symptom_text.lower() in ['quit', 'exit']:
            break
        candidates = hybrid_infer(symptom_text, vect, clf, kb=DEFAULT_KB, alpha=0.6, top_k=4)
        resp = generate_response(symptom_text, candidates)
        print('\n' + resp + '\n')

if __name__ == '__main__':
    main()
    