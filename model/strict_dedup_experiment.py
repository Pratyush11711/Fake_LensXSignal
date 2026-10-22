"""
FakeLens - Strict Deduplication & Near-Duplicate Contamination Experiment
Implements a strict, multi-stage deduplication pipeline:
1. Exact text deduplication
2. Normalized title deduplication
3. Normalized article body deduplication
4. Near-duplicate prefix/signature deduplication
5. Cross-split contamination verification (0 title & 0 prefix overlap)
6. Stratified 80/20 Train-Test split executed BEFORE TF-IDF fitting
7. Fitting TF-IDF ONLY on training split
8. Training Logistic Regression and computing real metrics
"""

import os
import sys
import json
import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import (
    accuracy_score,
    precision_score,
    recall_score,
    f1_score,
    classification_report,
    confusion_matrix
)

# Add parent directory
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.preprocess import clean_text


def run_strict_dedup_experiment(data_dir: str = "dataset"):
    print("=" * 70)
    print(" STRICT DEDUPLICATION & NEAR-DUPLICATE ELIMINATION EXPERIMENT")
    print("=" * 70)
    
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")
    
    # 1. Load Raw Data
    df_fake = pd.read_csv(fake_path)
    df_fake['label'] = 1
    df_fake['label_name'] = 'FAKE'
    
    df_true = pd.read_csv(true_path)
    df_true['label'] = 0
    df_true['label_name'] = 'REAL'
    
    df_raw = pd.concat([df_fake, df_true], ignore_index=True)
    raw_total = len(df_raw)
    print(f"[Stage 0] Raw dataset loaded: {raw_total:,} articles (Fake: {len(df_fake):,}, Real: {len(df_true):,})")
    
    # 2. Stage 1: Exact raw text deduplication
    df_raw['raw_content'] = df_raw['title'].fillna('') + " " + df_raw['text'].fillna('')
    df_stage1 = df_raw.drop_duplicates(subset=['raw_content']).reset_index(drop=True)
    stage1_removed = raw_total - len(df_stage1)
    print(f"[Stage 1] Removed exact raw content duplicates: {stage1_removed:,} (Remaining: {len(df_stage1):,})")
    
    # 3. Stage 2: Normalized title and body cleaning
    print("[Stage 2] Normalizing text (lowercasing, punctuation stripping, whitespace collapsing)...")
    df_stage1['norm_title'] = df_stage1['title'].fillna('').apply(lambda t: clean_text(t, strip_source=False))
    df_stage1['norm_text'] = df_stage1['text'].fillna('').apply(lambda t: clean_text(t, strip_source=False))
    df_stage1['norm_full'] = df_stage1['norm_title'] + " " + df_stage1['norm_text']
    
    # Filter out empty or trivially short articles (< 10 words)
    df_stage2 = df_stage1[df_stage1['norm_text'].str.split().str.len() >= 10].reset_index(drop=True)
    short_removed = len(df_stage1) - len(df_stage2)
    print(f" -> Removed trivially short articles (< 10 words): {short_removed:,} (Remaining: {len(df_stage2):,})")
    
    # 4. Stage 3: Duplicate normalized article body removal
    df_stage3 = df_stage2.drop_duplicates(subset=['norm_text']).reset_index(drop=True)
    stage3_removed = len(df_stage2) - len(df_stage3)
    print(f"[Stage 3] Removed duplicate normalized article bodies: {stage3_removed:,} (Remaining: {len(df_stage3):,})")
    
    # 5. Stage 4: Duplicate normalized title removal (removes wire updates / headline duplicates)
    df_stage4 = df_stage3.drop_duplicates(subset=['norm_title']).reset_index(drop=True)
    stage4_removed = len(df_stage3) - len(df_stage4)
    print(f"[Stage 4] Removed duplicate normalized titles: {stage4_removed:,} (Remaining: {len(df_stage4):,})")
    
    # 6. Stage 5: Near-duplicate prefix signature deduplication (first 120 chars of body)
    df_stage4['prefix_signature'] = df_stage4['norm_text'].str[:120].str.strip()
    df_clean = df_stage4.drop_duplicates(subset=['prefix_signature']).reset_index(drop=True)
    stage5_removed = len(df_stage4) - len(df_clean)
    print(f"[Stage 5] Removed near-duplicates by 120-char prefix signature: {stage5_removed:,} (Remaining: {len(df_clean):,})")
    
    total_removed = raw_total - len(df_clean)
    print("\n" + "-" * 70)
    print(f" DEDUPLICATION COMPLETE: Removed {total_removed:,} articles ({total_removed/raw_total*100:.2f}%)")
    print(f" Final Cleaned Dataset Size: {len(df_clean):,} articles")
    print(f" Class Distribution: Real: {(df_clean['label']==0).sum():,} ({(df_clean['label']==0).sum()/len(df_clean)*100:.1f}%), Fake: {(df_clean['label']==1).sum():,} ({(df_clean['label']==1).sum()/len(df_clean)*100:.1f}%)")
    print("-" * 70)
    
    # 7. Stratified 80/20 Train-Test Split (EXECUTED BEFORE TF-IDF)
    print("\n[Stage 6] Executing Stratified 80/20 Train-Test split (random_state=42)...")
    train_df, test_df = train_test_split(
        df_clean,
        test_size=0.20,
        random_state=42,
        stratify=df_clean['label']
    )
    
    print(f" -> Training set: {len(train_df):,} samples (Fake: {(train_df['label']==1).sum():,}, Real: {(train_df['label']==0).sum():,})")
    print(f" -> Testing set:  {len(test_df):,} samples (Fake: {(test_df['label']==1).sum():,}, Real: {(test_df['label']==0).sum():,})")
    
    # 8. Cross-split contamination audit
    train_titles_set = set(train_df['norm_title'])
    test_titles_set = set(test_df['norm_title'])
    title_overlap = train_titles_set.intersection(test_titles_set)
    
    train_prefix_set = set(train_df['prefix_signature'])
    test_prefix_set = set(test_df['prefix_signature'])
    prefix_overlap = train_prefix_set.intersection(test_prefix_set)
    
    print("\n[Cross-Split Contamination Audit]:")
    print(f" -> Title overlap between Train and Test: {len(title_overlap)} articles")
    print(f" -> 120-char prefix signature overlap:    {len(prefix_overlap)} articles")
    assert len(title_overlap) == 0, "Error: Title overlap exists across train/test!"
    assert len(prefix_overlap) == 0, "Error: Prefix overlap exists across train/test!"
    print(" -> Cross-split contamination check: PASSED (Zero title and zero prefix overlap).")
    
    # 9. Fit TF-IDF ONLY on training split
    print("\n[Stage 7] Fitting TF-IDF Vectorizer strictly on X_train...")
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
        stop_words='english'
    )
    
    X_train_tfidf = vectorizer.fit_transform(train_df['norm_full'])
    X_test_tfidf = vectorizer.transform(test_df['norm_full'])
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    # 10. Train Logistic Regression
    print("[Stage 8] Training Logistic Regression Classifier (max_iter=1000, C=1.0)...")
    clf = LogisticRegression(max_iter=1000, random_state=42, C=1.0, solver='lbfgs')
    clf.fit(X_train_tfidf, y_train)
    
    # 11. Evaluate on Test Set (Positive Class = FAKE)
    y_pred = clf.predict(X_test_tfidf)
    y_pred_proba = clf.predict_proba(X_test_tfidf)[:, 1]
    
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print("\n" + "=" * 70)
    print(" STRICT EXPERIMENT EVALUATION METRICS (FAKE = Positive Class)")
    print("=" * 70)
    print(f" Accuracy:  {acc * 100:.2f}% ({acc:.4f})")
    print(f" Precision: {prec * 100:.2f}% ({prec:.4f})")
    print(f" Recall:    {rec * 100:.2f}% ({rec:.4f})")
    print(f" F1 Score:  {f1 * 100:.2f}% ({f1:.4f})")
    print("\nConfusion Matrix:")
    print(f"  True Negatives  (REAL -> REAL): {tn:,}")
    print(f"  False Positives (REAL -> FAKE): {fp:,}")
    print(f"  False Negatives (FAKE -> REAL): {fn:,}")
    print(f"  True Positives  (FAKE -> FAKE): {tp:,}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    # Save results to JSON
    strict_results = {
        "experiment_name": "Strict Deduplication & Near-Duplicate Controlled Experiment",
        "deduplication_summary": {
            "raw_total_articles": int(raw_total),
            "exact_duplicates_removed": int(stage1_removed),
            "short_articles_removed": int(short_removed),
            "duplicate_bodies_removed": int(stage3_removed),
            "duplicate_titles_removed": int(stage4_removed),
            "prefix_near_duplicates_removed": int(stage5_removed),
            "total_articles_removed": int(total_removed),
            "final_clean_articles": int(len(df_clean)),
            "real_articles_count": int((df_clean['label']==0).sum()),
            "fake_articles_count": int((df_clean['label']==1).sum()),
            "train_set_size": int(len(train_df)),
            "test_set_size": int(len(test_df)),
            "cross_split_title_overlap": int(len(title_overlap)),
            "cross_split_prefix_overlap": int(len(prefix_overlap))
        },
        "metrics": {
            "accuracy": round(float(acc), 4),
            "precision": round(float(prec), 4),
            "recall": round(float(rec), 4),
            "f1_score": round(float(f1), 4),
            "accuracy_pct": f"{acc * 100:.2f}%",
            "precision_pct": f"{prec * 100:.2f}%",
            "recall_pct": f"{rec * 100:.2f}%",
            "f1_pct": f"{f1 * 100:.2f}%",
            "confusion_matrix": {
                "tn": int(tn),
                "fp": int(fp),
                "fn": int(fn),
                "tp": int(tp),
                "matrix": cm.tolist()
            }
        },
        "comparison_with_original_benchmark": {
            "original_benchmark_accuracy": 0.9914,
            "original_benchmark_f1": 0.9906,
            "strict_experiment_accuracy": round(float(acc), 4),
            "strict_experiment_f1": round(float(f1), 4),
            "accuracy_difference": round(float(acc - 0.9914), 4),
            "f1_difference": round(float(f1 - 0.9906), 4)
        }
    }
    
    os.makedirs("model", exist_ok=True)
    with open("model/strict_dedup_metrics.json", "w", encoding="utf-8") as f:
        json.dump(strict_results, f, indent=2)
        
    split_info = {
        "random_state": 42,
        "test_size": 0.20,
        "train_indices": train_df.index.tolist(),
        "test_indices": test_df.index.tolist()
    }
    with open("model/strict_split_indices.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f)
        
    print("Saved experiment results to 'model/strict_dedup_metrics.json' and 'model/strict_split_indices.json'.")
    print("=" * 70)
    return strict_results


if __name__ == "__main__":
    run_strict_dedup_experiment()
