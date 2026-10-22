"""
FakeLens - Model Training, Evaluation & Interpretability Pipeline
Implements:
1. Dataset loading (Fake.csv & True.csv) and validation
2. Text cleaning and preprocessing (shared module)
3. Stratified Train/Test split (80/20) executed BEFORE TF-IDF to avoid leakage
4. TF-IDF vectorization (ngram_range=(1,2), sublinear_tf=True)
5. Logistic Regression linear classification (interpretable baseline)
6. Evaluation: Accuracy, Precision, Recall, F1 (FAKE as positive class)
7. Confusion Matrix generation with TN, FP, FN, TP annotations
8. Dataset Source-Leakage analysis (original vs dateline-stripped)
9. Error analysis (misclassified samples)
10. Feature interpretability (top positive/negative coefficients)
"""

import os
import sys
import json
import joblib
import argparse
import numpy as np
import pandas as pd
import matplotlib
matplotlib.use('Agg')  # Non-interactive backend for generating plots
import matplotlib.pyplot as plt
import seaborn as sns

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

# Add parent directory to path to import preprocess module
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.preprocess import clean_text


def print_banner(msg: str):
    print("\n" + "=" * 60)
    print(f" {msg}")
    print("=" * 60)


def check_dataset_exists(data_dir: str = "dataset"):
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("\n" + "!" * 60)
        print(" [DATASET NOT FOUND]")
        print("!" * 60)
        print(" FakeLens requires the Kaggle Fake and Real News dataset:")
        print(f"  - Missing: {fake_path if not os.path.exists(fake_path) else ''}")
        print(f"  - Missing: {true_path if not os.path.exists(true_path) else ''}")
        print("\n Instructions:")
        print("  1. Download 'Fake.csv' and 'True.csv' from Kaggle or ISOT dataset.")
        print(f"  2. Place both CSV files in the '{data_dir}/' directory.")
        print("  3. Run 'python model/train_model.py' again.")
        print("!" * 60 + "\n")
        return False, fake_path, true_path
    
    return True, fake_path, true_path


def load_and_inspect_dataset(fake_path: str, true_path: str):
    print_banner("1. LOADING & INSPECTING DATASET")
    
    print(f"Loading Fake news from: {fake_path}")
    df_fake = pd.read_csv(fake_path)
    print(f" -> Fake news shape: {df_fake.shape}")
    
    print(f"Loading Real news from: {true_path}")
    df_true = pd.read_csv(true_path)
    print(f" -> Real news shape: {df_true.shape}")
    
    # Assign labels: 1 = FAKE (positive class), 0 = REAL
    df_fake['label'] = 1
    df_fake['label_name'] = 'FAKE'
    
    df_true['label'] = 0
    df_true['label_name'] = 'REAL'
    
    # Combine datasets
    df = pd.concat([df_fake, df_true], ignore_index=True)
    print(f"\nCombined dataset total rows: {len(df)}")
    
    # Combine title and text for rich context
    title_col = 'title' if 'title' in df.columns else None
    text_col = 'text' if 'text' in df.columns else None
    
    if title_col and text_col:
        df['raw_content'] = df['title'].fillna('') + " " + df['text'].fillna('')
    elif text_col:
        df['raw_content'] = df['text'].fillna('')
    else:
        df['raw_content'] = df.iloc[:, 0].fillna('')
        
    initial_count = len(df)
    
    # Inspect missing values
    missing_counts = df.isnull().sum().to_dict()
    print(f"Missing values per column: {missing_counts}")
    
    # Drop rows where raw_content is empty
    df = df[df['raw_content'].str.strip() != ''].reset_index(drop=True)
    
    # Duplicate inspection and removal
    duplicate_count = df.duplicated(subset=['raw_content']).sum()
    print(f"Duplicate articles found: {duplicate_count}")
    df = df.drop_duplicates(subset=['raw_content']).reset_index(drop=True)
    print(f"Dataset size after deduplication: {len(df)}")
    
    # Class distribution
    class_counts = df['label_name'].value_counts().to_dict()
    print(f"Class distribution: {class_counts}")
    
    # Save dataset statistics
    stats = {
        "fake_raw_count": int(len(df_fake)),
        "true_raw_count": int(len(df_true)),
        "total_raw_count": int(initial_count),
        "duplicates_removed": int(duplicate_count),
        "total_cleaned_count": int(len(df)),
        "fake_cleaned_count": int(class_counts.get('FAKE', 0)),
        "real_cleaned_count": int(class_counts.get('REAL', 0)),
        "columns": list(df.columns),
        "subjects": df['subject'].value_counts().to_dict() if 'subject' in df.columns else {}
    }
    
    os.makedirs("model", exist_ok=True)
    with open("model/dataset_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
        
    return df, stats


def train_pipeline(df: pd.DataFrame, strip_source_for_leakage_study: bool = True):
    print_banner("2. PREPROCESSING & TRAIN/TEST SPLIT")
    
    # Clean text using standard preprocessing
    print("Applying text preprocessing (lowercasing, url/html stripping, regex normalization)...")
    df['cleaned_text'] = df['raw_content'].apply(lambda t: clean_text(t, strip_source=False))
    
    # Stratified 80/20 train/test split - executed BEFORE vectorization
    print("Executing Stratified 80/20 Train-Test split (random_state=42)...")
    train_df, test_df = train_test_split(
        df,
        test_size=0.20,
        random_state=42,
        stratify=df['label']
    )
    
    print(f"Training set size: {len(train_df)} (Fake: {(train_df['label']==1).sum()}, Real: {(train_df['label']==0).sum()})")
    print(f"Testing set size:  {len(test_df)} (Fake: {(test_df['label']==1).sum()}, Real: {(test_df['label']==0).sum()})")
    
    # Save split indices for exact reproducibility in future RoBERTa model
    split_info = {
        "random_state": 42,
        "test_size": 0.20,
        "train_indices": train_df.index.tolist(),
        "test_indices": test_df.index.tolist()
    }
    with open("model/split_indices.json", "w", encoding="utf-8") as f:
        json.dump(split_info, f)
    print("Saved split indices to 'model/split_indices.json' for future RoBERTa comparison.")
    
    print_banner("3. TF-IDF VECTORIZATION")
    # TF-IDF configured with unigrams + bigrams, sublinear TF scaling
    vectorizer = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
        stop_words='english'
    )
    
    # FIT TF-IDF ONLY ON TRAINING DATA to prevent test leakage
    print("Fitting TfidfVectorizer on training set only...")
    X_train_tfidf = vectorizer.fit_transform(train_df['cleaned_text'])
    print(f"Transforming test set with fitted vectorizer... (TF-IDF Shape: {X_train_tfidf.shape})")
    X_test_tfidf = vectorizer.transform(test_df['cleaned_text'])
    
    y_train = train_df['label'].values
    y_test = test_df['label'].values
    
    print_banner("4. TRAINING LOGISTIC REGRESSION CLASSIFIER")
    # Logistic Regression linear model
    clf = LogisticRegression(
        max_iter=1000,
        random_state=42,
        C=1.0,
        solver='lbfgs'
    )
    clf.fit(X_train_tfidf, y_train)
    print("Model trained successfully.")
    
    # Predictions & Probabilities
    y_pred = clf.predict(X_test_tfidf)
    y_pred_proba = clf.predict_proba(X_test_tfidf)[:, 1]
    
    print_banner("5. MODEL EVALUATION (FAKE = Positive Class)")
    acc = accuracy_score(y_test, y_pred)
    prec = precision_score(y_test, y_pred, pos_label=1)
    rec = recall_score(y_test, y_pred, pos_label=1)
    f1 = f1_score(y_test, y_pred, pos_label=1)
    cm = confusion_matrix(y_test, y_pred)
    tn, fp, fn, tp = cm.ravel()
    
    print(f"Accuracy:  {acc * 100:.2f}%")
    print(f"Precision: {prec * 100:.2f}% (Fake class)")
    print(f"Recall:    {rec * 100:.2f}% (Fake class)")
    print(f"F1 Score:  {f1 * 100:.2f}%")
    print("\nConfusion Matrix Breakdown:")
    print(f"  True Negatives  (REAL correctly predicted as REAL): {tn}")
    print(f"  False Positives (REAL incorrectly predicted as FAKE): {fp}")
    print(f"  False Negatives (FAKE incorrectly predicted as REAL): {fn}")
    print(f"  True Positives  (FAKE correctly predicted as FAKE): {tp}")
    
    report_dict = classification_report(
        y_test, y_pred, target_names=['REAL', 'FAKE'], output_dict=True
    )
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=['REAL', 'FAKE']))
    
    # Save trained model & vectorizer
    joblib.dump(clf, "model/logistic_regression.pkl")
    joblib.dump(vectorizer, "model/tfidf_vectorizer.pkl")
    print("Saved 'model/logistic_regression.pkl' and 'model/tfidf_vectorizer.pkl'.")
    
    # Save comprehensive metrics JSON
    metrics_data = {
        "model_name": "TF-IDF + Logistic Regression",
        "model_type": "Linear Classifier",
        "vectorizer_config": {
            "ngram_range": [1, 2],
            "max_features": 5000,
            "min_df": 5,
            "max_df": 0.7,
            "sublinear_tf": True,
            "stop_words": "english"
        },
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
        },
        "classification_report": report_dict,
        "test_set_size": int(len(test_df)),
        "train_set_size": int(len(train_df))
    }
    
    with open("model/metrics.json", "w", encoding="utf-8") as f:
        json.dump(metrics_data, f, indent=2)
        
    # Generate and save Confusion Matrix Plot
    save_confusion_matrix_plot(cm, tn, fp, fn, tp)
    
    # Interpretability: Extract Top Features
    extract_top_coefficients(vectorizer, clf)
    
    # Error Analysis: Misclassified Samples
    perform_error_analysis(test_df, y_test, y_pred, y_pred_proba)
    
    # Leakage Study
    if strip_source_for_leakage_study:
        perform_leakage_study(df, train_df.index, test_df.index, metrics_data)
        
    print_banner("TRAINING PIPELINE COMPLETE")
    return metrics_data


def save_confusion_matrix_plot(cm, tn, fp, fn, tp):
    """Generates an academic-quality annotated confusion matrix plot."""
    os.makedirs("model", exist_ok=True)
    os.makedirs("static/img", exist_ok=True)
    
    plt.figure(figsize=(7, 6), dpi=300)
    plt.style.use('dark_background')
    
    # Custom annotations with numbers and descriptions
    annot = np.array([
        [f"TN: {tn}\n(Real → Real)", f"FP: {fp}\n(Real → Fake)"],
        [f"FN: {fn}\n(Fake → Real)", f"TP: {tp}\n(Fake → Fake)"]
    ])
    
    sns.heatmap(
        cm,
        annot=annot,
        fmt="",
        cmap="Blues",
        cbar=False,
        xticklabels=['Predicted REAL', 'Predicted FAKE'],
        yticklabels=['Actual REAL', 'Actual FAKE'],
        linewidths=2,
        linecolor='#1e293b',
        annot_kws={"size": 13, "weight": "bold", "color": "white"}
    )
    
    plt.title("Confusion Matrix — FakeLens (TF-IDF + Logistic Regression)", fontsize=13, weight='bold', pad=15, color='#10b981')
    plt.xlabel("Predicted Class", fontsize=11, labelpad=10, color='#94a3b8')
    plt.ylabel("Actual Class", fontsize=11, labelpad=10, color='#94a3b8')
    plt.tight_layout()
    
    plt.savefig("model/confusion_matrix.png", dpi=300, facecolor='#0f172a')
    plt.savefig("static/img/confusion_matrix.png", dpi=300, facecolor='#0f172a')
    plt.close()
    print("Saved Confusion Matrix plot to 'model/confusion_matrix.png' and 'static/img/confusion_matrix.png'.")


def extract_top_coefficients(vectorizer, clf, top_n: int = 25):
    """Extracts top words/n-grams with highest positive and negative coefficients."""
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = clf.coef_[0]
    
    # Top features for FAKE (positive coefficients push toward y=1)
    fake_indices = np.argsort(coefs)[::-1][:top_n]
    top_fake_features = [
        {"feature": str(feature_names[i]), "coefficient": round(float(coefs[i]), 4)}
        for i in fake_indices
    ]
    
    # Top features for REAL (negative coefficients push toward y=0)
    real_indices = np.argsort(coefs)[:top_n]
    top_real_features = [
        {"feature": str(feature_names[i]), "coefficient": round(float(coefs[i]), 4)}
        for i in real_indices
    ]
    
    top_features_data = {
        "fake_indicative_features": top_fake_features,
        "real_indicative_features": top_real_features
    }
    
    with open("model/top_features.json", "w", encoding="utf-8") as f:
        json.dump(top_features_data, f, indent=2)
    print("Saved top interpretable features to 'model/top_features.json'.")


def perform_error_analysis(test_df: pd.DataFrame, y_test, y_pred, y_pred_proba):
    """Saves misclassified test examples for academic error analysis."""
    test_eval_df = test_df.copy()
    test_eval_df['actual_label'] = y_test
    test_eval_df['actual_class'] = np.where(y_test == 1, 'FAKE', 'REAL')
    test_eval_df['predicted_label'] = y_pred
    test_eval_df['predicted_class'] = np.where(y_pred == 1, 'FAKE', 'REAL')
    test_eval_df['fake_probability'] = np.round(y_pred_proba, 4)
    test_eval_df['confidence'] = np.where(
        y_pred == 1,
        np.round(y_pred_proba * 100, 2),
        np.round((1 - y_pred_proba) * 100, 2)
    )
    
    # Misclassified samples
    errors = test_eval_df[test_eval_df['actual_label'] != test_eval_df['predicted_label']].copy()
    
    # Sample False Positives (REAL predicted as FAKE)
    fps = errors[errors['actual_class'] == 'REAL'].head(10)
    # Sample False Negatives (FAKE predicted as REAL)
    fns = errors[errors['actual_class'] == 'FAKE'].head(10)
    
    error_summary = {
        "total_misclassified": int(len(errors)),
        "error_rate": round(float(len(errors) / len(test_df)), 4),
        "false_positive_count": int((errors['actual_class'] == 'REAL').sum()),
        "false_negative_count": int((errors['actual_class'] == 'FAKE').sum()),
        "sample_false_positives": [
            {
                "title": str(row.get('title', 'N/A')),
                "snippet": str(row['raw_content'])[:250] + "...",
                "actual": "REAL",
                "predicted": "FAKE",
                "confidence": float(row['confidence'])
            }
            for _, row in fps.iterrows()
        ],
        "sample_false_negatives": [
            {
                "title": str(row.get('title', 'N/A')),
                "snippet": str(row['raw_content'])[:250] + "...",
                "actual": "FAKE",
                "predicted": "REAL",
                "confidence": float(row['confidence'])
            }
            for _, row in fns.iterrows()
        ]
    }
    
    # Save CSV and JSON
    cols_to_save = ['title', 'actual_class', 'predicted_class', 'confidence', 'raw_content']
    existing_cols = [c for c in cols_to_save if c in errors.columns]
    errors[existing_cols].to_csv("model/error_analysis.csv", index=False)
    
    with open("model/error_analysis.json", "w", encoding="utf-8") as f:
        json.dump(error_summary, f, indent=2)
    print("Saved error analysis to 'model/error_analysis.csv' and 'model/error_analysis.json'.")


def perform_leakage_study(df: pd.DataFrame, train_idx, test_idx, original_metrics: dict):
    """
    Academic Dataset Source-Leakage Study:
    ISOT True.csv articles frequently contain Reuters dateline headers.
    This trains a second model on text stripped of Reuters/dateline patterns,
    comparing performance to quantify stylistic leakage.
    """
    print_banner("6. DATASET SOURCE-LEAKAGE STUDY (Dateline Stripping)")
    print("Cleaning text with 'strip_source=True' to remove Reuters/datelines...")
    df['stripped_text'] = df['raw_content'].apply(lambda t: clean_text(t, strip_source=True))
    
    train_df = df.loc[train_idx]
    test_df = df.loc[test_idx]
    
    # Fit TF-IDF on stripped train text
    vec_stripped = TfidfVectorizer(
        ngram_range=(1, 2),
        max_features=5000,
        min_df=5,
        max_df=0.7,
        sublinear_tf=True,
        stop_words='english'
    )
    X_train_str = vec_stripped.fit_transform(train_df['stripped_text'])
    X_test_str = vec_stripped.transform(test_df['stripped_text'])
    
    clf_str = LogisticRegression(max_iter=1000, random_state=42, C=1.0)
    clf_str.fit(X_train_str, train_df['label'].values)
    
    y_pred_str = clf_str.predict(X_test_str)
    
    acc_str = accuracy_score(test_df['label'].values, y_pred_str)
    prec_str = precision_score(test_df['label'].values, y_pred_str, pos_label=1)
    rec_str = recall_score(test_df['label'].values, y_pred_str, pos_label=1)
    f1_str = f1_score(test_df['label'].values, y_pred_str, pos_label=1)
    cm_str = confusion_matrix(test_df['label'].values, y_pred_str)
    tn_s, fp_s, fn_s, tp_s = cm_str.ravel()
    
    leakage_data = {
        "explanation": (
            "The ISOT Fake News Dataset contains stylistic source artifacts (e.g., 'Reuters' datelines in real articles). "
            "Comparing the model trained on raw text vs dateline-stripped text quantifies the extent to which the linear classifier "
            "relies on stylistic metadata vs semantic content."
        ),
        "original_model": {
            "dataset_version": "Original Raw Text (Contains Source Artifacts)",
            "accuracy": original_metrics["accuracy"],
            "precision": original_metrics["precision"],
            "recall": original_metrics["recall"],
            "f1_score": original_metrics["f1_score"],
            "confusion_matrix": original_metrics["confusion_matrix"]
        },
        "source_stripped_model": {
            "dataset_version": "Cleaned / Dateline-Stripped Text",
            "accuracy": round(float(acc_str), 4),
            "precision": round(float(prec_str), 4),
            "recall": round(float(rec_str), 4),
            "f1_score": round(float(f1_str), 4),
            "confusion_matrix": {
                "tn": int(tn_s),
                "fp": int(fp_s),
                "fn": int(fn_s),
                "tp": int(tp_s)
            }
        },
        "deltas": {
            "accuracy_change": round(float(acc_str - original_metrics["accuracy"]), 4),
            "f1_change": round(float(f1_str - original_metrics["f1_score"]), 4)
        }
    }
    
    with open("model/leakage_comparison.json", "w", encoding="utf-8") as f:
        json.dump(leakage_data, f, indent=2)
    print("Saved Leakage Comparison study to 'model/leakage_comparison.json'.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens ML Training Pipeline")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Directory containing Fake.csv and True.csv")
    args = parser.parse_args()
    
    exists, fake_path, true_path = check_dataset_exists(args.data_dir)
    if not exists:
        sys.exit(1)
        
    df, stats = load_and_inspect_dataset(fake_path, true_path)
    train_pipeline(df)
