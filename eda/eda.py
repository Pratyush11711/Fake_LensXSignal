"""
FakeLens - Exploratory Data Analysis (EDA) Module
Generates descriptive statistics and publication-quality plots for academic viva and PPT presentations.
"""

import os
import sys
import json
import argparse
import numpy as np
import pandas as pd
from collections import Counter

import matplotlib
matplotlib.use('Agg')
import matplotlib.pyplot as plt
import seaborn as sns

# Add parent directory for preprocessing
sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from model.preprocess import clean_text


def run_eda(data_dir: str = "dataset"):
    fake_path = os.path.join(data_dir, "Fake.csv")
    true_path = os.path.join(data_dir, "True.csv")
    
    if not os.path.exists(fake_path) or not os.path.exists(true_path):
        print("\n" + "!" * 60)
        print(" [EDA ERROR: DATASET FILES MISSING]")
        print("!" * 60)
        print(f" Please place 'Fake.csv' and 'True.csv' inside '{data_dir}/'")
        print("!" * 60 + "\n")
        return None
        
    print(f"Loading Fake news from: {fake_path}")
    df_fake = pd.read_csv(fake_path)
    df_fake['label'] = 'FAKE'
    
    print(f"Loading Real news from: {true_path}")
    df_true = pd.read_csv(true_path)
    df_true['label'] = 'REAL'
    
    # Combined dataset
    df = pd.concat([df_fake, df_true], ignore_index=True)
    
    # Combined text for word counts
    df['full_text'] = df['title'].fillna('') + " " + df['text'].fillna('')
    df['word_count'] = df['full_text'].apply(lambda x: len(str(x).split()))
    df['char_count'] = df['full_text'].apply(lambda x: len(str(x)))
    df['title_length'] = df['title'].fillna('').apply(lambda x: len(str(x).split()))
    df['label_numeric'] = (df['label'] == 'FAKE').astype(int)

    import string
    punct_set = set(string.punctuation)

    def calc_upper_ratio(text):
        words = str(text).split()
        if not words:
            return 0.0
        upper_count = sum(1 for w in words if w.isupper() and len(w) > 1)
        return round(float(upper_count / len(words)), 4)

    def calc_punct_density(text):
        t = str(text)
        if not t:
            return 0.0
        punct_count = sum(1 for c in t if c in punct_set)
        return round(float(punct_count / len(t)), 4)

    df['uppercase_word_ratio'] = df['full_text'].apply(calc_upper_ratio)
    df['punctuation_density'] = df['full_text'].apply(calc_punct_density)

    # Numerical feature correlation matrix
    numerical_cols = [
        'word_count',
        'char_count',
        'title_length',
        'uppercase_word_ratio',
        'punctuation_density',
        'label_numeric'
    ]
    corr_matrix = df[numerical_cols].corr(method='pearson')

    # Output directories
    os.makedirs("eda/plots", exist_ok=True)
    os.makedirs("static/img/eda", exist_ok=True)
    
    # 1. Descriptive Statistics
    stats = {
        "dataset_shape": {
            "total_rows": int(len(df)),
            "total_columns": int(df.shape[1]),
            "columns": list(df.columns)
        },
        "class_distribution": {
            "fake_count": int(len(df_fake)),
            "real_count": int(len(df_true)),
            "fake_percentage": round(float(len(df_fake) / len(df) * 100), 2),
            "real_percentage": round(float(len(df_true) / len(df) * 100), 2)
        },
        "missing_values": df.isnull().sum().to_dict(),
        "duplicate_rows": int(df.duplicated().sum()),
        "duplicate_articles": int(df.duplicated(subset=['full_text']).sum()),
        "word_count_statistics": {
            "overall": {
                "mean": round(float(df['word_count'].mean()), 2),
                "median": round(float(df['word_count'].median()), 2),
                "std": round(float(df['word_count'].std()), 2),
                "min": int(df['word_count'].min()),
                "max": int(df['word_count'].max()),
                "p25": round(float(df['word_count'].quantile(0.25)), 2),
                "p75": round(float(df['word_count'].quantile(0.75)), 2),
                "p95": round(float(df['word_count'].quantile(0.95)), 2)
            },
            "fake_class": {
                "mean": round(float(df[df['label'] == 'FAKE']['word_count'].mean()), 2),
                "median": round(float(df[df['label'] == 'FAKE']['word_count'].median()), 2),
                "std": round(float(df[df['label'] == 'FAKE']['word_count'].std()), 2),
                "min": int(df[df['label'] == 'FAKE']['word_count'].min()),
                "max": int(df[df['label'] == 'FAKE']['word_count'].max())
            },
            "real_class": {
                "mean": round(float(df[df['label'] == 'REAL']['word_count'].mean()), 2),
                "median": round(float(df[df['label'] == 'REAL']['word_count'].median()), 2),
                "std": round(float(df[df['label'] == 'REAL']['word_count'].std()), 2),
                "min": int(df[df['label'] == 'REAL']['word_count'].min()),
                "max": int(df[df['label'] == 'REAL']['word_count'].max())
            }
        },
        "subject_distribution": df['subject'].value_counts().to_dict() if 'subject' in df.columns else {},
        "correlation_matrix_with_target": {
            col: round(float(corr_matrix.loc[col, 'label_numeric']), 4)
            for col in numerical_cols
        }
    }
    
    with open("eda/eda_stats.json", "w", encoding="utf-8") as f:
        json.dump(stats, f, indent=2)
    print("Saved descriptive stats to 'eda/eda_stats.json'.")
    
    # 2. Plotting Style Settings
    plt.style.use('dark_background')
    colors = {'FAKE': '#ef4444', 'REAL': '#10b981'}
    
    # -------------------------------------------------------------
    # Plot 1: FAKE vs REAL Class Distribution Bar Chart
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5), dpi=300)
    counts = df['label'].value_counts()
    bars = plt.bar(
        counts.index,
        counts.values,
        color=[colors.get(c, '#64748b') for c in counts.index],
        edgecolor='#1e293b',
        width=0.45
    )
    plt.title("Class Distribution (FAKE vs REAL)", fontsize=14, weight='bold', pad=15, color='white')
    plt.ylabel("Number of Articles", fontsize=11, color='#94a3b8')
    plt.xlabel("News Category", fontsize=11, color='#94a3b8')
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    
    # Annotate bar counts
    for bar in bars:
        yval = bar.get_height()
        pct = (yval / len(df)) * 100
        plt.text(
            bar.get_x() + bar.get_width() / 2.0,
            yval + (max(counts.values) * 0.02),
            f"{yval:,} ({pct:.1f}%)",
            ha='center',
            va='bottom',
            color='white',
            fontweight='bold',
            fontsize=10
        )
    plt.ylim(0, max(counts.values) * 1.15)
    plt.tight_layout()
    _save_plot("class_distribution.png")
    
    # -------------------------------------------------------------
    # Plot 2: Article Word Count Distribution Histogram / KDE
    # -------------------------------------------------------------
    plt.figure(figsize=(9, 5), dpi=300)
    # Clip extreme outliers (99th percentile) for clean presentation
    p99 = df['word_count'].quantile(0.99)
    df_clipped = df[df['word_count'] <= p99]
    
    sns.histplot(
        data=df_clipped,
        x='word_count',
        hue='label',
        palette=colors,
        kde=True,
        element='step',
        bins=50,
        alpha=0.4
    )
    plt.title("Article Word-Count Distribution (Clipped at 99th Percentile)", fontsize=14, weight='bold', pad=15, color='white')
    plt.xlabel("Word Count per Article", fontsize=11, color='#94a3b8')
    plt.ylabel("Frequency", fontsize=11, color='#94a3b8')
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.tight_layout()
    _save_plot("word_count_distribution.png")
    
    # -------------------------------------------------------------
    # Plot 3: Comparison of Article Lengths by Class (Box Plot)
    # -------------------------------------------------------------
    plt.figure(figsize=(7, 5), dpi=300)
    sns.boxplot(
        data=df_clipped,
        x='label',
        y='word_count',
        palette=colors,
        fliersize=2,
        width=0.4,
        boxprops=dict(alpha=0.85)
    )
    plt.title("Article Word-Length Comparison by Class", fontsize=14, weight='bold', pad=15, color='white')
    plt.xlabel("Class", fontsize=11, color='#94a3b8')
    plt.ylabel("Word Count", fontsize=11, color='#94a3b8')
    plt.grid(axis='y', linestyle='--', alpha=0.2)
    plt.tight_layout()
    _save_plot("article_length_by_class.png")
    
    # -------------------------------------------------------------
    # Plot 4: Top Frequent N-Grams by Class
    # -------------------------------------------------------------
    print("Computing top frequent n-grams for EDA...")
    fake_cleaned = df[df['label'] == 'FAKE']['full_text'].sample(min(3000, len(df_fake)), random_state=42).apply(lambda t: clean_text(t, strip_source=False))
    real_cleaned = df[df['label'] == 'REAL']['full_text'].sample(min(3000, len(df_true)), random_state=42).apply(lambda t: clean_text(t, strip_source=False))
    
    # Simple stopword list
    custom_stopwords = {'the', 'a', 'an', 'and', 'or', 'in', 'on', 'at', 'to', 'for', 'of', 'with', 'by', 'is', 'was', 'are', 'were', 'that', 'this', 'it', 'as', 'be', 'from', 'said', 'has', 'have', 'had', 'not', 'but', 'he', 'she', 'they', 'his', 'her', 'their', 'who', 'which', 'will', 'would', 'about', 'after', 'one', 'all', 'also', 'out', 'up', 'been', 'its', 'more', 'can', 'if', 'no', 'we', 'you', 'i', 'into', 'what', 'when'}
    
    def get_top_unigrams(corpus, top_k=10):
        words = []
        for doc in corpus:
            tokens = [w for w in doc.split() if w not in custom_stopwords and len(w) > 2]
            words.extend(tokens)
        return Counter(words).most_common(top_k)
        
    top_fake = get_top_unigrams(fake_cleaned, top_k=10)
    top_real = get_top_unigrams(real_cleaned, top_k=10)
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 5), dpi=300)
    
    # Fake top words
    y_fake = [item[0] for item in reversed(top_fake)]
    x_fake = [item[1] for item in reversed(top_fake)]
    axes[0].barh(y_fake, x_fake, color='#ef4444', alpha=0.85)
    axes[0].set_title("Top 10 Words in FAKE News", fontsize=12, weight='bold', color='white')
    axes[0].set_xlabel("Frequency (Sampled)", fontsize=10, color='#94a3b8')
    axes[0].grid(axis='x', linestyle='--', alpha=0.2)
    
    # Real top words
    y_real = [item[0] for item in reversed(top_real)]
    x_real = [item[1] for item in reversed(top_real)]
    axes[1].barh(y_real, x_real, color='#10b981', alpha=0.85)
    axes[1].set_title("Top 10 Words in REAL News", fontsize=12, weight='bold', color='white')
    axes[1].set_xlabel("Frequency (Sampled)", fontsize=10, color='#94a3b8')
    axes[1].grid(axis='x', linestyle='--', alpha=0.2)
    
    plt.tight_layout()
    _save_plot("top_ngrams_by_class.png")
    
    # -------------------------------------------------------------
    # Plot 5: Numerical Feature Correlation Heatmap
    # -------------------------------------------------------------
    plt.figure(figsize=(8, 6.5), dpi=300)
    corr_display = corr_matrix.copy()
    display_labels = ['Word Count', 'Char Count', 'Title Length', 'Upper Ratio', 'Punct Density', 'Label (Fake=1)']
    corr_display.columns = display_labels
    corr_display.index = display_labels
    
    sns.heatmap(
        corr_display,
        annot=True,
        fmt=".2f",
        cmap="coolwarm",
        vmin=-1,
        vmax=1,
        linewidths=1.5,
        linecolor='#1e293b',
        cbar_kws={'label': 'Pearson Correlation Coefficient'},
        annot_kws={"size": 10, "weight": "bold"}
    )
    plt.title("Correlation Heatmap of Text-Derived Numerical Features", fontsize=13, weight='bold', pad=15, color='white')
    plt.xticks(rotation=30, ha='right', fontsize=9, color='#94a3b8')
    plt.yticks(rotation=0, fontsize=9, color='#94a3b8')
    plt.tight_layout()
    _save_plot("correlation_heatmap.png")
    
    print("\nEDA Completed successfully. All plots generated in 'eda/plots/' and 'static/img/eda/'.")
    return stats


def _save_plot(filename: str):
    plt.savefig(f"eda/plots/{filename}", dpi=300, facecolor='#0f172a')
    plt.savefig(f"static/img/eda/{filename}", dpi=300, facecolor='#0f172a')
    plt.close()
    print(f" -> Generated plot: {filename}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="FakeLens EDA Generator")
    parser.add_argument("--data-dir", type=str, default="dataset", help="Dataset directory")
    args = parser.parse_args()
    run_eda(args.data_dir)
