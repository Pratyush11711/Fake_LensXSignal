"""
FakeLens Text Preprocessing Module
Provides reusable text cleaning and normalization functions for training,
EDA, leakage analysis, and real-time inference.
"""

import re
import string
import numpy as np

# Regular expressions compiled for high performance
HTML_TAG_PATTERN = re.compile(r"<.*?>", re.DOTALL)
URL_PATTERN = re.compile(r"https?://\S+|www\.\S+", re.IGNORECASE)
EMAIL_PATTERN = re.compile(r"\b[A-Za-z0-9._%+-]+@[A-Za-z0-9.-]+\.[A-Z|a-z]{2,}\b")

# Regex to detect and strip Reuters/dateline source artifacts (for ISOT dataset leakage analysis)
# Matches: "WASHINGTON (Reuters) -", "LONDON (Reuters) -", "Reuters -", "(Reuters)", etc.
REUTERS_DATELINE_PATTERN = re.compile(
    r"^\s*[A-Z\s,/\.\(\)]*\(Reuters\)\s*[-—–:]\s*|\b(Reuters)\b",
    re.IGNORECASE
)
GENERIC_DATELINE_PATTERN = re.compile(
    r"^[A-Z\s]{2,25}\s*\([A-Za-z\s]+\)\s*[-—–:]\s*"
)


def clean_text(
    text: str,
    strip_source: bool = False,
    lower: bool = True,
    remove_special_chars: bool = True
) -> str:
    """
    Standard text preprocessing pipeline.
    
    Args:
        text (str): Raw input text (headline or full article).
        strip_source (bool): If True, strips datelines / Reuters source indicators
                             to mitigate dataset style leakage.
        lower (bool): Whether to convert text to lowercase.
        remove_special_chars (bool): Whether to strip non-alphanumeric punctuation.
        
    Returns:
        str: Cleaned, normalized text string.
    """
    if text is None:
        return ""
    
    # Ensure string type
    text = str(text)
    
    # 1. Remove HTML tags
    text = HTML_TAG_PATTERN.sub(" ", text)
    
    # 2. Remove URLs and Email addresses
    text = URL_PATTERN.sub(" ", text)
    text = EMAIL_PATTERN.sub(" ", text)
    
    # 3. Optional dateline / source artifact stripping (for leakage study)
    if strip_source:
        text = REUTERS_DATELINE_PATTERN.sub(" ", text)
        text = GENERIC_DATELINE_PATTERN.sub(" ", text)
        
    # 4. Lowercase conversion
    if lower:
        text = text.lower()
        
    # 5. Normalize quotation marks and dashes
    text = re.sub(r'[\u2018\u2019\u201A\u201B`]', "'", text)
    text = re.sub(r'[\u201C\u201D\u201E\u201F"]', '"', text)
    text = re.sub(r'[\u2013\u2014]', '-', text)
    
    # 6. Remove unnecessary punctuation/special characters (preserving standard word characters)
    if remove_special_chars:
        # Keep letters, numbers, hyphens, and whitespace
        text = re.sub(r"[^a-zA-Z0-9\s\-]", " ", text)
        
    # 7. Normalize multiple whitespaces and newlines
    text = re.sub(r"\s+", " ", text).strip()
    
    return text


def extract_contributing_features(text: str, vectorizer, model, top_k: int = 8):
    """
    Explainability helper for Logistic Regression + TF-IDF.
    Extracts the n-grams present in the input text that have the highest positive
    (contributing to FAKE) or negative (contributing to REAL) coefficients.
    
    Args:
        text (str): Raw or preprocessed input text.
        vectorizer (TfidfVectorizer): Trained TF-IDF vectorizer.
        model (LogisticRegression): Trained Logistic Regression model.
        top_k (int): Number of top contributing terms to return.
        
    Returns:
        list of dict: [{"term": str, "weight": float, "influence": "FAKE"|"REAL"}]
    """
    if not hasattr(vectorizer, 'transform') or not hasattr(model, 'coef_'):
        return []
    
    cleaned = clean_text(text, strip_source=False)
    if not cleaned:
        return []
    
    # Transform single document
    tfidf_matrix = vectorizer.transform([cleaned])
    
    # Get non-zero feature indices in this document
    feature_indices = tfidf_matrix.nonzero()[1]
    if len(feature_indices) == 0:
        return []
    
    feature_names = np.array(vectorizer.get_feature_names_out())
    coefs = model.coef_[0]  # Shape (n_features,)
    
    contributions = []
    for idx in feature_indices:
        tfidf_val = tfidf_matrix[0, idx]
        coef_val = coefs[idx]
        score = float(tfidf_val * coef_val)
        term = str(feature_names[idx])
        influence = "FAKE" if coef_val > 0 else "REAL"
        contributions.append({
            "term": term,
            "weight": round(score, 4),
            "coefficient": round(float(coef_val), 4),
            "tfidf": round(float(tfidf_val), 4),
            "influence": influence
        })
        
    # Sort by absolute contribution score
    contributions.sort(key=lambda x: abs(x["weight"]), reverse=True)
    return contributions[:top_k]
