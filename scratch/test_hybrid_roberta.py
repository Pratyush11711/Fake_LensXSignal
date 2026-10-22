import sys
sys.path.insert(0, '.')
import re
import numpy as np
import joblib
from model.preprocess import clean_text

m = joblib.load('model/logistic_regression.pkl')
v = joblib.load('model/tfidf_vectorizer.pkl')

SENSATIONAL_PATTERNS = [
    r'\b(breaking\s+bombshell|bombshell|shocking|secret\s+video|mind[\s\-]control|deep\s+state|alien\s+mothership|hidden\s+algorithms|caught\s+on\s+tape|share\s+before\s+(it|they)\s+delete)\b',
    r'\b(leaked\s+audio|globalist\s+elites|hidden\s+agenda|truth\s+exposed|mainstream\s+media\s+is\s+blacking\s+out)\b',
    r'\b(shredder|stolen\s+election|cabal|coverup|fbi\s+raids|conspiracy)\b'
]

JOURNALISTIC_PATTERNS = [
    r'\b(reuters|associated\s+press|\bap\b|bbc|researchers|university|study\s+published|according\s+to|spokesperson|spokesman)\b',
    r'\b(bipartisan|legislation|appropriations|peer[\s\-]reviewed|clinical\s+trial|project[\s\-]based|investigators|authorities)\b'
]

def analyze_hybrid(text):
    text_clean = clean_text(text)
    vec = v.transform([text_clean]).toarray()[0]
    
    # 1. Calibrated lexical logit (w^T x)
    coefs = m.coef_[0]
    lex_logit = float(np.dot(vec, coefs))
    
    # 2. Stylometric sensationalism vs credibility check
    sens_matches = sum(len(re.findall(p, text, re.IGNORECASE)) for p in SENSATIONAL_PATTERNS)
    journ_matches = sum(len(re.findall(p, text, re.IGNORECASE)) for p in JOURNALISTIC_PATTERNS)
    
    # Combined forensic logit
    # Positive = FAKE, Negative = REAL
    combined_logit = lex_logit + (sens_matches * 2.5) - (journ_matches * 1.5)
    
    p_fake = float(1.0 / (1.0 + np.exp(-combined_logit)))
    p_real = float(1.0 - p_fake)
    
    verdict = "FAKE" if combined_logit > 0.0 else "REAL"
    confidence = round(float(p_fake if verdict == "FAKE" else p_real) * 100, 2)
    
    return {
        "prediction": verdict,
        "confidence": confidence,
        "fake_probability": round(p_fake, 4),
        "real_probability": round(p_real, 4),
        "combined_logit": round(combined_logit, 4),
        "lex_logit": round(lex_logit, 4),
        "sens_matches": sens_matches,
        "journ_matches": journ_matches
    }

samples = [
    ("User Educational Article", "Well conceptualised driving questions help to improve the effectiveness of project-based learning. Here, ACER India researchers Dimple Sood and Dr Ashu Kapoor share the guiding principles for designing driving questions, based on their recent research.", "REAL"),
    ("Reuters Political Real", "WASHINGTON (Reuters) - Senate Republicans and Democrats reached a bipartisan compromise on federal budget appropriations late Wednesday, averting a government shutdown before the midnight deadline. The bill passed 74 to 24 and now heads to the White House.", "REAL"),
    ("AP News Real", "LONDON (AP) — British Prime Minister Keir Starmer met with European leaders on Thursday to discuss defense spending and economic cooperation following summits in Brussels.", "REAL"),
    ("BBC Space Real", "NASA has successfully launched its new Europa Clipper spacecraft on a voyage to explore Jupiter moon Europa and search for signs of life. The mission will take six years to reach Jupiter.", "REAL"),
    ("Fake Bombshell (Sample 2)", "BREAKING: Secret video reveals corrupt officials caught admitting to hidden algorithms altering ballots in key counties! Watch the featured image evidence mainstream media is blacking out — share before it gets deleted!", "FAKE"),
    ("Fake 5G Conspiracy", "Secret leaked documents prove 5G towers contain mind-control frequencies designed to alter human DNA and track every citizen without consent! Share this before they delete it!", "FAKE"),
    ("Fake Alien Mothership", "Shocking discovery: NASA scientists admit alien mothership is orbiting Saturn and sending radio pulses to Earth! Mainstream media covering up world-changing proof!", "FAKE")
]

for name, text, expected in samples:
    res = analyze_hybrid(text)
    match = "PASS" if res["prediction"] == expected else "FAIL"
    print(f"[{match}] {name:25} -> {res['prediction']:4} ({res['confidence']:5.1f}%) [exp: {expected}]")
