import joblib
import numpy as np
import re
import os

# ── Absolute path fix ─────────────────────────────────────────
BASE_DIR    = os.path.dirname(os.path.abspath(__file__))
MODELS_DIR  = os.path.join(BASE_DIR, "models")

model       = joblib.load(os.path.join(MODELS_DIR, "spam_classifier.joblib"))
scaler      = joblib.load(os.path.join(MODELS_DIR, "scaler.joblib"))
vocabulary  = joblib.load(os.path.join(MODELS_DIR, "vocabulary.joblib"))
VOCAB_INDEX = {w: i for i, w in enumerate(vocabulary)}

def email_to_features(text):
    words  = re.findall(r'[a-z]+', text.lower())
    vector = np.zeros(len(vocabulary), dtype=np.float64)
    for w in words:
        if w in VOCAB_INDEX:
            vector[VOCAB_INDEX[w]] += 1
    return vector.reshape(1, -1)

def predict(text):
    features   = email_to_features(text)
    features_s = scaler.transform(features)
    prediction = model.predict(features_s)[0]

    confidence = None
    if hasattr(model, 'predict_proba'):
        proba      = model.predict_proba(features_s)[0]
        confidence = round(float(proba[prediction]) * 100, 2)

    matched = [w for w in re.findall(r'[a-z]+', text.lower())
               if w in VOCAB_INDEX]

    return {
        "label"        : "SPAM" if prediction == 1 else "HAM",
        "confidence"   : confidence,
        "vocab_hits"   : len(matched),
        "matched_words": matched[:20]
    }