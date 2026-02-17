# backend/app/drift_detector.py

"""
Domain Drift Detection Module (SE-SLM Requirement 3.7)

Drift shall be detected using:
- Embedding distribution shifts (via TF-IDF vectors)
- Vocabulary change rate
- Intent variance

Uses lightweight sklearn TF-IDF instead of transformer embeddings
for GGUF model compatibility.
"""

import numpy as np
from collections import Counter
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

# ============================================================
# In-memory drift state
# ============================================================

text_memory = []
vocab_memory = []
MAX_MEMORY = 20

# Lightweight TF-IDF vectorizer for semantic comparison
_vectorizer = TfidfVectorizer(max_features=500, stop_words="english")
_fitted = False


def initialize_memory(texts: list):
    """Load historical texts into memory to warm-start drift detection."""
    global text_memory, _fitted
    
    # Reset
    text_memory.clear()
    valid_texts = [t for t in texts if t and isinstance(t, str) and len(t.split()) > 2]
    
    # Keep only up to MAX_MEMORY
    text_memory.extend(valid_texts[-MAX_MEMORY:])
    
    # Pre-fit if we have enough data
    if len(text_memory) >= 2:
        try:
            _vectorizer.fit(text_memory)
            _fitted = True
            print(f"✅ Drift detector warmed up with {len(text_memory)} recent requests")
        except Exception as e:
            print(f"⚠️ Failed to warm-up drift detector: {e}")


# --------------------------------------------------
# TF-IDF Embedding Extraction
# --------------------------------------------------

def get_embedding(text):
    """Get TF-IDF vector for a text input."""
    global _fitted

    text_memory_snapshot = list(text_memory) + [text]

    if len(text_memory_snapshot) < 2:
        return np.zeros(500)

    try:
        vectors = _vectorizer.fit_transform(text_memory_snapshot)
        _fitted = True
        return vectors[-1].toarray().flatten()
    except Exception:
        return np.zeros(500)


# --------------------------------------------------
# Vocabulary Shift
# --------------------------------------------------

def compute_vocab_shift(text):
    tokens = text.lower().split()
    current = Counter(tokens)

    if len(vocab_memory) == 0:
        vocab_memory.append(current)
        return 0.0

    previous = vocab_memory[-1]
    diff = sum(abs(current[t] - previous.get(t, 0)) for t in current)
    total = sum(current.values()) + 1
    shift = diff / total

    vocab_memory.append(current)
    if len(vocab_memory) > MAX_MEMORY:
        vocab_memory.pop(0)

    return float(shift)


# --------------------------------------------------
# Intent Variance
# --------------------------------------------------

def compute_intent_variance():
    if len(text_memory) < 5:
        return 0.0

    recent_texts = text_memory[-5:]
    try:
        vectors = _vectorizer.fit_transform(recent_texts)
        return float(np.var(vectors.toarray()))
    except Exception:
        return 0.0


# --------------------------------------------------
# Main Drift Detection
# --------------------------------------------------

def detect_drift(text, threshold=0.35):
    """
    Drift score combines:
    - embedding semantic shift (TF-IDF cosine distance)
    - vocabulary shift
    - intent variance

    Returns: (drift_flag, drift_score, components)
    """

    # First request
    if len(text_memory) == 0:
        text_memory.append(text)
        return False, 0.0, {
            "embedding_shift": 0.0,
            "vocab_shift": 0.0,
            "intent_variance": 0.0
        }

    # --- Semantic shift via TF-IDF ---
    try:
        all_texts = list(text_memory) + [text]
        vectors = _vectorizer.fit_transform(all_texts)
        centroid = vectors[:-1].mean(axis=0)
        new_vec = vectors[-1]
        similarity = cosine_similarity(centroid, new_vec)[0][0]
        embedding_shift = max(0.0, 1.0 - similarity)
    except Exception:
        embedding_shift = 0.0

    # --- Vocab shift ---
    vocab_shift = compute_vocab_shift(text)

    # --- Intent variance ---
    intent_variance = compute_intent_variance()

    # Weighted drift score
    drift_score = (
        0.6 * embedding_shift +
        0.3 * vocab_shift +
        0.1 * intent_variance
    )

    # Update memory AFTER scoring
    text_memory.append(text)
    if len(text_memory) > MAX_MEMORY:
        text_memory.pop(0)

    drift_flag = drift_score > threshold

    components = {
        "embedding_shift": round(float(embedding_shift), 4),
        "vocab_shift": round(float(vocab_shift), 4),
        "intent_variance": round(float(intent_variance), 6)
    }

    return drift_flag, round(float(drift_score), 4), components


def get_drift_status():
    """Get current drift detection state."""
    return {
        "memory_size": len(text_memory),
        "max_memory": MAX_MEMORY,
        "vocab_memory_size": len(vocab_memory)
    }
