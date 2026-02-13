# backend/app/drift_detector.py

"""
Domain Drift Detection Module (SE-SLM Requirement 3.7)

Drift shall be detected using:
- Embedding distribution shifts
- Vocabulary change rate
- Intent variance
"""

import torch
import numpy as np
from collections import Counter
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
from sklearn.metrics.pairwise import cosine_similarity

MODEL_NAME = "google/flan-t5-small"

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

encoder = model.get_encoder()

embedding_memory = []
vocab_memory = []
MAX_MEMORY = 20


# --------------------------------------------------
# Embedding Extraction
# --------------------------------------------------

def get_embedding(text: str):
    inputs = tokenizer(text, return_tensors="pt", truncation=True)

    with torch.no_grad():
        outputs = encoder(**inputs)

    hidden = outputs.last_hidden_state
    embedding = hidden.mean(dim=1).squeeze().numpy()

    return embedding


# --------------------------------------------------
# Vocabulary Shift
# --------------------------------------------------

def compute_vocab_shift(text: str):
    tokens = tokenizer.tokenize(text)
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
# Intent Variance (embedding distribution variance)
# --------------------------------------------------

def compute_intent_variance(new_embedding):

    if len(embedding_memory) < 5:
        return 0.0

    recent = np.array(embedding_memory[-5:])
    return float(np.var(recent))


# --------------------------------------------------
# Main Drift Detection
# --------------------------------------------------

def detect_drift(text: str, threshold: float = 0.35):
    """
    Drift score combines:
    - embedding semantic shift
    - vocabulary shift
    - embedding variance trend

    Returns: (drift_flag, drift_score, components)
    """

    new_embedding = get_embedding(text)

    # First request
    if len(embedding_memory) == 0:
        embedding_memory.append(new_embedding)
        return False, 0.0, {
            "embedding_shift": 0.0,
            "vocab_shift": 0.0,
            "intent_variance": 0.0
        }

    # --- Semantic shift ---
    centroid = np.mean(embedding_memory, axis=0).reshape(1, -1)
    similarity = cosine_similarity(
        centroid,
        new_embedding.reshape(1, -1)
    )[0][0]

    embedding_shift = 1 - similarity

    # --- Vocab shift ---
    vocab_shift = compute_vocab_shift(text)

    # --- Intent variance ---
    intent_variance = compute_intent_variance(new_embedding)

    # Weighted drift score
    drift_score = (
        0.6 * embedding_shift +
        0.3 * vocab_shift +
        0.1 * intent_variance
    )

    # Update memory AFTER scoring
    embedding_memory.append(new_embedding)
    if len(embedding_memory) > MAX_MEMORY:
        embedding_memory.pop(0)

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
        "memory_size": len(embedding_memory),
        "max_memory": MAX_MEMORY,
        "vocab_memory_size": len(vocab_memory)
    }
