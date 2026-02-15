# backend/app/evolution/evaluator.py

"""
Candidate Evaluator (SE-SLM Evolution Engine)

Heuristic evaluation of pruning candidates based on
latency gain, memory gain, and risk from block importance scores.
"""


def evaluate_candidate(candidate, block_importance):
    """
    Heuristic evaluation — NO training.
    Uses .get() for safe key access to prevent KeyError.
    """

    pruned = candidate["prune_blocks"]

    latency_gain = len(pruned) * 5   # simulated ms gain
    memory_gain = len(pruned) * 3    # simulated MB gain

    # Safe access: use .get() with default 0.0 to prevent KeyError
    importance_values = [block_importance.get(b, 0.0) for b in pruned]

    if importance_values:
        risk = sum(importance_values) / len(importance_values)
    else:
        risk = 0.0

    score = (latency_gain * 0.5 + memory_gain * 0.3) - (risk * 10)

    return {
        "candidate": candidate,
        "latency_gain_ms": latency_gain,
        "memory_gain_mb": memory_gain,
        "risk": round(risk, 4),
        "score": round(score, 4)
    }
