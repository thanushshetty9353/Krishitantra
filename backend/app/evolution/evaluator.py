# backend/app/evolution/evaluator.py

def evaluate_candidate(candidate, block_importance):
    """
    Heuristic evaluation — NO training
    """

    pruned = candidate["prune_blocks"]

    latency_gain = len(pruned) * 5   # simulated ms gain
    memory_gain = len(pruned) * 3    # simulated MB gain

    risk = sum(block_importance[b] for b in pruned) / len(pruned)

    score = (latency_gain * 0.5 + memory_gain * 0.3) - (risk * 10)

    return {
        "candidate": candidate,
        "latency_gain_ms": latency_gain,
        "memory_gain_mb": memory_gain,
        "risk": round(risk, 2),
        "score": round(score, 2)
    }
