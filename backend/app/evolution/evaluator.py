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
    
    # Heuristic: 1 block ~ 2% speedup, 3MB memory (for 1.1B params)
    # We aim for >20% latency, >30% memory
    latency_gain = len(pruned) * 15   # simulated ms gain (roughly)
    latency_percent = (len(pruned) * 2.5) # approx % improvement

    memory_gain = len(pruned) * 5    # simulated MB gain
    memory_percent = (len(pruned) * 1.5) # approx % improvement

    # Safe access: use .get() with default 0.0 to prevent KeyError
    importance_values = [block_importance.get(b, 0.0) for b in pruned]

    if importance_values:
        risk = sum(importance_values) / len(importance_values)
    else:
        risk = 0.0

    # Boost score if we hit user targets
    target_bonus = 0.0
    if latency_percent >= 20: target_bonus += 0.5
    if memory_percent >= 30: target_bonus += 0.5

    score = (latency_gain * 0.8 + memory_gain * 0.5 + target_bonus) - (risk * 20)

    return {
        "candidate": candidate,
        "latency_gain_ms": latency_gain,
        "memory_gain_mb": memory_gain,
        "projected_latency_improvement": f"{latency_percent}%",
        "risk": round(risk, 4),
        "score": round(score, 4)
    }
