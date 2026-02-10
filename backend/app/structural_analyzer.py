# backend/app/structural_analyzer.py

PRUNE_THRESHOLD = 0.15
MAX_PRUNE_RATIO = 0.40

PROTECTED_BLOCKS = {
    "embedding",
    "output",
    "classifier",
    "safety"
}


def identify_prunable_blocks(block_importance: dict):
    """
    Phase 3:
    Decide WHAT CAN be pruned (not execute pruning)
    """
    candidates = {
        block: score
        for block, score in block_importance.items()
        if score < PRUNE_THRESHOLD
    }

    # Sort by lowest importance
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda x: x[1]
    )

    max_allowed = int(len(sorted_candidates) * MAX_PRUNE_RATIO)
    max_allowed = max(1, max_allowed) if sorted_candidates else 0

    selected = []

    for block, score in sorted_candidates:
        prefix = block.split(".")[0]
        if prefix in PROTECTED_BLOCKS:
            continue
        if len(selected) < max_allowed:
            selected.append(block)

    return selected


def compute_risk_score(selected_blocks: list, block_importance: dict):
    if not selected_blocks:
        return 0.0

    return round(
        sum(block_importance[b] for b in selected_blocks)
        / len(selected_blocks),
        2
    )
