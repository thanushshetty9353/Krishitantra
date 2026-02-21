# backend/app/evolution/candidate_generator.py

from itertools import combinations

def generate_candidates(prunable_blocks: list):
    """
    Generate multiple architecture configurations
    """
    candidates = []

    for r in range(1, min(6, len(prunable_blocks) + 1)):
        for combo in combinations(prunable_blocks, r):
            candidates.append({
                "prune_blocks": list(combo)
            })

    return candidates
