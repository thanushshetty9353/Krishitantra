# backend/app/evolution/orchestrator.py

import json
from pathlib import Path

from .candidate_generator import generate_candidates
from .evaluator import evaluate_candidate
from .evolution_logger import log_evolution

PHASE3_REPORT = Path("phase2_usage_report.json")


def run_evolution_cycle():
    # 1. Load Phase 3 output
    with open(PHASE3_REPORT) as f:
        report = json.load(f)

    prunable_blocks = report["structural_decisions"]["prune_attention_blocks"]
    block_importance = report["importance_scores"]["block_level"]["attention_heads"]

    # 2. Generate candidates
    candidates = generate_candidates(prunable_blocks)

    # 3. Evaluate
    evaluated = [
        evaluate_candidate(c, block_importance)
        for c in candidates
    ]

    # 4. Select best
    best = max(evaluated, key=lambda x: x["score"])

    # 5. Log evolution
    log_evolution(evaluated, best)

    return best
