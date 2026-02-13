# backend/app/evolution/orchestrator.py

import json
from pathlib import Path

from .candidate_generator import generate_candidates
from .evaluator import evaluate_candidate
from .evolution_logger import log_evolution
from .recompiler import recompile_model
from .validation_sandbox import run_validation
from .model_registry import register_model
from .rollback import backup_model, rollback_model

PHASE3_REPORT = Path("phase2_usage_report.json")


def run_evolution_cycle():
    """
    Executes:
    - Candidate selection
    - Recompile (quantization / pruning)
    - Validation gating
    - Rollback (if needed)
    - Registry update
    """

    if not PHASE3_REPORT.exists():
        return {"status": "SKIPPED", "reason": "Usage report not found"}

    with open(PHASE3_REPORT) as f:
        report = json.load(f)

    prunable_blocks = report["structural_decisions"]["prune_attention_blocks"]
    block_importance = report["importance_scores"]["block_level"]["attention_heads"]

    if not prunable_blocks:
        return {"status": "SKIPPED", "reason": "No prunable blocks"}

    # -------------------------
    # 1️⃣ Generate candidates
    # -------------------------
    candidates = generate_candidates(prunable_blocks)

    evaluated = [
        evaluate_candidate(candidate, block_importance)
        for candidate in candidates
    ]

    best = max(evaluated, key=lambda x: x["score"])

    # Log decision
    log_evolution(evaluated, best)

    try:
        # -------------------------
        # 2️⃣ Backup current model
        # -------------------------
        backup_model()

        # -------------------------
        # 3️⃣ Recompile optimized model
        # -------------------------
        architecture_diff, version = recompile_model()

        # -------------------------
        # 4️⃣ Run validation
        # -------------------------
        validation_result = run_validation()

        # -------------------------
        # 5️⃣ Validation gate
        # -------------------------
        if validation_result["status"] == "FAIL":
            rollback_model()

            return {
                "evolution_status": "REJECTED",
                "selected_candidate": best,
                "architecture_diff": architecture_diff,
                "validation": validation_result
            }

        # -------------------------
        # 6️⃣ Register new model
        # -------------------------
        register_model(
            version=version,
            architecture_diff=architecture_diff,
            validation_report=validation_result
        )

        return {
            "evolution_status": "APPROVED",
            "selected_candidate": best,
            "architecture_diff": architecture_diff,
            "validation": validation_result
        }

    except Exception as e:
        rollback_model()
        return {
            "evolution_status": "ERROR",
            "error": str(e)
        }


if __name__ == "__main__":
    result = run_evolution_cycle()
    print(json.dumps(result, indent=2))
