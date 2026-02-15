# backend/app/evolution/orchestrator.py

"""
Evolution Engine Orchestrator (SE-SLM Requirement 3.4)

The engine shall:
- Execute scheduled evolution cycles
- Generate compressed architectures
- Simulate pruning impact
- Produce candidate model graphs

Evolution Cycle (SE-SLM §6):
1. Collect telemetry
2. Aggregate metrics
3. Score components
4. Generate pruning plan
5. Recompile model
6. Validate
7. Deploy
"""

import json
import re
from pathlib import Path

from .candidate_generator import generate_candidates
from .evaluator import evaluate_candidate
from .evolution_logger import log_evolution
from .recompiler import recompile_model
from .validation_sandbox import run_validation
from .model_registry import register_model
from .rollback import backup_model, rollback_model
from .distillation import distill

PHASE3_REPORT = Path("phase2_usage_report.json")


def _parse_pruning_plan(best_candidate, report):
    """
    Parse the best candidate's prune_blocks into arguments
    that recompile_model() understands.

    Returns: (optimization_type, heads_to_prune, layers_to_remove)
    """
    prune_blocks = best_candidate.get("candidate", {}).get("prune_blocks", [])

    # Check structural recommendations for optimization type
    recommendations = report.get("structural_decisions", {}).get(
        "rewiring_recommendations", []
    )

    optimization_type = "head_pruning"  # default
    heads_to_prune = None
    layers_to_remove = None

    # Determine optimization from recommendations
    rec_types = [r.get("type", "") for r in recommendations]

    if "layer_pruning" in rec_types and "head_pruning" in rec_types:
        optimization_type = "all"
    elif "layer_pruning" in rec_types:
        optimization_type = "layer_pruning"
    elif "head_pruning" in rec_types:
        optimization_type = "head_pruning"

    # Parse block names to extract layer indices for head/layer pruning
    # Block names look like: "ffn_sparsity", "encoder.block.X", etc.
    encoder_layer_indices = set()
    for block_name in prune_blocks:
        # Try to extract encoder block indices
        match = re.search(r'encoder\.block\.(\d+)', block_name)
        if match:
            encoder_layer_indices.add(int(match.group(1)))

    if encoder_layer_indices:
        # Use extracted layer indices for pruning
        sorted_indices = sorted(encoder_layer_indices)
        if optimization_type in ("layer_pruning", "all"):
            layers_to_remove = sorted_indices
        # For head pruning, prune last head from identified layers
        heads_to_prune = {idx: [7] for idx in sorted_indices}
    else:
        # Default: prune from last encoder layers
        heads_to_prune = None  # recompiler will use defaults
        layers_to_remove = None

    return optimization_type, heads_to_prune, layers_to_remove


def run_evolution_cycle():
    """
    Full evolution cycle per SE-SLM §6:
    1. Read telemetry/usage report
    2. Generate candidates
    3. Evaluate candidates
    4. Select best candidate
    5. Backup current model
    6. Recompile with selected optimizations
    7. Validate in sandbox
    8. Deploy or rollback
    """

    # --- Step 1: Read usage report ---
    if not PHASE3_REPORT.exists():
        return {"status": "SKIPPED", "reason": "Usage report not found. Run /profiler/run first."}

    try:
        with open(PHASE3_REPORT) as f:
            report = json.load(f)
    except (json.JSONDecodeError, Exception) as e:
        return {"status": "ERROR", "reason": f"Failed to read usage report: {e}"}

    # Safely get structural decisions
    structural = report.get("structural_decisions", {})
    prunable_blocks = structural.get("prune_attention_blocks", [])
    block_importance = report.get("importance_scores", {}).get("block_level", {}).get("attention_heads", {})

    if not prunable_blocks:
        # If no prunable blocks found, still allow evolution with defaults
        prunable_blocks = ["default_pruning"]

    try:
        # --- Step 2 & 3: Generate and evaluate candidates ---
        candidates = generate_candidates(prunable_blocks)

        if not candidates:
            # Fallback: create a default candidate
            candidates = [{"prune_blocks": prunable_blocks[:1]}]

        evaluated = [
            evaluate_candidate(candidate, block_importance)
            for candidate in candidates
        ]

        # --- Step 4: Select best ---
        best = max(evaluated, key=lambda x: x["score"])

        # Log decision
        log_evolution(evaluated, best)

        # --- Step 5: Backup current model ---
        backup_model()

        # --- Step 6: Parse pruning plan and recompile ---
        optimization_type, heads_to_prune, layers_to_remove = _parse_pruning_plan(best, report)

        print(f"🧬 Evolution: optimization={optimization_type}, heads={heads_to_prune}, layers={layers_to_remove}")

        architecture_diff, version = recompile_model(
            optimization=optimization_type,
            heads_to_prune=heads_to_prune,
            layers_to_remove=layers_to_remove
        )

        # --- Step 7: Validate ---
        validation_result = run_validation()

        # --- Step 8: Validation gate ---
        if validation_result["status"] == "FAIL":
            rollback_model()

            return {
                "evolution_status": "REJECTED",
                "selected_candidate": best,
                "architecture_diff": architecture_diff,
                "validation": validation_result,
                "reason": "Validation failed — rolled back to previous model"
            }

        # --- Step 9: Register new model ---
        register_model(
            version=version,
            architecture_diff=architecture_diff,
            validation_report=validation_result
        )

        return {
            "evolution_status": "APPROVED",
            "selected_candidate": best,
            "architecture_diff": architecture_diff,
            "validation": validation_result,
            "new_version": version
        }

    except Exception as e:
        import traceback
        traceback.print_exc()
        rollback_model()
        return {
            "evolution_status": "ERROR",
            "error": str(e),
            "traceback": traceback.format_exc()
        }


if __name__ == "__main__":
    result = run_evolution_cycle()
    print(json.dumps(result, indent=2, default=str))
