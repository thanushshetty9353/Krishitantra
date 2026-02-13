# backend/app/usage_profiler.py

"""
Usage Profiling Module (SE-SLM Requirement 3.2)

Aggregates telemetry into:
- Frequent token paths
- Dormant neurons
- Redundant heads
- Rare vocabulary branches
"""

import json
from pathlib import Path
from collections import defaultdict

from backend.app.structural_analyzer import (
    identify_prunable_blocks,
    compute_risk_score,
    run_full_analysis
)

from backend.app.database import (
    get_aggregated_head_stats,
    get_aggregated_ffn_stats,
    get_telemetry_summary
)

# ==========================
# Path Resolution
# ==========================

BASE_DIR = Path(__file__).resolve().parent           # backend/app
OUTPUT_PATH = BASE_DIR.parent.parent / "phase2_usage_report.json"

# ==========================
# Configuration
# ==========================

DORMANT_THRESHOLD = 0.10  # <10% usage = dormant


# ==========================
# Importance Calculation
# ==========================

def compute_importance(usage_dict):
    """Normalized importance: usage / max_usage"""
    if not usage_dict:
        return {}

    max_usage = max(usage_dict.values())
    if max_usage == 0:
        return {k: 0.0 for k in usage_dict}

    return {
        component: round(count / max_usage, 4)
        for component, count in usage_dict.items()
    }


def find_dormant(importance_dict):
    """Components below dormant threshold."""
    return [
        component
        for component, score in importance_dict.items()
        if score < DORMANT_THRESHOLD
    ]


# ==========================
# Block-level Aggregation
# ==========================

def aggregate_to_block_level(importance_dict):
    """Aggregate fine-grained importance to block-level averages."""
    block_scores = defaultdict(list)

    for component, score in importance_dict.items():
        parts = component.split(".")
        # Use up to 3-level grouping for block key
        if len(parts) >= 3:
            block_key = ".".join(parts[:3])
        elif len(parts) >= 2:
            block_key = ".".join(parts[:2])
        else:
            block_key = parts[0]

        block_scores[block_key].append(score)

    return {
        block: round(sum(scores) / len(scores), 4)
        for block, scores in block_scores.items()
    }


# ==========================
# Frequent Token Path Analysis
# ==========================

def analyze_token_frequencies(token_freq: dict, top_n: int = 20):
    """Identify most and least frequent token paths."""
    if not token_freq:
        return {"frequent": [], "rare": []}

    sorted_tokens = sorted(
        token_freq.items(),
        key=lambda x: x[1],
        reverse=True
    )

    return {
        "frequent": sorted_tokens[:top_n],
        "rare": sorted_tokens[-top_n:] if len(sorted_tokens) > top_n else [],
        "total_unique_tokens": len(token_freq),
        "total_token_count": sum(token_freq.values())
    }


# ==========================
# Report Generation (Fixed)
# ==========================

def generate_usage_and_structure_report():
    """
    Phase 2 (Usage Profiling) + Phase 3 (Structural Analysis)
    Reads from the correct telemetry_structural table.
    """
    print("\n🚀 Running Phase 2 (Usage Profiling) + Phase 3 (Structural Analysis)\n")

    # -------- Read aggregated telemetry from DB --------
    head_stats = get_aggregated_head_stats()
    ffn_sparsity = get_aggregated_ffn_stats()
    telemetry_summary = get_telemetry_summary()

    # -------- Phase 2: Usage Profiling --------

    # Flatten head stats for importance scoring
    flat_head_usage = {}
    for layer, heads in head_stats.items():
        if isinstance(heads, dict):
            for head_id, value in heads.items():
                flat_head_usage[f"{layer}.head_{head_id}"] = float(value)

    head_importance = compute_importance(flat_head_usage)
    ffn_importance = compute_importance(ffn_sparsity)

    block_head_importance = aggregate_to_block_level(head_importance)
    block_ffn_importance = aggregate_to_block_level(ffn_importance)

    dormant_heads = find_dormant(head_importance)
    dormant_ffn = find_dormant(ffn_importance)

    # -------- Phase 3: Structural Analysis (full) --------
    analysis_result = run_full_analysis(head_stats, ffn_sparsity)

    # -------- Construct Report --------
    report = {
        "telemetry_summary": telemetry_summary,
        "importance_scores": {
            "fine_grained": {
                "attention_heads": head_importance,
                "ffn_layers": ffn_importance
            },
            "block_level": {
                "attention_heads": block_head_importance,
                "ffn_layers": block_ffn_importance
            }
        },
        "dormant_components": {
            "attention_heads": dormant_heads,
            "ffn_layers": dormant_ffn,
            "total_dormant": len(dormant_heads) + len(dormant_ffn)
        },
        "structural_decisions": {
            "prune_attention_blocks": analysis_result["prunable_attention_blocks"],
            "risk_score": analysis_result["pruning_risk_score"],
            "redundant_ffn_layers": analysis_result["redundant_ffn_layers"],
            "rewiring_recommendations": analysis_result["rewiring_recommendations"],
            "constraints_enforced": analysis_result["constraints"]
        }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Report generated successfully")
    print("📄 Output file:", OUTPUT_PATH)
    print("\n🔧 Structural Decisions:")
    print(json.dumps(report["structural_decisions"], indent=2))

    return report


# ==========================
# Entry Point
# ==========================

if __name__ == "__main__":
    generate_usage_and_structure_report()
