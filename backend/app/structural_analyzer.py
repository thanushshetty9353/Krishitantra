# backend/app/structural_analyzer.py

"""
Structural Analysis Module (SE-SLM Requirement 3.3)

The system shall:
- Identify prunable attention heads
- Detect redundant feedforward layers
- Score neuron importance
- Recommend rewiring opportunities
"""

PRUNE_THRESHOLD = 0.15
MAX_PRUNE_RATIO = 0.40
HIGH_SPARSITY_THRESHOLD = 0.70

PROTECTED_BLOCKS = {
    "embedding",
    "output",
    "classifier",
    "safety",
    "shared"
}


# ============================================================
# Attention Head Analysis
# ============================================================

def identify_prunable_blocks(block_importance: dict):
    """
    Identify attention heads / blocks that can be safely pruned.
    Respects safety constraints: max 40% pruning, protected blocks.
    """
    candidates = {
        block: score
        for block, score in block_importance.items()
        if score < PRUNE_THRESHOLD
    }

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
    """Compute average importance of selected blocks (lower = safer to prune)."""
    if not selected_blocks:
        return 0.0

    valid_blocks = [b for b in selected_blocks if b in block_importance]
    if not valid_blocks:
        return 0.0

    return round(
        sum(block_importance[b] for b in valid_blocks)
        / len(valid_blocks),
        4
    )


# ============================================================
# FFN / Feedforward Layer Analysis
# ============================================================

def detect_redundant_ffn_layers(ffn_sparsity: dict):
    """
    Detect FFN layers with high sparsity (many near-zero activations).
    High sparsity = neurons are largely dormant = redundant.
    """
    redundant = []

    for layer_name, avg_sparsity in ffn_sparsity.items():
        prefix = layer_name.split(".")[0]
        if prefix in PROTECTED_BLOCKS:
            continue

        if avg_sparsity > HIGH_SPARSITY_THRESHOLD:
            redundant.append({
                "layer": layer_name,
                "sparsity": round(avg_sparsity, 4),
                "recommendation": "prune" if avg_sparsity > 0.85 else "compress"
            })

    return sorted(redundant, key=lambda x: x["sparsity"], reverse=True)


# ============================================================
# Neuron Importance Scoring
# ============================================================

def score_neuron_importance(head_stats: dict):
    """
    Score each attention head/neuron group by total activation magnitude.
    Lower activation = less important = candidate for pruning.
    """
    scores = {}
    all_values = []

    for layer_name, heads in head_stats.items():
        if isinstance(heads, dict):
            for head_id, activation in heads.items():
                key = f"{layer_name}.head_{head_id}"
                scores[key] = float(activation)
                all_values.append(float(activation))

    # Normalize to 0-1 range
    if all_values:
        max_val = max(all_values) if max(all_values) > 0 else 1.0
        scores = {
            k: round(v / max_val, 4)
            for k, v in scores.items()
        }

    return scores


# ============================================================
# Rewiring Recommendations
# ============================================================

def recommend_rewiring(head_scores: dict, ffn_redundancy: list):
    """
    Generate architecture rewiring recommendations based on analysis.
    """
    recommendations = []

    # Head-based recommendations
    low_importance_heads = {
        k: v for k, v in head_scores.items()
        if v < PRUNE_THRESHOLD
    }

    if low_importance_heads:
        recommendations.append({
            "type": "head_pruning",
            "description": f"Remove {len(low_importance_heads)} low-importance attention heads",
            "targets": list(low_importance_heads.keys())[:10],
            "estimated_speedup": f"{len(low_importance_heads) * 2}%"
        })

    # FFN-based recommendations
    high_sparsity_layers = [r for r in ffn_redundancy if r["sparsity"] > 0.85]
    compressible_layers = [r for r in ffn_redundancy if r["recommendation"] == "compress"]

    if high_sparsity_layers:
        recommendations.append({
            "type": "layer_pruning",
            "description": f"Remove {len(high_sparsity_layers)} highly sparse FFN layers",
            "targets": [r["layer"] for r in high_sparsity_layers],
            "estimated_memory_saving": f"{len(high_sparsity_layers) * 5}%"
        })

    if compressible_layers:
        recommendations.append({
            "type": "sparse_rewiring",
            "description": f"Apply sparse rewiring to {len(compressible_layers)} FFN layers",
            "targets": [r["layer"] for r in compressible_layers],
            "estimated_speedup": f"{len(compressible_layers) * 3}%"
        })

    # Quantization recommendation (always applicable)
    recommendations.append({
        "type": "quantization",
        "description": "Apply INT8 dynamic quantization to all Linear layers",
        "estimated_memory_saving": "30-50%",
        "estimated_speedup": "10-20%"
    })

    return recommendations


# ============================================================
# Unified Analysis
# ============================================================

def run_full_analysis(head_stats: dict, ffn_sparsity: dict):
    """
    Run complete structural analysis covering all SE-SLM requirements:
    - Prunable attention heads
    - Redundant FFN layers
    - Neuron importance scores
    - Rewiring recommendations
    """

    # Score neurons
    neuron_scores = score_neuron_importance(head_stats)

    # Find prunable blocks
    block_importance = {}
    for layer_name, heads in head_stats.items():
        if isinstance(heads, dict):
            values = [float(v) for v in heads.values()]
            if values:
                max_val = max(values) if max(values) > 0 else 1.0
                block_importance[layer_name] = round(
                    sum(values) / (len(values) * max_val), 4
                )

    prunable_blocks = identify_prunable_blocks(block_importance)
    risk_score = compute_risk_score(prunable_blocks, block_importance)

    # Detect redundant FFN layers
    redundant_ffn = detect_redundant_ffn_layers(ffn_sparsity)

    # Generate recommendations
    recommendations = recommend_rewiring(neuron_scores, redundant_ffn)

    return {
        "prunable_attention_blocks": prunable_blocks,
        "pruning_risk_score": risk_score,
        "block_importance": block_importance,
        "neuron_importance_scores": dict(
            sorted(neuron_scores.items(), key=lambda x: x[1])[:20]
        ),
        "redundant_ffn_layers": redundant_ffn,
        "rewiring_recommendations": recommendations,
        "constraints": {
            "max_prune_ratio": MAX_PRUNE_RATIO,
            "protected_blocks": list(PROTECTED_BLOCKS),
            "prune_threshold": PRUNE_THRESHOLD
        }
    }
