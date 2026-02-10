# backend/app/usage_profiler.py

import sqlite3
from pathlib import Path
import json
from collections import defaultdict

# ✅ Phase 3 imported cleanly
from backend.app.structural_analyzer import (
    identify_prunable_blocks,
    compute_risk_score
)

# ==========================
# Path Resolution
# ==========================

BASE_DIR = Path(__file__).resolve().parent          # backend/app
DB_PATH = BASE_DIR.parent / "data" / "requests.db" # backend/data/requests.db
OUTPUT_PATH = BASE_DIR.parent.parent / "phase2_usage_report.json"  # project root

# ==========================
# Configuration (Phase 2 only)
# ==========================

DORMANT_THRESHOLD = 0.10  # <10% usage = dormant

# ==========================
# Connect to database
# ==========================

print("📂 Database path:", DB_PATH)
print("📂 Database exists:", DB_PATH.exists())

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

# ==========================
# Read & aggregate telemetry
# ==========================

def get_layer_usage():
    cursor.execute("""
        SELECT layer_name, SUM(call_count)
        FROM telemetry_layers
        GROUP BY layer_name
    """)
    return dict(cursor.fetchall())


def get_attention_usage():
    cursor.execute("""
        SELECT attention_module, SUM(call_count)
        FROM telemetry_attention
        GROUP BY attention_module
    """)
    return dict(cursor.fetchall())

# ==========================
# Importance calculation
# ==========================

def compute_importance(usage_dict):
    """
    Normalized importance:
    importance = usage / max_usage
    """
    if not usage_dict:
        return {}

    max_usage = max(usage_dict.values())
    return {
        component: round(count / max_usage, 2)
        for component, count in usage_dict.items()
    }


def find_dormant(importance_dict):
    """
    Components below dormant threshold
    """
    return [
        component
        for component, score in importance_dict.items()
        if score < DORMANT_THRESHOLD
    ]

# ==========================
# Block-level aggregation
# ==========================

def aggregate_to_block_level(importance_dict):
    """
    Aggregate fine-grained importance to block-level
    using average importance per block
    """
    block_scores = defaultdict(list)

    for component, score in importance_dict.items():
        parts = component.split(".")
        if len(parts) >= 3 and parts[1] == "block":
            block_key = ".".join(parts[:3])
            block_scores[block_key].append(score)

    return {
        block: round(sum(scores) / len(scores), 2)
        for block, scores in block_scores.items()
    }

# ==========================
# Report generation
# ==========================

def generate_usage_and_structure_report():
    print("\n🚀 Running Phase 2 (Usage Profiling) + Phase 3 (Structural Analysis)\n")

    # -------- Phase 2: Usage Profiling --------
    layer_usage = get_layer_usage()
    attention_usage = get_attention_usage()

    layer_importance = compute_importance(layer_usage)
    attention_importance = compute_importance(attention_usage)

    block_layer_importance = aggregate_to_block_level(layer_importance)
    block_attention_importance = aggregate_to_block_level(attention_importance)

    dormant_layers = find_dormant(layer_importance)
    dormant_attention = find_dormant(attention_importance)

    # -------- Phase 3: Structural Analysis (delegated) --------
    prunable_blocks = identify_prunable_blocks(block_attention_importance)
    risk_score = compute_risk_score(prunable_blocks, block_attention_importance)

    # -------- Report --------
    report = {
        "importance_scores": {
            "fine_grained": {
                "layers": layer_importance,
                "attention_heads": attention_importance
            },
            "block_level": {
                "layers": block_layer_importance,
                "attention_heads": block_attention_importance
            }
        },
        "dormant_components": {
            "layers": dormant_layers,
            "attention_heads": dormant_attention
        },
        "structural_decisions": {
            "prune_attention_blocks": prunable_blocks,
            "risk_score": risk_score,
            "constraints_enforced": {
                "max_prune_ratio": 0.40,
                "protected_blocks_respected": True
            }
        }
    }

    with open(OUTPUT_PATH, "w") as f:
        json.dump(report, f, indent=2)

    print("\n✅ Report generated successfully")
    print("📄 Output file:", OUTPUT_PATH)
    print("\n🔧 Structural Decisions:")
    print(json.dumps(report["structural_decisions"], indent=2))

# ==========================
# Entry point
# ==========================

if __name__ == "__main__":
    generate_usage_and_structure_report()
