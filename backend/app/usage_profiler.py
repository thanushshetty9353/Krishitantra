# backend/app/usage_profiler.py

import sqlite3
from pathlib import Path
import json
from collections import defaultdict

# ==========================
# Configuration
# ==========================

DB_PATH = Path("backend/data/requests.db")
DORMANT_THRESHOLD = 0.10   # <10% usage = dormant

# ==========================
# Connect to database
# ==========================

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
    Normalized importance score:
    importance = component_usage / max_usage
    """
    if not usage_dict:
        return {}

    max_usage = max(usage_dict.values())
    importance = {}

    for component, count in usage_dict.items():
        importance[component] = round(count / max_usage, 2)

    return importance


def find_dormant(importance_dict):
    """
    Components below threshold are considered dormant
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
    Aggregates fine-grained module importance to block-level importance.
    Block key examples:
    - encoder.block.3
    - decoder.block.5
    Uses average importance per block.
    """
    block_scores = defaultdict(list)

    for component, score in importance_dict.items():
        parts = component.split(".")
        if len(parts) >= 3 and parts[1] == "block":
            block_key = ".".join(parts[:3])  # encoder.block.X or decoder.block.X
            block_scores[block_key].append(score)

    aggregated = {
        block: round(sum(scores) / len(scores), 2)
        for block, scores in block_scores.items()
    }

    return aggregated

# ==========================
# Phase 2 report
# ==========================

def generate_phase2_report():
    # Raw aggregated usage
    layer_usage = get_layer_usage()
    attention_usage = get_attention_usage()

    # Fine-grained importance
    layer_importance = compute_importance(layer_usage)
    attention_importance = compute_importance(attention_usage)

    # Block-level importance (refinement)
    block_layer_importance = aggregate_to_block_level(layer_importance)
    block_attention_importance = aggregate_to_block_level(attention_importance)

    # Dormant components (fine-grained)
    dormant_layers = find_dormant(layer_importance)
    dormant_attention = find_dormant(attention_importance)

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
        }
    }

    with open("phase2_usage_report.json", "w") as f:
        json.dump(report, f, indent=2)

    print("✅ Phase 2 Usage Profiler Complete (with block-level aggregation)\n")
    print(json.dumps(report, indent=2))


# ==========================
# Entry point
# ==========================

if __name__ == "__main__":
    generate_phase2_report()
