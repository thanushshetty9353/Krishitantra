# backend/app/usage_profiler.py

import sqlite3
from pathlib import Path
import json
from collections import defaultdict

# ==========================
# Path Resolution (FIXED)
# ==========================

BASE_DIR = Path(__file__).resolve().parent          # backend/app
DB_PATH = BASE_DIR.parent / "data" / "requests.db" # backend/data/requests.db
OUTPUT_PATH = BASE_DIR.parent.parent / "phase2_usage_report.json"  # project root

# ==========================
# Configuration
# ==========================

DORMANT_THRESHOLD = 0.10        # <10% usage = dormant
PRUNE_THRESHOLD = 0.15          # structural prune threshold
MAX_PRUNE_RATIO = 0.40          # max 40% prune per cycle

PROTECTED_BLOCKS = {
    "embedding",
    "output",
    "classifier",
    "safety"
}

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
    try:
        cursor.execute("""
            SELECT layer_name, SUM(call_count)
            FROM telemetry_layers
            GROUP BY layer_name
        """)
        return dict(cursor.fetchall())
    except Exception as e:
        print("❌ Error reading telemetry_layers:", e)
        return {}


def get_attention_usage():
    try:
        cursor.execute("""
            SELECT attention_module, SUM(call_count)
            FROM telemetry_attention
            GROUP BY attention_module
        """)
        return dict(cursor.fetchall())
    except Exception as e:
        print("❌ Error reading telemetry_attention:", e)
        return {}

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
# Structural analysis logic
# ==========================

def identify_prune_candidates(importance_dict):
    """
    Rule:
    If importance < PRUNE_THRESHOLD → prune candidate
    """
    return {
        component: score
        for component, score in importance_dict.items()
        if score < PRUNE_THRESHOLD
    }


def enforce_pruning_constraints(candidates):
    """
    Enforces:
    - Max 40% prune
    - Protected blocks
    """
    sorted_candidates = sorted(
        candidates.items(),
        key=lambda x: x[1]  # lowest importance first
    )

    max_allowed = int(len(sorted_candidates) * MAX_PRUNE_RATIO)
    max_allowed = max(1, max_allowed) if sorted_candidates else 0

    selected = []
    skipped = []

    for component, score in sorted_candidates:
        prefix = component.split(".block.")[0]

        if prefix in PROTECTED_BLOCKS:
            skipped.append(component)
            continue

        if len(selected) < max_allowed:
            selected.append((component, score))

    return selected, skipped


def compute_risk_score(pruned_components):
    """
    Risk = average importance of pruned components
    """
    if not pruned_components:
        return 0.0

    return round(
        sum(score for _, score in pruned_components) / len(pruned_components),
        2
    )

# ==========================
# Report generation
# ==========================

def generate_usage_and_structure_report():
    print("\n🚀 Running usage profiling + structural analysis\n")

    # Read usage
    layer_usage = get_layer_usage()
    attention_usage = get_attention_usage()

    print("📊 Layer usage entries:", len(layer_usage))
    print("📊 Attention usage entries:", len(attention_usage))

    # Importance
    layer_importance = compute_importance(layer_usage)
    attention_importance = compute_importance(attention_usage)

    # Block-level importance
    block_layer_importance = aggregate_to_block_level(layer_importance)
    block_attention_importance = aggregate_to_block_level(attention_importance)

    # Dormant components
    dormant_layers = find_dormant(layer_importance)
    dormant_attention = find_dormant(attention_importance)

    # Structural decisions
    prune_candidates = identify_prune_candidates(block_attention_importance)
    selected_prunes, skipped = enforce_pruning_constraints(prune_candidates)
    risk_score = compute_risk_score(selected_prunes)

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
            "prune_attention_blocks": [c for c, _ in selected_prunes],
            "risk_score": risk_score,
            "constraints_enforced": {
                "max_prune_ratio": MAX_PRUNE_RATIO,
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
