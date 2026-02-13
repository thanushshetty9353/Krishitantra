# backend/app/evolution/model_registry.py

"""
Model Registry (SE-SLM Requirement 3.8)

The system shall maintain:
- Model lineage
- Compression ratios
- Accuracy deltas
- Evolution metadata
"""

import json
from pathlib import Path
from datetime import datetime

REGISTRY_PATH = Path("model_registry.json")
OPTIMIZED_DIR = Path("models/optimized")


def get_latest_version():
    """Get the name of the latest optimized model version."""
    if not OPTIMIZED_DIR.exists():
        return "base"

    versions = sorted(
        [v for v in OPTIMIZED_DIR.glob("v*") if v.is_dir() and v.name != "backup"],
        key=lambda x: int(x.name.replace("v", "").split(".")[0])
    )
    return versions[-1].name if versions else "base"


def get_previous_version(current_version: str):
    """Get the parent version for lineage tracking."""
    if not OPTIMIZED_DIR.exists():
        return "base"

    versions = sorted(
        [v for v in OPTIMIZED_DIR.glob("v*") if v.is_dir() and v.name != "backup"],
        key=lambda x: int(x.name.replace("v", "").split(".")[0])
    )

    version_names = [v.name for v in versions]

    if current_version in version_names:
        idx = version_names.index(current_version)
        return version_names[idx - 1] if idx > 0 else "base"

    return "base"


def register_model(version, architecture_diff, validation_report):
    """Register a new model version with full lineage and metrics."""

    parent = get_previous_version(version)

    entry = {
        "version": version,
        "parent_version": parent,
        "timestamp": datetime.utcnow().isoformat(),
        "lineage": build_lineage(version),
        "compression_percent": architecture_diff.get("reduction_percent", 0),
        "optimization": architecture_diff.get("optimizations",
                         [architecture_diff.get("optimization", "unknown")]),
        "base_parameters": architecture_diff.get("base_parameters", 0),
        "optimized_parameters": architecture_diff.get("optimized_parameters", 0),
        "base_size_mb": architecture_diff.get("base_size_mb", 0),
        "optimized_size_mb": architecture_diff.get("optimized_size_mb", 0),
        "accuracy_drop_percent": validation_report.get("accuracy_drop_percent", 0),
        "similarity_score": validation_report.get("similarity_score", 0),
        "hallucination_rate": validation_report.get("hallucination_rate", 0),
        "validation_status": validation_report.get("status", "unknown"),
        "trigger": architecture_diff.get("trigger", "domain_drift")
    }

    registry = get_registry()
    registry.append(entry)

    with open(REGISTRY_PATH, "w") as f:
        json.dump(registry, f, indent=2)

    print(f"📋 Registered model {version} (parent: {parent})")

    return entry


def get_registry():
    """Load the full model registry."""
    if REGISTRY_PATH.exists():
        try:
            with open(REGISTRY_PATH) as f:
                data = json.load(f)
                return data if isinstance(data, list) else [data]
        except (json.JSONDecodeError, Exception):
            return []
    return []


def get_model_entry(version: str):
    """Get a specific model version entry from the registry."""
    registry = get_registry()
    for entry in registry:
        if entry.get("version") == version:
            return entry
    return None


def build_lineage(version: str):
    """Build the full lineage chain up to the base model."""
    lineage = [version]
    current = version

    for _ in range(20):
        parent = get_previous_version(current)
        if parent == "base" or parent == current:
            lineage.append("base")
            break
        lineage.append(parent)
        current = parent

    return list(reversed(lineage))


def get_registry_summary():
    """Get a quick summary of the registry."""
    registry = get_registry()

    if not registry:
        return {
            "total_versions": 0,
            "latest_version": "base",
            "total_compression": 0
        }

    latest = registry[-1]
    return {
        "total_versions": len(registry),
        "latest_version": latest.get("version", "unknown"),
        "latest_compression_percent": latest.get("compression_percent", 0),
        "latest_accuracy_drop": latest.get("accuracy_drop_percent", 0),
        "latest_validation": latest.get("validation_status", "unknown"),
        "all_versions": [e.get("version", "unknown") for e in registry]
    }
