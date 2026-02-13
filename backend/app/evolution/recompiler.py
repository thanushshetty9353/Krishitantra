# backend/app/evolution/recompiler.py

"""
Self-Recompilation Module (SE-SLM Requirement 3.5)

The system shall recompile models using:
- Weight inheritance
- Graph rewriting
- Lightweight distillation

Supported optimizations:
- Head pruning
- Layer pruning
- Embedding compression
- Sparse rewiring
- Quantization
"""

import json
import os
import torch
from pathlib import Path
from transformers import AutoTokenizer, AutoModelForSeq2SeqLM

MODEL_NAME = "google/flan-t5-small"

OPTIMIZED_MODEL_DIR = Path("models/optimized")
OPTIMIZED_MODEL_DIR.mkdir(parents=True, exist_ok=True)


def get_next_version():
    versions = list(OPTIMIZED_MODEL_DIR.glob("v*"))

    if not versions:
        return "v1"

    numeric_versions = []

    for v in versions:
        name = v.name.replace("v", "")
        if name == "backup":
            continue
        try:
            major = int(name.split(".")[0])
            numeric_versions.append(major)
        except ValueError:
            continue

    if not numeric_versions:
        return "v1"

    return f"v{max(numeric_versions) + 1}"


def get_model_size_mb(model):
    """Calculate model size in MB."""
    param_size = sum(p.nelement() * p.element_size() for p in model.parameters())
    buffer_size = sum(b.nelement() * b.element_size() for b in model.buffers())
    return round((param_size + buffer_size) / (1024 * 1024), 2)


def get_param_count(model):
    """Get total parameter count."""
    return sum(p.numel() for p in model.parameters())


# ============================================================
# Head Pruning
# ============================================================

def prune_attention_heads(model, heads_to_prune: dict = None):
    """
    Prune specified attention heads from the model.
    heads_to_prune: {layer_idx: [head_indices]}
    """
    if not heads_to_prune:
        # Default: prune last head from last encoder layer
        num_layers = len(model.encoder.block)
        heads_to_prune = {num_layers - 1: [7]}  # prune head 7 from last layer

    try:
        model.prune_heads(heads_to_prune)
        return True, heads_to_prune
    except Exception:
        return False, {}


# ============================================================
# Layer Pruning
# ============================================================

def prune_layers(model, layer_indices: list = None):
    """
    Remove specified encoder layers from the model.
    Respects max 40% pruning constraint.
    """
    total_layers = len(model.encoder.block)
    max_removable = int(total_layers * 0.4)

    if not layer_indices:
        # Default: remove last encoder layer
        layer_indices = [total_layers - 1]

    # Enforce safety cap
    layer_indices = layer_indices[:max_removable]

    removed = []
    for idx in sorted(layer_indices, reverse=True):
        if idx < len(model.encoder.block) and len(model.encoder.block) > 2:
            del model.encoder.block[idx]
            removed.append(idx)

    # Update config
    model.config.num_layers = len(model.encoder.block)

    return removed


# ============================================================
# Quantization
# ============================================================

def apply_quantization(model):
    """Apply INT8 dynamic quantization to Linear layers."""
    quantized = torch.quantization.quantize_dynamic(
        model,
        {torch.nn.Linear},
        dtype=torch.qint8
    )
    return quantized


# ============================================================
# Main Recompiler
# ============================================================

def recompile_model(
    optimization: str = "quantization",
    heads_to_prune: dict = None,
    layers_to_remove: list = None
):
    """
    Full recompilation pipeline:
    1. Load base model
    2. Apply selected optimizations
    3. Save optimized model
    4. Generate architecture diff report
    """
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
    model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    model.eval()

    base_params = get_param_count(model)
    base_size_mb = get_model_size_mb(model)
    base_encoder_layers = len(model.encoder.block)

    optimizations_applied = []
    pruned_heads = {}
    removed_layers = []

    # Limit CPU threads for efficiency
    torch.set_num_threads(2)

    # ---- Apply optimizations ----

    if optimization in ("head_pruning", "all"):
        success, pruned = prune_attention_heads(model, heads_to_prune)
        if success:
            optimizations_applied.append("head_pruning")
            pruned_heads = pruned

    if optimization in ("layer_pruning", "all"):
        removed = prune_layers(model, layers_to_remove)
        if removed:
            optimizations_applied.append("layer_pruning")
            removed_layers = removed

    if optimization in ("quantization", "all"):
        model = apply_quantization(model)
        optimizations_applied.append("dynamic_int8_quantization")

    # ---- Save ----

    version = get_next_version()
    save_path = OPTIMIZED_MODEL_DIR / version
    save_path.mkdir(parents=True, exist_ok=True)

    model.save_pretrained(save_path)
    tokenizer.save_pretrained(save_path)

    # ---- Architecture diff report ----

    opt_params = get_param_count(model)
    opt_size_mb = get_model_size_mb(model)

    reduction_percent = round((1 - opt_params / base_params) * 100, 2) if base_params > 0 else 0

    diff = {
        "base_model": MODEL_NAME,
        "version": version,
        "optimizations": optimizations_applied,
        "base_parameters": base_params,
        "optimized_parameters": opt_params,
        "reduction_percent": reduction_percent,
        "base_size_mb": base_size_mb,
        "optimized_size_mb": opt_size_mb,
        "base_encoder_layers": base_encoder_layers,
        "optimized_encoder_layers": len(model.encoder.block) if hasattr(model, 'encoder') else base_encoder_layers,
        "pruned_heads": {str(k): v for k, v in pruned_heads.items()} if pruned_heads else {},
        "removed_layers": removed_layers,
        "cpu_threads": 2
    }

    with open(save_path / "architecture_diff.json", "w") as f:
        json.dump(diff, f, indent=2)

    print(f"\n🔥 Model recompiled as {version}")
    print(f"   Optimizations: {optimizations_applied}")
    print(f"   Parameters: {base_params:,} → {opt_params:,} ({reduction_percent}% reduction)")

    return diff, version
