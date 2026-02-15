# backend/app/evolution/recompiler.py

"""
Self-Recompilation Module (SE-SLM Requirement 3.5)

The system shall recompile models using:
- Weight inheritance
- Graph rewriting
- Lightweight distillation

Supported optimizations:
- Head pruning (T5-specific)
- Layer pruning (encoder block removal)
- Quantization (applied at load-time)
"""

import json
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
# Head Pruning (T5-specific)
# ============================================================

def prune_attention_heads(model, heads_to_prune: dict = None):
    """
    Prune specified attention heads from the T5 model.
    heads_to_prune: {layer_idx: [head_indices]}

    For google/flan-t5-small:
    - 8 encoder layers, 8 decoder layers
    - 8 attention heads per layer (d_model=512, d_kv=64)
    """
    num_encoder_layers = len(model.encoder.block)

    if not heads_to_prune:
        # Default: prune 2 least-important heads from last 2 encoder layers
        heads_to_prune = {
            num_encoder_layers - 1: [6, 7],  # last layer, prune heads 6 & 7
            num_encoder_layers - 2: [7],     # second-to-last, prune head 7
        }

    pruned_info = {}
    total_pruned = 0

    for layer_idx, head_list in heads_to_prune.items():
        if layer_idx >= num_encoder_layers or layer_idx < 0:
            continue

        block = model.encoder.block[layer_idx]
        attention = block.layer[0].SelfAttention

        # T5 pruning: zero out the weights for specified heads
        n_heads = attention.n_heads
        d_kv = attention.key_value_proj_dim

        valid_heads = [h for h in head_list if h < n_heads]
        if not valid_heads:
            continue

        # Zero out the query, key, value weights for pruned heads
        with torch.no_grad():
            for head_idx in valid_heads:
                start = head_idx * d_kv
                end = start + d_kv

                # Zero out Q, K, V projections for this head
                if hasattr(attention, 'q'):
                    attention.q.weight.data[:, start:end] = 0
                if hasattr(attention, 'k'):
                    attention.k.weight.data[:, start:end] = 0
                if hasattr(attention, 'v'):
                    attention.v.weight.data[:, start:end] = 0

        pruned_info[str(layer_idx)] = valid_heads
        total_pruned += len(valid_heads)

    if total_pruned > 0:
        print(f"✂️  Pruned {total_pruned} attention heads across {len(pruned_info)} layers")
        return True, pruned_info
    else:
        print("⚠️  No heads were pruned")
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

    # Enforce safety cap (SE-SLM §7: Max 40% pruning per cycle)
    layer_indices = layer_indices[:max_removable]

    removed = []
    for idx in sorted(layer_indices, reverse=True):
        if idx < len(model.encoder.block) and len(model.encoder.block) > 2:
            del model.encoder.block[idx]
            removed.append(idx)

    # Update config
    model.config.num_layers = len(model.encoder.block)

    if removed:
        print(f"✂️  Removed encoder layers: {removed} ({len(model.encoder.block)} remaining)")

    return removed


# ============================================================
# Main Recompiler
# ============================================================

def recompile_model(
    optimization: str = "all",
    heads_to_prune: dict = None,
    layers_to_remove: list = None
):
    """
    Full recompilation pipeline (SE-SLM §3.5):
    1. Load base model (weight inheritance)
    2. Apply head pruning (graph rewriting)
    3. Apply layer pruning if requested
    4. Save optimized model
    5. Generate architecture diff report

    NOTE: Quantization is applied at load-time in model.py
    (dynamically quantized models cannot be serialized).
    """
    print(f"\n🔧 Recompiling model with optimization: {optimization}")

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

    # ---- Apply head pruning ----
    if optimization in ("head_pruning", "all"):
        success, pruned = prune_attention_heads(model, heads_to_prune)
        if success:
            optimizations_applied.append("head_pruning")
            pruned_heads = pruned

    # ---- Apply layer pruning ----
    if optimization in ("layer_pruning", "all"):
        removed = prune_layers(model, layers_to_remove)
        if removed:
            optimizations_applied.append("layer_pruning")
            removed_layers = removed

    # If nothing was applied yet, do default head pruning
    if not optimizations_applied:
        success, pruned = prune_attention_heads(model)
        if success:
            optimizations_applied.append("head_pruning")
            pruned_heads = pruned

    # Note: quantization is applied at load-time for memory savings
    optimizations_applied.append("dynamic_int8_quantization_at_load")

    # ---- Save (non-quantized so save_pretrained works) ----
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
        "optimized_encoder_layers": len(model.encoder.block),
        "pruned_heads": pruned_heads,
        "removed_layers": removed_layers,
        "cpu_threads": 2
    }

    with open(save_path / "architecture_diff.json", "w") as f:
        json.dump(diff, f, indent=2)

    print(f"\n🔥 Model recompiled as {version}")
    print(f"   Optimizations: {optimizations_applied}")
    print(f"   Parameters: {base_params:,} → {opt_params:,} ({reduction_percent}% reduction)")
    print(f"   Encoder layers: {base_encoder_layers} → {len(model.encoder.block)}")

    return diff, version
