# backend/app/model.py

"""
Model Manager (SE-SLM)

Manages the base transformer SLM and optimized versions.
Handles model loading, inference, and structural telemetry hooks.
Applies dynamic quantization at load-time for optimized models.
"""

from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from pathlib import Path
from collections import defaultdict

BASE_MODEL = "google/flan-t5-small"
OPTIMIZED_DIR = Path("models/optimized")

MAX_NEW_TOKENS = 64

# ============================================================
# Structural Telemetry Storage
# ============================================================

# Head-level activation magnitude
head_activation_stats = defaultdict(lambda: defaultdict(float))

# FFN sparsity tracking
ffn_sparsity_stats = defaultdict(float)

# Token frequency tracking
token_frequency = defaultdict(int)


class ModelManager:
    def __init__(self):
        self.tokenizer = None
        self.model = None
        self.current_version = "base"
        self.load_base_model()

    # ============================================================
    # Model Loading
    # ============================================================

    def load_base_model(self):
        print("🔄 Loading base model...")
        self.tokenizer = AutoTokenizer.from_pretrained(BASE_MODEL)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)
        self.model.eval()
        self.current_version = "base"
        self._register_hooks()

    def load_latest_optimized(self):
        """
        Load the latest optimized model version.
        Applies dynamic INT8 quantization at load-time for performance.
        Falls back to base model on any failure.
        """
        if not OPTIMIZED_DIR.exists():
            print("⚠️  No optimized model directory found")
            return

        versions = sorted(
            [v for v in OPTIMIZED_DIR.glob("v*") if v.is_dir() and v.name != "backup"],
            key=lambda x: int(x.name.replace("v", "").split(".")[0])
        )

        if not versions:
            print("⚠️  No optimized model versions found")
            return

        latest = versions[-1]
        print(f"🔥 Loading optimized model: {latest.name}")

        try:
            self.tokenizer = AutoTokenizer.from_pretrained(latest)
            self.model = AutoModelForSeq2SeqLM.from_pretrained(latest)

            # Apply dynamic INT8 quantization at load-time
            # This gives memory/latency benefits without needing to save quantized weights
            try:
                self.model = torch.quantization.quantize_dynamic(
                    self.model,
                    {torch.nn.Linear},
                    dtype=torch.qint8
                )
                print(f"⚡ Applied INT8 quantization to {latest.name}")
            except Exception as qe:
                print(f"⚠️  Quantization skipped: {qe}")

            self.model.eval()
            self.current_version = latest.name
            self._register_hooks()
            print(f"✅ Model loaded: {self.current_version}")

        except Exception as e:
            print(f"❌ Failed to load optimized model {latest.name}: {e}")
            print("↩️  Falling back to base model...")
            self.load_base_model()

    # ============================================================
    # Hook Registration
    # ============================================================

    def _register_hooks(self):
        head_activation_stats.clear()
        ffn_sparsity_stats.clear()

        # Check if model has encoder attribute (non-quantized path)
        if not hasattr(self.model, 'named_modules'):
            return

        for name, module in self.model.named_modules():

            # Self-attention heads (encoder + decoder)
            if "SelfAttention" in name:
                module.register_forward_hook(self.attention_hook(name))

            # Cross-attention (decoder only)
            if "EncDecAttention" in name:
                module.register_forward_hook(self.attention_hook(name))

            # FFN layers
            if "DenseReluDense" in name:
                module.register_forward_hook(self.ffn_hook(name))

    # ============================================================
    # Attention Hook (Head-Level Importance)
    # ============================================================

    def attention_hook(self, layer_name):
        def hook(module, input, output):
            try:
                if isinstance(output, tuple):
                    output = output[0]

                if not isinstance(output, torch.Tensor):
                    return

                batch, seq_len, hidden_size = output.shape

                if hasattr(module, "n_heads"):
                    num_heads = module.n_heads
                else:
                    return

                head_dim = hidden_size // num_heads

                heads = output.view(batch, seq_len, num_heads, head_dim)

                # Mean absolute activation per head
                head_norm = heads.abs().mean(dim=(0, 1, 3))

                for idx, value in enumerate(head_norm):
                    head_activation_stats[layer_name][idx] += value.item()

            except Exception:
                pass

        return hook

    # ============================================================
    # FFN Hook (Neuron Sparsity)
    # ============================================================

    def ffn_hook(self, layer_name):
        def hook(module, input, output):
            try:
                if isinstance(output, tuple):
                    output = output[0]

                if not isinstance(output, torch.Tensor):
                    return

                sparsity = (output.abs() < 1e-6).float().mean().item()

                ffn_sparsity_stats[layer_name] += sparsity

            except Exception:
                pass

        return hook

    # ============================================================
    # Text Generation + Telemetry Extraction
    # ============================================================

    def generate(self, prompt: str):

        # Reset per-request stats
        head_activation_stats.clear()
        ffn_sparsity_stats.clear()

        inputs = self.tokenizer(
            prompt,
            return_tensors="pt",
            truncation=True
        )

        input_tokens = inputs["input_ids"].shape[1]

        # Track token frequency
        token_ids = inputs["input_ids"].flatten().tolist()
        for token_id in token_ids:
            token_frequency[token_id] += 1

        with torch.inference_mode():
            output_ids = self.model.generate(
                **inputs,
                max_new_tokens=MAX_NEW_TOKENS,
                do_sample=False,
                use_cache=True
            )

        output_tokens = output_ids.shape[1]

        response = self.tokenizer.decode(
            output_ids[0],
            skip_special_tokens=True
        )

        return (
            response.strip(),
            input_tokens,
            output_tokens,
            dict(head_activation_stats),
            {
                "ffn_sparsity": dict(ffn_sparsity_stats),
                "token_frequency": dict(token_frequency)
            }
        )


# ============================================================
# Global Runtime Manager
# ============================================================

model_manager = ModelManager()


def generate_text(prompt: str):
    return model_manager.generate(prompt)
