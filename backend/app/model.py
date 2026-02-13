# backend/app/model.py

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
        if not OPTIMIZED_DIR.exists():
            return

        versions = sorted(
            OPTIMIZED_DIR.glob("v*"),
            key=lambda x: int(x.name.replace("v", ""))
        )

        if not versions:
            return

        latest = versions[-1]
        print(f"🔥 Loading optimized model: {latest.name}")

        self.tokenizer = AutoTokenizer.from_pretrained(latest)
        self.model = AutoModelForSeq2SeqLM.from_pretrained(latest)
        self.model.eval()
        self.current_version = latest.name
        self._register_hooks()

    # ============================================================
    # Hook Registration
    # ============================================================

    def _register_hooks(self):
        head_activation_stats.clear()
        ffn_sparsity_stats.clear()

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
