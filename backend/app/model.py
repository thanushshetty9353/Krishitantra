# backend/app/model.py

"""
Model Manager (SE-SLM) - TinyLlama 1.1B Chat GGUF

Uses llama-cpp-python for efficient 4-bit quantized inference.
Provides structural telemetry hooks for SE-SLM compatibility.
"""

import os
import time
import hashlib
import random
from pathlib import Path
from collections import defaultdict
from huggingface_hub import hf_hub_download

GGUF_REPO = "TheBloke/TinyLlama-1.1B-Chat-v1.0-GGUF"
GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"
MODEL_DIR = Path("models/base")
OPTIMIZED_DIR = Path("models/optimized")

MAX_TOKENS = 256
CONTEXT_SIZE = 2048

# TinyLlama architecture constants for telemetry simulation
NUM_LAYERS = 22
NUM_HEADS = 32

# ============================================================
# Structural Telemetry Storage (SE-SLM 3.1 compatible)
# ============================================================

head_activation_stats = defaultdict(lambda: defaultdict(float))
ffn_sparsity_stats = defaultdict(float)
token_frequency = defaultdict(int)


def download_model():
    """Download the GGUF model if not already present."""
    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model_path = MODEL_DIR / GGUF_FILENAME
    if model_path.exists():
        print(f"[MODEL] Already exists: {model_path}")
        return str(model_path)
    print(f"[MODEL] Downloading {GGUF_FILENAME} from {GGUF_REPO}...")
    hf_hub_download(
        repo_id=GGUF_REPO,
        filename=GGUF_FILENAME,
        local_dir=str(MODEL_DIR),
        local_dir_use_symlinks=False
    )
    print(f"[MODEL] Downloaded to: {model_path}")
    return str(model_path)


def format_chat_prompt(user_message, system_prompt=None):
    """Format prompt using TinyLlama chat template."""
    if system_prompt is None:
        system_prompt = (
            "You are Krishitantra AI, a helpful, accurate, and concise assistant. "
            "Answer questions clearly and helpfully."
        )
    SYS = "<" + "|system|" + ">"
    USR = "<" + "|user|" + ">"
    AST = "<" + "|assistant|" + ">"
    EOS = "<" + "/s" + ">"
    prompt = SYS + "\n" + system_prompt + EOS + "\n"
    prompt += USR + "\n" + user_message + EOS + "\n"
    prompt += AST + "\n"
    return prompt


def _simulate_telemetry(prompt_text, output_text):
    """
    Generate simulated structural telemetry for GGUF models.
    GGUF models don't expose internal activations, so we simulate
    telemetry based on input/output characteristics for SE-SLM compatibility.
    """
    head_activation_stats.clear()
    ffn_sparsity_stats.clear()

    seed = int(hashlib.md5(prompt_text.encode()).hexdigest()[:8], 16)
    rng = random.Random(seed)

    for layer_idx in range(NUM_LAYERS):
        layer_name = f"model.layers.{layer_idx}.self_attn"
        for head_idx in range(NUM_HEADS):
            base = 0.3 + (layer_idx / NUM_LAYERS) * 0.5
            noise = rng.gauss(0, 0.1)
            activation = max(0.01, min(1.0, base + noise))
            head_activation_stats[layer_name][head_idx] = activation

    for layer_idx in range(NUM_LAYERS):
        layer_name = f"model.layers.{layer_idx}.mlp"
        base_sparsity = 0.6 - (layer_idx / NUM_LAYERS) * 0.3
        noise = rng.gauss(0, 0.05)
        ffn_sparsity_stats[layer_name] = max(0.0, min(1.0, base_sparsity + noise))

    words = prompt_text.lower().split()
    for w in words:
        token_id = hash(w) % 32000
        token_frequency[token_id] += 1


class ModelManager:
    def __init__(self):
        self.model = None
        self.current_version = "base"
        self.model_path = None
        self.load_base_model()

    def load_base_model(self):
        print("[MODEL] Loading base TinyLlama GGUF model...")
        from llama_cpp import Llama
        model_path = download_model()
        self.model = Llama(
            model_path=model_path,
            n_ctx=CONTEXT_SIZE,
            n_gpu_layers=0,
            n_threads=4,
            verbose=False
        )
        self.model_path = model_path
        self.current_version = "base"
        print(f"[MODEL] Loaded: TinyLlama 1.1B Q4_K_M (version={self.current_version})")

    def load_latest_optimized(self):
        """Load the latest optimized GGUF model version."""
        from llama_cpp import Llama

        if not OPTIMIZED_DIR.exists():
            print("[MODEL] No optimized model directory found")
            return

        versions = sorted(
            [v for v in OPTIMIZED_DIR.glob("v*") if v.is_dir() and v.name != "backup"],
            key=lambda x: int(x.name.replace("v", "").split(".")[0])
        )

        if not versions:
            print("[MODEL] No optimized versions found")
            return

        latest = versions[-1]
        gguf_files = list(latest.glob("*.gguf"))

        if gguf_files:
            model_path = str(gguf_files[0])
        else:
            model_path = str(MODEL_DIR / GGUF_FILENAME)

        print(f"[MODEL] Loading optimized model: {latest.name}")
        try:
            self.model = Llama(
                model_path=model_path,
                n_ctx=CONTEXT_SIZE,
                n_gpu_layers=0,
                n_threads=4,
                verbose=False
            )
            self.model_path = model_path
            self.current_version = latest.name
            print(f"[MODEL] Loaded optimized: {self.current_version}")
        except Exception as e:
            print(f"[MODEL] Failed to load optimized {latest.name}: {e}")
            print("[MODEL] Falling back to base model...")
            self.load_base_model()

    def generate(self, prompt):
        """Generate text and return with telemetry data."""
        head_activation_stats.clear()
        ffn_sparsity_stats.clear()

        formatted = format_chat_prompt(prompt)

        EOS = "<" + "/s" + ">"
        USR = "<" + "|user|" + ">"
        SYS = "<" + "|system|" + ">"

        output = self.model(
            formatted,
            max_tokens=MAX_TOKENS,
            temperature=0.7,
            top_p=0.9,
            top_k=40,
            repeat_penalty=1.1,
            stop=[EOS, USR, SYS],
            echo=False
        )

        response_text = output["choices"][0]["text"].strip()
        output_tokens = output["usage"]["completion_tokens"]
        input_tokens = output["usage"]["prompt_tokens"]

        _simulate_telemetry(prompt, response_text)

        layer_stats = dict(head_activation_stats)
        attention_stats = {
            "ffn_sparsity": dict(ffn_sparsity_stats),
            "token_frequency": dict(token_frequency)
        }

        return (
            response_text,
            input_tokens,
            output_tokens,
            layer_stats,
            attention_stats
        )


# ============================================================
# Global Runtime Manager
# ============================================================

model_manager = ModelManager()


def generate_text(prompt):
    return model_manager.generate(prompt)
