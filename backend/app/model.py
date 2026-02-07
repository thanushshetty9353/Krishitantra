from transformers import AutoTokenizer, AutoModelForSeq2SeqLM
import torch
from collections import defaultdict

# ==============================
# Phase 1: Base Transformer SLM
# + Runtime Telemetry Hooks
# ==============================

MODEL_NAME = "google/flan-t5-small"
MAX_NEW_TOKENS = 128

tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)
model = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
model.eval()

# ------------------------------
# Telemetry counters (per request)
# ------------------------------
layer_call_counts = defaultdict(int)
attention_call_counts = defaultdict(int)


def reset_telemetry_counters():
    layer_call_counts.clear()
    attention_call_counts.clear()


# ------------------------------
# Hook functions
# ------------------------------
def layer_hook(name):
    def hook(module, input, output):
        layer_call_counts[name] += 1
    return hook


def attention_hook(name):
    def hook(module, input, output):
        attention_call_counts[name] += 1
    return hook


# ------------------------------
# Register hooks (ONCE)
# ------------------------------
for name, module in model.named_modules():
    if "block" in name:
        module.register_forward_hook(layer_hook(name))

    if "SelfAttention" in name:
        module.register_forward_hook(attention_hook(name))


def generate_text(prompt: str, max_new_tokens: int = MAX_NEW_TOKENS):
    """
    Phase 1 Inference + Telemetry
    -----------------------------
    Returns:
    - generated text
    - input token count
    - output token count
    - layer usage stats
    - attention usage stats
    """

    reset_telemetry_counters()

    inputs = tokenizer(
        prompt,
        return_tensors="pt",
        truncation=True
    )

    input_tokens = inputs["input_ids"].shape[1]

    with torch.no_grad():
        output_ids = model.generate(
            **inputs,
            max_new_tokens=max_new_tokens,
            do_sample=False
        )

    output_tokens = output_ids.shape[1]

    response = tokenizer.decode(
        output_ids[0],
        skip_special_tokens=True
    )

    return (
        response.strip(),
        input_tokens,
        output_tokens,
        dict(layer_call_counts),
        dict(attention_call_counts)
    )
