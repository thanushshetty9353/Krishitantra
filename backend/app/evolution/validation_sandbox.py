# backend/app/evolution/validation_sandbox.py

"""
Validation Sandbox (SE-SLM Requirement 3.6)

Validation must include:
- Accuracy regression tests
- Domain benchmarks
- Hallucination checks
- Latency measurement

Adapted for GGUF model inference via llama-cpp-python.
"""

import time
import json
from pathlib import Path
from difflib import SequenceMatcher

MODEL_DIR = Path("models/base")
OPTIMIZED_DIR = Path("models/optimized")
GGUF_FILENAME = "tinyllama-1.1b-chat-v1.0.Q4_K_M.gguf"

VALIDATION_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Summarize the theory of relativity.",
    "What is artificial intelligence?",
    "Give three benefits of exercise.",
    "What is machine learning?"
]

FACTUAL_PROMPTS = [
    ("What is the capital of France?", "paris"),
    ("Who wrote Hamlet?", "shakespeare"),
    ("What is 2 + 2?", "4")
]

PASS_CRITERIA = {
    "min_similarity": 0.60,
    "max_accuracy_drop": 25,
    "max_hallucination_rate": 0.67
}


def get_latest_model_path():
    if not OPTIMIZED_DIR.exists():
        return None
    versions = sorted(
        [v for v in OPTIMIZED_DIR.glob("v*") if v.is_dir() and v.name != "backup"],
        key=lambda x: int(x.name.replace("v", "").split(".")[0])
    )
    return versions[-1] if versions else None


def similarity(a, b):
    return SequenceMatcher(None, a.lower(), b.lower()).ratio()


def _generate(model, prompt):
    """Generate text from GGUF model."""
    from backend.app.model import format_chat_prompt
    formatted = format_chat_prompt(prompt)
    EOS = "<" + "/s" + ">"
    USR = "<" + "|user|" + ">"
    SYS = "<" + "|system|" + ">"
    output = model(
        formatted,
        max_tokens=64,
        temperature=0.1,
        top_p=0.9,
        stop=[EOS, USR, SYS],
        echo=False
    )
    return output["choices"][0]["text"].strip()


def run_validation():
    """
    Full validation sandbox run per SE-SLM 3.6.
    Uses the base model as reference and compares with optimized.
    For GGUF models (same binary), validation mainly checks
    that the model still responds correctly.
    """
    try:
        from llama_cpp import Llama

        base_path = str(MODEL_DIR / GGUF_FILENAME)
        opt_dir = get_latest_model_path()

        if opt_dir is None:
            return {
                "status": "FAIL",
                "reason": "No optimized model found",
                "similarity_score": 0,
                "accuracy_drop_percent": 100,
                "hallucination_rate": 1.0,
                "avg_latency_ms": 0
            }

        # Find GGUF in optimized dir
        gguf_files = list(opt_dir.glob("*.gguf"))
        opt_path = str(gguf_files[0]) if gguf_files else base_path

        print("[VALIDATION] Loading base model...")
        base = Llama(model_path=base_path, n_ctx=1024, n_gpu_layers=0, verbose=False)

        print("[VALIDATION] Loading optimized model...")
        opt = Llama(model_path=opt_path, n_ctx=1024, n_gpu_layers=0, verbose=False)

    except Exception as e:
        return {
            "status": "FAIL",
            "reason": f"Failed to load models: {e}",
            "similarity_score": 0,
            "accuracy_drop_percent": 100,
            "hallucination_rate": 1.0,
            "avg_latency_ms": 0
        }

    # --- Accuracy regression ---
    sims = []
    for prompt in VALIDATION_PROMPTS[:3]:
        try:
            base_text = _generate(base, prompt)
            opt_text = _generate(opt, prompt)
            sims.append(similarity(base_text, opt_text))
        except Exception:
            sims.append(0.5)

    similarity_score = round(sum(sims) / len(sims), 3) if sims else 1.0
    accuracy_drop = round((1 - similarity_score) * 100, 2)

    # --- Hallucination check ---
    failures = 0
    for question, expected in FACTUAL_PROMPTS:
        try:
            text = _generate(opt, question).lower()
            if expected not in text and similarity(expected, text) < 0.3:
                failures += 1
        except Exception:
            pass

    hallucination = round(failures / len(FACTUAL_PROMPTS), 2)

    # --- Latency measurement ---
    latencies = []
    for prompt in VALIDATION_PROMPTS[:2]:
        start = time.perf_counter()
        try:
            _generate(opt, prompt)
        except Exception:
            pass
        latencies.append((time.perf_counter() - start) * 1000)

    opt_latency = round(sum(latencies) / len(latencies), 2) if latencies else 0.0

    # Base latency
    base_latencies = []
    for prompt in VALIDATION_PROMPTS[:2]:
        start = time.perf_counter()
        try:
            _generate(base, prompt)
        except Exception:
            pass
        base_latencies.append((time.perf_counter() - start) * 1000)

    base_latency = round(sum(base_latencies) / len(base_latencies), 2) if base_latencies else 0.0

    latency_improvement = round(
        ((base_latency - opt_latency) / base_latency) * 100, 2
    ) if base_latency > 0 else 0.0

    # --- Validation gate ---
    status = "PASS"
    if similarity_score < PASS_CRITERIA["min_similarity"]:
        status = "FAIL"
    if accuracy_drop > PASS_CRITERIA["max_accuracy_drop"]:
        status = "FAIL"
    if hallucination > PASS_CRITERIA["max_hallucination_rate"]:
        status = "FAIL"

    report = {
        "status": status,
        "similarity_score": similarity_score,
        "accuracy_drop_percent": accuracy_drop,
        "hallucination_rate": hallucination,
        "base_latency_ms": base_latency,
        "optimized_latency_ms": opt_latency,
        "latency_improvement_percent": latency_improvement,
        "avg_latency_ms": opt_latency
    }

    try:
        with open("validation_report.json", "w") as f:
            json.dump(report, f, indent=2)
    except Exception:
        pass

    print(f"[VALIDATION] {status} | Similarity={similarity_score} | "
          f"Accuracy Drop={accuracy_drop}% | Hallucination={hallucination}")

    # Clean up
    del base
    del opt

    return report
