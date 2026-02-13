# backend/app/evolution/validation_sandbox.py

import time
import json
import torch
from pathlib import Path
from transformers import AutoModelForSeq2SeqLM, AutoTokenizer
from difflib import SequenceMatcher

MODEL_NAME = "google/flan-t5-small"
OPTIMIZED_DIR = Path("models/optimized")

VALIDATION_PROMPTS = [
    "Explain quantum computing in simple terms.",
    "Summarize the theory of relativity.",
    "Translate 'good morning' to French.",
    "What is artificial intelligence?",
    "Give three benefits of exercise."
]

FACTUAL_PROMPTS = [
    ("What is the capital of France?", "Paris"),
    ("Who wrote Hamlet?", "Shakespeare"),
    ("What is 2 + 2?", "4")
]

PASS_CRITERIA = {
    "min_similarity": 0.85,
    "max_accuracy_drop": 10,
    "max_hallucination_rate": 0.5
}


def get_latest_model_path():
    versions = sorted(OPTIMIZED_DIR.glob("v*"), key=lambda x: int(x.name.replace("v", "")))
    return versions[-1]


def load_models():
    tokenizer = AutoTokenizer.from_pretrained(MODEL_NAME)

    base = AutoModelForSeq2SeqLM.from_pretrained(MODEL_NAME)
    opt = AutoModelForSeq2SeqLM.from_pretrained(get_latest_model_path())

    base.eval()
    opt.eval()

    return tokenizer, base, opt


def similarity(a, b):
    return SequenceMatcher(None, a, b).ratio()


def evaluate_regression(base, opt, tokenizer):
    sims = []
    for prompt in VALIDATION_PROMPTS:
        inputs = tokenizer(prompt, return_tensors="pt")

        with torch.inference_mode():
            base_out = base.generate(**inputs, max_new_tokens=64)
            opt_out = opt.generate(**inputs, max_new_tokens=64)

        base_text = tokenizer.decode(base_out[0], skip_special_tokens=True)
        opt_text = tokenizer.decode(opt_out[0], skip_special_tokens=True)

        sims.append(similarity(base_text, opt_text))

    return round(sum(sims) / len(sims), 3)


def hallucination_rate(model, tokenizer):
    failures = 0
    for question, expected in FACTUAL_PROMPTS:
        inputs = tokenizer(question, return_tensors="pt")

        with torch.inference_mode():
            output = model.generate(**inputs, max_new_tokens=32)

        text = tokenizer.decode(output[0], skip_special_tokens=True)

        if similarity(expected.lower(), text.lower()) < 0.6:
            failures += 1

    return round(failures / len(FACTUAL_PROMPTS), 2)


def run_validation():

    tokenizer, base, opt = load_models()

    similarity_score = evaluate_regression(base, opt, tokenizer)
    accuracy_drop = round((1 - similarity_score) * 100, 2)
    hallucination = hallucination_rate(opt, tokenizer)

    status = "PASS"

    if similarity_score < PASS_CRITERIA["min_similarity"]:
        status = "FAIL"

    if accuracy_drop > PASS_CRITERIA["max_accuracy_drop"]:
        status = "FAIL"

    if hallucination > PASS_CRITERIA["max_hallucination_rate"]:
        status = "FAIL"

    report = {
        "similarity_score": similarity_score,
        "accuracy_drop_percent": accuracy_drop,
        "hallucination_rate": hallucination,
        "status": status
    }

    with open("validation_report.json", "w") as f:
        json.dump(report, f, indent=2)

    return report
