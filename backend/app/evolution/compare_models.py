# backend/app/evolution/compare_models.py

from transformers import AutoModelForSeq2SeqLM
from pathlib import Path

BASE_MODEL = "google/flan-t5-small"
OPTIMIZED_DIR = Path("models/optimized")


def get_latest_optimized_path():
    """Find the latest optimized model version."""
    if not OPTIMIZED_DIR.exists():
        return None

    versions = sorted(
        [v for v in OPTIMIZED_DIR.glob("v*") if v.is_dir() and v.name != "backup"],
        key=lambda x: int(x.name.replace("v", "").split(".")[0])
    )
    return versions[-1] if versions else None


def compare_models():
    """Compare base model vs latest optimized model."""
    base = AutoModelForSeq2SeqLM.from_pretrained(BASE_MODEL)

    base_params = sum(p.numel() for p in base.parameters())
    base_encoder_layers = len(base.encoder.block)
    base_decoder_layers = len(base.decoder.block)

    result = {
        "base": {
            "model": BASE_MODEL,
            "parameters": base_params,
            "encoder_layers": base_encoder_layers,
            "decoder_layers": base_decoder_layers
        },
        "optimized": None,
        "compression": None
    }

    opt_path = get_latest_optimized_path()

    if opt_path:
        try:
            opt = AutoModelForSeq2SeqLM.from_pretrained(opt_path)
            opt_params = sum(p.numel() for p in opt.parameters())

            result["optimized"] = {
                "version": opt_path.name,
                "parameters": opt_params,
                "encoder_layers": len(opt.encoder.block),
                "decoder_layers": len(opt.decoder.block)
            }

            result["compression"] = {
                "parameter_reduction_percent": round(
                    (1 - opt_params / base_params) * 100, 2
                ),
                "size_ratio": round(opt_params / base_params, 4)
            }
        except Exception as e:
            result["optimized"] = {"error": str(e)}

    return result


if __name__ == "__main__":
    import json
    print(json.dumps(compare_models(), indent=2))
