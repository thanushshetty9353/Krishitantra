from fastapi import FastAPI
import time
import uuid

from backend.app.schemas import InferenceRequest, InferenceResponse
from backend.app.model import generate_text
from backend.app.database import (
    log_request_telemetry,
    log_layer_telemetry,
    log_attention_telemetry
)

app = FastAPI(
    title="Krishitantra",
    description="Base Transformer SLM with Runtime Telemetry Monitoring",
    version="1.0.0"
)

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    """
    Phase 1 Inference Endpoint
    --------------------------
    Collects runtime telemetry:
    - tokens
    - layers
    - attention
    - latency
    """

    request_id = str(uuid.uuid4())

    start_time = time.perf_counter()

    (
        output_text,
        input_tokens,
        output_tokens,
        layer_stats,
        attention_stats
    ) = generate_text(request.text)

    latency_ms = (time.perf_counter() - start_time) * 1000

    # Store telemetry
    log_request_telemetry(
        request_id,
        input_tokens,
        output_tokens,
        latency_ms
    )

    log_layer_telemetry(request_id, layer_stats)
    log_attention_telemetry(request_id, attention_stats)

    return InferenceResponse(response=output_text)
