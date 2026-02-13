# backend/app/main.py

"""
Krishitantra – Self-Evolving Small Language Model (SE-SLM)
Main FastAPI Application

Provides API endpoints for:
- Inference
- Runtime telemetry monitoring
- Usage profiling
- Structural analysis
- Evolution engine
- Model registry
- Drift detection
- Governance & rollback
- Prometheus metrics
"""

import time
import uuid
import threading
from pathlib import Path

from fastapi import FastAPI, HTTPException
from fastapi.responses import Response, FileResponse, HTMLResponse
from fastapi.staticfiles import StaticFiles
from fastapi.middleware.cors import CORSMiddleware

from backend.app.schemas import (
    InferenceRequest, InferenceResponse,
    HealthResponse, TelemetrySummary,
    EvolutionRequest, EvolutionResponse,
    RollbackRequest, RollbackResponse
)
from backend.app.model import generate_text, model_manager
from backend.app.database import (
    log_request_telemetry,
    log_layer_telemetry,
    log_attention_telemetry,
    log_drift_event,
    log_evolution_audit,
    get_recent_telemetry,
    get_structural_telemetry,
    get_drift_history,
    get_evolution_audit_log,
    get_telemetry_summary,
    get_aggregated_head_stats,
    get_aggregated_ffn_stats
)
from backend.app.drift_detector import detect_drift, get_drift_status
from backend.app.evolution.orchestrator import run_evolution_cycle
from backend.app.usage_profiler import generate_usage_and_structure_report
from backend.app.structural_analyzer import run_full_analysis
from backend.app.evolution.model_registry import (
    get_registry, get_model_entry, get_registry_summary
)
from backend.app.governance import (
    perform_rollback, get_audit_log, approve_evolution,
    reject_evolution, get_governance_summary
)

# ======================================================
# Prometheus Metrics
# ======================================================

from prometheus_client import Counter, Histogram, Gauge, generate_latest

REQUEST_COUNT = Counter(
    "inference_requests_total",
    "Total inference requests"
)

LATENCY = Histogram(
    "inference_latency_seconds",
    "Inference latency"
)

TOKEN_COUNT = Counter(
    "tokens_processed_total",
    "Total tokens processed"
)

DRIFT_EVENTS = Counter(
    "domain_drift_events_total",
    "Detected drift events"
)

ACTIVE_MODEL_VERSION = Gauge(
    "active_model_version_info",
    "Currently active model version (1 = active)",
    ["version"]
)

# ======================================================
# OpenTelemetry Tracing
# ======================================================

from opentelemetry import trace
from opentelemetry.sdk.trace import TracerProvider
from opentelemetry.sdk.trace.export import BatchSpanProcessor, ConsoleSpanExporter
from opentelemetry.instrumentation.fastapi import FastAPIInstrumentor

trace.set_tracer_provider(TracerProvider())
span_processor = BatchSpanProcessor(ConsoleSpanExporter())
trace.get_tracer_provider().add_span_processor(span_processor)

# ======================================================
# App Initialization
# ======================================================

app = FastAPI(
    title="Krishitantra – SE-SLM",
    description="Self-Evolving Small Language Model system with telemetry, "
                "evolution, and governance capabilities.",
    version="3.0.0"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

FastAPIInstrumentor.instrument_app(app)

# Track startup time
START_TIME = time.time()
TOTAL_REQUESTS = 0

# Set initial model version gauge
ACTIVE_MODEL_VERSION.labels(version=model_manager.current_version).set(1)

# Mount frontend static files
FRONTEND_DIR = Path(__file__).resolve().parent.parent.parent / "frontend"
if FRONTEND_DIR.exists():
    app.mount("/static", StaticFiles(directory=str(FRONTEND_DIR)), name="static")


# ======================================================
# Evolution Control (Stability Layer)
# ======================================================

DRIFT_THRESHOLD = 0.35
EVOLUTION_COOLDOWN = 600  # 10 minutes
LAST_EVOLUTION_TIME = 0
EVOLUTION_LOCK = threading.Lock()


def trigger_evolution():
    """
    Safe evolution trigger with cooldown + thread lock.
    Prevents infinite recompile loops.
    """
    global LAST_EVOLUTION_TIME

    with EVOLUTION_LOCK:
        current_time = time.time()

        if (current_time - LAST_EVOLUTION_TIME) < EVOLUTION_COOLDOWN:
            print("⏳ Evolution skipped (cooldown active)")
            return

        print("🚀 High drift detected → Starting evolution cycle")
        LAST_EVOLUTION_TIME = current_time

    result = run_evolution_cycle()
    print("🧠 Evolution result:", result)

    # Log to audit
    log_evolution_audit(
        action="auto_evolution",
        version=result.get("architecture_diff", {}).get("version", "unknown")
            if isinstance(result, dict) else "unknown",
        details=result if isinstance(result, dict) else {"result": str(result)},
        status=result.get("evolution_status", "UNKNOWN")
            if isinstance(result, dict) else "UNKNOWN",
        triggered_by="drift_detection"
    )


# ======================================================
# Root – Serve Frontend Dashboard
# ======================================================

@app.get("/", response_class=HTMLResponse)
def serve_dashboard():
    index_path = FRONTEND_DIR / "index.html"
    if index_path.exists():
        return FileResponse(str(index_path))
    return HTMLResponse(
        "<h1>Krishitantra SE-SLM</h1>"
        "<p>Frontend not found. API docs at <a href='/docs'>/docs</a></p>"
    )


# ======================================================
# Health Endpoint
# ======================================================

@app.get("/health")
def health():
    global TOTAL_REQUESTS
    summary = get_telemetry_summary()
    return {
        "status": "healthy",
        "model_version": model_manager.current_version,
        "uptime_seconds": round(time.time() - START_TIME, 2),
        "total_requests": summary.get("total_requests", 0),
        "model_name": "google/flan-t5-small"
    }


# ======================================================
# Inference Endpoint
# ======================================================

@app.post("/infer", response_model=InferenceResponse)
def infer(request: InferenceRequest):
    global TOTAL_REQUESTS
    TOTAL_REQUESTS += 1

    REQUEST_COUNT.inc()

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

    LATENCY.observe(latency_ms / 1000)
    TOKEN_COUNT.inc(input_tokens + output_tokens)

    # --------------------------------------------------
    # Drift Detection
    # --------------------------------------------------

    drift_flag, drift_score, drift_components = detect_drift(request.text)

    if drift_flag:
        DRIFT_EVENTS.inc()

        # Log drift event to database
        log_drift_event(
            drift_score=drift_score,
            drift_flag=drift_flag,
            input_text=request.text,
            embedding_shift=drift_components.get("embedding_shift", 0),
            vocab_shift=drift_components.get("vocab_shift", 0),
            intent_variance=drift_components.get("intent_variance", 0)
        )

        if drift_score > DRIFT_THRESHOLD:
            threading.Thread(
                target=trigger_evolution,
                daemon=True
            ).start()

    # --------------------------------------------------
    # Telemetry Logging
    # --------------------------------------------------

    log_request_telemetry(
        request_id,
        input_tokens,
        output_tokens,
        latency_ms
    )

    log_layer_telemetry(request_id, layer_stats)
    log_attention_telemetry(request_id, attention_stats)

    return InferenceResponse(
        response=output_text,
        request_id=request_id,
        latency_ms=round(latency_ms, 2),
        drift_detected=drift_flag,
        drift_score=drift_score
    )


# ======================================================
# Telemetry Endpoints
# ======================================================

@app.get("/telemetry")
def get_telemetry(limit: int = 50):
    return {
        "summary": get_telemetry_summary(),
        "recent_requests": get_recent_telemetry(limit)
    }


@app.get("/telemetry/structural")
def get_structural(limit: int = 20):
    return {
        "structural_telemetry": get_structural_telemetry(limit)
    }


# ======================================================
# Profiler Endpoints
# ======================================================

@app.post("/profiler/run")
def run_profiler():
    try:
        report = generate_usage_and_structure_report()
        return {"status": "OK", "report": report}
    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


@app.get("/profiler/report")
def get_profiler_report():
    report_path = Path("phase2_usage_report.json")
    if not report_path.exists():
        return {"status": "NOT_FOUND", "message": "Run /profiler/run first"}

    import json
    with open(report_path) as f:
        return json.load(f)


# ======================================================
# Structural Analysis Endpoint
# ======================================================

@app.get("/analysis")
def get_analysis():
    head_stats = get_aggregated_head_stats()
    ffn_stats = get_aggregated_ffn_stats()

    if not head_stats and not ffn_stats:
        return {
            "status": "NO_DATA",
            "message": "Run some inference requests first to generate telemetry data"
        }

    result = run_full_analysis(head_stats, ffn_stats)
    return {"status": "OK", "analysis": result}


# ======================================================
# Evolution Endpoints
# ======================================================

@app.post("/evolve")
def trigger_evolution_manual(request: EvolutionRequest = None):
    triggered_by = request.triggered_by if request else "manual"

    try:
        result = run_evolution_cycle()

        log_evolution_audit(
            action="manual_evolution",
            version=result.get("architecture_diff", {}).get("version", "unknown")
                if isinstance(result, dict) else "unknown",
            details=result if isinstance(result, dict) else {"result": str(result)},
            status=result.get("evolution_status", "UNKNOWN")
                if isinstance(result, dict) else "UNKNOWN",
            triggered_by=triggered_by
        )

        # Reload model if evolution was approved
        if isinstance(result, dict) and result.get("evolution_status") == "APPROVED":
            model_manager.load_latest_optimized()
            ACTIVE_MODEL_VERSION.labels(
                version=model_manager.current_version
            ).set(1)

        return {"status": "OK", "result": result}

    except Exception as e:
        raise HTTPException(status_code=500, detail=str(e))


# ======================================================
# Model Registry Endpoints
# ======================================================

@app.get("/registry")
def view_registry():
    return {
        "summary": get_registry_summary(),
        "models": get_registry()
    }


@app.get("/registry/{version}")
def view_model_version(version: str):
    entry = get_model_entry(version)
    if not entry:
        raise HTTPException(status_code=404, detail=f"Version {version} not found")
    return entry


# ======================================================
# Drift Endpoints
# ======================================================

@app.get("/drift")
def view_drift(limit: int = 50):
    return {
        "detector_status": get_drift_status(),
        "history": get_drift_history(limit)
    }


# ======================================================
# Governance Endpoints
# ======================================================

@app.get("/governance/audit")
def view_audit_log(limit: int = 50):
    return {
        "summary": get_governance_summary(),
        "audit_log": get_audit_log(limit)
    }


@app.post("/governance/rollback")
def rollback(request: RollbackRequest):
    result = perform_rollback(
        target_version=request.target_version,
        reason=request.reason
    )

    if result.get("status") == "OK":
        model_manager.load_latest_optimized()
        ACTIVE_MODEL_VERSION.labels(
            version=model_manager.current_version
        ).set(1)

    return result


@app.post("/governance/approve/{version}")
def approve(version: str):
    return approve_evolution(version)


@app.post("/governance/reject/{version}")
def reject(version: str, reason: str = "Not specified"):
    return reject_evolution(version, reason)


# ======================================================
# Prometheus Metrics Endpoint
# ======================================================

@app.get("/metrics")
def metrics():
    return Response(generate_latest(), media_type="text/plain")
