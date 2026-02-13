# backend/app/schemas.py

from pydantic import BaseModel
from typing import Optional, Any


# =====================================================
# Inference
# =====================================================

class InferenceRequest(BaseModel):
    text: str


class InferenceResponse(BaseModel):
    response: str
    request_id: Optional[str] = None
    latency_ms: Optional[float] = None
    drift_detected: Optional[bool] = None
    drift_score: Optional[float] = None


# =====================================================
# Health
# =====================================================

class HealthResponse(BaseModel):
    status: str
    model_version: str
    uptime_seconds: float
    total_requests: int


# =====================================================
# Telemetry
# =====================================================

class TelemetrySummary(BaseModel):
    total_requests: Optional[int] = 0
    avg_latency_ms: Optional[float] = 0.0
    min_latency_ms: Optional[float] = 0.0
    max_latency_ms: Optional[float] = 0.0
    total_input_tokens: Optional[int] = 0
    total_output_tokens: Optional[int] = 0


# =====================================================
# Evolution
# =====================================================

class EvolutionRequest(BaseModel):
    triggered_by: str = "manual"


class EvolutionResponse(BaseModel):
    evolution_status: str
    details: Optional[Any] = None


# =====================================================
# Governance
# =====================================================

class RollbackRequest(BaseModel):
    target_version: Optional[str] = None
    reason: str = "Manual rollback"


class RollbackResponse(BaseModel):
    status: str
    message: str
    rolled_back_to: Optional[str] = None


# =====================================================
# Drift
# =====================================================

class DriftStatus(BaseModel):
    current_memory_size: int
    recent_drift_events: int
    avg_drift_score: Optional[float] = 0.0
