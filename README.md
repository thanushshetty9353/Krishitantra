# Krishitantra – Self-Evolving Small Language Model (SE-SLM)

A full-featured Self-Evolving Small Language Model system that monitors, profiles, analyzes, evolves, and governs a transformer-based language model in real-time.

## Architecture

```
┌─────────────────────────────────────────────┐
│             Frontend Dashboard              │
│   (Overview, Telemetry, Drift, Governance)  │
├─────────────────────────────────────────────┤
│              FastAPI Backend                │
│  /infer  /telemetry  /evolve  /governance   │
├─────────────┬───────────┬───────────────────┤
│  Telemetry  │   Usage   │    Structural     │
│  Monitor    │  Profiler │    Analyzer       │
├─────────────┼───────────┼───────────────────┤
│  Evolution  │   Drift   │   Governance      │
│  Engine     │ Detector  │   Manager         │
├─────────────┴───────────┴───────────────────┤
│        Model (google/flan-t5-small)         │
└─────────────────────────────────────────────┘
```

## Tech Stack

- **Language**: Python 3.10+
- **Framework**: FastAPI + Uvicorn
- **AI/ML**: PyTorch, Hugging Face Transformers
- **Model**: google/flan-t5-small (Seq2Seq)
- **Telemetry**: OpenTelemetry, Prometheus
- **Database**: SQLite
- **Frontend**: HTML/CSS/JavaScript (Dark-themed dashboard)

## Setup

```bash
# Install dependencies
pip install -r requirements.txt

# Run the server
uvicorn backend.app.main:app --reload --host 0.0.0.0 --port 8000
```

Open **http://localhost:8000** to access the dashboard.

## API Endpoints

| Endpoint | Method | Description |
|---|---|---|
| `/` | GET | Dashboard UI |
| `/health` | GET | System health check |
| `/infer` | POST | Run model inference |
| `/telemetry` | GET | View telemetry data |
| `/telemetry/structural` | GET | Structural telemetry |
| `/profiler/run` | POST | Run usage profiler |
| `/profiler/report` | GET | View profiling report |
| `/analysis` | GET | Structural analysis |
| `/evolve` | POST | Trigger evolution cycle |
| `/registry` | GET | Model registry |
| `/registry/{version}` | GET | Specific model version |
| `/drift` | GET | Drift detection status |
| `/governance/audit` | GET | Audit trail |
| `/governance/rollback` | POST | Rollback model |
| `/metrics` | GET | Prometheus metrics |

## System Components

1. **Runtime Telemetry Monitor** – Logs token activations, attention head utilization, layer execution frequency, latency and memory metrics
2. **Usage Profiler** – Aggregates telemetry into frequent token paths, dormant neurons, redundant heads
3. **Structural Analyzer** – Identifies prunable heads, detects redundant FFN layers, scores neuron importance, recommends rewiring
4. **Evolution Engine** – Executes scheduled evolution cycles with head/layer pruning, quantization, and embedding compression
5. **Self-Recompilation** – Recompiles models with weight inheritance, graph rewriting, and lightweight distillation
6. **Validation Sandbox** – Runs accuracy regression tests, hallucination checks, and latency measurement
7. **Drift Detector** – Detects embedding distribution shifts, vocabulary change rate, and intent variance
8. **Model Registry** – Maintains model lineage, compression ratios, accuracy deltas, and evolution metadata
9. **Governance Manager** – Supports instant rollback, evolution audit logs, and change approvals
