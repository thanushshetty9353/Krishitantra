# SE-SLM – Phase 0

This project implements Phase 0 of a Self-Evolving Small Language Model (SE-SLM) system.

## Phase 0 Objective
Establish a baseline transformer-based inference system that:
- Loads a pretrained Small Language Model (SLM)
- Exposes inference via FastAPI
- Persists request metadata for future telemetry

## Tech Stack
- Python
- FastAPI
- Hugging Face Transformers
- PyTorch
- SQLite

## Model
- distilgpt2 (Hugging Face)

## API Endpoint
POST /infer

### Input
{
  "text": "Explain transformers"
}

### Output
{
  "response": "Transformers are neural network models..."
}

This phase serves as the baseline system for future optimization and self-evolution phases.
