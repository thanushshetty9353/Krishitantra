import sqlite3
import uuid
from datetime import datetime
from pathlib import Path

# Ensure data directory exists
db_path = Path("backend/data/requests.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(db_path, check_same_thread=False)
cursor = conn.cursor()

# ==============================
# Phase 1: Telemetry Tables
# ==============================

cursor.execute("""
CREATE TABLE IF NOT EXISTS telemetry_requests (
    request_id TEXT PRIMARY KEY,
    input_tokens INTEGER,
    output_tokens INTEGER,
    latency_ms REAL,
    timestamp TEXT
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS telemetry_layers (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    layer_name TEXT,
    call_count INTEGER
)
""")

cursor.execute("""
CREATE TABLE IF NOT EXISTS telemetry_attention (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    request_id TEXT,
    attention_module TEXT,
    call_count INTEGER
)
""")

conn.commit()


def log_request_telemetry(
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float
):
    cursor.execute(
        """
        INSERT INTO telemetry_requests
        VALUES (?, ?, ?, ?, ?)
        """,
        (
            request_id,
            input_tokens,
            output_tokens,
            latency_ms,
            datetime.utcnow().isoformat()
        )
    )
    conn.commit()


def log_layer_telemetry(request_id: str, layer_stats: dict):
    for layer_name, count in layer_stats.items():
        cursor.execute(
            """
            INSERT INTO telemetry_layers (request_id, layer_name, call_count)
            VALUES (?, ?, ?)
            """,
            (request_id, layer_name, count)
        )
    conn.commit()


def log_attention_telemetry(request_id: str, attention_stats: dict):
    for attn_name, count in attention_stats.items():
        cursor.execute(
            """
            INSERT INTO telemetry_attention (request_id, attention_module, call_count)
            VALUES (?, ?, ?)
            """,
            (request_id, attn_name, count)
        )
    conn.commit()
