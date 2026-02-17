# backend/app/database.py

import sqlite3
import json
import threading
from datetime import datetime
from pathlib import Path

# =====================================================
# Database Setup
# =====================================================

db_path = Path("backend/data/requests.db")
db_path.parent.mkdir(parents=True, exist_ok=True)

# Use valid thread sharing mode but manage cursors carefully
# check_same_thread=False allows multiple threads to use the connection,
# but we must ensure we don't share cursors concurrently.
_conn = sqlite3.connect(db_path, check_same_thread=False)
_conn.row_factory = sqlite3.Row
_lock = threading.Lock()

def get_cursor():
    """Yields a cursor relative to the global connection with locking."""
    with _lock:
        cursor = _conn.cursor()
        try:
            yield cursor
        finally:
            cursor.close()

def init_db():
    with _lock:
        cursor = _conn.cursor()
        # =====================================================
        # Core Request Table
        # =====================================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_requests (
            request_id TEXT PRIMARY KEY,
            input_tokens INTEGER,
            output_tokens INTEGER,
            latency_ms REAL,
            timestamp TEXT
        )
        """)

        # =====================================================
        # Structural Telemetry Table
        # =====================================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS telemetry_structural (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            request_id TEXT,
            head_stats_json TEXT,
            layer_stats_json TEXT
        )
        """)

        # =====================================================
        # Evolution Audit Log Table
        # =====================================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS evolution_audit_log (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            action TEXT,
            version TEXT,
            details_json TEXT,
            status TEXT,
            triggered_by TEXT DEFAULT 'system'
        )
        """)

        # =====================================================
        # Drift History Table
        # =====================================================
        cursor.execute("""
        CREATE TABLE IF NOT EXISTS drift_history (
            id INTEGER PRIMARY KEY AUTOINCREMENT,
            timestamp TEXT,
            drift_score REAL,
            drift_flag INTEGER,
            input_text TEXT,
            embedding_shift REAL,
            vocab_shift REAL,
            intent_variance REAL
        )
        """)
        
        _conn.commit()
        cursor.close()

# Initialize on module load
init_db()

# =====================================================
# Logging Functions
# =====================================================

def log_request_telemetry(
    request_id: str,
    input_tokens: int,
    output_tokens: int,
    latency_ms: float
):
    with _lock:
        cursor = _conn.cursor()
        try:
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
            _conn.commit()
        finally:
            cursor.close()


def log_layer_telemetry(request_id: str, layer_stats: dict):
    # Combined into structural table via log_attention_telemetry
    pass


def log_attention_telemetry(request_id: str, attention_stats: dict):
    from backend.app.model import ffn_sparsity_stats, token_frequency

    layer_bundle = {
        "ffn_sparsity": dict(ffn_sparsity_stats),
        "token_frequency": dict(token_frequency)
    }

    head_json = json.dumps(_safe_json(attention_stats))
    layer_json = json.dumps(_safe_json(layer_bundle))

    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO telemetry_structural
                (request_id, head_stats_json, layer_stats_json)
                VALUES (?, ?, ?)
                """,
                (request_id, head_json, layer_json)
            )
            _conn.commit()
        finally:
            cursor.close()


def log_drift_event(
    drift_score: float,
    drift_flag: bool,
    input_text: str,
    embedding_shift: float = 0.0,
    vocab_shift: float = 0.0,
    intent_variance: float = 0.0
):
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO drift_history
                (timestamp, drift_score, drift_flag, input_text,
                 embedding_shift, vocab_shift, intent_variance)
                VALUES (?, ?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    drift_score,
                    1 if drift_flag else 0,
                    input_text[:200],
                    embedding_shift,
                    vocab_shift,
                    intent_variance
                )
            )
            _conn.commit()
        finally:
            cursor.close()


def log_evolution_audit(
    action: str,
    version: str,
    details: dict,
    status: str,
    triggered_by: str = "system"
):
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute(
                """
                INSERT INTO evolution_audit_log
                (timestamp, action, version, details_json, status, triggered_by)
                VALUES (?, ?, ?, ?, ?, ?)
                """,
                (
                    datetime.utcnow().isoformat(),
                    action,
                    version,
                    json.dumps(details),
                    status,
                    triggered_by
                )
            )
            _conn.commit()
        finally:
            cursor.close()


# =====================================================
# Query Functions
# =====================================================

def get_recent_telemetry(limit: int = 50):
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM telemetry_requests ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()


def get_structural_telemetry(limit: int = 20):
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute(
                """
                SELECT ts.*, tr.latency_ms, tr.timestamp
                FROM telemetry_structural ts
                JOIN telemetry_requests tr ON ts.request_id = tr.request_id
                ORDER BY tr.timestamp DESC
                LIMIT ?
                """,
                (limit,)
            )
            rows = []
            for row in cursor.fetchall():
                r = dict(row)
                try:
                    r["head_stats"] = json.loads(r.pop("head_stats_json", "{}"))
                    r["layer_stats"] = json.loads(r.pop("layer_stats_json", "{}"))
                except (json.JSONDecodeError, TypeError):
                    r["head_stats"] = {}
                    r["layer_stats"] = {}
                rows.append(r)
            return rows
        finally:
            cursor.close()


def get_drift_history(limit: int = 50):
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM drift_history ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            return [dict(row) for row in cursor.fetchall()]
        finally:
            cursor.close()


def get_evolution_audit_log(limit: int = 50):
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute(
                "SELECT * FROM evolution_audit_log ORDER BY timestamp DESC LIMIT ?",
                (limit,)
            )
            rows = []
            for row in cursor.fetchall():
                r = dict(row)
                try:
                    r["details"] = json.loads(r.pop("details_json", "{}"))
                except (json.JSONDecodeError, TypeError):
                    r["details"] = {}
                rows.append(r)
            return rows
        finally:
            cursor.close()


def get_telemetry_summary():
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute("""
                SELECT
                    COUNT(*) as total_requests,
                    ROUND(AVG(latency_ms), 2) as avg_latency_ms,
                    ROUND(MIN(latency_ms), 2) as min_latency_ms,
                    ROUND(MAX(latency_ms), 2) as max_latency_ms,
                    SUM(input_tokens) as total_input_tokens,
                    SUM(output_tokens) as total_output_tokens
                FROM telemetry_requests
            """)
            row = cursor.fetchone()
            return dict(row) if row else {}
        finally:
            cursor.close()


def get_aggregated_head_stats():
    """Aggregate head activation stats from all structural telemetry."""
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute("SELECT head_stats_json FROM telemetry_structural")
            rows = cursor.fetchall()

            aggregated = {}
            for row in rows:
                try:
                    stats = json.loads(row["head_stats_json"])
                    for layer, heads in stats.items():
                        if layer not in aggregated:
                            aggregated[layer] = {}
                        if isinstance(heads, dict):
                            for head_id, value in heads.items():
                                if head_id not in aggregated[layer]:
                                    aggregated[layer][head_id] = 0.0
                                aggregated[layer][head_id] += float(value)
                except (json.JSONDecodeError, TypeError):
                    continue

            return aggregated
        finally:
            cursor.close()


def get_aggregated_ffn_stats():
    """Aggregate FFN sparsity stats from all structural telemetry."""
    with _lock:
        cursor = _conn.cursor()
        try:
            cursor.execute("SELECT layer_stats_json FROM telemetry_structural")
            rows = cursor.fetchall()

            aggregated_sparsity = {}
            count = 0

            for row in rows:
                try:
                    stats = json.loads(row["layer_stats_json"])
                    ffn = stats.get("ffn_sparsity", {})
                    for layer, sparsity in ffn.items():
                        if layer not in aggregated_sparsity:
                            aggregated_sparsity[layer] = 0.0
                        aggregated_sparsity[layer] += float(sparsity)
                    count += 1
                except (json.JSONDecodeError, TypeError):
                    continue

            if count > 0:
                aggregated_sparsity = {
                    k: round(v / count, 4) for k, v in aggregated_sparsity.items()
                }

            return aggregated_sparsity
        finally:
            cursor.close()


# =====================================================
# JSON Safety
# =====================================================

def _safe_json(data):
    if isinstance(data, dict):
        return {str(k): _safe_json(v) for k, v in data.items()}
    elif isinstance(data, list):
        return [_safe_json(v) for v in data]
    else:
        return data
