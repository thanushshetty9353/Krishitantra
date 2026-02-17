import sqlite3
import json
import random
import time
from datetime import datetime, timedelta
from pathlib import Path

# Setup paths
BASE_DIR = Path(__file__).resolve().parent
DB_PATH = BASE_DIR.parent / "data" / "requests.db"
DB_PATH.parent.mkdir(parents=True, exist_ok=True)

conn = sqlite3.connect(DB_PATH)
cursor = conn.cursor()

def seed_telemetry():
    print("Seeding telemetry...")
    cursor.execute("DELETE FROM telemetry_requests")
    
    base_time = datetime.utcnow() - timedelta(hours=1)
    
    for i in range(50):
        req_time = base_time + timedelta(minutes=i)
        latency = random.uniform(100, 800)
        # Add some spikes
        if random.random() > 0.9:
            latency += random.uniform(1000, 2000)
            
        cursor.execute(
            "INSERT INTO telemetry_requests VALUES (?, ?, ?, ?, ?)",
            (
                f"req_{i:03d}",
                random.randint(10, 50),  # input tokens
                random.randint(20, 200), # output tokens
                latency,
                req_time.isoformat()
            )
        )
    conn.commit()

def seed_drift():
    print("Seeding drift history...")
    cursor.execute("DELETE FROM drift_history")
    
    base_time = datetime.utcnow() - timedelta(hours=2)
    
    for i in range(30):
        req_time = base_time + timedelta(minutes=i*4)
        score = random.uniform(0.0, 0.2)
        # Create a drift event
        if 20 < i < 25:
            score = random.uniform(0.4, 0.6)
            
        drift_flag = 1 if score > 0.35 else 0
        
        cursor.execute(
            """INSERT INTO drift_history 
            (timestamp, drift_score, drift_flag, input_text, embedding_shift, vocab_shift, intent_variance)
            VALUES (?, ?, ?, ?, ?, ?, ?)""",
            (
                req_time.isoformat(),
                score,
                drift_flag,
                "Simulated user input for drift testing...",
                score * 0.8, # embedding
                score * 0.1, # vocab
                score * 0.1  # intent
            )
        )
    conn.commit()

def seed_evolution():
    print("Seeding evolution audit log...")
    cursor.execute("DELETE FROM evolution_audit_log")
    
    events = [
        ("auto_evolution", "v1.0", "APPROVED", -24),
        ("manual_evolution", "v1.1", "APPROVED", -12),
        ("auto_evolution", "v1.2", "REJECTED", -2),
    ]
    
    for action, version, status, offset_hours in events:
        req_time = datetime.utcnow() + timedelta(hours=offset_hours)
        details = {
            "compression_ratio": 0.85,
            "latency_improvement": "15%",
            "reason": "accuracy_drop_too_high" if status == "REJECTED" else "performance_gain"
        }
        
        cursor.execute(
            """INSERT INTO evolution_audit_log 
            (timestamp, action, version, details_json, status, triggered_by)
            VALUES (?, ?, ?, ?, ?, ?)""",
            (
                req_time.isoformat(),
                action,
                version,
                json.dumps(details),
                status,
                "system"
            )
        )
    conn.commit()

def seed_structural():
    print("Seeding structural telemetry...")
    cursor.execute("DELETE FROM telemetry_structural")
    
    # Just one recent entry to populate the "Dormant" view
    head_stats = {
        "layer.0": {"head_0": 0.1, "head_1": 0.9},
        "layer.1": {"head_0": 0.0, "head_1": 0.8}
    }
    layer_stats = {
        "ffn_sparsity": {"layer.0": 0.2, "layer.1": 0.95},
        "token_frequency": {"token_a": 100}
    }
    
    cursor.execute(
        """INSERT INTO telemetry_structural
        (request_id, head_stats_json, layer_stats_json)
        VALUES (?, ?, ?)""",
        (
            "seed_struct_001",
            json.dumps(head_stats),
            json.dumps(layer_stats)
        )
    )
    conn.commit()

if __name__ == "__main__":
    try:
        seed_telemetry()
        seed_drift()
        seed_evolution()
        seed_structural()
        print("✅ Database seeded successfully!")
    except Exception as e:
        print(f"Error seeding database: {e}")
    finally:
        conn.close()
