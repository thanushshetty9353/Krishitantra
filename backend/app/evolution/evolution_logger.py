# backend/app/evolution/evolution_logger.py

"""
Evolution Audit Logger
Persists evolution events both to JSON file and SQLite database.
"""

import json
from datetime import datetime
from backend.app.database import log_evolution_audit


def log_evolution(results, selected, path="model_evolution_history.json"):
    """
    Log evolution cycle results.
    Writes to both JSON file and SQLite audit table.
    """
    log_entry = {
        "timestamp": datetime.utcnow().isoformat(),
        "candidates_evaluated": len(results),
        "candidates": results,
        "selected": selected
    }

    # --- JSON file log ---
    try:
        with open(path, "r") as f:
            history = json.load(f)
            if not isinstance(history, list):
                history = [history]
    except (FileNotFoundError, json.JSONDecodeError):
        history = []

    history.append(log_entry)

    with open(path, "w") as f:
        json.dump(history, f, indent=2)

    # --- Database audit log ---
    try:
        log_evolution_audit(
            action="evolution_candidate_selected",
            version=selected.get("candidate", {}).get("prune_blocks", ["unknown"])[0]
                if isinstance(selected, dict) else "unknown",
            details={
                "candidates_evaluated": len(results),
                "selected_score": selected.get("score", 0) if isinstance(selected, dict) else 0,
                "selected_candidate": selected
            },
            status="LOGGED",
            triggered_by="evolution_engine"
        )
    except Exception as e:
        print(f"⚠️  DB audit log failed: {e}")

    print(f"📝 Evolution logged: {len(results)} candidates, best score: "
          f"{selected.get('score', 'N/A') if isinstance(selected, dict) else 'N/A'}")
