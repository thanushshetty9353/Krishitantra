# backend/app/evolution/evolution_logger.py

import json
from datetime import datetime

def log_evolution(results, selected, path="model_evolution_history.json"):
    log = {
        "timestamp": datetime.utcnow().isoformat(),
        "candidates": results,
        "selected": selected
    }

    with open(path, "w") as f:
        json.dump(log, f, indent=2)
