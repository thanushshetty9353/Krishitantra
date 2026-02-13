# backend/app/evolution/rollback.py

import shutil
from pathlib import Path

OPTIMIZED_DIR = Path("models/optimized")
BACKUP_PATH = OPTIMIZED_DIR / "backup"


def get_latest_version_path():
    """Dynamically find the latest optimized model version."""
    if not OPTIMIZED_DIR.exists():
        return None

    versions = sorted(
        [v for v in OPTIMIZED_DIR.glob("v*") if v.is_dir() and v.name != "backup"],
        key=lambda x: int(x.name.replace("v", "").split(".")[0])
    )

    return versions[-1] if versions else None


def backup_model():
    """Backup the current latest model before evolution."""
    latest = get_latest_version_path()

    if latest and latest.exists():
        BACKUP_PATH.mkdir(parents=True, exist_ok=True)
        shutil.copytree(latest, BACKUP_PATH, dirs_exist_ok=True)
        print(f"📦 Backed up model {latest.name} → backup/")
        return True

    print("⚠️  No model to backup")
    return False


def rollback_model():
    """Rollback to the backed-up model version."""
    latest = get_latest_version_path()

    if BACKUP_PATH.exists() and latest:
        shutil.copytree(BACKUP_PATH, latest, dirs_exist_ok=True)
        print(f"⏪ Rolled back to backup → {latest.name}")
        return True

    print("⚠️  No backup available for rollback")
    return False


def rollback_to_version(target_version: str):
    """Rollback to a specific version by making it the active model."""
    target_path = OPTIMIZED_DIR / target_version

    if not target_path.exists():
        return {"status": "FAIL", "reason": f"Version {target_version} not found"}

    latest = get_latest_version_path()
    if latest:
        backup_model()

    return {
        "status": "OK",
        "rolled_back_to": target_version,
        "path": str(target_path)
    }
