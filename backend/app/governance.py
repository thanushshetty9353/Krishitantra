# backend/app/governance.py

"""
Governance Manager (SE-SLM Requirement 3.9)

The system shall support:
- Instant rollback
- Evolution audit logs
- Change approvals
"""

from backend.app.database import (
    log_evolution_audit,
    get_evolution_audit_log
)
from backend.app.evolution.rollback import (
    rollback_model,
    rollback_to_version,
    backup_model,
    get_latest_version_path
)
from backend.app.evolution.model_registry import get_registry


# ============================================================
# Rollback Operations
# ============================================================

def perform_rollback(target_version: str = None, reason: str = "Manual rollback"):
    """
    Perform instant rollback to backup or specific version.
    Logs the action in audit trail.
    """
    current = get_latest_version_path()
    current_name = current.name if current else "unknown"

    if target_version:
        result = rollback_to_version(target_version)
        action = f"rollback_to_{target_version}"
    else:
        success = rollback_model()
        result = {
            "status": "OK" if success else "FAIL",
            "rolled_back_to": "backup"
        }
        action = "rollback_to_backup"

    # Log audit event
    log_evolution_audit(
        action=action,
        version=current_name,
        details={
            "reason": reason,
            "target": target_version or "backup",
            "result": result
        },
        status=result.get("status", "OK"),
        triggered_by="governance"
    )

    return result


# ============================================================
# Audit Log
# ============================================================

def get_audit_log(limit: int = 50):
    """Retrieve complete evolution audit trail."""
    return get_evolution_audit_log(limit)


# ============================================================
# Change Approval
# ============================================================

def approve_evolution(version: str, approver: str = "admin"):
    """Approve a model evolution (post-validation)."""
    log_evolution_audit(
        action="evolution_approved",
        version=version,
        details={"approver": approver},
        status="APPROVED",
        triggered_by=approver
    )
    return {"status": "APPROVED", "version": version, "approver": approver}


def reject_evolution(version: str, reason: str, rejector: str = "admin"):
    """Reject a model evolution and trigger rollback."""
    # Rollback
    rollback_model()

    log_evolution_audit(
        action="evolution_rejected",
        version=version,
        details={
            "rejector": rejector,
            "reason": reason
        },
        status="REJECTED",
        triggered_by=rejector
    )

    return {
        "status": "REJECTED",
        "version": version,
        "reason": reason,
        "rollback_performed": True
    }


# ============================================================
# Governance Summary
# ============================================================

def get_governance_summary():
    """Get summary of governance state."""
    registry = get_registry()
    audit = get_audit_log(limit=10)

    latest_version = get_latest_version_path()

    return {
        "current_model": latest_version.name if latest_version else "base",
        "total_evolutions": len(registry),
        "recent_audit_events": len(audit),
        "last_audit_action": audit[0]["action"] if audit else "none",
        "registry_versions": [
            entry.get("version", "unknown") for entry in registry
        ]
    }
