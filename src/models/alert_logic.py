"""Alert Logic Service - Alert Decision Making

Determines when to create alerts for investigator review.
"""

from typing import Dict, Optional


def should_create_alert(risk_score: float, threshold: float = 70.0) -> bool:
    """Should we alert the investigator?
    
    Args:
        risk_score: Provider risk score (0-100%)
        threshold: Risk threshold for alert (default 70%)
    
    Returns:
        True if alert should be created, False otherwise
    """
    return risk_score > threshold


def get_alert_priority(risk_score: float, flagged_claim_count: int = 0) -> str:
    """Determine alert priority level.
    
    Args:
        risk_score: Risk score 0-100%
        flagged_claim_count: Number of flagged claims
    
    Returns:
        Priority: 'CRITICAL', 'HIGH', or 'MEDIUM'
    """
    if risk_score > 85 or flagged_claim_count > 20:
        return "CRITICAL"
    elif risk_score > 75:
        return "HIGH"
    else:
        return "MEDIUM"


def get_routing_decision(risk_score: float, category: str) -> Dict:
    """Get complete routing decision including SLA.
    
    Args:
        risk_score: Risk score 0-100%
        category: Risk category (LOW/MEDIUM/HIGH)
    
    Returns:
        Dict with action, queue, sla_hours, and reasoning
    """
    if risk_score > 70:
        priority = get_alert_priority(risk_score)
        return {
            "action": "CREATE_ALERT",
            "queue": "INVESTIGATOR",
            "priority": priority,
            "sla_hours": {"CRITICAL": 4, "HIGH": 12, "MEDIUM": 24}.get(priority, 24),
            "reasoning": f"Risk score {risk_score:.1f}% exceeds threshold"
        }
    elif risk_score > 30:
        return {
            "action": "STANDARD_REVIEW",
            "queue": "REVIEW_QUEUE",
            "sla_hours": 48,
            "reasoning": f"Risk score {risk_score:.1f}% requires standard review"
        }
    else:
        return {
            "action": "AUTO_APPROVE",
            "queue": "APPROVED",
            "sla_hours": 2,
            "reasoning": f"Risk score {risk_score:.1f}% is low risk"
        }


def should_escalate_alert(priority: str, hours_unreviewed: float) -> bool:
    """Check if alert should be escalated.
    
    Args:
        priority: Alert priority level
        hours_unreviewed: Hours since alert created
    
    Returns:
        True if should escalate
    """
    thresholds = {"CRITICAL": 48, "HIGH": 120, "MEDIUM": 240}
    threshold_hours = thresholds.get(priority, 240)
    return hours_unreviewed > threshold_hours
