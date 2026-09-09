"""Taxonomy module defining customer intents, escalation categories, and policies."""

from enum import Enum
from typing import Dict, List


class IntentType(str, Enum):
    """Customer intent categories defined directly from Amazon customer support interactions."""

    ORDER_STATUS_DELIVERY = "ORDER_STATUS_DELIVERY"
    RETURN_REFUND = "RETURN_REFUND"
    DAMAGED_DEFECTIVE_WRONG = "DAMAGED_DEFECTIVE_WRONG"
    ACCOUNT_PAYMENT_SECURITY = "ACCOUNT_PAYMENT_SECURITY"
    PRODUCT_SERVICE_INQUIRY = "PRODUCT_SERVICE_INQUIRY"
    ESCALATION_COMPLAINT = "ESCALATION_COMPLAINT"
    FEEDBACK_CHITCHAT = "FEEDBACK_CHITCHAT"


class EscalationDecision(str, Enum):
    """Routing decision for incoming messages."""

    AUTO_HANDLE = "AUTO_HANDLE"
    ESCALATE_TO_HUMAN = "ESCALATE_TO_HUMAN"


INTENT_DESCRIPTIONS: Dict[IntentType, str] = {
    IntentType.ORDER_STATUS_DELIVERY: (
        "Inquiries regarding shipping status, delivery dates, tracking numbers, "
        "carrier delays, or packages marked delivered but missing."
    ),
    IntentType.RETURN_REFUND: (
        "Requests for return labels, instructions for returning merchandise, "
        "refund timelines, refund status checks, or replacement requests."
    ),
    IntentType.DAMAGED_DEFECTIVE_WRONG: (
        "Reports of damaged packaging or products, defective hardware/items upon arrival, "
        "receiving the wrong item/size, or missing parts from an order."
    ),
    IntentType.ACCOUNT_PAYMENT_SECURITY: (
        "Account access troubles, password resets, Prime membership charges/cancellations, "
        "unauthorized transactions, payment declines, or gift card balance issues."
    ),
    IntentType.PRODUCT_SERVICE_INQUIRY: (
        "Questions regarding product availability, specifications, compatibility, digital "
        "services (Prime Video, Kindle, Fire TV app issues), or general policies."
    ),
    IntentType.ESCALATION_COMPLAINT: (
        "High-friction complaints, repeat failed service attempts, severe agent dissatisfaction, "
        "threats of chargeback/legal action, or explicit demands for a human manager."
    ),
    IntentType.FEEDBACK_CHITCHAT: (
        "Expressions of thanks, general compliments, greetings, light banter, or neutral "
        "comments that do not describe an active problem requiring resolution."
    ),
}

# Policy guidelines defining what must be escalated
ESCALATION_RULES: List[Dict[str, str]] = [
    {
        "trigger": "Financial / Security Risk",
        "reason": "Account access, password resets, unauthorized charges, or account closure require authenticated human review.",
    },
    {
        "trigger": "Missing Package Marked Delivered",
        "reason": "Packages marked as delivered but missing require carrier GPS trace and manual refund/replacement authorization.",
    },
    {
        "trigger": "Severe Frustration & Multi-Agent Failure",
        "reason": "Customer mentions prior failed contacts or threatens churn/legal/chargeback; requires human empathy and supervisor discretion.",
    },
    {
        "trigger": "High-Value Damaged / Safety Hazard",
        "reason": "Damaged electronics, hazardous defects, or high-value claims require human damage review and carrier claim initiation.",
    },
    {
        "trigger": "Policy Exception / Repeated Delay",
        "reason": "Delivery delayed past multiple promised ETAs or returns outside normal return window requiring managerial exception.",
    },
]
