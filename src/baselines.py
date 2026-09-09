"""Baselines module implementing Trivial and Simple benchmark baselines.

1. Trivial Baseline:
   - Intent: Majority class predictor (always predicts ORDER_STATUS_DELIVERY)
   - Escalation: Never escalate (always AUTO_HANDLE)
   - Reply: Static canned generic message

2. Simple Baseline:
   - Intent: Standard TF-IDF + Logistic Regression (uncalibrated, no safety rules)
   - Escalation: Naive keyword search heuristic ("lawyer", "stolen", "manager", "chargeback")
   - Reply: Verbatim raw nearest-neighbor historical reply without LLM conditioning
"""

from typing import Any, Dict, List, Optional
import numpy as np
from pydantic import BaseModel
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.data_loader import load_knowledge_base
from src.retriever import HistoricalResolutionRetriever
from src.taxonomy import IntentType, EscalationDecision


class BaselineResult(BaseModel):
    customer_text: str
    intent: str
    escalation_decision: str
    escalation_reason: str
    drafted_reply: str


class TrivialBaseline:
    """Baseline 1: Majority class intent + Static canned response + Never escalate."""

    def __init__(self):
        self.majority_intent = IntentType.ORDER_STATUS_DELIVERY.value
        self.canned_reply = (
            "Thank you for contacting Amazon Help! Please check your orders page or "
            "contact customer service for assistance. ^AmazonBot"
        )

    def process(self, customer_text: str) -> BaselineResult:
        return BaselineResult(
            customer_text=customer_text,
            intent=self.majority_intent,
            escalation_decision=EscalationDecision.AUTO_HANDLE.value,
            escalation_reason="Trivial baseline policy: all inquiries marked as auto-handled by default.",
            drafted_reply=self.canned_reply,
        )


class SimpleBaseline:
    """Baseline 2: Basic TF-IDF Logistic Regression + Naive keyword escalation + Raw top-1 reply."""

    def __init__(self):
        kb_data = load_knowledge_base()
        texts = [it["customer_text"] for it in kb_data]

        from scripts.prepare_dataset import classify_intent_heuristic
        labels = [classify_intent_heuristic(t).value for t in texts]

        self.vectorizer = TfidfVectorizer(max_features=2000, stop_words="english")
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=500)
        self.clf.fit(X, labels)

        self.retriever = HistoricalResolutionRetriever()
        self.escalate_keywords = ["lawyer", "sue", "manager", "stolen", "chargeback", "bbb", "fraud"]

    def process(self, customer_text: str) -> BaselineResult:
        t_low = customer_text.lower()

        # Naive intent prediction
        X = self.vectorizer.transform([customer_text])
        pred_intent = self.clf.predict(X)[0]

        # Naive escalation
        if any(kw in t_low for kw in self.escalate_keywords):
            esc_dec = EscalationDecision.ESCALATE_TO_HUMAN.value
            esc_reason = "Keyword match for high-friction term."
        else:
            esc_dec = EscalationDecision.AUTO_HANDLE.value
            esc_reason = "No high-friction keyword detected."

        # Top-1 nearest neighbor verbatim reply
        retrieved = self.retriever.retrieve(customer_text, top_k=1)
        raw_reply = (
            retrieved[0]["response_text"]
            if retrieved
            else "Please reach out to our customer support team for help. ^CS"
        )

        return BaselineResult(
            customer_text=customer_text,
            intent=pred_intent,
            escalation_decision=esc_dec,
            escalation_reason=esc_reason,
            drafted_reply=raw_reply,
        )
