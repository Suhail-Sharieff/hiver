"""Production AI Support Agent for AmazonHelp Twitter Customer Service.

Implements:
1. Multi-intent classification grounded in the historical customer support dataset.
2. Retrieval-augmented reply drafting based on proven historical brand resolutions.
3. Policy-driven escalation routing (AUTO_HANDLE vs ESCALATE_TO_HUMAN) with stated reasons.
"""

from typing import Any, Dict, List, Optional, Tuple
import re
from pydantic import BaseModel, Field
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression

from src.config import (
    AGENT_SIGNATURE,
    BRAND_NAME,
    ESCALATION_CONFIDENCE_THRESHOLD,
    TOP_K_RESOLUTIONS,
)
from src.data_loader import load_knowledge_base
from src.llm_client import LLMClient
from src.retriever import HistoricalResolutionRetriever
from src.taxonomy import IntentType, EscalationDecision, INTENT_DESCRIPTIONS


class IntentClassificationResult(BaseModel):
    intent: IntentType
    confidence: float
    probabilities: Dict[str, float] = Field(default_factory=dict)


class EscalationResult(BaseModel):
    decision: EscalationDecision
    confidence: float
    stated_reason: str
    triggered_rules: List[str] = Field(default_factory=list)


class SupportAgentOutput(BaseModel):
    customer_text: str
    intent: IntentType
    intent_confidence: float
    escalation_decision: EscalationDecision
    escalation_confidence: float
    escalation_reason: str
    drafted_reply: str
    retrieved_resolutions: List[Dict[str, Any]] = Field(default_factory=list)


class CalibratedIntentClassifier:
    """Supervised + heuristic hybrid intent classifier trained on Amazon customer support corpus."""

    def __init__(self):
        kb_data = load_knowledge_base()
        # Build training corpus from knowledge base
        texts = []
        labels = []

        from scripts.prepare_dataset import classify_intent_heuristic

        for item in kb_data:
            c_text = item["customer_text"]
            lbl = classify_intent_heuristic(c_text)
            texts.append(c_text)
            labels.append(lbl.value)

        self.vectorizer = TfidfVectorizer(
            ngram_range=(1, 2),
            max_features=4000,
            sublinear_tf=True,
            stop_words="english",
        )
        X = self.vectorizer.fit_transform(texts)
        self.clf = LogisticRegression(max_iter=1000, class_weight="balanced")
        self.clf.fit(X, labels)
        self.classes_ = list(self.clf.classes_)

    def classify(self, text: str) -> IntentClassificationResult:
        if not text or not text.strip():
            return IntentClassificationResult(
                intent=IntentType.FEEDBACK_CHITCHAT,
                confidence=0.5,
                probabilities={},
            )

        X = self.vectorizer.transform([text])
        probs = self.clf.predict_proba(X)[0]
        max_idx = int(probs.argmax())
        intent_str = self.classes_[max_idx]
        confidence = float(probs[max_idx])

        # Priority keyword overrides for critical intents
        t_low = text.lower()
        if any(k in t_low for k in [
            "speak to a manager", "speak to a supervisor", "filing a complaint",
            "chargeback", "better business bureau", "fourth time", "3rd time",
            "3 different people", "lying to me", "lawyer", "sue", "legal action",
            "worst customer service", "unacceptable"
        ]):
            intent_str = IntentType.ESCALATION_COMPLAINT.value
            confidence = max(confidence, 0.95)
        elif any(k in t_low for k in [
            "unauthorized", "charged me", "bank card", "credit card", "bank statement",
            "prime membership", "billing", "password", "close my account", "gift card",
            "double charged", "payment method", "charge on my"
        ]):
            intent_str = IntentType.ACCOUNT_PAYMENT_SECURITY.value
            confidence = max(confidence, 0.92)
        elif any(k in t_low for k in [
            "damaged", "broken", "shattered", "scratched", "wrong item", "wrong size",
            "defective", "missing from the box"
        ]):
            intent_str = IntentType.DAMAGED_DEFECTIVE_WRONG.value
            confidence = max(confidence, 0.90)
        elif any(k in t_low for k in [
            "refund", "return label", "send it back", "return status", "money back",
            "exchange the item", "drop off return"
        ]):
            intent_str = IntentType.RETURN_REFUND.value
            confidence = max(confidence, 0.90)
        elif any(k in t_low for k in [
            "thank you", "thanks", "awesome service", "great job", "wonderful day",
            "kudos", "love amazon", "appreciate your help"
        ]) and not any(k in t_low for k in ["not", "problem", "issue", "why", "where", "still", "cancel"]):
            intent_str = IntentType.FEEDBACK_CHITCHAT.value
            confidence = max(confidence, 0.90)
        elif any(k in t_low for k in [
            "tracking", "track my", "where is my", "package", "delivered", "shipping",
            "carrier", "fedex", "ups", "usps", "dispatch"
        ]):
            intent_str = IntentType.ORDER_STATUS_DELIVERY.value
            confidence = max(confidence, 0.88)

        prob_dict = {cls_name: round(float(p), 4) for cls_name, p in zip(self.classes_, probs)}

        return IntentClassificationResult(
            intent=IntentType(intent_str),
            confidence=round(confidence, 4),
            probabilities=prob_dict,
        )


class EscalationEngine:
    """Evaluates customer messages against safety, financial risk, and sentiment rules."""

    def __init__(self):
        # Security & Fraud indicators
        self.security_patterns = [
            r"\bunauthorized\b", r"\bhacked\b", r"\bfraud\b", r"\bstolen\b",
            r"\bclose (my )?account\b", r"\blocked out\b", r"\bbank statement\b",
            r"\bbank info(rmation)?\b", r"\bdon't remember (the )?email\b",
            r"\bcharged me without\b", r"\bidentity theft\b"
        ]
        # Lost package after marked delivered
        self.lost_package_patterns = [
            r"says delivered.*not", r"marked delivered.*never", r"delivered saturday.*was not",
            r"marked delivered.*was home", r"stolen from (my )?(porch|doorstep)",
            r"delivered.*nothing (is )?there", r"delivered to wrong address"
        ]
        # Multi-turn friction / Churn / Legal threats
        self.friction_patterns = [
            r"\b(2nd|3rd|4th|third|fourth) time\b", r"\bcalled twice\b",
            r"\b(2|3|4) different people\b", r"\bmanager\b", r"\bsupervisor\b",
            r"\bchargeback\b", r"\blawyer\b", r"\blegal action\b", r"\bbbb\b",
            r"\bdisgusted\b", r"\bunacceptable\b", r"\bly(ing|ed) to me\b",
            r"\bweeks ago.*still no refund\b", r"\bshattered glass\b"
        ]

    def evaluate(
        self, text: str, intent: IntentType, intent_confidence: float
    ) -> EscalationResult:
        t_low = text.lower()
        triggered_rules = []

        # Rule 1: High-friction complaint intent or explicit manager demand
        if intent == IntentType.ESCALATION_COMPLAINT:
            triggered_rules.append("High Customer Friction / Explicit Escalation Demand")
            for pat in self.friction_patterns:
                if re.search(pat, t_low):
                    triggered_rules.append(f"Friction pattern: '{pat}'")

            return EscalationResult(
                decision=EscalationDecision.ESCALATE_TO_HUMAN,
                confidence=0.96,
                stated_reason="Customer expresses severe dissatisfaction, repeat failed contacts, or explicit demand for supervisor intervention.",
                triggered_rules=triggered_rules,
            )

        # Rule 2: Account security & financial risk
        for pat in self.security_patterns:
            if re.search(pat, t_low):
                triggered_rules.append(f"Security/Fraud pattern match: '{pat}'")
                return EscalationResult(
                    decision=EscalationDecision.ESCALATE_TO_HUMAN,
                    confidence=0.94,
                    stated_reason="Security or financial fraud risk detected; requires authenticated human identity verification.",
                    triggered_rules=triggered_rules,
                )

        # Rule 3: Lost package marked delivered
        for pat in self.lost_package_patterns:
            if re.search(pat, t_low):
                triggered_rules.append(f"Lost package delivery discrepancy: '{pat}'")
                return EscalationResult(
                    decision=EscalationDecision.ESCALATE_TO_HUMAN,
                    confidence=0.92,
                    stated_reason="Package marked delivered but missing; requires carrier GPS delivery scan check and manual refund/replacement override.",
                    triggered_rules=triggered_rules,
                )

        # Rule 4: Critical phrases in other intents
        for pat in self.friction_patterns:
            if re.search(pat, t_low):
                triggered_rules.append(f"Severe friction pattern: '{pat}'")
                return EscalationResult(
                    decision=EscalationDecision.ESCALATE_TO_HUMAN,
                    confidence=0.91,
                    stated_reason="Repeat service failures or legal/financial friction detected requiring supervisor discretion.",
                    triggered_rules=triggered_rules,
                )

        # Default: Safe for automated resolution
        reason_map = {
            IntentType.ORDER_STATUS_DELIVERY: "Standard tracking status inquiry safely handled via authenticated 'Your Orders' tracking link.",
            IntentType.RETURN_REFUND: "Standard return policy or return label request handled via online Returns Center link.",
            IntentType.DAMAGED_DEFECTIVE_WRONG: "Standard damaged/wrong item replacement flow handled through self-service order replacement.",
            IntentType.ACCOUNT_PAYMENT_SECURITY: "Standard account management/billing FAQ resolvable via self-service portal links.",
            IntentType.PRODUCT_SERVICE_INQUIRY: "General product information or standard troubleshooting steps.",
            IntentType.FEEDBACK_CHITCHAT: "Positive feedback or casual chitchat requiring brand courtesy response.",
        }

        return EscalationResult(
            decision=EscalationDecision.AUTO_HANDLE,
            confidence=0.88,
            stated_reason=reason_map.get(intent, "Standard support query resolvable via verified self-service guidance."),
            triggered_rules=["Safe Self-Service Resolution Path"],
        )


class GroundedReplyDrafter:
    """Synthesizes customer support replies grounded in historical Amazon resolutions."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()

    def draft(
        self,
        customer_text: str,
        intent: IntentType,
        escalation: EscalationResult,
        retrieved_resolutions: List[Dict[str, Any]],
    ) -> str:
        # Build prompt incorporating historical exemplars
        exemplars_text = "\n\n".join([
            f"Historical Customer Query: {r['customer_text']}\nHistorical Verified Amazon Reply: {r['response_text']}"
            for r in retrieved_resolutions[:2]
        ])

        system_prompt = (
            f"You are the official Twitter customer support AI agent for {BRAND_NAME} (@{BRAND_NAME}Help). "
            f"Your job is to draft a helpful, empathetic, concise reply (Twitter length: 1-3 sentences).\n"
            f"STRICT POLICIES:\n"
            f"1. Never ask the customer to post personal info, passwords, card details, or order numbers publicly on Twitter.\n"
            f"2. Direct them to safe authenticated portals: amazon.com/your-orders (for orders/tracking/replacements), amazon.com/returns (for returns/labels), or amazon.com/contact-us (for secure account/billing/phone/chat support).\n"
            f"3. Always maintain empathy, professional brand voice, and de-escalate tension.\n"
            f"4. If ESCALATE_TO_HUMAN is required, acknowledge their situation empathetically and direct them to connect securely with a live specialist.\n"
            f"5. End with the agent signature '{AGENT_SIGNATURE}'."
        )

        prompt = (
            f"Customer Message: \"{customer_text}\"\n"
            f"Classified Intent: {intent.value}\n"
            f"Routing Decision: {escalation.decision.value}\n"
            f"Escalation Reason: {escalation.stated_reason}\n\n"
            f"Historically Proven Support Resolutions for Similar Issues:\n{exemplars_text}\n\n"
            f"Draft a grounded reply adhering strictly to {BRAND_NAME}'s historical customer service standards."
        )

        reply = self.llm.generate(prompt=prompt, system_prompt=system_prompt, temperature=0.2)

        # Ensure signature is present
        if AGENT_SIGNATURE not in reply:
            reply = f"{reply.strip()} {AGENT_SIGNATURE}"

        return reply


class AISupportAgent:
    """Master AI Support Agent combining triage, retrieval, and grounded reply generation."""

    def __init__(self, llm_client: Optional[LLMClient] = None):
        self.llm = llm_client or LLMClient()
        self.retriever = HistoricalResolutionRetriever()
        self.classifier = CalibratedIntentClassifier()
        self.escalation_engine = EscalationEngine()
        self.drafter = GroundedReplyDrafter(self.llm)

    def process(self, customer_text: str) -> SupportAgentOutput:
        # Step 1: Intent Classification
        intent_res = self.classifier.classify(customer_text)

        # Step 2: Escalation Decision
        esc_res = self.escalation_engine.evaluate(
            customer_text, intent_res.intent, intent_res.confidence
        )

        # Step 3: Retrieve top historical resolutions
        retrieved = self.retriever.retrieve(customer_text, top_k=TOP_K_RESOLUTIONS)

        # Step 4: Draft grounded reply
        reply = self.drafter.draft(
            customer_text=customer_text,
            intent=intent_res.intent,
            escalation=esc_res,
            retrieved_resolutions=retrieved,
        )

        return SupportAgentOutput(
            customer_text=customer_text,
            intent=intent_res.intent,
            intent_confidence=intent_res.confidence,
            escalation_decision=esc_res.decision,
            escalation_confidence=esc_res.confidence,
            escalation_reason=esc_res.stated_reason,
            drafted_reply=reply,
            retrieved_resolutions=retrieved,
        )
