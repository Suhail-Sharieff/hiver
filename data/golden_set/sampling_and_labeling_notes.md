# Golden Evaluation Set: Sampling & Labeling Methodology

## 1. Dataset Source & Context
- **Source Corpus:** `thoughtvector/customer-support-on-twitter` (Kaggle benchmark dataset), containing ~3 million multi-turn customer service tweets.
- **Brand Selected:** `@AmazonHelp` (Amazon Customer Support).
- **Domain:** E-commerce customer service spanning order tracking, logistics delays, refunds/returns, product defects, account billing/Prime membership, digital services (Prime Video/Kindle), and high-churn escalations.

## 2. Sampling Strategy
A multi-stage stratified sampling approach was applied:
1. **Thread Reconstruction:** Identified verified customer inquiries (`inbound=True`) explicitly paired with corresponding brand replies (`author_id=AmazonHelp`, `in_response_to_tweet_id`).
2. **Quality & Noise Filtering:**
   - Removed Twitter handles (`@AmazonHelp`, customer handles) and redundant spacing.
   - Filtered out non-English interactions (AmazonHelp supports 6+ languages; interactions were filtered using English stopword density and character set analysis).
   - Removed single-token/low-content queries (<15 characters).
3. **Intent Stratification:** Curated 200 representative customer interactions distributed across 7 core intents:
   - `ORDER_STATUS_DELIVERY`: 38 examples (19.0%)
   - `RETURN_REFUND`: 34 examples (17.0%)
   - `ACCOUNT_PAYMENT_SECURITY`: 32 examples (16.0%)
   - `DAMAGED_DEFECTIVE_WRONG`: 30 examples (15.0%)
   - `PRODUCT_SERVICE_INQUIRY`: 26 examples (13.0%)
   - `ESCALATION_COMPLAINT`: 24 examples (12.0%)
   - `FEEDBACK_CHITCHAT`: 16 examples (8.0%)
   **Total:** 200 hand-verified examples.

## 3. Ground Truth Annotation Protocol
Each example was labeled according to a standardized annotation guide:
- **True Intent:** The primary operational category the message falls under. Ambiguous multi-intent messages were categorized by the highest-priority actionable request.
- **True Escalation:**
  - `AUTO_HANDLE`: Standard FAQ, self-service tracking guidance (`amazon.com/your-orders`), standard return label guidance, general device/streaming troubleshooting, and courteous feedback acknowledgments.
  - `ESCALATE_TO_HUMAN`:
    - Financial & Security Risk: Unauthorized charges, compromised accounts, payment disputes, account closures.
    - Physical Delivery Failure: Package marked "delivered" but missing (requires carrier GPS trace and manual agent replacement/refund authorization).
    - SLA Breach & Multi-Agent Failure: Repeat contacts ("3rd agent", "called twice"), high customer hostility, legal/chargeback threats.
    - Safety & High-Value Damage: Shattered glass, battery fire hazards, or damaged high-ticket merchandise.
- **Escalation Reason:** Explicit justification documenting why human supervisor discretion or security verification is mandatory.
- **Human Reference Reply:** Preserved authentic resolution tweet from AmazonHelp agents.
- **Human Rubric Benchmark:** 50 randomly sampled items include gold-standard human expert ratings across all 5 evaluation dimensions (1-5 scale) to measure Human-Judge Agreement (Cohen's Kappa and Pearson correlation).
