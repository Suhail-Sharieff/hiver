"""Data Loader and Ingestion module for Customer Support Twitter Dataset.

Handles downloading, filtering, cleaning, thread reconstruction, and indexing
of AmazonHelp customer interactions from the thoughtvector/customer-support-on-twitter dataset.
"""

import csv
import io
import json
import re
import sys
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import requests

from src.config import (
    KB_DATA_PATH,
    RAW_DATA_PATH,
    GOLDEN_SET_PATH,
    SAMPLING_NOTES_PATH,
    TARGET_BRAND,
)
from src.taxonomy import IntentType, EscalationDecision

DATASET_STREAM_URL = "https://huggingface.co/datasets/SunidhiSriram/twcs/resolve/main/twcs.csv"

COMMON_EN_WORDS = {
    "the", "to", "and", "my", "is", "it", "for", "you", "in", "on", "this", "me",
    "have", "with", "that", "your", "was", "ordered", "order", "delivery", "delivered",
    "package", "amazon", "refund", "prime", "help", "account", "item", "customer",
    "service", "received", "not", "can", "cant", "cannot", "why", "been", "still",
    "when", "day", "days", "dm", "got", "from", "get", "will", "would", "like"
}


def clean_tweet_text(text: str) -> str:
    """Clean mentions, redundant whitespace, and normalize text."""
    if not text:
        return ""
    # Remove user handles like @AmazonHelp or @12345
    cleaned = re.sub(r"@[A-Za-z0-9_]+\s*", "", text)
    # Normalize multiple whitespace characters
    cleaned = re.sub(r"\s+", " ", cleaned).strip()
    return cleaned


def is_valid_english(text: str) -> bool:
    """Check if the text is predominantly English and has sufficient content."""
    if len(text) < 15:
        return False
    # Check ASCII character ratio
    ascii_chars = sum(1 for c in text if ord(c) < 128)
    if ascii_chars / len(text) < 0.85:
        return False
    # Check common English word vocabulary presence
    words = set(re.findall(r"[a-zA-Z]+", text.lower()))
    if len(words) < 3:
        return False
    overlap = words.intersection(COMMON_EN_WORDS)
    return len(overlap) >= 2


def fetch_raw_amazon_pairs(byte_range: int = 12000000) -> List[Dict[str, str]]:
    """Stream a slice of twcs.csv and extract matched customer -> AmazonHelp pairs."""
    headers = {"Range": f"bytes=0-{byte_range}"}
    try:
        response = requests.get(DATASET_STREAM_URL, headers=headers, timeout=30)
        response.raise_for_status()
        content = response.content.decode("utf-8", errors="replace")
    except Exception as e:
        print(f"[WARN] Could not fetch remote dataset: {e}. Falling back to cached data if present.")
        return []

    reader = csv.DictReader(io.StringIO(content))
    customer_tweets: Dict[str, Dict] = {}
    amazon_tweets: List[Dict] = []

    for row in reader:
        author = row.get("author_id")
        if author == TARGET_BRAND:
            amazon_tweets.append(row)
        elif row.get("inbound") == "True":
            t_id = row.get("tweet_id")
            if t_id:
                customer_tweets[t_id] = row

    pairs: List[Dict[str, str]] = []
    seen_cust = set()

    for at in amazon_tweets:
        in_resp = at.get("in_response_to_tweet_id")
        if in_resp and in_resp in customer_tweets:
            cust_raw = customer_tweets[in_resp].get("text", "")
            resp_raw = at.get("text", "")

            cust_clean = clean_tweet_text(cust_raw)
            resp_clean = clean_tweet_text(resp_raw)

            if cust_clean in seen_cust:
                continue
            seen_cust.add(cust_clean)

            if is_valid_english(cust_clean) and is_valid_english(resp_clean):
                pairs.append({
                    "tweet_id": in_resp,
                    "customer_text": cust_clean,
                    "response_text": resp_clean,
                    "created_at": customer_tweets[in_resp].get("created_at", "")
                })

    return pairs


def load_knowledge_base() -> List[Dict[str, str]]:
    """Load historical resolutions knowledge base for RAG grounding."""
    if not KB_DATA_PATH.exists():
        raise FileNotFoundError(
            f"Knowledge base not found at {KB_DATA_PATH}. Run pipeline data preparation first."
        )
    with open(KB_DATA_PATH, "r", encoding="utf-8") as f:
        return json.load(f)


def load_golden_set() -> List[Dict]:
    """Load the hand-labeled Golden Evaluation Set."""
    if not GOLDEN_SET_PATH.exists():
        raise FileNotFoundError(
            f"Golden evaluation set not found at {GOLDEN_SET_PATH}. Run pipeline data preparation first."
        )
    with open(GOLDEN_SET_PATH, "r", encoding="utf-8") as f:
        return json.load(f)
