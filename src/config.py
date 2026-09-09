"""Configuration module for Hiver AI Support Agent.

Defines project paths, model configurations, intent taxonomy, and threshold
constants.
"""

from enum import Enum
import os
from pathlib import Path
from dotenv import load_dotenv

# Load environment variables
load_dotenv()

# Base directories
BASE_DIR = Path(__file__).resolve().parent.parent
DATA_DIR = BASE_DIR / "data"
RAW_DATA_PATH = DATA_DIR / "raw" / "amazon_pairs_raw.json"
KB_DATA_PATH = DATA_DIR / "knowledge_base" / "historical_resolutions.json"
GOLDEN_SET_PATH = DATA_DIR / "golden_set" / "golden_eval_200.json"
SAMPLING_NOTES_PATH = (
    DATA_DIR / "golden_set" / "sampling_and_labeling_notes.md"
)
REPORTS_DIR = BASE_DIR / "reports"

# Brand settings
TARGET_BRAND = "AmazonHelp"
BRAND_NAME = "Amazon"
AGENT_SIGNATURE = "^HiverBot"

# LLM Provider settings
LLM_PROVIDER = os.getenv("LLM_PROVIDER", "mock").lower()
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY", "")
ANTHROPIC_API_KEY = os.getenv("ANTHROPIC_API_KEY", "")
GEMINI_API_KEY = os.getenv("GEMINI_API_KEY", "")
GROQ_API_KEY = os.getenv("GROQ_API_KEY", "")

OPENAI_MODEL = os.getenv("OPENAI_MODEL", "gpt-4o-mini")
GEMINI_MODEL = os.getenv("GEMINI_MODEL", "gemini-2.5-flash")
GROQ_MODEL = os.getenv("GROQ_MODEL", "llama-3.3-70b-versatile")

# Retrieval parameters
TOP_K_RESOLUTIONS = 3
TFIDF_MAX_FEATURES = 5000

# Escalation thresholds
ESCALATION_CONFIDENCE_THRESHOLD = 0.65
