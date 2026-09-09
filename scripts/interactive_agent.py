"""Interactive CLI for testing the AI Support Agent live on custom customer inquiries."""

import json
import sys
from pathlib import Path

# Ensure project root is in sys.path
sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.agent import AISupportAgent


def run_interactive():
    print("\n========================================================")
    print("   AMAZON CUSTOMER SUPPORT AI AGENT - INTERACTIVE CLI   ")
    print("========================================================")
    print("Type any incoming customer message to test the agent.")
    print("Type 'exit' or 'quit' to end.\n")

    agent = AISupportAgent()

    while True:
        try:
            query = input("\n[Customer Tweet] > ").strip()
            if not query:
                continue
            if query.lower() in ["exit", "quit", "q"]:
                print("Exiting interactive CLI. Goodbye!")
                break

            res = agent.process(query)

            print("\n---------------- AGENT TRIAGE & RESPONSE ----------------")
            print(f"Predicted Intent     : {res.intent.value} (Confidence: {res.intent_confidence:.1%})")
            print(f"Escalation Decision  : {res.escalation_decision.value} (Confidence: {res.escalation_confidence:.1%})")
            print(f"Escalation Reason    : {res.escalation_reason}")
            print(f"\nDrafted Reply        :\n{res.drafted_reply}")
            print("\nTop Retrieved Historical Resolution:")
            if res.retrieved_resolutions:
                top_r = res.retrieved_resolutions[0]
                print(f"  [Similarity: {top_r['similarity_score']:.3f}] Customer: {top_r['customer_text']}")
                print(f"  Verified Reply: {top_r['response_text']}")
            print("---------------------------------------------------------")
        except KeyboardInterrupt:
            print("\nExiting interactive CLI.")
            break
        except Exception as e:
            print(f"[ERROR] {e}")


if __name__ == "__main__":
    run_interactive()
