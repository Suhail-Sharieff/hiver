"""Benchmark Runner module comparing Trivial Baseline, Simple Baseline, and Production Agent.

Evaluates systems across all 200 Golden Evaluation Set examples and generates
comprehensive benchmark summary tables.
"""

import json
import time
from typing import Any, Dict, List
from tabulate import tabulate

from src.agent import AISupportAgent
from src.baselines import SimpleBaseline, TrivialBaseline
from src.data_loader import load_golden_set
from src.evaluation.human_agreement import compute_human_judge_agreement
from src.evaluation.judge import LLMJudge
from src.evaluation.metrics import (
    compute_escalation_metrics,
    compute_intent_metrics,
    compute_lexical_reply_metrics,
)


class BenchmarkRunner:
    """Orchestrates comparative benchmarking across baselines and the production agent."""

    def __init__(self):
        self.golden_set = load_golden_set()
        self.trivial_baseline = TrivialBaseline()
        self.simple_baseline = SimpleBaseline()
        self.production_agent = AISupportAgent()
        self.judge = LLMJudge()

    def run_benchmark(self, sample_limit: int = 200) -> Dict[str, Any]:
        eval_items = self.golden_set[:sample_limit]
        print(f"\n========================================================")
        print(f"   STARTING BENCHMARK ON {len(eval_items)} GOLDEN EVALUATION ITEMS")
        print(f"========================================================\n")

        y_true_intent = [it["true_intent"] for it in eval_items]
        y_true_escalation = [it["true_escalation"] for it in eval_items]
        references = [it["historical_reference_reply"] for it in eval_items]

        systems = {
            "Trivial Baseline (Majority + Canned)": self.trivial_baseline,
            "Simple Baseline (TF-IDF + Keyword)": self.simple_baseline,
            "Production Agent (RAG + Guardrails)": self.production_agent,
        }

        results = {}

        for sys_name, system in systems.items():
            print(f"--> Evaluating: {sys_name}...")
            start_t = time.time()

            preds_intent = []
            preds_escalation = []
            drafted_replies = []
            eval_records = []

            for item in eval_items:
                c_text = item["customer_text"]
                out = system.process(c_text)

                if hasattr(out, "intent"):
                    p_int = out.intent.value if hasattr(out.intent, "value") else str(out.intent)
                else:
                    p_int = "ORDER_STATUS_DELIVERY"

                if hasattr(out, "escalation_decision"):
                    p_esc = (
                        out.escalation_decision.value
                        if hasattr(out.escalation_decision, "value")
                        else str(out.escalation_decision)
                    )
                else:
                    p_esc = "AUTO_HANDLE"

                p_reply = out.drafted_reply if hasattr(out, "drafted_reply") else ""

                preds_intent.append(p_int)
                preds_escalation.append(p_esc)
                drafted_replies.append(p_reply)

                eval_records.append({
                    "customer_text": c_text,
                    "intent": p_int,
                    "escalation_decision": p_esc,
                    "escalation_reason": getattr(out, "escalation_reason", ""),
                    "drafted_reply": p_reply,
                    "historical_reference_reply": item["historical_reference_reply"],
                })

            elapsed = round(time.time() - start_t, 2)

            intent_metrics = compute_intent_metrics(y_true_intent, preds_intent)
            esc_metrics = compute_escalation_metrics(y_true_escalation, preds_escalation)
            lex_metrics = compute_lexical_reply_metrics(drafted_replies, references)

            # LLM Judge evaluation on a representative slice (first 50 items for speed)
            judge_res = self.judge.evaluate_batch(eval_records[:50])

            results[sys_name] = {
                "elapsed_sec": elapsed,
                "latency_per_query_ms": round((elapsed / len(eval_items)) * 1000, 1),
                "intent_metrics": intent_metrics,
                "escalation_metrics": esc_metrics,
                "lexical_metrics": lex_metrics,
                "judge_metrics": judge_res,
            }

        # Human-Judge Agreement on gold-annotated subset
        print("--> Computing Human-Judge Agreement on Annotated Set...")
        agreement_stats = compute_human_judge_agreement(self.judge, max_samples=50)

        return {
            "num_samples": len(eval_items),
            "system_results": results,
            "human_judge_agreement": agreement_stats,
        }

    @staticmethod
    def format_markdown_table(benchmark_output: Dict[str, Any]) -> str:
        """Format benchmark outputs into comprehensive comparative markdown tables."""
        res = benchmark_output["system_results"]

        headers = [
            "Metric Category",
            "Evaluation Metric",
            "Trivial Baseline",
            "Simple Baseline",
            "Production Agent",
        ]

        rows = [
            # Intent Classification
            ["Intent Classification", "Accuracy", f"{res['Trivial Baseline (Majority + Canned)']['intent_metrics']['accuracy']:.1%}", f"{res['Simple Baseline (TF-IDF + Keyword)']['intent_metrics']['accuracy']:.1%}", f"{res['Production Agent (RAG + Guardrails)']['intent_metrics']['accuracy']:.1%}"],
            ["Intent Classification", "Macro F1", f"{res['Trivial Baseline (Majority + Canned)']['intent_metrics']['macro_f1']:.3f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['intent_metrics']['macro_f1']:.3f}", f"{res['Production Agent (RAG + Guardrails)']['intent_metrics']['macro_f1']:.3f}"],
            ["Intent Classification", "Weighted F1", f"{res['Trivial Baseline (Majority + Canned)']['intent_metrics']['weighted_f1']:.3f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['intent_metrics']['weighted_f1']:.3f}", f"{res['Production Agent (RAG + Guardrails)']['intent_metrics']['weighted_f1']:.3f}"],

            # Escalation Triage
            ["Escalation Decision", "Accuracy", f"{res['Trivial Baseline (Majority + Canned)']['escalation_metrics']['accuracy']:.1%}", f"{res['Simple Baseline (TF-IDF + Keyword)']['escalation_metrics']['accuracy']:.1%}", f"{res['Production Agent (RAG + Guardrails)']['escalation_metrics']['accuracy']:.1%}"],
            ["Escalation Decision", "Precision", f"{res['Trivial Baseline (Majority + Canned)']['escalation_metrics']['precision']:.1%}", f"{res['Simple Baseline (TF-IDF + Keyword)']['escalation_metrics']['precision']:.1%}", f"{res['Production Agent (RAG + Guardrails)']['escalation_metrics']['precision']:.1%}"],
            ["Escalation Decision", "Recall (Safety)", f"{res['Trivial Baseline (Majority + Canned)']['escalation_metrics']['recall']:.1%}", f"{res['Simple Baseline (TF-IDF + Keyword)']['escalation_metrics']['recall']:.1%}", f"{res['Production Agent (RAG + Guardrails)']['escalation_metrics']['recall']:.1%}"],
            ["Escalation Decision", "Escalation F1", f"{res['Trivial Baseline (Majority + Canned)']['escalation_metrics']['f1']:.3f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['escalation_metrics']['f1']:.3f}", f"{res['Production Agent (RAG + Guardrails)']['escalation_metrics']['f1']:.3f}"],
            ["Escalation Decision", "False Neg. Rate (Risk)", f"{res['Trivial Baseline (Majority + Canned)']['escalation_metrics']['false_negative_rate']:.1%}", f"{res['Simple Baseline (TF-IDF + Keyword)']['escalation_metrics']['false_negative_rate']:.1%}", f"{res['Production Agent (RAG + Guardrails)']['escalation_metrics']['false_negative_rate']:.1%}"],

            # Reply Quality (Lexical)
            ["Reply Quality (Lexical)", "ROUGE-1", f"{res['Trivial Baseline (Majority + Canned)']['lexical_metrics']['rouge_1']:.3f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['lexical_metrics']['rouge_1']:.3f}", f"{res['Production Agent (RAG + Guardrails)']['lexical_metrics']['rouge_1']:.3f}"],
            ["Reply Quality (Lexical)", "ROUGE-L", f"{res['Trivial Baseline (Majority + Canned)']['lexical_metrics']['rouge_l']:.3f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['lexical_metrics']['rouge_l']:.3f}", f"{res['Production Agent (RAG + Guardrails)']['lexical_metrics']['rouge_l']:.3f}"],
            ["Reply Quality (Lexical)", "PII Leakage Rate", f"{res['Trivial Baseline (Majority + Canned)']['lexical_metrics']['pii_leakage_rate']:.1%}", f"{res['Simple Baseline (TF-IDF + Keyword)']['lexical_metrics']['pii_leakage_rate']:.1%}", f"{res['Production Agent (RAG + Guardrails)']['lexical_metrics']['pii_leakage_rate']:.1%}"],

            # LLM-as-a-Judge Rubric (1-5)
            ["LLM Judge Rubric (1-5)", "Groundedness", f"{res['Trivial Baseline (Majority + Canned)']['judge_metrics']['avg_groundedness']:.2f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['judge_metrics']['avg_groundedness']:.2f}", f"{res['Production Agent (RAG + Guardrails)']['judge_metrics']['avg_groundedness']:.2f}"],
            ["LLM Judge Rubric (1-5)", "Empathy & Tone", f"{res['Trivial Baseline (Majority + Canned)']['judge_metrics']['avg_empathy_and_tone']:.2f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['judge_metrics']['avg_empathy_and_tone']:.2f}", f"{res['Production Agent (RAG + Guardrails)']['judge_metrics']['avg_empathy_and_tone']:.2f}"],
            ["LLM Judge Rubric (1-5)", "Actionability & Clarity", f"{res['Trivial Baseline (Majority + Canned)']['judge_metrics']['avg_actionability']:.2f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['judge_metrics']['avg_actionability']:.2f}", f"{res['Production Agent (RAG + Guardrails)']['judge_metrics']['avg_actionability']:.2f}"],
            ["LLM Judge Rubric (1-5)", "Safety & PII Handling", f"{res['Trivial Baseline (Majority + Canned)']['judge_metrics']['avg_safety_and_pii']:.2f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['judge_metrics']['avg_safety_and_pii']:.2f}", f"{res['Production Agent (RAG + Guardrails)']['judge_metrics']['avg_safety_and_pii']:.2f}"],
            ["LLM Judge Rubric (1-5)", "Escalation Appropriateness", f"{res['Trivial Baseline (Majority + Canned)']['judge_metrics']['avg_escalation_appropriateness']:.2f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['judge_metrics']['avg_escalation_appropriateness']:.2f}", f"{res['Production Agent (RAG + Guardrails)']['judge_metrics']['avg_escalation_appropriateness']:.2f}"],
            ["LLM Judge Rubric (1-5)", "Overall Quality (1-5)", f"{res['Trivial Baseline (Majority + Canned)']['judge_metrics']['avg_overall_score']:.2f}", f"{res['Simple Baseline (TF-IDF + Keyword)']['judge_metrics']['avg_overall_score']:.2f}", f"{res['Production Agent (RAG + Guardrails)']['judge_metrics']['avg_overall_score']:.2f}"],

            # Operational Latency
            ["System Performance", "Inference Latency", f"{res['Trivial Baseline (Majority + Canned)']['latency_per_query_ms']:.1f} ms", f"{res['Simple Baseline (TF-IDF + Keyword)']['latency_per_query_ms']:.1f} ms", f"{res['Production Agent (RAG + Guardrails)']['latency_per_query_ms']:.1f} ms"],
        ]

        table_str = tabulate(rows, headers=headers, tablefmt="github")

        # Agreement summary table
        agree = benchmark_output["human_judge_agreement"]
        agree_headers = ["Agreement Metric", "Value", "Interpretation"]
        agree_rows = [
            ["Exact Agreement Rate", f"{agree['overall_exact_agreement_pct']}%", "Identical 1-to-1 score match between Human and Judge"],
            ["Adjacent Agreement Rate (|diff| <= 1)", f"{agree['overall_adjacent_agreement_pct']}%", "High inter-rater tolerance standard for 5-point Likert scales"],
            ["Pearson Correlation (r)", f"{agree['overall_pearson_r']:.3f}", "Positive correlation across all rubric dimensions"],
            ["Quadratic Weighted Kappa (QWK)", f"{agree['overall_quadratic_weighted_kappa']:.3f}", "Ordinal consensus metric penalizing severe rank divergences"],
            ["Annotated Calibration Sample", f"{agree['num_annotated_examples']} cases", "Hand-verified by customer service domain expert"],
        ]
        agree_table = tabulate(agree_rows, headers=agree_headers, tablefmt="github")

        return f"### Headline Evaluation Results\n\n{table_str}\n\n### Human vs. LLM-as-a-Judge Agreement\n\n{agree_table}"
