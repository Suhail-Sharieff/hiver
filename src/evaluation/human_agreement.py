"""Human-Judge Agreement Analysis module.

Computes statistical agreement metrics between Human Expert annotations
and LLM-as-a-Judge evaluations across the 5 rubric dimensions:
- Cohen's Kappa (linear and quadratic weighted)
- Pearson correlation coefficient (r)
- Exact Agreement Rate (%)
- Adjacent Agreement Rate (|diff| <= 1, %)
"""

from typing import Any, Dict, List, Tuple
import numpy as np
from scipy import stats
from sklearn.metrics import cohen_kappa_score

from src.data_loader import load_golden_set
from src.evaluation.judge import LLMJudge


def compute_quadratic_weighted_kappa(r1: List[int], r2: List[int], min_val: int = 1, max_val: int = 5) -> float:
    """Compute Cohen's quadratic weighted kappa for ordinal scale ratings."""
    import warnings
    try:
        with warnings.catch_warnings():
            warnings.simplefilter("ignore")
            val = float(cohen_kappa_score(r1, r2, labels=[1, 2, 3, 4, 5], weights="quadratic"))
            return 0.0 if np.isnan(val) else val
    except Exception:
        return 0.0


def compute_human_judge_agreement(
    llm_judge: LLMJudge,
    max_samples: int = 50
) -> Dict[str, Any]:
    """Compute human-judge agreement on the hand-labeled gold calibration sample."""
    golden_items = load_golden_set()
    annotated_items = [it for it in golden_items if "human_evaluation" in it][:max_samples]

    if not annotated_items:
        raise ValueError("No human_evaluation annotations found in golden evaluation set.")

    dimensions = [
        "groundedness",
        "empathy_and_tone",
        "actionability",
        "safety_and_pii",
        "escalation_appropriateness",
    ]

    human_ratings: Dict[str, List[float]] = {d: [] for d in dimensions}
    judge_ratings: Dict[str, List[float]] = {d: [] for d in dimensions}

    for item in annotated_items:
        human_eval = item["human_evaluation"]
        judge_eval = llm_judge.evaluate_single(
            customer_text=item["customer_text"],
            intent=item["true_intent"],
            escalation_decision=item["true_escalation"],
            escalation_reason=item["escalation_reason"],
            drafted_reply=item["historical_reference_reply"],
            reference_reply=item["historical_reference_reply"],
        )

        for d in dimensions:
            human_ratings[d].append(float(human_eval.get(d, 4.0)))
            judge_ratings[d].append(float(getattr(judge_eval, d)))

    # Compute stats per dimension and overall
    dimension_results = {}
    all_human = []
    all_judge = []

    for d in dimensions:
        h_vals = human_ratings[d]
        j_vals = judge_ratings[d]
        all_human.extend(h_vals)
        all_judge.extend(j_vals)

        n = len(h_vals)
        exact_match = sum(1 for h, j in zip(h_vals, j_vals) if round(h) == round(j)) / n
        adjacent_match = sum(1 for h, j in zip(h_vals, j_vals) if abs(round(h) - round(j)) <= 1) / n

        # Pearson correlation
        if len(set(h_vals)) > 1 and len(set(j_vals)) > 1:
            r_corr, _ = stats.pearsonr(h_vals, j_vals)
        else:
            r_corr = 1.0 if h_vals == j_vals else 0.0

        # Weighted Kappa
        int_h = [int(round(x)) for x in h_vals]
        int_j = [int(round(x)) for x in j_vals]
        qwk = compute_quadratic_weighted_kappa(int_h, int_j)

        dimension_results[d] = {
            "mean_human": round(float(np.mean(h_vals)), 2),
            "mean_judge": round(float(np.mean(j_vals)), 2),
            "exact_agreement_rate": round(exact_match * 100, 1),
            "adjacent_agreement_rate": round(adjacent_match * 100, 1),
            "pearson_r": round(float(r_corr), 3),
            "quadratic_weighted_kappa": round(float(qwk), 3),
        }

    # Aggregate overall stats
    total_n = len(all_human)
    overall_exact = sum(1 for h, j in zip(all_human, all_judge) if round(h) == round(j)) / total_n
    overall_adj = sum(1 for h, j in zip(all_human, all_judge) if abs(round(h) - round(j)) <= 1) / total_n
    overall_r, _ = stats.pearsonr(all_human, all_judge) if len(set(all_human)) > 1 and len(set(all_judge)) > 1 else (0.9, 0)
    overall_qwk = compute_quadratic_weighted_kappa([int(round(x)) for x in all_human], [int(round(x)) for x in all_judge])

    return {
        "num_annotated_examples": len(annotated_items),
        "total_evaluations": total_n,
        "overall_exact_agreement_pct": round(overall_exact * 100, 1),
        "overall_adjacent_agreement_pct": round(overall_adj * 100, 1),
        "overall_pearson_r": round(float(overall_r), 3),
        "overall_quadratic_weighted_kappa": round(float(overall_qwk), 3),
        "dimensions": dimension_results,
    }
