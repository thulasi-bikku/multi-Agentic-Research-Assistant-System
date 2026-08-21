"""
Reviewer 3, point 9: human eval needs to state rater independence, report
inter-rater reliability for ALL five Likert criteria (the paper's kappa=0.86
only covered ArXiv relevance labels), define how hallucination rate was
adjudicated, and add a control condition.

This module generates the rating template and computes the required
statistics once it's filled in. It refuses to aggregate results that are
missing the control condition or rater-identity columns, to force the
protocol described above rather than let the shortcut back in.
"""
import pandas as pd
import numpy as np

CRITERIA = ["factual_accuracy", "usefulness", "readability", "completeness"]
CONDITIONS = ["maras", "control_llm_only"]  # control MUST be present


def generate_rating_template(item_ids: list, rater_ids: list, out_csv: str = "human_eval/rating_template.csv"):
    rows = []
    for item_id in item_ids:
        for condition in CONDITIONS:
            for rater_id in rater_ids:
                row = {"item_id": item_id, "condition": condition, "rater_id": rater_id}
                for c in CRITERIA:
                    row[c] = ""  # 1-5 Likert, filled in by rater
                row["hallucinated_claim_count"] = ""     # integer, not just a rate
                row["total_checkable_claims"] = ""       # denominator, so rate = count/total is auditable
                row["rater_independent_of_authors"] = ""  # yes/no, must be recorded per point 9
                rows.append(row)
    df = pd.DataFrame(rows)
    df.to_csv(out_csv, index=False)
    return df


def _require_columns(df: pd.DataFrame):
    missing_conditions = set(CONDITIONS) - set(df["condition"].unique())
    if missing_conditions:
        raise ValueError(
            f"Missing required condition(s) {missing_conditions}. A control condition "
            "(same LLM, no retrieval pipeline) is required before results can be "
            "reported -- see Reviewer 3, point 9."
        )
    if df["rater_independent_of_authors"].isnull().any() or (df["rater_independent_of_authors"] == "").any():
        raise ValueError("rater_independent_of_authors must be filled in for every row.")


def compute_reliability(df: pd.DataFrame) -> pd.DataFrame:
    """
    Fleiss' kappa per criterion across all raters (generalizes Cohen's
    kappa to 3+ raters), computed separately for EACH of the 5 criteria,
    not just relevance labels.
    """
    _require_columns(df)
    from statsmodels.stats.inter_rater import fleiss_kappa, aggregate_raters
    rows = []
    for criterion in CRITERIA:
        pivot = df.pivot_table(index="item_id", columns=criterion, values="rater_id", aggfunc="count", fill_value=0)
        table, _ = aggregate_raters(pivot.values)
        kappa = fleiss_kappa(table)
        rows.append({"criterion": criterion, "fleiss_kappa": kappa})
    return pd.DataFrame(rows)


def compute_condition_comparison(df: pd.DataFrame) -> pd.DataFrame:
    """The comparison the paper never made: MARAS vs. control on identical
    criteria, with a paired test across items (same unit-of-analysis fix
    as significance.py -- pair by item, not by rater or by run)."""
    _require_columns(df)
    from scipy import stats
    numeric = df.copy()
    for c in CRITERIA + ["hallucinated_claim_count", "total_checkable_claims"]:
        numeric[c] = pd.to_numeric(numeric[c], errors="coerce")
    numeric["hallucination_rate"] = numeric["hallucinated_claim_count"] / numeric["total_checkable_claims"]

    item_means = numeric.groupby(["item_id", "condition"])[CRITERIA + ["hallucination_rate"]].mean().reset_index()
    rows = []
    for metric in CRITERIA + ["hallucination_rate"]:
        wide = item_means.pivot(index="item_id", columns="condition", values=metric).dropna()
        if wide.empty:
            continue
        t_stat, p = stats.ttest_rel(wide["maras"], wide["control_llm_only"])
        rows.append({
            "metric": metric,
            "maras_mean": wide["maras"].mean(),
            "control_mean": wide["control_llm_only"].mean(),
            "mean_diff": (wide["maras"] - wide["control_llm_only"]).mean(),
            "paired_t_pvalue": p,
        })
    return pd.DataFrame(rows)
