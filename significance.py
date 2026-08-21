"""
Reviewer 3, point 5: the paper tested across 10 repeated runs of a
deterministic system, which measures nothing meaningful. The correct unit
of analysis for IR significance testing is the QUERY: for each query,
compute metric_A(query) - metric_B(query), and test whether that paired
difference is systematically non-zero across queries.
"""
import numpy as np
import pandas as pd
from scipy import stats


def paired_query_test(metric_a: pd.Series, metric_b: pd.Series, alpha: float = 0.05):
    """
    metric_a, metric_b: per-query metric values (same query order/index)
    for the two systems being compared, e.g. hybrid NDCG@10 vs BM25-only
    NDCG@10 on identical queries.
    """
    a = metric_a.values
    b = metric_b.values
    diff = a - b

    t_stat, t_p = stats.ttest_rel(a, b)
    try:
        w_stat, w_p = stats.wilcoxon(a, b)
    except ValueError:
        # all differences are zero
        w_stat, w_p = np.nan, 1.0

    # paired Cohen's d
    d = diff.mean() / diff.std(ddof=1) if diff.std(ddof=1) > 0 else 0.0

    return {
        "n_queries": len(a),
        "mean_diff": diff.mean(),
        "t_stat": t_stat, "t_pvalue": t_p,
        "wilcoxon_stat": w_stat, "wilcoxon_pvalue": w_p,
        "cohens_d_paired": d,
        "significant_at_alpha_uncorrected": t_p < alpha,
    }


def multiple_comparisons_correction(pvalues: list, method: str = "holm", alpha: float = 0.05):
    """Holm-Bonferroni correction across the several baseline comparisons
    (hybrid vs BM25-only, hybrid vs semantic-only, hybrid vs TF-IDF,
    hybrid vs dense retriever, ...) run per dataset -- correcting for the
    fact that testing many comparisons inflates the false-positive rate,
    which the paper's design did not address."""
    from statsmodels.stats.multitest import multipletests
    reject, corrected_p, _, _ = multipletests(pvalues, alpha=alpha, method=method)
    return reject, corrected_p


def run_all_comparisons(system_metrics: dict, reference: str, dataset_name: str,
                         metric_name: str = "ndcg@10", alpha: float = 0.05, method: str = "holm"):
    """
    system_metrics: {system_name: per_query_df} where per_query_df has a
    `metric_name` column indexed by query_id, all sharing the same queries.
    reference: name of the system to compare every other system against
               (typically 'hybrid_full').
    Returns a results DataFrame with corrected significance.
    """
    ref_series = system_metrics[reference].set_index("query_id")[metric_name]
    rows = []
    others = [s for s in system_metrics if s != reference]
    pvals = []
    for other in others:
        other_series = system_metrics[other].set_index("query_id")[metric_name]
        common = ref_series.index.intersection(other_series.index)
        result = paired_query_test(ref_series.loc[common], other_series.loc[common], alpha)
        result["dataset"] = dataset_name
        result["comparison"] = f"{reference}_vs_{other}"
        rows.append(result)
        pvals.append(result["t_pvalue"])

    if pvals:
        reject, corrected = multiple_comparisons_correction(pvals, method, alpha)
        for row, rej, corr_p in zip(rows, reject, corrected):
            row["corrected_pvalue"] = corr_p
            row["significant_after_correction"] = bool(rej)

    return pd.DataFrame(rows)
