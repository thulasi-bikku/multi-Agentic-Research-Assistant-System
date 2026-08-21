"""
Standard IR evaluation via pytrec_eval (the same library used to produce
official BEIR leaderboard numbers, so results are directly comparable to
published SOTA -- Reviewer 3, point 1).
"""
import pytrec_eval
import numpy as np
import pandas as pd


def _sanitize_run(run: dict) -> dict:
    """
    pytrec_eval's C extension requires every score to be a native Python
    float and every id to be a native str -- numpy.float32/float64 (which
    is what raw sklearn/sentence-transformers output, e.g. in baselines.py)
    causes a bare 'Unable to extract query/object scores' TypeError with no
    indication of which value caused it. Sanitizing once, centrally, here
    means every caller (hybrid ranking, TF-IDF baseline, dense-retriever
    baseline, anything added later) is protected, instead of requiring
    every score-producing function to remember to cast itself. Also drops
    any NaN scores, which pytrec_eval also can't handle.
    """
    clean = {}
    for qid, docs in run.items():
        clean[qid] = {
            str(doc_id): float(score) for doc_id, score in docs.items()
            if score == score  # filters NaN (NaN != NaN)
        }
    return clean


def evaluate_run(qrels: dict, run: dict, cfg) -> dict:
    """
    Returns {query_id: {metric: value}} for every query -- this per-query
    dict is what significance.py and sensitivity.py consume, instead of
    the paper's per-seed repeated-run numbers (Reviewer 3, point 5).
    """
    run = _sanitize_run(run)
    measures = {
        f"ndcg_cut.{cfg.ndcg_cutoff}",
        f"P.{cfg.precision_cutoff}",
        f"recall.{cfg.recall_cutoff}",
        f"recip_rank",
    }
    evaluator = pytrec_eval.RelevanceEvaluator(qrels, measures)
    results = evaluator.evaluate(run)
    return results


def per_query_dataframe(results: dict, cfg) -> pd.DataFrame:
    rows = []
    for qid, metrics in results.items():
        rows.append({
            "query_id": qid,
            "ndcg@10": metrics.get(f"ndcg_cut_{cfg.ndcg_cutoff}", np.nan),
            "precision@10": metrics.get(f"P_{cfg.precision_cutoff}", np.nan),
            "recall@100": metrics.get(f"recall_{cfg.recall_cutoff}", np.nan),
            "mrr": metrics.get("recip_rank", np.nan),
        })
    return pd.DataFrame(rows)


def dataset_summary(per_query_df: pd.DataFrame) -> dict:
    p = per_query_df["precision@10"].mean()
    r = per_query_df["recall@100"].mean()
    f1 = 0.0 if (p + r) == 0 else 2 * p * r / (p + r)
    return {
        "precision@10": p,
        "recall@100": r,
        "f1": f1,
        "mrr": per_query_df["mrr"].mean(),   # computed once, no separate "1.000" claim anywhere
        "ndcg@10": per_query_df["ndcg@10"].mean(),
        "n_queries": len(per_query_df),
    }


def aggregate(per_dataset_summaries: dict) -> dict:
    """
    per_dataset_summaries: {dataset_name: summary_dict from dataset_summary()}
    Returns both macro (unweighted mean across datasets) and micro
    (query-count-weighted mean) averages, EXPLICITLY LABELED --
    Reviewer 3, point 2. The paper reported one unlabeled pooled number.
    """
    df = pd.DataFrame(per_dataset_summaries).T
    macro = df[["precision@10", "recall@100", "f1", "mrr", "ndcg@10"]].mean()
    weights = df["n_queries"]
    micro = (df[["precision@10", "recall@100", "f1", "mrr", "ndcg@10"]]
             .multiply(weights, axis=0).sum() / weights.sum())
    return {
        "per_dataset": df,
        "macro_average (unweighted across datasets)": macro,
        "micro_average (weighted by query count)": micro,
    }


def check_leakage(qrels: dict, queries: dict, corpus: dict, top_doc_per_query: dict,
                   near_duplicate_threshold: float = 0.95) -> pd.DataFrame:
    """
    Flags queries whose top-ranked document is (near-)identical to the
    query string itself -- the classic cause of an inflated/perfect MRR
    that Reviewer 3 point 3 asked to have investigated rather than
    reported as a real result. Uses a cheap token-Jaccard check; if any
    rows come back here, treat MRR near 1.0 as a data-leakage bug, not
    a system strength.
    """
    from difflib import SequenceMatcher
    flagged = []
    for qid, doc_id in top_doc_per_query.items():
        if qid not in queries or doc_id not in corpus:
            continue
        q_text = queries[qid]
        d_text = f"{corpus[doc_id].get('title','')} {corpus[doc_id].get('text','')}"[:len(q_text) * 3]
        ratio = SequenceMatcher(None, q_text.lower(), d_text.lower()).ratio()
        if ratio >= near_duplicate_threshold:
            flagged.append({"query_id": qid, "doc_id": doc_id, "similarity_ratio": ratio})
    return pd.DataFrame(flagged)


def top_doc_per_query(run: dict) -> dict:
    return {qid: max(docs.items(), key=lambda x: x[1])[0] for qid, docs in run.items() if docs}
