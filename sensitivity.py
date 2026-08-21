"""
Reviewer 3, point 6: Table 5 had no numbers and didn't say whether weights
were tuned on the test set. This module:
  1. splits queries into val/test disjoint sets and asserts the split,
  2. grid-searches (alpha, beta, gamma) on the VAL split only,
  3. reports actual F1 / NDCG@10 numbers for each config,
  4. the chosen config is then evaluated once on the held-out TEST split
     elsewhere (evaluate.py) -- never re-tuned on test.
"""
import itertools
import numpy as np
import pandas as pd
from retrieval.hybrid import HybridRetriever
from evaluate import evaluate_run, per_query_dataframe, dataset_summary


def split_queries(queries: dict, qrels: dict, val_fraction: float = 0.2, seed: int = 42):
    rng = np.random.default_rng(seed)
    qids = list(queries.keys())
    rng.shuffle(qids)
    n_val = int(len(qids) * val_fraction)
    val_ids, test_ids = set(qids[:n_val]), set(qids[n_val:])
    assert val_ids.isdisjoint(test_ids), "validation and test query sets must be disjoint"

    def _subset(d, ids):
        return {k: v for k, v in d.items() if k in ids}

    val = (_subset(queries, val_ids), _subset(qrels, val_ids))
    test = (_subset(queries, test_ids), _subset(qrels, test_ids))
    return val, test


def weight_grid(step: float = 0.05):
    """All (alpha, beta, gamma) triples on the step-grid that sum to 1."""
    vals = np.round(np.arange(0.0, 1.0 + step, step), 2)
    grid = []
    for a, b in itertools.product(vals, vals):
        g = round(1.0 - a - b, 2)
        if 0.0 <= g <= 1.0:
            grid.append((round(a, 2), round(b, 2), g))
    return grid


def run_sensitivity(bm25_scores: dict, semantic_scores: dict, temporal_scores: dict,
                     val_qrels: dict, cfg, grid=None) -> pd.DataFrame:
    grid = grid or weight_grid(cfg.sensitivity_grid_step)
    rows = []
    for alpha, beta, gamma in grid:
        retriever = HybridRetriever(bm25_scores, semantic_scores, temporal_scores, alpha, beta, gamma)
        run = retriever.rank_all(top_k=cfg.retrieval_depth)
        run = {qid: run[qid] for qid in val_qrels if qid in run}
        results = evaluate_run(val_qrels, run, cfg)
        df = per_query_dataframe(results, cfg)
        summary = dataset_summary(df)
        rows.append({"alpha": alpha, "beta": beta, "gamma": gamma, **summary})
    result_df = pd.DataFrame(rows).sort_values("ndcg@10", ascending=False).reset_index(drop=True)
    return result_df


def best_config(sensitivity_df: pd.DataFrame, metric: str = "ndcg@10") -> tuple:
    row = sensitivity_df.sort_values(metric, ascending=False).iloc[0]
    return float(row["alpha"]), float(row["beta"]), float(row["gamma"])
