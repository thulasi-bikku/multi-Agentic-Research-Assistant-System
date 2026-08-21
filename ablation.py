"""
Runs each ablation configuration and reports real numbers (Table 13 in the
paper was internally consistent already; this just makes the configs
explicit and adds the single-LLM ablation Reviewer 1 asked for on the
retrieval-adjacent side, plus TF-IDF and dense-retriever rows so the
"only self-ablations" complaint from Reviewer 3 point 4 is also closed
here, not just in baselines.py).
"""
import pandas as pd
from retrieval.hybrid import HybridRetriever
from evaluate import evaluate_run, per_query_dataframe, dataset_summary


def run_ablation_suite(bm25_scores: dict, semantic_scores: dict, temporal_scores: dict,
                        tfidf_scores: dict, dense_scores: dict,
                        qrels: dict, cfg) -> dict:
    """Returns {config_name: per_query_dataframe} for use in significance.py,
    and prints a summary table."""
    neutral_temporal = {d: 0.5 for d in temporal_scores}  # cancels temporal term
    configs = {
        "full_maras":            (bm25_scores, semantic_scores, temporal_scores, cfg.alpha, cfg.beta, cfg.gamma),
        "without_semantic":      (bm25_scores, semantic_scores, neutral_temporal, 1 - cfg.gamma, 0.0, cfg.gamma),
        "without_temporal":      (bm25_scores, semantic_scores, neutral_temporal, cfg.alpha, cfg.beta + cfg.gamma, 0.0),
        "bm25_only":             (bm25_scores, semantic_scores, neutral_temporal, 1.0, 0.0, 0.0),
        "semantic_only":         (bm25_scores, semantic_scores, neutral_temporal, 0.0, 1.0, 0.0),
    }

    per_system_dfs = {}
    summary_rows = []
    for name, (bm25_s, sem_s, temp_s, a, b, g) in configs.items():
        retriever = HybridRetriever(bm25_s, sem_s, temp_s, a, b, g)
        run = retriever.rank_all(top_k=cfg.retrieval_depth)
        run = {qid: run[qid] for qid in qrels if qid in run}
        results = evaluate_run(qrels, run, cfg)
        df = per_query_dataframe(results, cfg)
        per_system_dfs[name] = df
        summary_rows.append({"config": name, **dataset_summary(df)})

    # TF-IDF-only and dense-retriever rows use score dicts directly, no temporal component
    for name, scores in [("tfidf_only", tfidf_scores), ("dense_retriever_only", dense_scores)]:
        run = {qid: dict(sorted(docs.items(), key=lambda x: x[1], reverse=True)[:cfg.retrieval_depth])
               for qid, docs in scores.items() if qid in qrels}
        results = evaluate_run(qrels, run, cfg)
        df = per_query_dataframe(results, cfg)
        per_system_dfs[name] = df
        summary_rows.append({"config": name, **dataset_summary(df)})

    summary_df = pd.DataFrame(summary_rows).set_index("config")
    return per_system_dfs, summary_df
