"""
Orchestrates the whole re-evaluation. Run on a machine with internet
access to the BEIR dataset host. Writes CSV + Markdown tables to --out.

Example:
    python run_full_pipeline.py --datasets trec-covid scifact nfcorpus --out results/
"""
import argparse
import os
import pandas as pd

from config import RunConfig
import data_loader
from retrieval.bm25 import BM25Retriever
from retrieval.semantic import SemanticRetriever
from retrieval.temporal import TemporalScorer
from retrieval.hybrid import HybridRetriever
import baselines
import evaluate as ev
import significance as sig
import sensitivity as sens
import ablation


def run_one_dataset(name: str, cfg: RunConfig, out_dir: str):
    print(f"\n===== {name} =====")
    if name == "msmarco":
        corpus, queries, qrels = data_loader.load_msmarco_subset(n_queries=cfg.msmarco_query_subset)
    elif name == "arxiv-cs":
        print("Skipping arxiv-cs in automatic run: requires local Kaggle CSV + human qrels. "
              "See data_loader.load_arxiv_cs().")
        return None
    else:
        corpus, queries, qrels = data_loader.load_beir_dataset(name)

    print(f"corpus={len(corpus)} queries={len(queries)}")
    print(f"protocol: depth={cfg.retrieval_depth}, pool={cfg.candidate_pool}, "
          f"qrels={cfg.qrels_source}, NDCG@{cfg.ndcg_cutoff}, "
          f"P@{cfg.precision_cutoff}, R@{cfg.recall_cutoff}")

    (val_q, val_qrels), (test_q, test_qrels) = sens.split_queries(
        queries, qrels, cfg.val_query_fraction, cfg.random_seed)

    bm25 = BM25Retriever(corpus)
    semantic = SemanticRetriever(corpus, cfg.semantic_model_name)
    temporal = TemporalScorer(corpus, cfg.lambda_decay)
    temporal_scores = temporal.score_all_docs()

    bm25_scores = bm25.score_all(queries)
    semantic_scores = semantic.score_all(queries)

    # --- sensitivity analysis on VAL split only (Reviewer 3, point 6) ---
    sensitivity_df = sens.run_sensitivity(bm25_scores, semantic_scores, temporal_scores,
                                           val_qrels, cfg)
    sensitivity_df.to_csv(f"{out_dir}/{name}_sensitivity.csv", index=False)
    best_alpha, best_beta, best_gamma = sens.best_config(sensitivity_df)
    print(f"best weights on VAL split: alpha={best_alpha} beta={best_beta} gamma={best_gamma}")

    # --- final evaluation on TEST split with chosen weights ---
    hybrid = HybridRetriever(bm25_scores, semantic_scores, temporal_scores,
                              best_alpha, best_beta, best_gamma)
    run = hybrid.rank_all(top_k=cfg.retrieval_depth)
    test_run = {qid: run[qid] for qid in test_qrels if qid in run}
    results = ev.evaluate_run(test_qrels, test_run, cfg)
    per_query_df = ev.per_query_dataframe(results, cfg)
    summary = ev.dataset_summary(per_query_df)
    print("test-set summary:", summary)

    # --- leakage check (Reviewer 3, point 3) ---
    top_docs = ev.top_doc_per_query(test_run)
    leakage_df = ev.check_leakage(test_qrels, test_q, corpus, top_docs)
    if len(leakage_df):
        print(f"WARNING: {len(leakage_df)} queries show possible leakage (near-duplicate top doc)")
        leakage_df.to_csv(f"{out_dir}/{name}_leakage_flags.csv", index=False)

    # --- baselines (Reviewer 3, point 4) ---
    tfidf_scores = baselines.tfidf_only_scores(corpus, test_q)
    dense_scores = baselines.dense_retriever_scores(corpus, test_q, cfg.dense_baseline_model_name)

    per_system_dfs, ablation_summary = ablation.run_ablation_suite(
        {q: bm25_scores[q] for q in test_q}, {q: semantic_scores[q] for q in test_q},
        temporal_scores, tfidf_scores, dense_scores, test_qrels, cfg)
    per_system_dfs["full_maras"] = per_query_df  # replace with tuned-weight version
    ablation_summary.loc["full_maras"] = summary
    ablation_summary.to_csv(f"{out_dir}/{name}_ablation_and_baselines.csv")
    print(ablation_summary)

    if name in baselines.PUBLISHED_BM25_NDCG10:
        published = baselines.PUBLISHED_BM25_NDCG10[name]
        measured = ablation_summary.loc["bm25_only", "ndcg@10"]
        print(f"sanity check: published BM25 NDCG@10={published:.3f} vs "
              f"locally measured={measured:.3f} (large gaps suggest a protocol bug)")

    # --- significance testing (Reviewer 3, point 5) ---
    sig_df = sig.run_all_comparisons(per_system_dfs, reference="full_maras",
                                      dataset_name=name, metric_name="ndcg@10",
                                      alpha=cfg.alpha_significance, method=cfg.correction_method)
    sig_df.to_csv(f"{out_dir}/{name}_significance.csv", index=False)
    print(sig_df[["comparison", "mean_diff", "t_pvalue", "corrected_pvalue", "significant_after_correction"]])

    return {"summary": summary, "n_queries": summary["n_queries"]}


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument("--datasets", nargs="+", default=["trec-covid", "scifact", "nfcorpus"])
    parser.add_argument("--out", default="results")
    args = parser.parse_args()

    os.makedirs(args.out, exist_ok=True)
    cfg = RunConfig(datasets=args.datasets)

    per_dataset_summaries = {}
    for name in cfg.datasets:
        result = run_one_dataset(name, cfg, args.out)
        if result:
            per_dataset_summaries[name] = result["summary"]

    if per_dataset_summaries:
        agg = ev.aggregate(per_dataset_summaries)
        agg["per_dataset"].to_csv(f"{args.out}/all_datasets_primary_table.csv")
        print("\n=== PRIMARY TABLE (per-dataset, report this, not a single pooled number) ===")
        print(agg["per_dataset"])
        print("\nmacro-average (unweighted across datasets):")
        print(agg["macro_average (unweighted across datasets)"])
        print("\nmicro-average (weighted by query count):")
        print(agg["micro_average (weighted by query count)"])


if __name__ == "__main__":
    main()
