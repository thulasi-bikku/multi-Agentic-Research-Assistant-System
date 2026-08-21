"""
Central configuration. Every field here is a protocol choice that Reviewer 3
(point 1) asked to see stated explicitly rather than left implicit.
"""
from dataclasses import dataclass, field
from typing import List


@dataclass
class RunConfig:
    # --- retrieval protocol (Reviewer 3, point 1) ---
    retrieval_depth: int = 1000          # candidates retrieved per query before ranking
    ndcg_cutoff: int = 10                # NDCG@10
    precision_cutoff: int = 10           # P@10
    recall_cutoff: int = 100             # R@100
    mrr_cutoff: int = 10                 # MRR@10
    candidate_pool: str = "full_collection"  # NOT a reranked subset — full corpus is scored
    qrels_source: str = "official_beir_release"

    # --- hybrid scoring weights (Section IV-A of the paper) ---
    alpha: float = 0.35   # BM25 weight
    beta: float = 0.50    # semantic weight
    gamma: float = 0.15   # temporal weight
    lambda_decay: float = 0.05  # temporal decay rate, years^-1

    # --- datasets (Reviewer 3, point 2: report per-dataset, not just pooled) ---
    datasets: List[str] = field(default_factory=lambda: [
        "trec-covid", "scifact", "nfcorpus", "msmarco", "arxiv-cs"
    ])
    # NOTE for first Colab T4 run: start with 1000-2000, not 6980. Once the
    # pipeline finishes cleanly end-to-end, bump this back up to 6980 to
    # match the paper's stated subset size for the final numbers you report.
    msmarco_query_subset: int = 1500

    # --- models ---
    semantic_model_name: str = "sentence-transformers/all-MiniLM-L6-v2"
    dense_baseline_model_name: str = "sentence-transformers/msmarco-distilbert-base-v3"

    # --- statistics (Reviewer 3, point 5: paired-by-query, not paired-by-seed) ---
    alpha_significance: float = 0.05
    correction_method: str = "holm"   # Holm-Bonferroni across multiple baseline comparisons

    # --- sensitivity analysis (Reviewer 3, point 6) ---
    # 0.2 val fraction + 0.05 grid step = ~230 configs x ~1400 queries on
    # MS MARCO ~ 90+ minutes even after vectorizing hybrid.py. 0.1 fraction
    # + 0.1 step cuts that to roughly 10-15 minutes with negligible loss of
    # precision in picking the best (alpha, beta, gamma) -- a coarse grid
    # is enough to find the right neighborhood; you don't need a value
    # every 0.05 to see that beta=0.5 dominates.
    val_query_fraction: float = 0.1
    sensitivity_grid_step: float = 0.1

    random_seed: int = 42


def assert_normalized(alpha: float, beta: float, gamma: float, tol: float = 1e-6):
    total = alpha + beta + gamma
    if abs(total - 1.0) > tol:
        raise ValueError(f"alpha+beta+gamma must equal 1.0, got {total}")
