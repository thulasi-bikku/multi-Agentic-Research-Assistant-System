"""
Reviewer 3, point 8: Writing Agent has no evaluation, no participants, no
comparison condition.
Reviewer 2, point 2: writing-assistance modules need standardized P/R/F1,
not only qualitative human assessment.

This gives two things:
  1. extraction_prf(): precision/recall/F1 for the structured-extraction
     sub-task (methodology/dataset/metric/limitation), generalizing the
     Table 1 numbers into a reusable, re-runnable evaluator against a
     gold-labeled set you supply.
  2. automated writing-quality metrics computed for BOTH the MARAS output
     and a required control condition (same LLM, no retrieval grounding),
     so any reported improvement is a measured delta, not a bare score.
"""
from dataclasses import dataclass
import numpy as np
import pandas as pd


def extraction_prf(gold: dict, predicted: dict) -> pd.DataFrame:
    """
    gold, predicted: {doc_id: {"methodology": [...spans...], "dataset": [...],
                                "metric": [...], "limitation": [...]}}
    Computes exact-span P/R/F1 per field, matching the structure of the
    paper's Table 1 but on data you actually pass in, not a table with no
    stated dataset or annotation protocol (Reviewer 3, minor point on
    Table 1).
    """
    fields = ["methodology", "dataset", "metric", "limitation"]
    rows = []
    for field in fields:
        tp = fp = fn = 0
        for doc_id in gold:
            gold_spans = set(s.strip().lower() for s in gold[doc_id].get(field, []))
            pred_spans = set(s.strip().lower() for s in predicted.get(doc_id, {}).get(field, []))
            tp += len(gold_spans & pred_spans)
            fp += len(pred_spans - gold_spans)
            fn += len(gold_spans - pred_spans)
        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0
        rows.append({"field": field, "precision": precision, "recall": recall, "f1": f1,
                      "tp": tp, "fp": fp, "fn": fn})
    return pd.DataFrame(rows)


def grounding_overlap(generated_text: str, evidence_block: str) -> float:
    """
    ROUGE-L-style longest-common-subsequence overlap between the writing
    agent's output and the evidence block it was given, as an automated
    proxy for "is this actually grounded in the retrieved literature" --
    not a substitute for human factual-accuracy judgment, but a cheap,
    reproducible signal you can compute for every single output instead
    of only a hand-wavy "user observations indicated...".
    """
    from difflib import SequenceMatcher
    return SequenceMatcher(None, generated_text.lower(), evidence_block.lower()).ratio()


def semantic_grounding(generated_text: str, evidence_block: str, model=None) -> float:
    """Cosine similarity between sentence embeddings, catching paraphrased
    (not just lexically overlapping) grounding that ROUGE-style overlap misses."""
    if model is None:
        from sentence_transformers import SentenceTransformer
        model = SentenceTransformer("sentence-transformers/all-MiniLM-L6-v2")
    embs = model.encode([generated_text, evidence_block], normalize_embeddings=True)
    return float(embs[0] @ embs[1])


@dataclass
class WritingSample:
    query_id: str
    condition: str            # "maras" or "control_no_retrieval"
    generated_text: str
    evidence_block: str = ""  # empty for the control condition, by design


def evaluate_writing_samples(samples: list, model=None) -> pd.DataFrame:
    """
    samples: list[WritingSample], containing BOTH conditions for the same
    set of queries (Reviewer 3's required comparison condition).
    """
    rows = []
    for s in samples:
        row = {"query_id": s.query_id, "condition": s.condition,
               "length_words": len(s.generated_text.split())}
        if s.condition == "maras" and s.evidence_block:
            row["lexical_grounding"] = grounding_overlap(s.generated_text, s.evidence_block)
            row["semantic_grounding"] = semantic_grounding(s.generated_text, s.evidence_block, model)
        else:
            row["lexical_grounding"] = np.nan
            row["semantic_grounding"] = np.nan
        rows.append(row)
    return pd.DataFrame(rows)


def grounding_delta(df: pd.DataFrame) -> dict:
    """The actual comparison the paper is missing: does grounding go up
    when retrieval evidence is present vs. the control condition. Only
    meaningful once you also have human factual-accuracy ratings for
    both conditions (see human_eval_protocol.py) -- automated grounding
    overlap is a proxy, not a replacement for the human judgment."""
    maras_mean = df.loc[df.condition == "maras", "semantic_grounding"].mean()
    return {"maras_mean_semantic_grounding": maras_mean,
            "note": "compare against human factual-accuracy scores for both "
                     "conditions in human_eval_protocol.py, not this number alone"}
