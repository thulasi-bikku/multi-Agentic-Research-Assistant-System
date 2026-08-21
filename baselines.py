"""
Reviewer 3, point 4: "The only comparators are the authors' own ablations."
Reviewer 1, point 1: needs a single-unified-LLM ablation baseline.

This module adds real external comparators, not just internal ablations,
and carries published literature numbers so they can be printed alongside
the locally-measured ones for context (never presented as if they were
measured on the exact same run -- protocol differences are noted).
"""
import numpy as np
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


# Published BM25 baselines from the original BEIR paper (Thakur et al. 2021,
# NeurIPS Datasets & Benchmarks) and dataset-specific leaderboards, reported
# as NDCG@10. Cite these, do not claim they were reproduced locally unless
# you actually re-ran BM25 yourself with the identical BEIR pipeline (which
# `retrieval/bm25.py` does, and its output should be checked against these
# as a sanity bound before trusting the hybrid numbers).
PUBLISHED_BM25_NDCG10 = {
    "trec-covid": 0.656,
    "scifact": 0.665,
    "nfcorpus": 0.325,
    "msmarco": 0.228,
}


def tfidf_only_scores(corpus: dict, queries: dict) -> dict:
    doc_ids = list(corpus.keys())
    texts = [f"{corpus[d].get('title','')} {corpus[d].get('text','')}" for d in doc_ids]
    vectorizer = TfidfVectorizer(max_features=50000)
    doc_matrix = vectorizer.fit_transform(texts)

    out = {}
    for qid, q in queries.items():
        q_vec = vectorizer.transform([q])
        sims = cosine_similarity(q_vec, doc_matrix).flatten()
        out[qid] = {doc_id: float(s) for doc_id, s in zip(doc_ids, sims)}
    return out


def dense_retriever_scores(corpus: dict, queries: dict,
                            model_name: str = "sentence-transformers/msmarco-distilbert-base-v3"):
    """A genuinely different, modern dense retriever (trained specifically
    for retrieval, not just general sentence similarity like MiniLM), so
    the comparison in point 4 isn't hybrid-vs-its-own-components."""
    from sentence_transformers import SentenceTransformer
    model = SentenceTransformer(model_name)
    doc_ids = list(corpus.keys())
    texts = [f"{corpus[d].get('title','')} {corpus[d].get('text','')}" for d in doc_ids]
    doc_emb = model.encode(texts, batch_size=64, normalize_embeddings=True, show_progress_bar=True)

    qids = list(queries.keys())
    q_emb = model.encode([queries[q] for q in qids], normalize_embeddings=True, show_progress_bar=True)
    sims = q_emb @ doc_emb.T
    return {qid: {doc_id: float(s) for doc_id, s in zip(doc_ids, sims[i])} for i, qid in enumerate(qids)}


def single_llm_baseline(queries: dict, corpus_sample_size: int = 0, llm_call_fn=None):
    """
    Reviewer 1: ablation against a single unified LLM with NO agent
    decomposition and NO retrieval grounding -- just "answer this research
    question" in one shot. Pass an `llm_call_fn(prompt: str) -> str`
    (e.g. wrapping the Anthropic API) to actually generate outputs; this
    function only builds the fair-comparison harness, since the actual
    generation call needs your API credentials.

    Score this baseline's OUTPUT on the same automated writing metrics
    used in writing_eval.py (grounding overlap, factual-claim checking)
    so "MARAS vs single LLM" is measured on identical criteria, not
    retrieval metrics the single LLM was never designed to produce.
    """
    if llm_call_fn is None:
        raise ValueError("Provide llm_call_fn to actually run this baseline.")
    outputs = {}
    for qid, q in queries.items():
        prompt = (
            f"Answer the following research question directly, using only your own "
            f"knowledge (no external documents provided): {q}"
        )
        outputs[qid] = llm_call_fn(prompt)
    return outputs
