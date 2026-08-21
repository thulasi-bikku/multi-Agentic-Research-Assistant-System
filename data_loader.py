"""
Loads corpus/queries/qrels in BEIR's standard (corpus, queries, qrels) format:
    corpus: {doc_id: {"title": str, "text": str, "year": int|None}}
    queries: {query_id: str}
    qrels:  {query_id: {doc_id: relevance_int}}

BEIR datasets (trec-covid, scifact, nfcorpus, msmarco) download automatically
via the `beir` package the first time they're used. ArXiv-CS is not part of
BEIR, so `load_arxiv_cs` expects a local CSV (matching the Kaggle
Cornell-University/arxiv metadata dump referenced in the paper's Data
Availability section) with columns: id, title, abstract, categories,
update_date. It builds pseudo-qrels via category + query-term matching if
you don't already have human-labeled qrels — replace with real qrels if
you have them; do not report retrieval metrics on synthetic qrels as if
they were gold-standard.
"""
import os
import random
import pandas as pd

BEIR_DATASETS = {"trec-covid", "scifact", "nfcorpus", "msmarco"}


def load_beir_dataset(name: str, data_dir: str = "./beir_data"):
    from beir import util
    from beir.datasets.data_loader import GenericDataLoader

    assert name in BEIR_DATASETS, f"{name} is not a BEIR dataset, use load_arxiv_cs"
    url = f"https://public.ukp.informatik.tu-darmstadt.de/thakur/BEIR/datasets/{name}.zip"
    out_dir = util.download_and_unzip(url, data_dir)
    corpus, queries, qrels = GenericDataLoader(data_folder=out_dir).load(split="test")

    # normalize field name / attach a fake "year" if the dataset has none —
    # temporal relevance is only meaningful where publication dates exist.
    for doc_id, doc in corpus.items():
        doc.setdefault("year", None)
    return corpus, queries, qrels


def load_msmarco_subset(data_dir: str = "./beir_data", n_queries: int = 6980, seed: int = 42):
    corpus, queries, qrels = load_beir_dataset("msmarco", data_dir)
    if n_queries is not None and n_queries < len(queries):
        rng = random.Random(seed)
        keep_ids = set(rng.sample(list(queries.keys()), n_queries))
        queries = {qid: q for qid, q in queries.items() if qid in keep_ids}
        qrels = {qid: r for qid, r in qrels.items() if qid in keep_ids}
    return corpus, queries, qrels


def load_arxiv_cs(csv_path: str, query_terms_path: str = None, n_docs: int = 24500, seed: int = 42):
    """
    Loads the Kaggle arXiv metadata dump. You must download it yourself
    (Kaggle requires authentication this sandbox/pipeline does not have):
    https://www.kaggle.com/datasets/Cornell-University/arxiv

    Filters to cs.* categories, samples n_docs, and expects a queries file
    (one query per line) if you have real research questions with human
    relevance judgments. Without human judgments, do NOT report P/R/NDCG
    on this split -- report it as a demonstration only, per Reviewer 3
    point 9's logic about unvalidated ground truth.
    """
    if not os.path.exists(csv_path):
        raise FileNotFoundError(
            f"{csv_path} not found. Download the arXiv metadata CSV from Kaggle "
            "(Cornell-University/arxiv) and pass its path here."
        )
    df = pd.read_csv(csv_path)
    df = df[df["categories"].astype(str).str.contains("cs\\.", regex=True, na=False)]
    df["year"] = pd.to_datetime(df["update_date"], errors="coerce").dt.year
    if n_docs is not None and n_docs < len(df):
        df = df.sample(n=n_docs, random_state=seed)

    corpus = {
        str(row.id): {"title": str(row.title), "text": str(row.abstract), "year": row.year}
        for row in df.itertuples()
    }

    queries, qrels = {}, {}
    if query_terms_path and os.path.exists(query_terms_path):
        with open(query_terms_path) as f:
            for i, line in enumerate(f):
                line = line.strip()
                if line:
                    queries[f"q{i}"] = line
        # NOTE: qrels must come from human annotation (paper: 3 domain experts,
        # kappa=0.86). This loader does not fabricate them.
    return corpus, queries, qrels
