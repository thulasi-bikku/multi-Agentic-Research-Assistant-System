from rank_bm25 import BM25Okapi


def bm25_rank(query, docs):
    tokenized_docs = [doc.split() for doc in docs]
    bm25 = BM25Okapi(tokenized_docs)
    return bm25.get_scores(query.split())