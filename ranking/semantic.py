# ranking/semantic.py

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity


def semantic_rank(query, docs):
    """
    Lightweight semantic similarity (NO torch, NO transformers)
    """

    if not docs:
        return []

    texts = docs + [query]

    vectorizer = TfidfVectorizer(stop_words='english')
    tfidf_matrix = vectorizer.fit_transform(texts)

    query_vec = tfidf_matrix[-1]
    doc_vecs = tfidf_matrix[:-1]

    scores = cosine_similarity(query_vec, doc_vecs)[0]

    return scores