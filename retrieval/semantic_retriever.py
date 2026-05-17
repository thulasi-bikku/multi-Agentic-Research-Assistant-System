import numpy as np
from sentence_transformers import SentenceTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

_model = None
_model_failed = False


def _paper_text(paper):
    return f"{paper.get('title', '')} {paper.get('abstract', '')}".strip()


def _get_model():
    global _model, _model_failed

    if _model_failed:
        return None

    if _model is not None:
        return _model

    try:
        _model = SentenceTransformer("all-MiniLM-L6-v2")
        return _model
    except Exception as exc:
        _model_failed = True
        print("Semantic model unavailable, using TF-IDF fallback:", exc)
        return None


def _tfidf_ranking(query, papers, top_k):
    texts = [_paper_text(paper) for paper in papers]

    if not texts:
        return []

    try:
        vectorizer = TfidfVectorizer(stop_words="english")
        matrix = vectorizer.fit_transform([query] + texts)
        scores = cosine_similarity(matrix[0:1], matrix[1:]).ravel()
    except Exception:
        query_terms = set(query.lower().split())
        scores = np.array([
            len(query_terms.intersection(set(text.lower().split())))
            for text in texts
        ], dtype=float)

    indexes = np.argsort(scores)[::-1][:top_k]
    return [(papers[index], float(scores[index])) for index in indexes]


def hybrid_ranking(query, papers, top_k=15):
    if not papers:
        return []

    top_k = min(top_k, len(papers))
    texts = [_paper_text(paper) for paper in papers]
    model = _get_model()

    if model is None:
        return _tfidf_ranking(query, papers, top_k)

    try:
        embeddings = model.encode(texts, convert_to_numpy=True)
        query_embedding = model.encode([query], convert_to_numpy=True)[0]

        scores = np.dot(embeddings, query_embedding)
        indexes = np.argsort(scores)[::-1][:top_k]

        return [(papers[index], float(scores[index])) for index in indexes]

    except Exception as exc:
        print("Semantic ranking error, using TF-IDF fallback:", exc)
        return _tfidf_ranking(query, papers, top_k)