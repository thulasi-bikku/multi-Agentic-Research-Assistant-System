from retrieval.semantic_retriever import hybrid_ranking
from agents.query_domain_agent import detect_domains


def rank_papers(query, papers, top_k=5):

    domains = detect_domains(query)

    ranked = hybrid_ranking(query, papers, top_k=len(papers))

    boosted = []

    for p, score in ranked:
        text = (p["title"] + " " + p["abstract"]).lower()

        for d in domains:
            if d in text:
                score += 0.2   # multi-domain boost

        boosted.append((p, score))

    boosted.sort(key=lambda x: x[1], reverse=True)

    final = []
    for i, (p, s) in enumerate(boosted[:top_k]):
        p["score"] = round(s, 4)
        p["rank"] = i + 1
        final.append(p)

    return final