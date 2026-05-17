import time
from retrieval.arxiv_api import fetch_arxiv_papers as search_arxiv
from retrieval.semantic_retriever import hybrid_ranking
from agents.query_domain_agent import detect_domains


# =========================
# FILTER CONFIG
# =========================
TITLE_FILTER_KEYWORDS = ["survey", "review", "overview"]
ABSTRACT_FILTER_PHRASES = ["this survey", "this review"]


# =========================
# HELPERS
# =========================
def _is_survey_or_review(paper):
    title = paper.get("title", "").lower()
    abstract = paper.get("abstract", "").lower()

    return any(k in title for k in TITLE_FILTER_KEYWORDS) or \
           any(p in abstract for p in ABSTRACT_FILTER_PHRASES)


# =========================
# 🔥 IMPROVED SUBQUERIES
# =========================
def generate_subqueries(query):
    q = query.lower()
    subs = [query]

    # =========================
    # CORE AI / ML (MISSING FIX)
    # =========================
    if "ai" in q or "machine learning" in q:
        subs.append("machine learning models healthcare prediction")
        subs.append("deep learning clinical decision support systems")

    # =========================
    # HEALTHCARE / CDSS
    # =========================
    if any(k in q for k in ["clinical", "healthcare", "medical"]):
        subs.append("clinical decision support systems machine learning healthcare")
        subs.append("ai for medical diagnosis prediction healthcare")

    # =========================
    # EXPLAINABILITY
    # =========================
    if any(k in q for k in ["explainable", "interpretability"]):
        subs.append("explainable ai healthcare models interpretability")
        subs.append("xai methods shap lime clinical decision support")

    # =========================
    # 🔥 FAIRNESS (YOU WERE WEAK HERE)
    # =========================
    if "fairness" in q or "bias" in q:
        subs.append("fairness in machine learning healthcare bias mitigation")
        subs.append("ethical ai healthcare bias evaluation clinical systems")

    return list(dict.fromkeys(subs))

# =========================
# 🔥 SOFT DOMAIN FILTER (FIXED)
# =========================
def _enforce_domain(query, paper):
    domains = detect_domains(query)

    text = (paper.get("title", "") + " " + paper.get("abstract", "")).lower()

    score = 0


    # AI / ML
    if any(k in domains for k in ["ml", "xai"]):
        if any(k in text for k in ["machine learning", "deep learning", "model", "ai"]):
            score += 1
    # SECURITY
    if "security" in domains:
        if any(k in text for k in ["intrusion", "anomaly", "cyber", "malware"]):
            score += 1

    # XAI
    if "xai" in domains:
        if any(k in text for k in ["explainable", "interpretability", "xai"]):
            score += 1

    # HEALTHCARE
    if "healthcare" in domains:
        if any(k in text for k in ["medical", "clinical", "healthcare"]):
            score += 1

    # VISION
    if "vision" in domains:
        if any(k in text for k in ["image", "vision", "lidar", "camera"]):
            score += 1

    # BLOCKCHAIN
    if "blockchain" in domains:
        if "blockchain" in text:
            score += 1

    # 🔥 KEY CHANGE: VERY SOFT FILTER
    return score >= 1


# =========================
# MAIN FUNCTION
# =========================
def retrieve_papers(query, max_results=5, time_filter="All"):

    start_time = time.time()

    try:
        # =========================
        # STEP 1: MULTI QUERY
        # =========================
        subqueries = generate_subqueries(query)
        print("🔍 Subqueries:", subqueries)

        fetch_size = max_results * 8
        all_papers = []

        # =========================
        # STEP 2: FETCH
        # =========================
        for sq in subqueries:
            results = search_arxiv(sq, fetch_size // len(subqueries))
            all_papers.extend(results)

        # 🔥 FALLBACK (CRITICAL)
        if not all_papers:
            print("⚠️ fallback retrieval")
            all_papers = search_arxiv("machine learning", max_results * 3)

        # =========================
        # STEP 3: REMOVE DUPLICATES
        # =========================
        seen = set()
        papers = []

        for p in all_papers:
            title = p.get("title", "")
            if title not in seen:
                seen.add(title)
                papers.append(p)

        # =========================
        # STEP 4: REMOVE SURVEYS
        # =========================
        papers = [p for p in papers if not _is_survey_or_review(p)]

        if not papers:
            return []

        # =========================
        # STEP 5: RANKING
        # =========================
        ranked = hybrid_ranking(query, papers, top_k=len(papers))
        ranked_papers = [p for p, _ in ranked]

        # =========================
        # STEP 6: DOMAIN FILTER
        # =========================
        filtered = [p for p in ranked_papers if _enforce_domain(query, p)]

        if len(filtered) < max_results:
            print("⚠️ Relaxing domain filter")
            filtered = ranked_papers[:max_results]

        # =========================
        # STEP 7: TIME FILTER
        # =========================
        if time_filter != "All":
            try:
                year_limit = int(time_filter.split("-")[0])

                temp = [
                    p for p in filtered
                    if int(p.get("year", 0)) >= year_limit
                ]

                if temp:
                    filtered = temp

            except:
                pass

        # =========================
        # FINAL
        # =========================
        final = filtered[:max_results]

        for i, p in enumerate(final):
            p["rank"] = i + 1

        elapsed = round(time.time() - start_time, 2)
        print(f"⏱️ Retrieval completed in {elapsed}s")

        return final

    except Exception as e:
        print("❌ Retrieval Error:", e)
        return []



def filter_papers(papers, domain, threshold=1):

    filtered = []

    for p in papers:
        text = (p.get("title", "") + " " + p.get("abstract", "")).lower()

        score = sum(1 for w in domain.split() if w in text)

        if score >= threshold:
            filtered.append(p)

    return filtered