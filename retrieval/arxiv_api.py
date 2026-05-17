import arxiv
from datetime import datetime


# =========================
# SMART KEYWORD EXTRACTION
# =========================
def extract_keywords(query):
    words = query.lower().split()

    # remove useless words
    stopwords = {
        "the", "and", "for", "with", "using",
        "based", "systems", "models", "approach"
    }

    keywords = [w for w in words if w not in stopwords and len(w) > 3]

    return keywords[:6]  # allow more signal


# =========================
# BUILD ARXIV QUERY (FIXED)
# =========================
def build_arxiv_query(query):

    keywords = extract_keywords(query)

    if not keywords:
        return f'all:"{query}"'

    # 🔥 BETTER MATCHING (AND + OR MIX)
    keyword_query = " AND ".join(
        [f'(ti:"{k}" OR abs:"{k}")' for k in keywords]
    )

    # 🔥 REMOVE HARD DOMAIN RESTRICTION
    return keyword_query


# =========================
# TIME FILTER
# =========================
def min_year_from_time_filter(time_filter):
    if not time_filter or time_filter == "All":
        return None

    try:
        return int(str(time_filter).split("-")[0])
    except:
        return None


# =========================
# FETCH PAPERS (FIXED)
# =========================
def fetch_arxiv_papers(query, max_papers=5, time_filter="All"):

    search_query = build_arxiv_query(query)
    min_year = min_year_from_time_filter(time_filter)

    search = arxiv.Search(
        query=search_query,
        max_results=max(max_papers * 5, 30),
        sort_by=arxiv.SortCriterion.Relevance
    )

    client = arxiv.Client()

    papers = []

    for paper in client.results(search):
        try:
            # time filter
            if min_year and paper.published.year < min_year:
                continue

            arxiv_id = paper.entry_id.split("/")[-1]

            papers.append({
                "title": paper.title,
                "abstract": paper.summary,
                "year": paper.published.year,
                "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}"
            })

            if len(papers) >= max_papers:
                break

        except:
            continue

    # 🔥 FALLBACK IF EMPTY
    if not papers:
        try:
            fallback = arxiv.Search(
                query=f"all:{query}",
                max_results=max_papers,
                sort_by=arxiv.SortCriterion.Relevance
            )

            for paper in client.results(fallback):
                arxiv_id = paper.entry_id.split("/")[-1]

                papers.append({
                    "title": paper.title,
                    "abstract": paper.summary,
                    "year": paper.published.year,
                    "pdf_url": f"https://arxiv.org/pdf/{arxiv_id}"
                })

                if len(papers) >= max_papers:
                    break
        except:
            pass

    return papers


# =========================
# PUBLICATION TREND (FIXED)
# =========================
def get_publication_trend(query, start_year=2020):

    client = arxiv.Client()
    current_year = datetime.now().year

    search = arxiv.Search(
        query=f"all:{query}",
        max_results=400,
        sort_by=arxiv.SortCriterion.Relevance
    )

    year_counts = {year: 0 for year in range(start_year, current_year + 1)}

    try:
        for paper in client.results(search):
            year = paper.published.year

            if year in year_counts:
                year_counts[year] += 1

    except Exception as e:
        print("Trend Error:", e)

    return [(year, year_counts[year]) for year in sorted(year_counts)]


# =========================
# PUBLIC API
# =========================
def search_arxiv(query, max_results=5, time_filter="All"):
    return fetch_arxiv_papers(query, max_results, time_filter)