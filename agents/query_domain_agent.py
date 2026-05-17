
def refine_query(query: str):
    """
    Keep query clean — DO NOT overload.
    """
    q = query.lower().strip()

    # remove duplicate words
    words = list(dict.fromkeys(q.split()))

    return " ".join(words)


def generate_subqueries(query: str):
    """
    Split complex query into multiple focused queries
    """

    q = query.lower()
    subs = []

    # base query
    subs.append(query)
    if "fairness" in q:
        subs.append("fairness in machine learning healthcare bias mitigation")
        subs.append("ethical ai clinical decision support bias fairness healthcare")

    # =========================
    # HEALTHCARE
    # =========================
    if "clinical" in q or "healthcare" in q:
        subs.append("clinical decision support systems machine learning healthcare")
        subs.append("ai for medical diagnosis prediction models healthcare")

    # =========================
    # EXPLAINABILITY
    # =========================
    if "explainable" in q or "fairness" in q:
        subs.append("explainable ai models interpretability machine learning")
        subs.append("xai methods shap lime model explanation")

    # =========================
    # INTRUSION DETECTION
    # =========================
    if "intrusion" in q or "ids" in q:
        subs.append("intrusion detection system machine learning")
        subs.append("network anomaly detection deep learning cyber security")

    # =========================
    # FEDERATED LEARNING
    # =========================
    if "federated" in q:
        subs.append("federated learning privacy distributed training")
        subs.append("secure federated learning healthcare data")

    # =========================
    # BLOCKCHAIN
    # =========================
    if "blockchain" in q:
        subs.append("blockchain distributed ledger consensus smart contracts")

    # =========================
    # AUTONOMOUS / VISION
    # =========================
    if "autonomous" in q or "vehicle" in q:
        subs.append("autonomous driving perception lidar camera fusion")
        subs.append("multi modal sensor fusion object detection autonomous driving")

    return subs

from collections import defaultdict


def detect_domains(text: str):
    text = text.lower()
    scores = defaultdict(int)
    domains = set()

    # =========================
    # HARD RULES (CRITICAL FIX)
    # =========================

    # SECURITY (force detect)
    if any(k in text for k in [
        "intrusion",
        "intrusion detection",
        "intrusion detection system",
        "ids",
        "anomaly detection",
        "cyber attack",
        "malware",
        "network security"
    ]):
        domains.add("security")

    # XAI (force detect)
    if any(k in text for k in [
        "explainable",
        "interpretability",
        "explanation",
        "transparent",
        "xai"
    ]):
        domains.add("xai")

    # =========================
    # SOFT SCORING
    # =========================

    DOMAIN_KEYWORDS = {
        "distributed": [
            "distributed", "consensus", "replication", "byzantine", "paxos", "raft"
        ],
        "ml": [
            "machine learning", "deep learning", "neural", "transformer", "model", "classification"
        ],
        "healthcare": [
            "healthcare", "medical", "clinical", "diagnosis", "patient"
        ],
        "blockchain": [
            "blockchain", "ledger", "smart contract", "decentralized"
        ],
        "vision": [
            "image", "video", "vision", "mri", "ct", "radiology"
        ],
        "nlp": [
            "language model", "text", "nlp", "bert"
        ],
    }

    for domain, keywords in DOMAIN_KEYWORDS.items():
        for kw in keywords:
            if kw in text:
                scores[domain] += 2 if len(kw.split()) > 1 else 1

    # select domains with enough score
    for d, s in scores.items():
        if s >= 2:
            domains.add(d)

    # =========================
    # FALLBACK (IMPORTANT)
    # =========================
    if not domains:
        # take top 2 instead of empty
        fallback = sorted(scores.items(), key=lambda x: x[1], reverse=True)[:2]
        domains = {d for d, _ in fallback}

    return list(domains)