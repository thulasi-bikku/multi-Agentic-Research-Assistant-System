import os
import re

import fitz
import requests
from dotenv import load_dotenv
from groq import Groq

load_dotenv()

client = Groq(api_key=os.getenv("GROQ_API_KEY")) if os.getenv("GROQ_API_KEY") else None
_llm_failed = False

FIELD_DEFAULTS = {
    "problem": "Not specified",
    "method": "Not specified",
    "dataset": "Not specified",
    "performance": "Not specified",
    "application": "Not specified",
    "limitations": "Not specified",
}

KEYWORD_GROUPS = {
    "problem": ["problem", "challenge", "issue", "gap", "difficulty", "need", "limitation"],
    "method": ["propose", "proposed", "model", "framework", "approach", "method", "architecture", "algorithm"],
    "dataset": ["dataset", "benchmark", "evaluated on", "trained on", "tested on", "using the", "corpus"],
    "performance": ["accuracy", "f1", "precision", "recall", "auc", "improvement", "outperform", "%"],
    "application": ["application", "used for", "applied to", "task", "domain", "system"],
    "limitations": ["limitations", "future work", "drawbacks", "constraint", "weakness", "fails", "limited"],
}

KNOWN_DATASETS = [
    "ImageNet", "CIFAR-10", "CIFAR-100", "MNIST", "COCO", "KITTI", "nuScenes",
    "Waymo", "SQuAD", "GLUE", "SuperGLUE", "MIMIC", "MIMIC-III", "MIMIC-IV",
    "PhysioNet", "UCI", "KDD", "NSL-KDD", "UNSW-NB15", "CICIDS2017",
]


def clean_text(text, limit=9000):
    text = re.sub(r"\s+", " ", text or "")
    return text.strip()[:limit]


def _sentences(text):
    parts = re.split(r"(?<=[.!?])\s+(?=[A-Z0-9])", clean_text(text, limit=20000))
    cleaned = []
    for part in parts:
        part = part.strip(" -:\t\n")
        words = part.split()
        if 8 <= len(words) <= 70 and not part.endswith("-"):
            cleaned.append(part)
    return cleaned


def extract_pdf_text(url, max_pages=8):
    if not url:
        return ""

    try:
        response = requests.get(url, timeout=12)
        response.raise_for_status()
        doc = fitz.open(stream=response.content, filetype="pdf")
        text = " ".join(doc[i].get_text() for i in range(min(max_pages, len(doc))))
        return clean_text(text, limit=6000)
    except Exception:
        return ""


def _score_sentence(sentence, field):
    lower = sentence.lower()
    score = 0

    for keyword in KEYWORD_GROUPS[field]:
        if keyword in lower:
            score += 4 if " " in keyword else 2

    if field == "performance" and re.search(r"\b\d+(\.\d+)?\s?%|\b0\.\d+\b", lower):
        score += 5
    if field == "dataset" and _dataset_names(sentence):
        score += 5
    if field == "limitations" and any(k in lower for k in ["future", "limited", "however", "although"]):
        score += 3
    if field == "method" and any(k in lower for k in ["we propose", "this paper proposes", "we present"]):
        score += 4

    # Penalize citation fragments, headings, and broken extraction noise.
    if len(sentence.split()) < 10:
        score -= 3
    if re.search(r"\[[0-9,\s]+\]", sentence):
        score -= 1
    if sentence.count(",") > 8:
        score -= 1

    return score


def _best_sentence(text, field):
    ranked = sorted(
        ((s, _score_sentence(s, field)) for s in _sentences(text)),
        key=lambda item: item[1],
        reverse=True,
    )
    if ranked and ranked[0][1] > 0:
        return ranked[0][0]
    return "Not specified"


def _dataset_names(text):
    names = []
    for name in KNOWN_DATASETS:
        if re.search(rf"\b{re.escape(name)}\b", text, flags=re.IGNORECASE):
            names.append(name)

    patterns = [
        r"(?:evaluated|trained|tested|validated)\s+on\s+([A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+){0,3})",
        r"(?:using|with)\s+the\s+([A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+){0,3})\s+(?:dataset|benchmark|corpus)",
        r"([A-Z][A-Za-z0-9_\-]+(?:\s+[A-Z][A-Za-z0-9_\-]+){0,3})\s+(?:dataset|benchmark|corpus)",
    ]

    for pattern in patterns:
        for match in re.findall(pattern, text):
            candidate = match.strip(" ,.;:")
            if 2 <= len(candidate) <= 60 and candidate.lower() not in {"this", "our", "the"}:
                names.append(candidate)

    return list(dict.fromkeys(names))


def extract_dataset(text):
    names = _dataset_names(text)
    if names:
        return ", ".join(names[:4])

    sentence = _best_sentence(text, "dataset")
    return sentence if sentence != "Not specified" else "Not specified"


def extract_performance(text):
    sentences = _sentences(text)
    candidates = []
    for sentence in sentences:
        lower = sentence.lower()
        has_metric = any(k in lower for k in KEYWORD_GROUPS["performance"])
        has_number = bool(re.search(r"\b\d+(\.\d+)?\s?%|\b0\.\d+\b", lower))
        has_comparison = any(k in lower for k in ["outperform", "improve", "better", "state-of-the-art", "compared"])
        if has_metric and (has_number or has_comparison):
            candidates.append((sentence, _score_sentence(sentence, "performance")))

    if candidates:
        return sorted(candidates, key=lambda item: item[1], reverse=True)[0][0]
    return _best_sentence(text, "performance")



def parse_output(text):
    fields = dict(FIELD_DEFAULTS)
    current_key = None

    for raw_line in (text or "").splitlines():
        line = raw_line.strip().strip("-")
        if not line:
            continue

        match = re.match(r"^(problem|method|dataset|performance|application|limitations)\s*:\s*(.*)$", line, re.I)
        if match:
            current_key = match.group(1).lower()
            value = match.group(2).strip()
            if value:
                fields[current_key] = value
            continue

        if current_key and fields[current_key] == "Not specified":
            fields[current_key] = line

    return fields


def _heuristic_analysis(paper, text):
    abstract = paper.get("abstract", "")
    useful_text = clean_text(f"{abstract} {text}", limit=12000)

    return {
        "title": paper.get("title", ""),
        "problem": _best_sentence(useful_text, "problem"),
        "method": _best_sentence(useful_text, "method"),
        "dataset": extract_dataset(useful_text),
        "performance": extract_performance(useful_text),
        "application": _best_sentence(useful_text, "application"),
        "limitations": _best_sentence(useful_text, "limitations"),
    }


def _merge_with_heuristics(parsed, heuristics):
    merged = {}
    for key in FIELD_DEFAULTS:
        value = parsed.get(key, "Not specified")
        if not value or value.lower() in {"not specified", "none", "n/a"}:
            value = heuristics.get(key, "Not specified")
        merged[key] = value or "Not specified"
    return merged


def analyze_paper(paper):
    global _llm_failed

    title = paper.get("title", "")
    abstract = paper.get("abstract", "")
    pdf_text = extract_pdf_text(paper.get("pdf_url"), max_pages=8) if paper.get("pdf_url") else ""

    source_text = clean_text(
        f"Title: {title}\n\nAbstract: {abstract}\n\nFirst PDF pages: {pdf_text}",
        limit=9000,
    )
    heuristics = _heuristic_analysis(paper, source_text)

    if client is None or _llm_failed:
        return heuristics

    prompt = f"""
Extract high-quality structured research information from the paper text.

Return exactly this format:
Problem: <specific challenge or research gap, one sentence>
Method: <specific proposed method/framework/model, one sentence>
Dataset: <dataset/benchmark names or evaluation setting; Not specified if absent>
Performance: <metrics, numerical results, or comparison/improvement; Not specified if absent>
Application: <main task/domain/use case, one sentence>
Limitations: <stated limitation, constraint, weakness, or future work; Not specified if absent>

Rules:
- Prefer the abstract, then use the first PDF pages for missing context.
- Select complete, informative sentences; do not copy broken fragments.
- Use only the supplied text.
- Avoid generic phrases such as "the paper discusses".

Paper text:
{source_text}
"""

    try:
        response = client.chat.completions.create(
            model="llama-3.1-8b-instant",
            messages=[{"role": "user", "content": prompt}],
            temperature=0,
            max_tokens=650,
        )
        parsed = parse_output(response.choices[0].message.content)
        fields = _merge_with_heuristics(parsed, heuristics)
    except Exception:
        _llm_failed = True
        fields = heuristics

    return {
        "title": title,
        "problem": fields["problem"],
        "method": fields["method"],
        "dataset": fields["dataset"],
        "performance": fields["performance"],
        "application": fields["application"],
        "limitations": fields["limitations"],
    }


def analyze_multiple(papers):
    return [analyze_paper(paper) for paper in papers]
