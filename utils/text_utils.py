from __future__ import annotations


def normalize_text(text: str) -> str:
    return " ".join(text.lower().strip().split())


def dedupe_keep_order(items: list[str]) -> list[str]:
    seen = set()
    deduped = []
    for item in items:
        if item not in seen:
            deduped.append(item)
            seen.add(item)
    return deduped
