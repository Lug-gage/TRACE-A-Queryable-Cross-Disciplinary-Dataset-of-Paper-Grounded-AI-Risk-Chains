import re
from collections import Counter
from typing import Any, Dict, Iterable


STOPWORDS = {
    "the",
    "and",
    "for",
    "that",
    "with",
    "this",
    "from",
    "are",
    "was",
    "were",
    "will",
    "can",
    "could",
    "may",
    "our",
    "their",
    "about",
    "into",
    "using",
    "used",
    "use",
    "its",
    "has",
    "have",
    "not",
    "but",
    "they",
    "such",
    "these",
    "those",
    "which",
    "when",
    "where",
    "more",
    "than",
    "also",
}


def _tokens(text: str) -> Counter:
    words = re.findall(r"[a-zA-Z][a-zA-Z0-9_-]+", text.lower())
    return Counter(w for w in words if w not in STOPWORDS and len(w) > 2)


def _flatten_strings(value: Any) -> Iterable[str]:
    if isinstance(value, str):
        yield value
    elif isinstance(value, dict):
        for item in value.values():
            yield from _flatten_strings(item)
    elif isinstance(value, list):
        for item in value:
            yield from _flatten_strings(item)


def evaluate_against_impact(schema: Dict[str, Any], impact: str) -> Dict[str, Any]:
    predicted_text = " ".join(_flatten_strings(schema.get("risk_schema", {}).get("impact", [])))
    predicted_text += " " + str(schema.get("final_risk_summary", ""))

    gold = _tokens(impact or "")
    pred = _tokens(predicted_text)
    overlap = set(gold) & set(pred)

    precision = len(overlap) / len(set(pred)) if pred else 0.0
    recall = len(overlap) / len(set(gold)) if gold else 0.0
    f1 = 2 * precision * recall / (precision + recall) if precision + recall else 0.0

    return {
        "impact_keyword_precision": round(precision, 4),
        "impact_keyword_recall": round(recall, 4),
        "impact_keyword_f1": round(f1, 4),
        "overlap_terms": sorted(overlap)[:50],
        "gold_impact_present": bool((impact or "").strip()),
    }

