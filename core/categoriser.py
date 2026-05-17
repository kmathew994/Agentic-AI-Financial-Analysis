"""
Hybrid transaction categoriser.
Phase 1: Fast regex rule-matching (no API call).
Phase 2: LLM fallback for low-confidence entries (confidence < CONFIDENCE_THRESHOLD).
"""

from __future__ import annotations

import json
import re
from typing import Any

CONFIDENCE_THRESHOLD = 0.70

CATEGORIES = [
    "groceries", "dining", "transport", "utilities", "entertainment",
    "healthcare", "shopping", "subscriptions", "income", "transfers", "other",
]

# Pattern lists: each entry is a (regex, subcategory | None) tuple
_RULES: dict[str, list[tuple[str, str | None]]] = {
    "groceries": [
        (r"woolworths", "supermarket"),
        (r"coles\b", "supermarket"),
        (r"\biga\b", "supermarket"),
        (r"aldi\b", "supermarket"),
        (r"harris farm", "fresh produce"),
        (r"foodworks", "supermarket"),
        (r"costco", "wholesale"),
    ],
    "dining": [
        (r"uber eats", "delivery"),
        (r"menulog", "delivery"),
        (r"doordash", "delivery"),
        (r"deliveroo", "delivery"),
        (r"mcdonald", "fast food"),
        (r"\bkfc\b", "fast food"),
        (r"domino", "fast food"),
        (r"guzman", "fast food"),
        (r"subway\b", "fast food"),
        (r"cafe\b", "cafe"),
        (r"coffee", "cafe"),
        (r"restaurant", "restaurant"),
        (r"dining", "restaurant"),
        (r"thai\b", "restaurant"),
        (r"sushi", "restaurant"),
        (r"pizza", "restaurant"),
        (r"bistro", "restaurant"),
        (r"bar\b", "bar"),
    ],
    "transport": [
        (r"opal card", "public transport"),
        (r"uber trip", "rideshare"),
        (r"\bbp\b", "fuel"),
        (r"shell\b", "fuel"),
        (r"caltex", "fuel"),
        (r"ampol", "fuel"),
        (r"7-eleven fuel", "fuel"),
        (r"parking", "parking"),
        (r"e-toll", "toll"),
        (r"\btoll\b", "toll"),
    ],
    "subscriptions": [
        (r"netflix", "streaming"),
        (r"spotify", "music"),
        (r"disney\+", "streaming"),
        (r"binge\b", "streaming"),
        (r"stan\b", "streaming"),
        (r"paramount", "streaming"),
        (r"apple.*subscri", "software"),
        (r"google one", "cloud storage"),
        (r"microsoft 365", "software"),
        (r"adobe\b", "software"),
        (r"amazon prime", "shopping/streaming"),
    ],
    "utilities": [
        (r"\bagl\b", "electricity/gas"),
        (r"origin energy", "electricity/gas"),
        (r"sydney water", "water"),
        (r"optus\b", "phone/internet"),
        (r"telstra\b", "phone/internet"),
        (r"vodafone", "phone/internet"),
        (r"\btpg\b", "phone/internet"),
        (r"aussie broadband", "internet"),
        (r"electricity", "electricity"),
        (r"rates\b", "council rates"),
    ],
    "healthcare": [
        (r"priceline", "pharmacy"),
        (r"chemist warehouse", "pharmacy"),
        (r"terry white", "pharmacy"),
        (r"pharmacy", "pharmacy"),
        (r"\bbupa\b", "health insurance"),
        (r"medibank", "health insurance"),
        (r"\bhcf\b", "health insurance"),
        (r"\bnib\b", "health insurance"),
        (r"medical centre", "gp"),
        (r"dental", "dental"),
        (r"pathology", "diagnostics"),
    ],
    "entertainment": [
        (r"hoyts", "cinema"),
        (r"event cinemas", "cinema"),
        (r"village cinema", "cinema"),
        (r"\bgym\b", "gym"),
        (r"fitness", "gym"),
        (r"goodlife", "gym"),
        (r"anytime fitness", "gym"),
        (r"ticketek", "events"),
        (r"ticketmaster", "events"),
    ],
    "shopping": [
        (r"\bamazon\b", "online retail"),
        (r"jbhifi", "electronics"),
        (r"harvey norman", "electronics"),
        (r"officeworks", "stationery"),
        (r"david jones", "department store"),
        (r"\bmyer\b", "department store"),
        (r"\bkmart\b", "discount retail"),
        (r"\btarget\b", "discount retail"),
        (r"bunnings", "hardware"),
        (r"\bikea\b", "homewares"),
        (r"\bebay\b", "online retail"),
        (r"dan murphy", "liquor"),
        (r"\bbws\b", "liquor"),
        (r"liquorland", "liquor"),
        (r"\buniqlo\b", "clothing"),
        (r"\bzara\b", "clothing"),
    ],
    "income": [
        (r"salary", None),
        (r"payroll", None),
        (r"\bwages\b", None),
        (r"pay from", None),
        (r"medicare.*refund", "government"),
        (r"ato refund", "government"),
        (r"tax refund", "government"),
        (r"centrelink", "government"),
        (r"dividend", "investment"),
        (r"interest earned", "investment"),
    ],
    "transfers": [
        (r"\btransfer\b", None),
        (r"savings", None),
        (r"\bbpay\b", "bill payment"),
        (r"pay anyone", None),
        (r"\bosko\b", None),
        (r"\bpayid\b", None),
        (r"direct debit", "direct debit"),
    ],
}


def rule_categorise(description: str, amount: float) -> dict[str, Any]:
    """
    Attempt rule-based categorisation.
    Returns category, subcategory, confidence, and reasoning.
    """
    desc_lower = description.lower()
    for category, patterns in _RULES.items():
        for pattern, subcategory in patterns:
            if re.search(pattern, desc_lower):
                return {
                    "category": category,
                    "subcategory": subcategory,
                    "confidence": 0.90,
                    "reasoning": f"Matched pattern /{pattern}/ → {category}",
                }
    return {
        "category": "other",
        "subcategory": None,
        "confidence": 0.40,
        "reasoning": f"No pattern matched '{description}'",
    }


def llm_categorise(description: str, amount: float, llm: Any) -> dict[str, Any]:
    """
    Use the Claude LLM to categorise a transaction.
    Falls back to 'other' if parsing fails.
    """
    from langchain_core.messages import HumanMessage, SystemMessage

    system = SystemMessage(content=(
        "You are a financial transaction categoriser for Australian bank data. "
        "Respond ONLY with valid JSON — no preamble, no markdown fences.\n"
        f"Valid categories: {', '.join(CATEGORIES)}"
    ))
    prompt = (
        f"Categorise this bank transaction:\n"
        f"Description: {description}\n"
        f"Amount: {'credit' if amount > 0 else 'debit'} ${abs(amount):.2f}\n\n"
        'Return JSON: {"category": "...", "subcategory": "..." or null, "confidence": 0.0-1.0, "reasoning": "..."}'
    )
    try:
        resp = llm.invoke([system, HumanMessage(content=prompt)])
        raw = resp.content.strip().lstrip("```json").rstrip("```").strip()
        result = json.loads(raw)
        if result.get("category") not in CATEGORIES:
            result["category"] = "other"
        return result
    except Exception as e:
        return {
            "category": "other",
            "subcategory": None,
            "confidence": 0.30,
            "reasoning": f"LLM fallback failed: {e}",
        }


def categorise_transactions(
    transactions: list[dict[str, Any]],
    llm: Any | None = None,
) -> list[dict[str, Any]]:
    """
    Categorise a full list of transactions.
    Uses rule_categorise first; if confidence < threshold and llm provided, uses llm_categorise.
    """
    results = []
    for t in transactions:
        result = rule_categorise(t["description"], t["amount"])
        if result["confidence"] < CONFIDENCE_THRESHOLD and llm is not None:
            result = llm_categorise(t["description"], t["amount"], llm)
        results.append({**t, **result})
    return results
