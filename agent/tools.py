import json
import re
from typing import Any

import numpy as np
import pandas as pd
from langchain_core.tools import tool
from scipy import stats


# ── Helpers ───────────────────────────────────────────────────────────────────

CATEGORY_PATTERNS: dict[str, list[str]] = {
    "groceries": [
        r"woolworths", r"coles", r"\biga\b", r"aldi", r"harris farm",
        r"foodworks", r"spar", r"costco",
    ],
    "dining": [
        r"uber eats", r"menulog", r"doordash", r"deliveroo",
        r"mcdonald", r"kfc\b", r"domino", r"guzman", r"subway",
        r"thai", r"sushi", r"cafe", r"restaurant", r"dining", r"bar\b",
        r"pizza", r"noodle", r"bistro",
    ],
    "transport": [
        r"opal card", r"uber trip", r"\bbp\b", r"shell", r"caltex",
        r"ampol", r"7-eleven fuel", r"metro trains", r"bus fare",
        r"parking", r"toll", r"e-toll",
    ],
    "subscriptions": [
        r"netflix", r"spotify", r"disney\+", r"apple.*subscription",
        r"google one", r"microsoft 365", r"adobe", r"amazon prime",
        r"binge\b", r"stan\b", r"paramount",
    ],
    "utilities": [
        r"agl\b", r"origin energy", r"sydney water", r"city water",
        r"optus", r"telstra", r"vodafone", r"tpg\b", r"aussie broadband",
        r"electricity", r"gas bill", r"rates\b",
    ],
    "healthcare": [
        r"priceline", r"chemist warehouse", r"terry white", r"pharmacy",
        r"bupa\b", r"medibank", r"hcf\b", r"nib\b",
        r"doctor", r"medical centre", r"pathology", r"dental",
    ],
    "entertainment": [
        r"hoyts", r"event cinemas", r"village cinema",
        r"gym", r"fitness", r"goodlife", r"anytime fitness",
        r"ticketek", r"ticketmaster", r"zoo\b", r"aquarium",
    ],
    "shopping": [
        r"amazon\b", r"jbhifi", r"harvey norman", r"officeworks",
        r"david jones", r"myer\b", r"kmart", r"target\b",
        r"bunnings", r"ikea", r"uniqlo", r"zara\b",
        r"ebay\b", r"catch\.com", r"the good guys",
        r"dan murphy", r"bws\b", r"liquorland",
    ],
    "income": [
        r"salary", r"payroll", r"wages", r"pay from",
        r"medicare.*refund", r"ato refund", r"tax refund", r"centrelink",
        r"dividend", r"interest earned",
    ],
    "transfers": [
        r"transfer", r"savings", r"bpay", r"pay anyone",
        r"osko", r"payid", r"direct debit",
    ],
}


def _rule_categorise(description: str) -> dict[str, Any]:
    desc_lower = description.lower()
    for category, patterns in CATEGORY_PATTERNS.items():
        for pattern in patterns:
            if re.search(pattern, desc_lower):
                return {"category": category, "subcategory": None, "confidence": 0.90}
    return {"category": "other", "subcategory": None, "confidence": 0.45}


# ── Tools ──────────────────────────────────────────────────────────────────────

@tool
def parse_csv_transactions(csv_content: str) -> dict:
    """
    Parse raw CSV bank transaction data into structured format.
    Handles various Australian bank CSV formats (CBA, ANZ, Westpac, NAB).
    Returns: {transactions: list, columns_detected: list, row_count: int, date_range: dict}
    """
    try:
        from io import StringIO
        df = pd.read_csv(StringIO(csv_content))

        # Normalise column names
        df.columns = [c.strip().lower().replace(" ", "_") for c in df.columns]

        # Map known column aliases
        col_map = {
            "transaction_date": "date", "trans_date": "date",
            "narration": "description", "details": "description",
            "withdrawal": "debit", "deposit": "credit",
            "running_balance": "balance",
        }
        df.rename(columns=col_map, inplace=True)

        # Ensure required columns exist
        for col in ["date", "description"]:
            if col not in df.columns:
                return {"error": f"Missing required column: {col}", "transactions": []}

        # Compute amount: positive = credit/income, negative = debit/expense
        if "debit" in df.columns and "credit" in df.columns:
            df["amount"] = df["credit"].fillna(0).astype(float) - df["debit"].fillna(0).astype(float)
        elif "amount" in df.columns:
            df["amount"] = pd.to_numeric(df["amount"], errors="coerce").fillna(0)
        else:
            return {"error": "Cannot determine transaction amounts from columns", "transactions": []}

        # Normalise dates
        df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
        df = df.dropna(subset=["date"])

        transactions = []
        for i, row in df.iterrows():
            transactions.append({
                "id": i + 1,
                "date": row["date"],
                "description": str(row.get("description", "")).strip(),
                "amount": round(float(row["amount"]), 2),
                "balance": round(float(row.get("balance", 0) or 0), 2),
                "is_income": float(row["amount"]) > 0,
            })

        dates = [t["date"] for t in transactions]
        return {
            "transactions": transactions,
            "columns_detected": list(df.columns),
            "row_count": len(transactions),
            "date_range": {"start": min(dates), "end": max(dates)},
        }
    except Exception as e:
        return {"error": str(e), "transactions": []}


@tool
def categorise_transaction(description: str, amount: float) -> dict:
    """
    Categorise a single transaction using rule-based pattern matching.
    Categories: groceries, dining, transport, utilities, entertainment,
                healthcare, shopping, subscriptions, income, transfers, other
    Returns: {category: str, subcategory: str | None, confidence: float, reasoning: str}
    """
    result = _rule_categorise(description)
    reasoning = (
        f"Matched '{description}' to '{result['category']}' "
        f"via pattern rules (confidence={result['confidence']})"
    ) if result["confidence"] >= 0.7 else (
        f"No strong pattern match for '{description}' — defaulting to 'other'"
    )
    return {**result, "reasoning": reasoning}


@tool
def detect_spending_anomalies(transactions_json: str) -> dict:
    """
    Run statistical anomaly detection on transaction data.
    Uses Z-score (threshold: 2.0 std) + IQR method (1.5× fence) per overall expenses.
    Returns: {anomalies: list, method_used: str, threshold: float, stats: dict}
    """
    transactions = json.loads(transactions_json)
    expenses = [abs(t["amount"]) for t in transactions if t["amount"] < 0]

    if len(expenses) < 5:
        return {"anomalies": [], "method_used": "insufficient_data", "threshold": 0, "stats": {}}

    arr = np.array(expenses)
    mean_val = float(np.mean(arr))
    std_val = float(np.std(arr))
    q1 = float(np.percentile(arr, 25))
    q3 = float(np.percentile(arr, 75))
    iqr = q3 - q1
    upper_fence = q3 + 1.5 * iqr

    anomalies = []
    for t in transactions:
        if t["amount"] >= 0:
            continue
        abs_amt = abs(t["amount"])
        z = (abs_amt - mean_val) / std_val if std_val > 0 else 0
        is_anomaly = z > 2.0 or abs_amt > upper_fence

        if is_anomaly:
            severity = "HIGH" if abs_amt > 1000 else "MEDIUM" if abs_amt > 300 else "LOW"
            anomalies.append({
                **t,
                "z_score": round(z, 2),
                "severity": severity,
                "flags": (["z_score_high"] if z > 2.0 else []) + (["iqr_outlier"] if abs_amt > upper_fence else []),
            })

    return {
        "anomalies": sorted(anomalies, key=lambda x: abs(x["amount"]), reverse=True),
        "method_used": "z_score_and_iqr",
        "threshold": round(upper_fence, 2),
        "stats": {
            "mean": round(mean_val, 2),
            "std": round(std_val, 2),
            "q1": round(q1, 2),
            "q3": round(q3, 2),
            "iqr": round(iqr, 2),
            "upper_fence": round(upper_fence, 2),
        },
    }


@tool
def calculate_spending_summary(transactions_json: str) -> dict:
    """
    Compute aggregate spending statistics by category and time period.
    Returns: {by_category, by_week, by_merchant, total_income, total_expenses, savings_rate}
    """
    transactions = json.loads(transactions_json)
    df = pd.DataFrame(transactions)
    df["date"] = pd.to_datetime(df["date"])
    df["week"] = df["date"].dt.strftime("%Y-W%V")

    by_category: dict[str, float] = {}
    by_week: dict[str, float] = {}
    by_merchant: dict[str, float] = {}
    total_income = 0.0
    total_expenses = 0.0

    for _, row in df.iterrows():
        amt = float(row["amount"])
        if amt > 0:
            total_income += amt
        else:
            total_expenses += abs(amt)
            cat = str(row.get("category", "other"))
            by_category[cat] = by_category.get(cat, 0) + abs(amt)
            week = str(row["week"])
            by_week[week] = by_week.get(week, 0) + abs(amt)
            merchant = str(row["description"])
            by_merchant[merchant] = by_merchant.get(merchant, 0) + abs(amt)

    savings_rate = ((total_income - total_expenses) / total_income * 100) if total_income > 0 else 0

    top_merchants = dict(
        sorted(by_merchant.items(), key=lambda x: x[1], reverse=True)[:10]
    )

    return {
        "by_category": {k: round(v, 2) for k, v in sorted(by_category.items(), key=lambda x: x[1], reverse=True)},
        "by_week": {k: round(v, 2) for k, v in sorted(by_week.items())},
        "by_merchant": {k: round(v, 2) for k, v in top_merchants.items()},
        "total_income": round(total_income, 2),
        "total_expenses": round(total_expenses, 2),
        "savings_rate": round(savings_rate, 1),
    }


@tool
def generate_budget_recommendations(summary_json: str) -> dict:
    """
    Analyse spending patterns and generate rule-based budget recommendations.
    Returns: {recommendations: list, priority: str, potential_savings: float}
    """
    summary = json.loads(summary_json)
    by_cat = summary.get("by_category", {})
    total_expenses = summary.get("total_expenses", 1)
    savings_rate = summary.get("savings_rate", 0)

    recommendations = []
    potential_savings = 0.0

    dining = by_cat.get("dining", 0)
    if dining > 300:
        saving = round(dining * 0.3, 2)
        recommendations.append({
            "category": "dining",
            "insight": f"Dining/takeaway spend is ${dining:.2f}. Cooking at home 3× more per week could save ~${saving:.2f}.",
            "potential_saving": saving,
        })
        potential_savings += saving

    subs = by_cat.get("subscriptions", 0)
    if subs > 50:
        saving = round(subs * 0.4, 2)
        recommendations.append({
            "category": "subscriptions",
            "insight": f"You have ${subs:.2f} in subscriptions. Auditing and cancelling unused ones could save ~${saving:.2f}.",
            "potential_saving": saving,
        })
        potential_savings += saving

    shopping = by_cat.get("shopping", 0)
    if shopping > 200:
        saving = round(shopping * 0.25, 2)
        recommendations.append({
            "category": "shopping",
            "insight": f"Discretionary shopping is ${shopping:.2f}. A 24-hour rule before purchases could reduce this by ~${saving:.2f}.",
            "potential_saving": saving,
        })
        potential_savings += saving

    if savings_rate < 20:
        recommendations.append({
            "category": "savings",
            "insight": f"Your savings rate is {savings_rate}%. Automating a transfer of 20% of income on payday is recommended.",
            "potential_saving": 0,
        })

    priority = "HIGH" if savings_rate < 10 else "MEDIUM" if savings_rate < 20 else "LOW"

    return {
        "recommendations": recommendations,
        "priority": priority,
        "potential_savings": round(potential_savings, 2),
    }
