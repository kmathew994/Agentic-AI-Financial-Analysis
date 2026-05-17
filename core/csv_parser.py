"""
Standalone CSV parser for Australian bank transaction files.
Supports: CBA, ANZ, Westpac, NAB and generic debit/credit split formats.
"""

from __future__ import annotations

import re
from io import StringIO
from typing import Any

import pandas as pd


# ── Column alias maps per bank ─────────────────────────────────────────────────

_COLUMN_ALIASES: dict[str, str] = {
    # Generic
    "transaction_date": "date",
    "trans_date": "date",
    "value_date": "date",
    "narration": "description",
    "details": "description",
    "memo": "description",
    "particulars": "description",
    "withdrawal": "debit",
    "withdrawal_amt": "debit",
    "deposit": "credit",
    "deposit_amt": "credit",
    "running_balance": "balance",
    "closing_balance": "balance",
    # CBA
    "transaction_amount": "amount",
    # ANZ
    "debit_amount": "debit",
    "credit_amount": "credit",
}


def _normalise_columns(df: pd.DataFrame) -> pd.DataFrame:
    df.columns = [c.strip().lower().replace(" ", "_").replace("-", "_") for c in df.columns]
    df.rename(columns=_COLUMN_ALIASES, inplace=True)
    return df


def _resolve_amount(df: pd.DataFrame) -> pd.Series:
    """Return a signed amount series: positive = credit, negative = debit."""
    if "amount" in df.columns:
        return pd.to_numeric(df["amount"], errors="coerce").fillna(0)
    if "debit" in df.columns and "credit" in df.columns:
        debit = pd.to_numeric(df["debit"], errors="coerce").fillna(0)
        credit = pd.to_numeric(df["credit"], errors="coerce").fillna(0)
        return credit - debit
    raise ValueError(
        "Cannot determine transaction amounts — expected 'amount' or 'debit'/'credit' columns."
    )


def parse_bank_csv(csv_content: str) -> tuple[list[dict[str, Any]], dict[str, Any]]:
    """
    Parse a bank CSV string into a list of normalised transaction dicts.

    Returns:
        transactions: list of dicts with keys:
            id, date, description, amount, balance, is_income
        metadata: dict with columns_detected, row_count, date_range
    """
    # Strip BOM and leading whitespace
    cleaned = csv_content.lstrip("\ufeff").strip()

    # Skip header comment lines (some banks prefix with # or *)
    lines = [l for l in cleaned.splitlines() if not l.startswith(("#", "*"))]
    cleaned = "\n".join(lines)

    try:
        df = pd.read_csv(StringIO(cleaned))
    except Exception as e:
        raise ValueError(f"pandas could not parse CSV: {e}") from e

    df = _normalise_columns(df)

    # Require at least date + description
    for required in ("date", "description"):
        if required not in df.columns:
            raise ValueError(f"Missing required column: '{required}' (after normalisation)")

    df["amount"] = _resolve_amount(df)
    df["balance"] = pd.to_numeric(df.get("balance", 0), errors="coerce").fillna(0)
    df["date"] = pd.to_datetime(df["date"], dayfirst=True, errors="coerce").dt.strftime("%Y-%m-%d")
    df = df.dropna(subset=["date"])
    df["description"] = df["description"].astype(str).str.strip()

    transactions: list[dict[str, Any]] = []
    for i, row in df.iterrows():
        amt = float(row["amount"])
        transactions.append({
            "id": int(i) + 1,
            "date": row["date"],
            "description": row["description"],
            "amount": round(amt, 2),
            "balance": round(float(row["balance"]), 2),
            "is_income": amt > 0,
        })

    dates = [t["date"] for t in transactions]
    metadata = {
        "columns_detected": list(df.columns.tolist()),
        "row_count": len(transactions),
        "date_range": {"start": min(dates) if dates else None, "end": max(dates) if dates else None},
    }
    return transactions, metadata
