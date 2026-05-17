"""
Statistical anomaly detection for bank transactions.
Methods: Z-score (global) + IQR (per-category).
"""

from __future__ import annotations

from typing import Any

import numpy as np
import pandas as pd


def detect_anomalies(
    transactions: list[dict[str, Any]],
    z_threshold: float = 2.0,
    iqr_multiplier: float = 1.5,
) -> dict[str, Any]:
    """
    Detect anomalous transactions using two complementary methods:

    1. Global Z-score: flags transactions whose absolute amount deviates
       more than `z_threshold` standard deviations from the mean.
    2. Per-category IQR: flags transactions that exceed the category's
       Q3 + `iqr_multiplier` × IQR upper fence.

    Returns a dict with:
        anomalies: list of flagged transaction dicts (enriched with z_score, severity, flags)
        stats: global descriptive statistics
        category_stats: per-category IQR fences
        method_used: str
    """
    expenses = [t for t in transactions if t["amount"] < 0]

    if len(expenses) < 5:
        return {
            "anomalies": [],
            "stats": {},
            "category_stats": {},
            "method_used": "insufficient_data",
        }

    amounts = np.array([abs(t["amount"]) for t in expenses])

    # ── Global stats ──────────────────────────────────────────────────────────
    mean_val = float(np.mean(amounts))
    std_val = float(np.std(amounts))
    q1_global = float(np.percentile(amounts, 25))
    q3_global = float(np.percentile(amounts, 75))
    iqr_global = q3_global - q1_global
    global_fence = q3_global + iqr_multiplier * iqr_global

    global_stats = {
        "mean": round(mean_val, 2),
        "std": round(std_val, 2),
        "q1": round(q1_global, 2),
        "q3": round(q3_global, 2),
        "iqr": round(iqr_global, 2),
        "upper_fence": round(global_fence, 2),
        "n": len(expenses),
    }

    # ── Per-category IQR fences ───────────────────────────────────────────────
    df = pd.DataFrame(expenses)
    cat_stats: dict[str, dict] = {}
    cat_fences: dict[str, float] = {}

    for cat, grp in df.groupby("category"):
        cat_amounts = grp["amount"].abs().values
        if len(cat_amounts) < 3:
            continue
        cq1 = float(np.percentile(cat_amounts, 25))
        cq3 = float(np.percentile(cat_amounts, 75))
        ciqr = cq3 - cq1
        fence = cq3 + iqr_multiplier * ciqr
        cat_fences[str(cat)] = fence
        cat_stats[str(cat)] = {
            "q1": round(cq1, 2),
            "q3": round(cq3, 2),
            "iqr": round(ciqr, 2),
            "upper_fence": round(fence, 2),
            "n": len(cat_amounts),
        }

    # ── Flag anomalies ────────────────────────────────────────────────────────
    anomalies = []
    for t in expenses:
        abs_amt = abs(t["amount"])
        z = (abs_amt - mean_val) / std_val if std_val > 0 else 0.0
        cat = str(t.get("category", "other"))
        cat_fence = cat_fences.get(cat, global_fence)

        flags = []
        if z > z_threshold:
            flags.append("global_z_score")
        if abs_amt > cat_fence:
            flags.append("category_iqr")
        if abs_amt > global_fence:
            flags.append("global_iqr")

        if not flags:
            continue

        severity = "HIGH" if abs_amt > 1000 else "MEDIUM" if abs_amt > 300 else "LOW"

        anomalies.append({
            **t,
            "z_score": round(z, 2),
            "severity": severity,
            "flags": flags,
        })

    # Sort: HIGH first, then by amount descending
    severity_order = {"HIGH": 0, "MEDIUM": 1, "LOW": 2}
    anomalies.sort(key=lambda x: (severity_order[x["severity"]], -abs(x["amount"])))

    return {
        "anomalies": anomalies,
        "stats": global_stats,
        "category_stats": cat_stats,
        "method_used": "z_score_and_iqr",
        "z_threshold": z_threshold,
        "iqr_multiplier": iqr_multiplier,
    }
