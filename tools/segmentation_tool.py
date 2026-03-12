from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

# OTIF service level targets by ABC segment
OTIF_TARGETS: Dict[str, float] = {
    "A": 95.0,  # Top-revenue SKUs — tight service level
    "B": 92.0,  # Mid-revenue SKUs
    "C": 92.0,  # Tail SKUs
}

# Cumulative revenue cutoffs for ABC classification
ABC_CUTOFFS = {
    "A": 80.0,  # First 80% of revenue
    "B": 95.0,  # Next 15% (80-95%)
    # C: remaining 5%
}

_SERVICE_COST_NOTE = (
    "Service targets are tied to safety stock and forecast error. "
    "Pushing OTIF above target requires disproportionate inventory investment — "
    "higher service levels sit on the steep part of the service-cost curve."
)


def get_sku_segments(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Classify SKUs into ABC groups by cumulative revenue contribution.

    Group A: top 80% of revenue  → OTIF target 95%
    Group B: next 15% of revenue → OTIF target 92%
    Group C: bottom 5% of revenue→ OTIF target 92%
    """
    from tools.forecast_tool import load_forecast_data

    df = load_forecast_data().copy()

    if "actual_revenue" not in df.columns:
        return {"metric": "sku_segments", "summary": "Revenue column not available.", "records": []}

    if category:
        df = df[df["product_type"].str.lower() == category.lower()]

    if df.empty:
        return {"metric": "sku_segments", "category": category,
                "summary": "No data found.", "records": []}

    sku_rev = (
        df.groupby(["sku", "product_type"])
        .agg(
            total_revenue=("actual_revenue", "sum"),
            total_units=("actual_sales_qty", "sum"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    total_rev = sku_rev["total_revenue"].sum()
    sku_rev["revenue_pct"] = sku_rev["total_revenue"] / total_rev * 100
    sku_rev["cumulative_revenue_pct"] = sku_rev["revenue_pct"].cumsum()

    def _assign_group(cum_pct: float) -> str:
        if cum_pct <= ABC_CUTOFFS["A"]:
            return "A"
        elif cum_pct <= ABC_CUTOFFS["B"]:
            return "B"
        return "C"

    sku_rev["segment"] = sku_rev["cumulative_revenue_pct"].apply(_assign_group)
    sku_rev["otif_target"] = sku_rev["segment"].map(OTIF_TARGETS)

    counts = sku_rev["segment"].value_counts().to_dict()

    return {
        "metric": "sku_segments",
        "category": category,
        "total_skus": len(sku_rev),
        "segment_counts": counts,
        "otif_targets": OTIF_TARGETS,
        "summary": (
            f"ABC segmentation: {counts.get('A', 0)} Group A SKUs (OTIF target {OTIF_TARGETS['A']}%), "
            f"{counts.get('B', 0)} Group B and {counts.get('C', 0)} Group C SKUs "
            f"(OTIF target {OTIF_TARGETS['B']}%)."
        ),
        "records": sku_rev.round(2).to_dict(orient="records"),
    }


def assess_otif_vs_target(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Compare actual SKU-level OTIF against their ABC-segment target.

    Below target  → flag for root cause investigation
    Above target, non-A → flag for service level cost review
      (excess OTIF in tail SKUs drives unnecessary safety stock)
    """
    from tools.otif_tool import load_otif_data

    # Build segment lookup
    seg_result = get_sku_segments(category)
    if not seg_result.get("records"):
        return {
            "metric": "otif_vs_target",
            "summary": "Cannot assess — no segmentation data.",
            "below_target": [],
            "review_for_savings": [],
        }

    segment_lookup: Dict[str, Dict] = {
        row["sku"]: {"segment": row["segment"], "otif_target": row["otif_target"]}
        for row in seg_result["records"]
    }

    # Compute actual OTIF per SKU directly
    df = load_otif_data().copy()
    if category:
        df = df[df["product_type"].str.lower() == category.lower()]

    if df.empty or "sku" not in df.columns:
        return {
            "metric": "otif_vs_target",
            "summary": "No OTIF data to compare against targets.",
            "below_target": [],
            "review_for_savings": [],
        }

    sku_otif = (
        df.groupby(["sku", "product_type"])["otif_flag"]
        .agg(actual_otif=lambda x: round(x.mean() * 100, 2), order_count="count")
        .reset_index()
    )

    below_target = []
    review_for_savings = []

    for _, row in sku_otif.iterrows():
        sku = row["sku"]
        actual = float(row["actual_otif"])
        seg_info = segment_lookup.get(sku, {"segment": "C", "otif_target": OTIF_TARGETS["C"]})
        segment = seg_info["segment"]
        target = float(seg_info["otif_target"])
        gap = round(actual - target, 2)

        entry = {
            "sku": sku,
            "product_type": row["product_type"],
            "segment": segment,
            "actual_otif": actual,
            "target_otif": target,
            "gap": gap,
            "order_count": int(row["order_count"]),
        }

        if actual < target:
            below_target.append(entry)
        elif actual > target and segment != "A":
            review_for_savings.append({
                **entry,
                "note": (
                    f"OTIF is {gap:+.1f}pts above the {target:.0f}% target for a Group {segment} SKU. "
                    "Excess service level likely drives unnecessary safety stock. "
                    "Review if planned service level can be reduced to free working capital."
                ),
            })

    # Sort: below target by largest gap first; savings by largest excess first
    below_target.sort(key=lambda x: x["gap"])
    review_for_savings.sort(key=lambda x: x["gap"], reverse=True)

    parts = []
    if below_target:
        parts.append(f"{len(below_target)} SKUs are below their OTIF target — root cause needed")
    if review_for_savings:
        parts.append(
            f"{len(review_for_savings)} non-Group-A SKUs are over-serving — "
            "review for inventory cost savings"
        )

    return {
        "metric": "otif_vs_target",
        "category": category,
        "below_target_count": len(below_target),
        "savings_review_count": len(review_for_savings),
        "summary": "; ".join(parts) if parts else "All SKUs are meeting their OTIF targets.",
        "service_cost_note": _SERVICE_COST_NOTE,
        "below_target": below_target[:10],
        "review_for_savings": review_for_savings[:10],
    }


if __name__ == "__main__":
    print(get_sku_segments(category="haircare"))
    print(assess_otif_vs_target(category="haircare"))
