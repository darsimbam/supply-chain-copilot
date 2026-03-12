from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
INVENTORY_FILE = DATA_DIR / "inventory.csv"

_inventory_cache: Optional[pd.DataFrame] = None


def load_inventory_data() -> pd.DataFrame:
    """Load inventory data from CSV (cached after first load)."""
    global _inventory_cache
    if _inventory_cache is not None:
        return _inventory_cache

    if not INVENTORY_FILE.exists():
        raise FileNotFoundError(f"Inventory file not found: {INVENTORY_FILE}")

    df = pd.read_csv(INVENTORY_FILE)

    required_cols = {"snapshot_date", "product_type", "sku", "stock_on_hand_units", "avg_weekly_sales_units"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required inventory columns: {missing}")

    _inventory_cache = df
    return _inventory_cache


def _apply_filters(
    df: pd.DataFrame,
    category: Optional[str],
    week: Optional[str],
    sku: Optional[str],
) -> pd.DataFrame:
    if category:
        df = df[df["product_type"].str.lower() == category.lower()]
    if week:
        df = df[df["snapshot_date"].astype(str) == str(week)]
    if sku:
        df = df[df["sku"].str.upper() == sku.upper()]
    return df


def _compute_weeks_of_cover(df: pd.DataFrame) -> pd.DataFrame:
    """Add weeks_of_cover column, preferring pre-computed days_of_cover when available."""
    df = df.copy()
    if "days_of_cover" in df.columns:
        df["weeks_of_cover"] = df["days_of_cover"] / 7
    else:
        df["weeks_of_cover"] = df.apply(
            lambda row: row["stock_on_hand_units"] / row["avg_weekly_sales_units"]
            if row["avg_weekly_sales_units"] > 0 else 999,
            axis=1,
        )
    return df


def get_inventory_risk(
    category: Optional[str] = None,
    week: Optional[str] = None,
    sku: Optional[str] = None,
    low_cover_threshold: float = 2.0,
) -> Dict[str, Any]:
    """Identify SKUs with low stock cover (in weeks)."""
    df = _apply_filters(load_inventory_data(), category, week, sku)

    if df.empty:
        return {"metric": "inventory_risk", "summary": "No inventory data found.", "records": []}

    df = _compute_weeks_of_cover(df)
    risky = df[df["weeks_of_cover"] < low_cover_threshold].copy().sort_values("weeks_of_cover")

    return {
        "metric": "inventory_risk",
        "category": category,
        "week": week,
        "sku": sku,
        "risk_count": int(len(risky)),
        "summary": f"{len(risky)} SKUs are below {low_cover_threshold:.1f} weeks of cover.",
        "risky_skus": risky[
            ["sku", "product_type", "stock_on_hand_units", "avg_weekly_sales_units", "weeks_of_cover"]
        ].round(2).to_dict(orient="records"),
    }


def get_stock_summary(
    category: Optional[str] = None,
    week: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Return overall inventory summary with status breakdown."""
    df = _apply_filters(load_inventory_data(), category, week, sku)

    if df.empty:
        return {"metric": "stock_summary", "summary": "No inventory data found."}

    df = _compute_weeks_of_cover(df)
    total_stock = float(df["stock_on_hand_units"].sum())
    avg_demand = float(df["avg_weekly_sales_units"].sum())
    overall_weeks_cover = total_stock / avg_demand if avg_demand > 0 else 0.0

    result: Dict[str, Any] = {
        "metric": "stock_summary",
        "category": category,
        "week": week,
        "sku": sku,
        "total_stock": round(total_stock, 2),
        "total_avg_weekly_demand": round(avg_demand, 2),
        "overall_weeks_cover": round(overall_weeks_cover, 2),
        "summary": f"Total stock is {total_stock:.0f} units with {overall_weeks_cover:.2f} weeks of cover.",
    }

    if "inventory_status" in df.columns:
        result["status_breakdown"] = df["inventory_status"].value_counts().to_dict()

    return result


def get_healthy_skus(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    detail: str = "inventory",
    min_cover_threshold: float = 2.0,
) -> Dict[str, Any]:
    """
    Return SKUs with sufficient stock cover.

    detail options:
      "count"     - counts only, no per-SKU rows
      "inventory" - stock_on_hand, weeks_of_cover, inventory_status (default)
      "sales"     - avg_weekly_sales, reorder_point, safety_stock
      "orders"    - same as inventory; caller can cross-ref PO data externally
      "full"      - all available columns
    """
    df = _apply_filters(load_inventory_data(), category, None, sku)

    if df.empty:
        return {
            "metric": "healthy_skus",
            "category": category,
            "sku": sku,
            "healthy_count": 0,
            "summary": "No inventory data found.",
            "records": [],
        }

    df = _compute_weeks_of_cover(df)

    if "inventory_status" in df.columns:
        healthy = df[
            (df["weeks_of_cover"] >= min_cover_threshold) |
            (df["inventory_status"].str.lower().isin(["ok", "overstock"]))
        ].copy()
    else:
        healthy = df[df["weeks_of_cover"] >= min_cover_threshold].copy()

    healthy = healthy.sort_values("weeks_of_cover", ascending=False)

    if detail == "count":
        breakdown = (
            healthy["inventory_status"].value_counts().to_dict()
            if "inventory_status" in healthy.columns else {}
        )
        return {
            "metric": "healthy_skus",
            "category": category,
            "sku": sku,
            "healthy_count": len(healthy),
            "status_breakdown": breakdown,
            "summary": f"{len(healthy)} SKUs have sufficient stock (>= {min_cover_threshold:.1f} weeks of cover).",
        }

    base_cols = ["sku", "product_type"]

    if detail == "sales":
        extra = ["avg_weekly_sales_units", "reorder_point_units", "safety_stock_units", "weeks_of_cover"]
    elif detail == "full":
        extra = [c for c in healthy.columns if c not in base_cols]
    else:  # "inventory" or "orders"
        extra = ["stock_on_hand_units", "avg_weekly_sales_units", "weeks_of_cover"]
        if "inventory_status" in healthy.columns:
            extra.append("inventory_status")
        if "location" in healthy.columns:
            extra.append("location")

    cols = [c for c in (base_cols + extra) if c in healthy.columns]
    records = healthy[cols].round(2).to_dict(orient="records")

    top_sku = healthy.iloc[0]["sku"] if not healthy.empty else "N/A"
    top_cover = round(float(healthy.iloc[0]["weeks_of_cover"]), 1) if not healthy.empty else 0.0

    return {
        "metric": "healthy_skus",
        "category": category,
        "sku": sku,
        "detail": detail,
        "healthy_count": len(healthy),
        "summary": (
            f"{len(healthy)} SKUs have sufficient stock. "
            f"Best covered: {top_sku} with {top_cover:.1f} weeks of cover."
        ),
        "records": records,
    }


def get_inventory_turns(
    category: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inventory turns = (avg_weekly_sales * 52) / stock_on_hand.
    Higher = leaner; < 4 turns/year is typically a concern.
    """
    df = _apply_filters(load_inventory_data(), category, None, sku)

    if df.empty:
        return {"metric": "inventory_turns", "summary": "No inventory data found."}

    df = df.copy()
    df["annual_sales_units"] = df["avg_weekly_sales_units"] * 52
    df["inventory_turns"] = df.apply(
        lambda r: round(r["annual_sales_units"] / r["stock_on_hand_units"], 2)
        if r["stock_on_hand_units"] > 0 else 0.0,
        axis=1,
    )

    avg_turns = round(float(df["inventory_turns"].mean()), 2)
    low_turns = df[df["inventory_turns"] < 4].copy()

    cols = ["sku", "product_type", "stock_on_hand_units", "avg_weekly_sales_units", "inventory_turns"]
    cols = [c for c in cols if c in df.columns]

    return {
        "metric": "inventory_turns",
        "category": category,
        "sku": sku,
        "avg_inventory_turns": avg_turns,
        "low_turns_count": len(low_turns),
        "summary": (
            f"Average inventory turns: {avg_turns:.1f}x/year. "
            f"{len(low_turns)} SKUs below 4 turns/year (slow-moving)."
        ),
        "slowest_skus": low_turns[cols].sort_values("inventory_turns").head(10).round(2).to_dict(orient="records"),
    }


def get_days_of_supply(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    critical_threshold: int = 14,
) -> Dict[str, Any]:
    """
    Days of supply per SKU. Uses pre-computed days_of_cover when available.
    < 14 days = critical; > 90 days = excess.
    """
    df = _apply_filters(load_inventory_data(), category, None, sku)

    if df.empty:
        return {"metric": "days_of_supply", "summary": "No inventory data found."}

    if "days_of_cover" not in df.columns:
        df = df.copy()
        df["days_of_cover"] = df.apply(
            lambda r: round(r["stock_on_hand_units"] / (r["avg_weekly_sales_units"] / 7), 1)
            if r["avg_weekly_sales_units"] > 0 else 999,
            axis=1,
        )

    avg_dos = round(float(df["days_of_cover"].mean()), 1)
    critical = df[df["days_of_cover"] < critical_threshold]
    excess = df[df["days_of_cover"] > 90]

    cols = ["sku", "product_type", "stock_on_hand_units", "days_of_cover"]
    if "inventory_status" in df.columns:
        cols.append("inventory_status")
    cols = [c for c in cols if c in df.columns]

    return {
        "metric": "days_of_supply",
        "category": category,
        "sku": sku,
        "avg_days_of_supply": avg_dos,
        "critical_count": len(critical),
        "excess_count": len(excess),
        "summary": (
            f"Average days of supply: {avg_dos:.0f} days. "
            f"{len(critical)} SKUs below {critical_threshold} days (critical). "
            f"{len(excess)} SKUs above 90 days (excess)."
        ),
        "critical_skus": critical[cols].sort_values("days_of_cover").head(10).round(1).to_dict(orient="records"),
    }


def get_inventory_accuracy(
    category: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Inventory accuracy proxy via availability_pct.
    < 90% availability indicates potential system vs. physical count discrepancy.
    """
    df = _apply_filters(load_inventory_data(), category, None, sku)

    if df.empty or "availability_pct" not in df.columns:
        return {"metric": "inventory_accuracy", "summary": "No availability data found."}

    avg_acc = round(float(df["availability_pct"].mean()), 2)
    low_acc = df[df["availability_pct"] < 90].copy()

    cols = ["sku", "product_type", "availability_pct"]
    if "inventory_status" in df.columns:
        cols.append("inventory_status")
    cols = [c for c in cols if c in df.columns]

    return {
        "metric": "inventory_accuracy",
        "category": category,
        "sku": sku,
        "avg_availability_pct": avg_acc,
        "low_accuracy_count": len(low_acc),
        "summary": (
            f"Average inventory availability: {avg_acc:.1f}%. "
            f"{len(low_acc)} SKUs below 90% (potential accuracy issues)."
        ),
        "low_accuracy_skus": low_acc[cols].sort_values("availability_pct").head(10).round(2).to_dict(orient="records"),
    }


def get_excess_obsolete(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    excess_weeks: float = 13.0,
) -> Dict[str, Any]:
    """
    Identify excess and obsolete inventory.
    Excess: weeks_of_cover > excess_weeks (default 13 = ~3 months).
    Overstock: inventory_status == 'Overstock'.
    """
    df = _apply_filters(load_inventory_data(), category, None, sku)

    if df.empty:
        return {"metric": "excess_obsolete", "summary": "No inventory data found."}

    df = _compute_weeks_of_cover(df)
    excess = df[df["weeks_of_cover"] > excess_weeks].copy()

    overstock_count = 0
    if "inventory_status" in df.columns:
        overstock_count = int((df["inventory_status"].str.lower() == "overstock").sum())

    excess_units = float(excess["stock_on_hand_units"].sum())

    cols = ["sku", "product_type", "stock_on_hand_units", "weeks_of_cover"]
    if "inventory_status" in excess.columns:
        cols.append("inventory_status")
    cols = [c for c in cols if c in excess.columns]

    return {
        "metric": "excess_obsolete",
        "category": category,
        "sku": sku,
        "excess_sku_count": len(excess),
        "overstock_sku_count": overstock_count,
        "excess_units": round(excess_units, 0),
        "summary": (
            f"{len(excess)} SKUs have excess stock (> {excess_weeks:.0f} weeks of cover) "
            f"representing {excess_units:.0f} units. "
            f"{overstock_count} SKUs flagged as Overstock."
        ),
        "records": excess[cols].sort_values("weeks_of_cover", ascending=False).head(10).round(2).to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(get_inventory_risk(category="haircare"))
    print(get_stock_summary(category="haircare"))
    print(get_healthy_skus(category="haircare"))
    print(get_inventory_turns(category="haircare"))
    print(get_days_of_supply(category="haircare"))
    print(get_inventory_accuracy(category="haircare"))
    print(get_excess_obsolete(category="haircare"))
