from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
OTIF_FILE = DATA_DIR / "otif.csv"

_otif_cache: Optional[pd.DataFrame] = None


def load_otif_data() -> pd.DataFrame:
    """Load OTIF data from CSV (cached after first load)."""
    global _otif_cache
    if _otif_cache is not None:
        return _otif_cache

    if not OTIF_FILE.exists():
        raise FileNotFoundError(f"OTIF file not found: {OTIF_FILE}")

    df = pd.read_csv(OTIF_FILE)

    required_cols = {"product_type", "customer_request_date", "otif_flag"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required OTIF columns: {missing}")

    df["customer_request_date"] = pd.to_datetime(df["customer_request_date"])
    df["week"] = df["customer_request_date"].dt.strftime("%G-W%V")
    df["otif_flag"] = df["otif_flag"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
    )

    _otif_cache = df
    return _otif_cache


def _aggregate_weekly(df: pd.DataFrame) -> pd.DataFrame:
    """Aggregate PO-level data into weekly OTIF rates."""
    return (
        df.groupby("week")["otif_flag"]
        .agg(otif_rate=lambda x: round(x.mean() * 100, 2), order_count="count")
        .reset_index()
        .sort_values("week")
    )


def _apply_filters(df: pd.DataFrame, category: Optional[str], sku: Optional[str]) -> pd.DataFrame:
    if category:
        df = df[df["product_type"].str.lower() == category.lower()]
    if sku and "sku" in df.columns:
        df = df[df["sku"].str.upper() == sku.upper()]
    return df


def get_otif_trend(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    weeks: int = 4,
) -> Dict[str, Any]:
    """Return OTIF trend over the most recent N weeks."""
    df = _apply_filters(load_otif_data(), category, sku)

    if df.empty:
        return {"metric": "otif_trend", "category": category, "sku": sku,
                "summary": "No OTIF data found.", "records": []}

    weekly = _aggregate_weekly(df)
    recent = weekly.tail(weeks)
    latest_otif = float(recent["otif_rate"].iloc[-1])
    avg_otif = float(recent["otif_rate"].mean())

    return {
        "metric": "otif_trend",
        "category": category,
        "sku": sku,
        "latest_otif": round(latest_otif, 2),
        "average_otif": round(avg_otif, 2),
        "summary": (
            f"Latest OTIF is {latest_otif:.2f}% vs {avg_otif:.2f}% average "
            f"over last {len(recent)} weeks."
        ),
        "records": recent.to_dict(orient="records"),
    }


def get_otif_drop(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    compare_last_n: int = 2,
) -> Dict[str, Any]:
    """Compare last OTIF point versus the previous one."""
    df = _apply_filters(load_otif_data(), category, sku)

    if df.empty:
        return {"metric": "otif_drop", "category": category, "sku": sku,
                "summary": "No OTIF data found."}

    weekly = _aggregate_weekly(df)

    if len(weekly) < compare_last_n:
        return {"metric": "otif_drop", "category": category, "sku": sku,
                "summary": "Not enough OTIF history for comparison."}

    last_values = weekly["otif_rate"].tail(compare_last_n).tolist()
    current = float(last_values[-1])
    previous = float(last_values[-2])
    delta = current - previous
    direction = "down" if delta < 0 else "up" if delta > 0 else "flat"

    return {
        "metric": "otif_drop",
        "category": category,
        "sku": sku,
        "current_otif": round(current, 2),
        "previous_otif": round(previous, 2),
        "delta": round(delta, 2),
        "direction": direction,
        "summary": f"OTIF is {direction}: {previous:.2f}% -> {current:.2f}% ({delta:+.2f} pts).",
    }


def get_sku_otif_ranking(category: Optional[str] = None, n: int = 10) -> Dict[str, Any]:
    """Rank all SKUs by OTIF rate."""
    df = _apply_filters(load_otif_data(), category, None)

    if df.empty or "sku" not in df.columns:
        return {"metric": "sku_otif_ranking", "category": category,
                "summary": "No SKU-level OTIF data found.", "best_skus": [], "worst_skus": []}

    sku_otif = (
        df.groupby(["sku", "product_type"])["otif_flag"]
        .agg(otif_rate=lambda x: round(x.mean() * 100, 2), order_count="count")
        .reset_index()
        .sort_values("otif_rate", ascending=False)
    )

    best = sku_otif.head(n)
    worst = sku_otif.tail(n).sort_values("otif_rate")

    return {
        "metric": "sku_otif_ranking",
        "category": category,
        "summary": (
            f"Best OTIF SKU: {best.iloc[0]['sku']} at {best.iloc[0]['otif_rate']:.1f}%. "
            f"Worst: {worst.iloc[0]['sku']} at {worst.iloc[0]['otif_rate']:.1f}%."
            if not best.empty else "No data."
        ),
        "best_otif_skus": best.to_dict(orient="records"),
        "worst_otif_skus": worst.to_dict(orient="records"),
    }


def get_otif_by_supplier(
    category: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Break down OTIF rates by supplier."""
    df = _apply_filters(load_otif_data(), category, sku)

    if df.empty or "supplier_name" not in df.columns:
        return {"metric": "otif_by_supplier", "category": category, "sku": sku,
                "summary": "No supplier-level OTIF data found.", "records": []}

    supplier_otif = (
        df.groupby("supplier_name")["otif_flag"]
        .agg(otif_rate=lambda x: round(x.mean() * 100, 2), order_count="count")
        .reset_index()
        .sort_values("otif_rate")
    )

    worst = supplier_otif.iloc[0] if not supplier_otif.empty else None

    return {
        "metric": "otif_by_supplier",
        "category": category,
        "sku": sku,
        "summary": (
            f"Worst supplier OTIF: {worst['supplier_name']} at {worst['otif_rate']:.1f}%."
            if worst is not None else "No data."
        ),
        "records": supplier_otif.to_dict(orient="records"),
    }


def get_fill_rate(
    category: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Fill rate = shipped_qty / order_qty per PO.
    In-full rate = % of POs where in_full == True.
    Benchmark: fill rate >= 95% is healthy; < 90% is a concern.
    """
    df = _apply_filters(load_otif_data(), category, sku)

    if df.empty:
        return {"metric": "fill_rate", "category": category, "sku": sku,
                "summary": "No OTIF data found.", "records": []}

    result: Dict[str, Any] = {
        "metric": "fill_rate",
        "category": category,
        "sku": sku,
    }

    # In-full rate from boolean flag
    if "in_full" in df.columns:
        in_full = df["in_full"].map(
            lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
        )
        in_full_rate = round(float(in_full.mean() * 100), 2)
        result["in_full_rate_pct"] = in_full_rate
    else:
        in_full_rate = None
        result["in_full_rate_pct"] = None

    # Quantity fill rate from shipped/order quantities
    if "shipped_qty" in df.columns and "order_qty" in df.columns:
        valid = df[(df["order_qty"] > 0)].copy()
        valid["line_fill_rate"] = valid["shipped_qty"] / valid["order_qty"] * 100
        avg_fill_rate = round(float(valid["line_fill_rate"].mean()), 2)
        low_fill = valid[valid["line_fill_rate"] < 90]
        result["avg_qty_fill_rate_pct"] = avg_fill_rate
        result["low_fill_rate_count"] = int(len(low_fill))

        summary_parts = []
        if in_full_rate is not None:
            summary_parts.append(f"In-full rate: {in_full_rate:.1f}%")
        summary_parts.append(
            f"Avg quantity fill rate: {avg_fill_rate:.1f}%. "
            f"{len(low_fill)} POs below 90% fill rate."
        )
        result["summary"] = " | ".join(summary_parts)
    else:
        result["summary"] = (
            f"In-full rate: {in_full_rate:.1f}%." if in_full_rate is not None
            else "Shipped/order quantity data not available."
        )

    return result


if __name__ == "__main__":
    print(get_otif_trend(category="Haircare"))
    print(get_otif_drop(category="Haircare"))
    print(get_sku_otif_ranking(category="Haircare"))
    print(get_otif_by_supplier(category="Haircare"))
    print(get_fill_rate(category="Haircare"))
