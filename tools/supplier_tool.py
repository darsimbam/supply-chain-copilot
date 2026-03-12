from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUPPLIERS_FILE = DATA_DIR / "suppliers.csv"
PO_FILE = DATA_DIR / "purchase_orders.csv"

_suppliers_cache: Optional[pd.DataFrame] = None
_po_cache: Optional[pd.DataFrame] = None


def load_suppliers_data() -> pd.DataFrame:
    """Load supplier master data (cached after first load)."""
    global _suppliers_cache
    if _suppliers_cache is not None:
        return _suppliers_cache

    if not SUPPLIERS_FILE.exists():
        raise FileNotFoundError(f"Suppliers file not found: {SUPPLIERS_FILE}")

    df = pd.read_csv(SUPPLIERS_FILE)

    required_cols = {"supplier_name", "product_type", "supplier_lead_time_days", "defect_rate_pct"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required suppliers columns: {missing}")

    _suppliers_cache = df
    return _suppliers_cache


def load_purchase_orders_data() -> pd.DataFrame:
    """Load purchase orders data (cached after first load)."""
    global _po_cache
    if _po_cache is not None:
        return _po_cache

    if not PO_FILE.exists():
        raise FileNotFoundError(f"Purchase orders file not found: {PO_FILE}")

    df = pd.read_csv(PO_FILE)

    required_cols = {"po_id", "supplier_name", "product_type", "po_status",
                     "expected_receipt_date", "actual_receipt_date", "days_late"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required PO columns: {missing}")

    df["expected_receipt_date"] = pd.to_datetime(df["expected_receipt_date"], errors="coerce")
    df["actual_receipt_date"] = pd.to_datetime(df["actual_receipt_date"], errors="coerce")

    _po_cache = df
    return _po_cache


def _apply_po_filters(
    df: pd.DataFrame,
    category: Optional[str],
    sku: Optional[str],
    supplier: Optional[str],
) -> pd.DataFrame:
    if category:
        df = df[df["product_type"].str.lower() == category.lower()]
    if sku and "sku" in df.columns:
        df = df[df["sku"].str.upper() == sku.upper()]
    if supplier:
        df = df[df["supplier_name"].str.upper() == supplier.upper()]
    return df


def get_supplier_delays(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    supplier: Optional[str] = None,
) -> Dict[str, Any]:
    """Find delayed purchase orders and supplier delay statistics."""
    po_df = _apply_po_filters(load_purchase_orders_data(), category, sku, supplier)

    if po_df.empty:
        return {"metric": "supplier_delays", "summary": "No supplier PO data found."}

    delayed = po_df[po_df["days_late"] > 0].copy()

    delay_summary = (
        delayed.groupby("supplier_name")
        .agg(
            late_po_count=("po_id", "count"),
            avg_delay_days=("days_late", "mean"),
        )
        .reset_index()
        .sort_values(["late_po_count", "avg_delay_days"], ascending=[False, False])
    )

    return {
        "metric": "supplier_delays",
        "category": category,
        "sku": sku,
        "late_po_count": int(len(delayed)),
        "summary": f"{len(delayed)} purchase orders are delayed.",
        "supplier_delay_summary": delay_summary.round(2).to_dict(orient="records"),
        "delayed_pos": delayed[
            ["po_id", "supplier_name", "product_type",
             "expected_receipt_date", "actual_receipt_date", "days_late"]
        ].astype(str).to_dict(orient="records"),
    }


def get_supplier_performance(
    category: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Return supplier performance overview with on-time rate computed from PO data."""
    df = load_suppliers_data().copy()
    po_df = load_purchase_orders_data().copy()

    if category:
        df = df[df["product_type"].str.lower() == category.lower()]
        po_df = po_df[po_df["product_type"].str.lower() == category.lower()]
    if sku and "sku" in po_df.columns:
        po_df = po_df[po_df["sku"].str.upper() == sku.upper()]

    if df.empty:
        return {"metric": "supplier_performance", "summary": "No supplier data found."}

    if not po_df.empty:
        on_time_rates = (
            po_df.groupby("supplier_name")["days_late"]
            .apply(lambda x: round((x <= 0).mean() * 100, 2))
            .reset_index(name="on_time_rate")
        )
        df = df.merge(on_time_rates, on="supplier_name", how="left")
        # Leave as NaN (not 0%) for suppliers with no PO history — 0% would be misleading
    else:
        df["on_time_rate"] = None

    ranked = df.dropna(subset=["on_time_rate"])
    worst_suppliers = ranked.sort_values("on_time_rate").head(5)

    avg_on_time = round(float(ranked["on_time_rate"].mean()), 2) if not ranked.empty else None
    avg_lead_time = round(float(df["supplier_lead_time_days"].mean()), 2)

    summary = (
        f"Average supplier on-time rate is {avg_on_time:.2f}% "
        f"with average lead time of {avg_lead_time:.2f} days."
        if avg_on_time is not None
        else f"No PO history available. Average lead time is {avg_lead_time:.2f} days."
    )

    return {
        "metric": "supplier_performance",
        "category": category,
        "sku": sku,
        "average_on_time_rate": avg_on_time,
        "average_lead_time_days": avg_lead_time,
        "summary": summary,
        "worst_suppliers": worst_suppliers[
            ["supplier_name", "product_type", "supplier_lead_time_days",
             "defect_rate_pct", "supplier_risk_level", "on_time_rate"]
        ].to_dict(orient="records"),
    }


def get_open_pos(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    supplier: Optional[str] = None,
) -> Dict[str, Any]:
    """Return all open/pending purchase orders, optionally filtered."""
    po_df = _apply_po_filters(load_purchase_orders_data(), category, sku, supplier)

    if po_df.empty:
        return {"metric": "open_pos", "summary": "No PO data found.", "records": []}

    open_statuses = {"open", "pending", "in transit", "in-transit", "ordered"}
    open_pos = po_df[po_df["po_status"].str.lower().isin(open_statuses)].copy()

    if open_pos.empty:
        return {
            "metric": "open_pos",
            "category": category,
            "sku": sku,
            "supplier": supplier,
            "open_count": 0,
            "summary": "No open purchase orders found.",
            "records": [],
        }

    open_pos = open_pos.sort_values("expected_receipt_date")

    cols = ["po_id", "supplier_name", "product_type", "order_qty",
            "expected_receipt_date", "po_status", "days_late"]
    if "sku" in open_pos.columns:
        cols.insert(2, "sku")
    cols = [c for c in cols if c in open_pos.columns]

    return {
        "metric": "open_pos",
        "category": category,
        "sku": sku,
        "supplier": supplier,
        "open_count": int(len(open_pos)),
        "summary": f"{len(open_pos)} open purchase orders in flight.",
        "records": open_pos[cols].astype(str).to_dict(orient="records"),
    }


def get_order_cycle_time(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    supplier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Order cycle time = actual_receipt_date - po_date in days.
    Benchmark: cycle time <= supplier_lead_time_days is on target.
    """
    po_df = _apply_po_filters(load_purchase_orders_data(), category, sku, supplier)

    if po_df.empty:
        return {"metric": "order_cycle_time", "summary": "No PO data found."}

    if "po_date" not in po_df.columns:
        return {"metric": "order_cycle_time", "summary": "po_date column not available."}

    po_df = po_df.copy()
    po_df["po_date"] = pd.to_datetime(po_df["po_date"], errors="coerce")

    completed = po_df.dropna(subset=["po_date", "actual_receipt_date"]).copy()
    if completed.empty:
        return {
            "metric": "order_cycle_time",
            "category": category,
            "sku": sku,
            "supplier": supplier,
            "summary": "No completed POs with both po_date and actual_receipt_date.",
        }

    completed["cycle_time_days"] = (
        completed["actual_receipt_date"] - completed["po_date"]
    ).dt.days

    avg_cycle = round(float(completed["cycle_time_days"].mean()), 1)
    max_cycle = int(completed["cycle_time_days"].max())
    min_cycle = int(completed["cycle_time_days"].min())

    by_supplier = (
        completed.groupby("supplier_name")["cycle_time_days"]
        .agg(avg_cycle_days="mean", po_count="count")
        .reset_index()
        .sort_values("avg_cycle_days", ascending=False)
        .round(1)
    )

    return {
        "metric": "order_cycle_time",
        "category": category,
        "sku": sku,
        "supplier": supplier,
        "avg_cycle_time_days": avg_cycle,
        "max_cycle_time_days": max_cycle,
        "min_cycle_time_days": min_cycle,
        "po_count": int(len(completed)),
        "summary": (
            f"Average order cycle time: {avg_cycle:.1f} days "
            f"(range: {min_cycle}–{max_cycle} days across {len(completed)} completed POs)."
        ),
        "by_supplier": by_supplier.to_dict(orient="records"),
    }


def get_freight_cost(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    supplier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Freight cost analysis from shipping_cost and freight_cost columns.
    Returns totals and per-unit averages by supplier and category.
    """
    po_df = _apply_po_filters(load_purchase_orders_data(), category, sku, supplier)

    if po_df.empty:
        return {"metric": "freight_cost", "summary": "No PO data found."}

    cost_cols = [c for c in ["shipping_cost", "freight_cost"] if c in po_df.columns]
    if not cost_cols:
        return {"metric": "freight_cost", "summary": "No freight/shipping cost columns found in PO data."}

    po_df = po_df.copy()
    po_df["total_freight"] = po_df[cost_cols].fillna(0).sum(axis=1)

    total_cost = round(float(po_df["total_freight"].sum()), 2)
    avg_cost_per_po = round(float(po_df["total_freight"].mean()), 2)

    # Per-unit cost where order_qty is available
    if "order_qty" in po_df.columns:
        valid = po_df[po_df["order_qty"] > 0].copy()
        valid["freight_per_unit"] = valid["total_freight"] / valid["order_qty"]
        avg_per_unit = round(float(valid["freight_per_unit"].mean()), 4)
    else:
        avg_per_unit = None

    by_supplier = (
        po_df.groupby("supplier_name")["total_freight"]
        .agg(total_freight="sum", avg_freight_per_po="mean", po_count="count")
        .reset_index()
        .sort_values("total_freight", ascending=False)
        .round(2)
    )

    summary = (
        f"Total freight cost: {total_cost:,.2f}. "
        f"Average per PO: {avg_cost_per_po:,.2f}."
    )
    if avg_per_unit is not None:
        summary += f" Average per unit: {avg_per_unit:.4f}."

    return {
        "metric": "freight_cost",
        "category": category,
        "sku": sku,
        "supplier": supplier,
        "total_freight_cost": total_cost,
        "avg_freight_per_po": avg_cost_per_po,
        "avg_freight_per_unit": avg_per_unit,
        "cost_columns_used": cost_cols,
        "summary": summary,
        "by_supplier": by_supplier.head(10).to_dict(orient="records"),
    }


def get_lead_time_variability(
    category: Optional[str] = None,
    supplier: Optional[str] = None,
) -> Dict[str, Any]:
    """
    Lead time variability = std deviation of actual lead time (po_date → actual_receipt_date).
    High variability (std > 7 days) indicates unreliable suppliers.
    Also reports days_late std dev as a simpler proxy when po_date is unavailable.
    """
    po_df = _apply_po_filters(load_purchase_orders_data(), category, None, supplier)

    if po_df.empty:
        return {"metric": "lead_time_variability", "summary": "No PO data found."}

    po_df = po_df.copy()

    # Prefer actual cycle time; fall back to days_late std as proxy
    if "po_date" in po_df.columns:
        po_df["po_date"] = pd.to_datetime(po_df["po_date"], errors="coerce")
        completed = po_df.dropna(subset=["po_date", "actual_receipt_date"]).copy()
        completed["actual_lead_time"] = (
            completed["actual_receipt_date"] - completed["po_date"]
        ).dt.days
        metric_col = "actual_lead_time"
        data = completed
    else:
        data = po_df.dropna(subset=["days_late"])
        metric_col = "days_late"

    if data.empty:
        return {
            "metric": "lead_time_variability",
            "category": category,
            "supplier": supplier,
            "summary": "No completed PO records to compute variability.",
        }

    overall_std = round(float(data[metric_col].std()), 1)
    overall_mean = round(float(data[metric_col].mean()), 1)

    by_supplier = (
        data.groupby("supplier_name")[metric_col]
        .agg(mean_days="mean", std_days="std", po_count="count")
        .reset_index()
        .sort_values("std_days", ascending=False)
        .round(1)
    )
    by_supplier["std_days"] = by_supplier["std_days"].fillna(0)

    high_var = by_supplier[by_supplier["std_days"] > 7]

    return {
        "metric": "lead_time_variability",
        "category": category,
        "supplier": supplier,
        "overall_mean_days": overall_mean,
        "overall_std_days": overall_std,
        "high_variability_supplier_count": int(len(high_var)),
        "summary": (
            f"Mean lead time: {overall_mean:.1f} days, std dev: {overall_std:.1f} days. "
            f"{len(high_var)} supplier(s) with high variability (std > 7 days)."
        ),
        "by_supplier": by_supplier.head(10).to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(get_supplier_delays(category="haircare"))
    print(get_supplier_performance(category="haircare"))
    print(get_open_pos(category="haircare"))
    print(get_order_cycle_time(category="haircare"))
    print(get_freight_cost(category="haircare"))
    print(get_lead_time_variability(category="haircare"))
