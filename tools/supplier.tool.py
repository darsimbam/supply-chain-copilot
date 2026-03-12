from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
SUPPLIERS_FILE = DATA_DIR / "suppliers.csv"
PO_FILE = DATA_DIR / "purchase_orders.csv"


def load_suppliers_data() -> pd.DataFrame:
    """Load supplier master / performance data."""
    if not SUPPLIERS_FILE.exists():
        raise FileNotFoundError(f"Suppliers file not found: {SUPPLIERS_FILE}")

    df = pd.read_csv(SUPPLIERS_FILE)

    required_cols = {"supplier", "category", "lead_time_days", "on_time_rate"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required suppliers columns: {missing}")

    return df


def load_purchase_orders_data() -> pd.DataFrame:
    """Load purchase orders data."""
    if not PO_FILE.exists():
        raise FileNotFoundError(f"Purchase orders file not found: {PO_FILE}")

    df = pd.read_csv(PO_FILE)

    required_cols = {"po_number", "supplier", "category", "status", "scheduled_date", "actual_date"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required PO columns: {missing}")

    df["scheduled_date"] = pd.to_datetime(df["scheduled_date"], errors="coerce")
    df["actual_date"] = pd.to_datetime(df["actual_date"], errors="coerce")

    return df


def get_supplier_delays(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Find delayed purchase orders and supplier delay statistics.
    """
    po_df = load_purchase_orders_data().copy()

    if category:
        po_df = po_df[po_df["category"].str.lower() == category.lower()]

    if po_df.empty:
        return {
            "metric": "supplier_delays",
            "summary": "No supplier PO data found."
        }

    po_df["delay_days"] = (po_df["actual_date"] - po_df["scheduled_date"]).dt.days
    delayed = po_df[po_df["delay_days"] > 0].copy()

    delay_summary = (
        delayed.groupby("supplier")
        .agg(
            late_po_count=("po_number", "count"),
            avg_delay_days=("delay_days", "mean")
        )
        .reset_index()
        .sort_values(["late_po_count", "avg_delay_days"], ascending=[False, False])
    )

    return {
        "metric": "supplier_delays",
        "category": category,
        "late_po_count": int(len(delayed)),
        "summary": f"{len(delayed)} purchase orders are delayed.",
        "supplier_delay_summary": delay_summary.round(2).to_dict(orient="records"),
        "delayed_pos": delayed[["po_number", "supplier", "category", "scheduled_date", "actual_date", "delay_days"]]
        .astype(str)
        .to_dict(orient="records")
    }


def get_supplier_performance(category: Optional[str] = None) -> Dict[str, Any]:
    """
    Return supplier performance overview.
    """
    df = load_suppliers_data().copy()

    if category:
        df = df[df["category"].str.lower() == category.lower()]

    if df.empty:
        return {
            "metric": "supplier_performance",
            "summary": "No supplier data found."
        }

    worst_suppliers = df.sort_values("on_time_rate").head(5)

    return {
        "metric": "supplier_performance",
        "category": category,
        "average_on_time_rate": round(float(df["on_time_rate"].mean()), 2),
        "average_lead_time_days": round(float(df["lead_time_days"].mean()), 2),
        "summary": (
            f"Average supplier on-time rate is {df['on_time_rate'].mean():.2f}% "
            f"with average lead time of {df['lead_time_days'].mean():.2f} days."
        ),
        "worst_suppliers": worst_suppliers.to_dict(orient="records")
    }


if __name__ == "__main__":
    print(get_supplier_delays(category="Haircare"))
    print(get_supplier_performance(category="Haircare"))