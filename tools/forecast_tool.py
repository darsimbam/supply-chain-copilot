from pathlib import Path
from typing import Optional, Dict, Any
import pandas as pd

DATA_DIR = Path(__file__).resolve().parents[1] / "data"
FORECAST_FILE = DATA_DIR / "forecast.csv"

_forecast_cache: Optional[pd.DataFrame] = None


def load_forecast_data() -> pd.DataFrame:
    """Load forecast data from CSV (cached after first load)."""
    global _forecast_cache
    if _forecast_cache is not None:
        return _forecast_cache

    if not FORECAST_FILE.exists():
        raise FileNotFoundError(f"Forecast file not found: {FORECAST_FILE}")

    df = pd.read_csv(FORECAST_FILE)

    required_cols = {"week_start_date", "product_type", "sku", "forecast_qty", "actual_sales_qty"}
    missing = required_cols - set(df.columns)
    if missing:
        raise ValueError(f"Missing required forecast columns: {missing}")

    _forecast_cache = df
    return _forecast_cache


def _apply_filters(
    df: pd.DataFrame,
    category: Optional[str],
    week: Optional[str],
    sku: Optional[str],
) -> pd.DataFrame:
    if category:
        df = df[df["product_type"].str.lower() == category.lower()]
    if week:
        df = df[df["week_start_date"].astype(str) == str(week)]
    if sku:
        df = df[df["sku"].str.upper() == sku.upper()]
    return df


def get_forecast_error(
    category: Optional[str] = None,
    week: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Return MAPE and worst SKUs for a given category/week/sku."""
    df = _apply_filters(load_forecast_data(), category, week, sku)

    if df.empty:
        return {"metric": "forecast_error", "category": category, "week": week, "sku": sku,
                "summary": "No forecast data found.", "records": []}

    df = df[df["actual_sales_qty"] != 0].copy()
    df["abs_error"] = (df["actual_sales_qty"] - df["forecast_qty"]).abs()
    df["abs_pct_error"] = (df["abs_error"] / df["actual_sales_qty"]) * 100

    mape = float(df["abs_pct_error"].mean())
    worst_skus = (
        df.sort_values("abs_pct_error", ascending=False)
        .head(5)[["sku", "product_type", "week_start_date", "forecast_qty",
                  "actual_sales_qty", "abs_pct_error"]]
    )

    return {
        "metric": "forecast_error",
        "category": category,
        "week": week,
        "sku": sku,
        "mape": round(mape, 2),
        "summary": f"Forecast MAPE is {mape:.2f}%.",
        "worst_skus": worst_skus.to_dict(orient="records"),
    }


def get_bias(
    category: Optional[str] = None,
    week: Optional[str] = None,
    sku: Optional[str] = None,
) -> Dict[str, Any]:
    """Return forecast bias: positive = over-forecast, negative = under-forecast."""
    df = _apply_filters(load_forecast_data(), category, week, sku)

    if df.empty:
        return {"metric": "forecast_bias", "category": category, "week": week, "sku": sku,
                "summary": "No forecast data found."}

    df = df.copy()
    df["error"] = df["forecast_qty"] - df["actual_sales_qty"]
    bias = float(df["error"].mean())
    interpretation = "Over-forecasting" if bias > 0 else "Under-forecasting" if bias < 0 else "Balanced"

    return {
        "metric": "forecast_bias",
        "category": category,
        "week": week,
        "sku": sku,
        "bias": round(bias, 2),
        "interpretation": interpretation,
        "summary": f"{interpretation} detected with average bias of {bias:.2f} units.",
    }


def get_top_skus(category: Optional[str] = None, n: int = 10) -> Dict[str, Any]:
    """Rank SKUs by total actual sales volume and forecast accuracy."""
    df = _apply_filters(load_forecast_data(), category, None, None)

    if df.empty:
        return {"metric": "top_skus", "category": category,
                "summary": "No forecast data found.", "top_skus": [], "bottom_skus": []}

    valid = df[df["actual_sales_qty"] != 0].copy()
    valid["abs_pct_error"] = (
        (valid["actual_sales_qty"] - valid["forecast_qty"]).abs() / valid["actual_sales_qty"]
    ) * 100

    sku_summary = (
        valid.groupby(["sku", "product_type"])
        .agg(
            total_actual_sales=("actual_sales_qty", "sum"),
            avg_forecast_accuracy=("abs_pct_error", lambda x: round(100 - x.mean(), 2)),
            weeks_of_data=("week_start_date", "count"),
        )
        .reset_index()
        .sort_values("total_actual_sales", ascending=False)
    )

    top = sku_summary.head(n)
    bottom = sku_summary.tail(n).sort_values("total_actual_sales")

    return {
        "metric": "top_skus",
        "category": category,
        "summary": (
            f"Top SKU by sales: {top.iloc[0]['sku']} with {top.iloc[0]['total_actual_sales']:.0f} units sold."
            if not top.empty else "No data."
        ),
        "top_skus_by_sales": top.to_dict(orient="records"),
        "bottom_skus_by_sales": bottom.to_dict(orient="records"),
    }


def get_top_skus_by_revenue(
    category: Optional[str] = None,
    sku: Optional[str] = None,
    n: int = 10,
) -> Dict[str, Any]:
    """Rank SKUs by total actual revenue."""
    df = _apply_filters(load_forecast_data(), category, None, sku)

    if "actual_revenue" not in df.columns or df.empty:
        return {"metric": "top_skus_by_revenue", "category": category, "sku": sku,
                "summary": "No revenue data available.", "records": []}

    revenue_summary = (
        df.groupby(["sku", "product_type"])
        .agg(
            total_revenue=("actual_revenue", "sum"),
            total_units_sold=("actual_sales_qty", "sum"),
            weeks_of_data=("week_start_date", "count"),
        )
        .reset_index()
        .sort_values("total_revenue", ascending=False)
    )

    top = revenue_summary.head(n)

    return {
        "metric": "top_skus_by_revenue",
        "category": category,
        "sku": sku,
        "summary": (
            f"Top revenue SKU: {top.iloc[0]['sku']} with "
            f"${top.iloc[0]['total_revenue']:,.0f} in total revenue."
            if not top.empty else "No data."
        ),
        "records": top.round(2).to_dict(orient="records"),
    }


if __name__ == "__main__":
    print(get_forecast_error(category="haircare"))
    print(get_bias(category="haircare"))
    print(get_top_skus(category="haircare"))
    print(get_top_skus_by_revenue(category="haircare"))
