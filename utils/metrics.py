# utils/metrics.py

from __future__ import annotations

from typing import Optional

import numpy as np
import pandas as pd


def safe_divide(numerator: float, denominator: float, default: float = 0.0) -> float:
    """
    Safely divide two numbers.
    Returns `default` if denominator is 0 or NaN.
    """
    if denominator is None or pd.isna(denominator) or denominator == 0:
        return default
    return float(numerator) / float(denominator)


def percentage(value: float, decimals: int = 2) -> float:
    """
    Convert decimal to percentage.
    Example: 0.956 -> 95.6
    """
    return round(value * 100, decimals)


def calculate_otif(delivered_on_time_in_full: float, total_orders: float) -> float:
    """
    Calculate OTIF as a percentage.
    """
    return percentage(safe_divide(delivered_on_time_in_full, total_orders))


def calculate_fill_rate(delivered_qty: float, ordered_qty: float) -> float:
    """
    Calculate fill rate as a percentage.
    """
    return percentage(safe_divide(delivered_qty, ordered_qty))


def calculate_forecast_error(actual: float, forecast: float) -> float:
    """
    Absolute forecast error.
    """
    if pd.isna(actual) or pd.isna(forecast):
        return 0.0
    return abs(float(actual) - float(forecast))


def calculate_ape(actual: float, forecast: float) -> float:
    """
    Absolute Percentage Error (APE) in %.
    """
    error = calculate_forecast_error(actual, forecast)
    return percentage(safe_divide(error, actual))


def calculate_mape(df: pd.DataFrame, actual_col: str, forecast_col: str) -> float:
    """
    Calculate Mean Absolute Percentage Error (MAPE) in %.
    Ignores rows where actual = 0 to avoid distortion/divide-by-zero.
    """
    if df.empty:
        return 0.0

    valid_df = df[df[actual_col] != 0].copy()
    if valid_df.empty:
        return 0.0

    valid_df["ape"] = (
        (valid_df[actual_col] - valid_df[forecast_col]).abs() / valid_df[actual_col].abs()
    ) * 100

    return round(valid_df["ape"].mean(), 2)


def calculate_wape(df: pd.DataFrame, actual_col: str, forecast_col: str) -> float:
    """
    Calculate Weighted Absolute Percentage Error (WAPE) in %.
    Better than MAPE in many supply chain cases.
    """
    if df.empty:
        return 0.0

    numerator = (df[actual_col] - df[forecast_col]).abs().sum()
    denominator = df[actual_col].abs().sum()

    return round(percentage(safe_divide(numerator, denominator)), 2)


def calculate_bias(df: pd.DataFrame, actual_col: str, forecast_col: str) -> float:
    """
    Forecast bias in %.
    Positive = over-forecast
    Negative = under-forecast
    """
    if df.empty:
        return 0.0

    numerator = (df[forecast_col] - df[actual_col]).sum()
    denominator = df[actual_col].sum()

    return round(percentage(safe_divide(numerator, denominator)), 2)


def calculate_days_of_supply(stock_qty: float, avg_daily_demand: float) -> float:
    """
    Days of Supply = Current Stock / Average Daily Demand
    """
    return round(safe_divide(stock_qty, avg_daily_demand), 2)


def calculate_stock_cover(stock_qty: float, weekly_demand: float) -> float:
    """
    Stock cover in weeks.
    """
    return round(safe_divide(stock_qty, weekly_demand), 2)


def calculate_inventory_value(stock_qty: float, unit_cost: float) -> float:
    """
    Inventory value = stock quantity * unit cost
    """
    if pd.isna(stock_qty) or pd.isna(unit_cost):
        return 0.0
    return round(float(stock_qty) * float(unit_cost), 2)


def calculate_late_po_rate(late_pos: float, total_pos: float) -> float:
    """
    Late purchase order rate in %.
    """
    return percentage(safe_divide(late_pos, total_pos))


def calculate_supplier_otif(on_time_pos: float, total_pos: float) -> float:
    """
    Supplier OTIF / On-time delivery rate in %.
    """
    return percentage(safe_divide(on_time_pos, total_pos))


def add_otif_flag(
    df: pd.DataFrame,
    delivered_col: str = "delivered_in_full_on_time",
    total_col: str = "total_orders",
    threshold: float = 95.0,
) -> pd.DataFrame:
    """
    Add OTIF % and risk flag columns to dataframe.
    """
    result = df.copy()
    result["otif_pct"] = result.apply(
        lambda row: calculate_otif(row[delivered_col], row[total_col]), axis=1
    )
    result["otif_flag"] = np.where(result["otif_pct"] < threshold, "Risk", "Healthy")
    return result


def add_forecast_metrics(
    df: pd.DataFrame,
    actual_col: str = "actual_demand",
    forecast_col: str = "forecast_demand",
) -> pd.DataFrame:
    """
    Add forecast error and APE columns.
    """
    result = df.copy()
    result["forecast_error"] = (result[actual_col] - result[forecast_col]).abs()

    result["ape_pct"] = np.where(
        result[actual_col] == 0,
        0.0,
        (result["forecast_error"] / result[actual_col].abs()) * 100,
    )

    return result


def add_inventory_metrics(
    df: pd.DataFrame,
    stock_col: str = "stock_qty",
    avg_daily_demand_col: str = "avg_daily_demand",
    unit_cost_col: Optional[str] = "unit_cost",
    dos_threshold: float = 7.0,
) -> pd.DataFrame:
    """
    Add inventory KPIs such as days of supply and inventory value.
    """
    result = df.copy()

    result["days_of_supply"] = result.apply(
        lambda row: calculate_days_of_supply(row[stock_col], row[avg_daily_demand_col]),
        axis=1,
    )

    result["inventory_risk"] = np.where(
        result["days_of_supply"] < dos_threshold, "Low Cover", "Healthy"
    )

    if unit_cost_col and unit_cost_col in result.columns:
        result["inventory_value"] = result.apply(
            lambda row: calculate_inventory_value(row[stock_col], row[unit_cost_col]),
            axis=1,
        )

    return result


def summarize_otif(df: pd.DataFrame, otif_col: str = "otif_pct") -> dict:
    """
    Return OTIF summary stats.
    """
    if df.empty or otif_col not in df.columns:
        return {
            "average_otif": 0.0,
            "min_otif": 0.0,
            "max_otif": 0.0,
            "risk_count": 0,
        }

    return {
        "average_otif": round(df[otif_col].mean(), 2),
        "min_otif": round(df[otif_col].min(), 2),
        "max_otif": round(df[otif_col].max(), 2),
        "risk_count": int((df[otif_col] < 95).sum()),
    }


def summarize_forecast(df: pd.DataFrame) -> dict:
    """
    Return forecasting summary metrics.
    Requires columns: actual_demand, forecast_demand
    """
    if df.empty:
        return {"mape": 0.0, "wape": 0.0, "bias": 0.0}

    return {
        "mape": calculate_mape(df, "actual_demand", "forecast_demand"),
        "wape": calculate_wape(df, "actual_demand", "forecast_demand"),
        "bias": calculate_bias(df, "actual_demand", "forecast_demand"),
    }


def summarize_inventory(df: pd.DataFrame, dos_col: str = "days_of_supply") -> dict:
    """
    Return inventory summary stats.
    """
    if df.empty or dos_col not in df.columns:
        return {
            "avg_days_of_supply": 0.0,
            "low_cover_count": 0,
        }

    return {
        "avg_days_of_supply": round(df[dos_col].mean(), 2),
        "low_cover_count": int((df[dos_col] < 7).sum()),
    }


def summarize_suppliers(
    df: pd.DataFrame,
    late_col: str = "late_pos",
    total_col: str = "total_pos",
) -> dict:
    """
    Return supplier summary stats.
    """
    if df.empty:
        return {
            "late_po_rate": 0.0,
            "total_late_pos": 0,
            "total_pos": 0,
        }

    total_late = df[late_col].sum()
    total_pos = df[total_col].sum()

    return {
        "late_po_rate": calculate_late_po_rate(total_late, total_pos),
        "total_late_pos": int(total_late),
        "total_pos": int(total_pos),
    }