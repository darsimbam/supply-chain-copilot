"""
app.py — Baselstrasse Supply Chain Copilot
Dashboard: KPI cards → trend charts → category analytics → supplier panel → exception table
"""

import sys
from pathlib import Path

import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# ── Path setup ──────────────────────────────────────────────────────────────
PROJECT_ROOT = Path(__file__).resolve().parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from tools.forecast_tool import load_forecast_data
from tools.inventory_tool import load_inventory_data
from tools.otif_tool import load_otif_data
from tools.supplier_tool import load_purchase_orders_data, load_suppliers_data

# ── Page config ──────────────────────────────────────────────────────────────
st.set_page_config(
    page_title="Baselstrasse Supply Chain Copilot",
    page_icon="📦",
    layout="wide",
)

# ── Baselstrasse colour palette ───────────────────────────────────────────────
_BG      = "#0f172a"
_SURFACE = "#1e293b"
_ACCENT  = "#63b3ed"
_WHITE   = "#ffffff"
_MUTED   = "#a0aec0"
_RED     = "#fc8181"
_AMBER   = "#f6ad55"
_GREEN   = "#68d391"

# ── CSS ───────────────────────────────────────────────────────────────────────
st.markdown(f"""
<style>
.stApp, [data-testid="stAppViewContainer"] {{
    background-color: {_BG}; color: {_WHITE};
}}
[data-testid="stSidebar"] {{
    background-color: {_BG};
    border-right: 1px solid {_SURFACE};
}}
[data-testid="stSidebar"] * {{ color: {_MUTED} !important; }}
[data-testid="stSidebar"] h1,
[data-testid="stSidebar"] h2,
[data-testid="stSidebar"] h3 {{ color: {_ACCENT} !important; }}
h1, h2, h3, h4 {{ color: {_WHITE} !important; }}

/* title accent bar */
.bs-title {{
    border-left: 4px solid {_ACCENT};
    padding-left: 14px;
    margin-bottom: 2px;
}}
.bs-brand {{
    color: {_MUTED};
    font-size: 0.76rem;
    letter-spacing: 0.09em;
    text-transform: uppercase;
    padding-left: 18px;
    margin-bottom: 22px;
}}

/* KPI card */
.kpi-card {{
    background: {_SURFACE};
    border-top: 3px solid {_ACCENT};
    border-radius: 6px;
    padding: 14px 18px;
    text-align: center;
}}
.kpi-label {{
    color: {_MUTED};
    font-size: 0.70rem;
    text-transform: uppercase;
    letter-spacing: 0.09em;
    margin-bottom: 5px;
}}
.kpi-value {{
    color: {_WHITE};
    font-size: 1.55rem;
    font-weight: 700;
    line-height: 1.1;
}}
.kpi-sub {{
    color: {_MUTED};
    font-size: 0.72rem;
    margin-top: 3px;
}}
.kpi-red   {{ border-top-color: {_RED}   !important; }}
.kpi-amber {{ border-top-color: {_AMBER} !important; }}
.kpi-green {{ border-top-color: {_GREEN} !important; }}

/* section header */
.section-header {{
    color: {_ACCENT};
    font-size: 0.72rem;
    font-weight: 700;
    letter-spacing: 0.11em;
    text-transform: uppercase;
    border-left: 3px solid {_ACCENT};
    padding-left: 8px;
    margin-bottom: 10px;
    margin-top: 6px;
}}

/* divider */
.bs-divider {{
    border: none;
    border-top: 1px solid {_SURFACE};
    margin: 20px 0;
}}

/* exception table */
[data-testid="stDataFrame"] {{ border: 1px solid {_SURFACE} !important; border-radius: 6px; }}

/* buttons */
.stButton > button {{
    background-color: {_SURFACE} !important;
    color: {_ACCENT} !important;
    border: 1px solid {_ACCENT} !important;
    border-radius: 5px !important;
}}
.stButton > button:hover {{
    background-color: {_ACCENT} !important;
    color: {_BG} !important;
}}
</style>
""", unsafe_allow_html=True)


# ── Plotly dark theme ─────────────────────────────────────────────────────────
_LAYOUT = dict(
    paper_bgcolor=_SURFACE,
    plot_bgcolor=_SURFACE,
    font_color=_MUTED,
    margin=dict(t=28, b=28, l=8, r=8),
    height=240,
    showlegend=True,
    legend=dict(font=dict(size=10), bgcolor="rgba(0,0,0,0)"),
)
_AXIS = dict(gridcolor="#334155", linecolor="#334155", tickfont=dict(size=10))
_COLORS = [_ACCENT, "#f6ad55", "#68d391", "#fc8181", "#b794f4", "#76e4f7"]


# ── Data loaders (cached) ─────────────────────────────────────────────────────
@st.cache_data(show_spinner=False)
def _otif() -> pd.DataFrame:
    df = load_otif_data()
    df["customer_request_date"] = pd.to_datetime(df["customer_request_date"])
    df["week"] = df["customer_request_date"].dt.to_period("W").astype(str)
    df["otif_flag"] = df["otif_flag"].map(
        lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
    )
    if "in_full" in df.columns:
        df["in_full"] = df["in_full"].map(
            lambda x: x if isinstance(x, bool) else str(x).strip().lower() == "true"
        )
    return df


@st.cache_data(show_spinner=False)
def _forecast() -> pd.DataFrame:
    return load_forecast_data()


@st.cache_data(show_spinner=False)
def _inventory() -> pd.DataFrame:
    df = load_inventory_data()
    if "days_of_cover" not in df.columns:
        df["days_of_cover"] = df.apply(
            lambda r: r["stock_on_hand_units"] / (r["avg_weekly_sales_units"] / 7)
            if r["avg_weekly_sales_units"] > 0 else 999, axis=1
        )
    df["weeks_of_cover"] = df["days_of_cover"] / 7
    df["annual_sales"] = df["avg_weekly_sales_units"] * 52
    df["inventory_turns"] = df.apply(
        lambda r: r["annual_sales"] / r["stock_on_hand_units"]
        if r["stock_on_hand_units"] > 0 else 0.0, axis=1
    )
    return df


@st.cache_data(show_spinner=False)
def _po() -> pd.DataFrame:
    df = load_purchase_orders_data()
    if "shipping_cost" in df.columns or "freight_cost" in df.columns:
        cost_cols = [c for c in ["shipping_cost", "freight_cost"] if c in df.columns]
        df["total_freight"] = df[cost_cols].fillna(0).sum(axis=1)
    else:
        df["total_freight"] = 0.0
    return df


@st.cache_data(show_spinner=False)
def _suppliers() -> pd.DataFrame:
    return load_suppliers_data()


# ── Helper: mini chart ────────────────────────────────────────────────────────
def _fig(h=240):
    fig = go.Figure()
    fig.update_layout(**{**_LAYOUT, "height": h})
    fig.update_xaxes(**_AXIS)
    fig.update_yaxes(**_AXIS)
    return fig


def _kpi(label, value, sub="", color="default"):
    cls = f"kpi-{color}" if color != "default" else ""
    return (
        f'<div class="kpi-card {cls}">'
        f'<div class="kpi-label">{label}</div>'
        f'<div class="kpi-value">{value}</div>'
        f'<div class="kpi-sub">{sub}</div>'
        f'</div>'
    )


# ── Load data ─────────────────────────────────────────────────────────────────
with st.spinner("Loading supply chain data..."):
    otif_df    = _otif()
    fc_df      = _forecast()
    inv_df     = _inventory()
    po_df      = _po()
    supp_df    = _suppliers()


# ── Header ────────────────────────────────────────────────────────────────────
st.markdown(
    '<div class="bs-title"><h1 style="margin:0;font-size:1.55rem;">'
    'Baselstrasse Supply Chain Copilot</h1></div>',
    unsafe_allow_html=True,
)
st.markdown(
    '<div class="bs-brand">Baselstrasse Co. LTD &nbsp;|&nbsp; Supply Chain Control Tower</div>',
    unsafe_allow_html=True,
)

# ── Sidebar: category filter ──────────────────────────────────────────────────
with st.sidebar:
    st.markdown(
        f'<h3 style="color:{_ACCENT};border-left:3px solid {_ACCENT};padding-left:10px;">'
        'Filters</h3>', unsafe_allow_html=True
    )
    categories = sorted(otif_df["product_type"].dropna().unique())
    selected = st.selectbox("Product type", ["All"] + categories)

    st.markdown('<hr style="border-color:#1e293b;"/>', unsafe_allow_html=True)
    st.page_link("pages/1_Copilot.py", label="Open Copilot →", icon="🤖")
    st.markdown(
        f'<p style="color:{_MUTED};font-size:0.73rem;margin-top:20px;">'
        'Baselstrasse Co. LTD | Supply Chain Intelligence</p>',
        unsafe_allow_html=True,
    )


def _filter(df: pd.DataFrame, col="product_type") -> pd.DataFrame:
    if selected != "All" and col in df.columns:
        return df[df[col].str.lower() == selected.lower()].copy()
    return df.copy()


otif_f  = _filter(otif_df)
fc_f    = _filter(fc_df)
inv_f   = _filter(inv_df)
po_f    = _filter(po_df)
supp_f  = _filter(supp_df)


# ═══════════════════════════════════════════════════════════════════════════════
# TOP — KPI CARDS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<div class="section-header">Key Performance Indicators</div>', unsafe_allow_html=True)

# Compute summary metrics
otif_rate    = round(otif_f["otif_flag"].mean() * 100, 1) if not otif_f.empty else None
fill_rate    = round(otif_f["in_full"].mean() * 100, 1) if "in_full" in otif_f.columns and not otif_f.empty else None
fc_valid     = fc_f[fc_f["actual_sales_qty"] != 0].copy() if not fc_f.empty else pd.DataFrame()
if not fc_valid.empty:
    fc_valid["ape"] = (fc_valid["forecast_qty"] - fc_valid["actual_sales_qty"]).abs() / fc_valid["actual_sales_qty"] * 100
    mape = round(fc_valid["ape"].mean(), 1)
    bias_val = round((fc_valid["forecast_qty"] - fc_valid["actual_sales_qty"]).mean(), 1)
else:
    mape = bias_val = None

# Supplier OTD from POs
if not po_f.empty:
    otd_rate = round((po_f["days_late"] <= 0).mean() * 100, 1)
    open_pos = int((po_f["po_status"].str.lower().isin({"open", "pending", "ordered", "in transit", "in-transit"})).sum())
    prem_freight = round(po_f["total_freight"].sum(), 0)
else:
    otd_rate = open_pos = prem_freight = None

# Inventory
if not inv_f.empty:
    avg_dos     = round(inv_f["days_of_cover"].mean(), 1)
    stockout_ct = int((inv_f["days_of_cover"] < 14).sum())
    excess_ct   = int((inv_f["days_of_cover"] > 90).sum())
else:
    avg_dos = stockout_ct = excess_ct = None


def _fmt(v, suffix="", prefix=""):
    return f"{prefix}{v:,.1f}{suffix}" if v is not None else "—"

def _color_otif(v):
    if v is None: return "default"
    return "green" if v >= 95 else "amber" if v >= 90 else "red"

def _color_mape(v):
    if v is None: return "default"
    return "green" if v <= 10 else "amber" if v <= 20 else "red"


k1, k2, k3, k4, k5, k6, k7, k8 = st.columns(8)
k1.markdown(_kpi("OTIF Rate",      _fmt(otif_rate, "%"),  "target ≥ 95%",   _color_otif(otif_rate)),  unsafe_allow_html=True)
k2.markdown(_kpi("Fill Rate",      _fmt(fill_rate, "%"),  "target ≥ 95%",   _color_otif(fill_rate)),  unsafe_allow_html=True)
k3.markdown(_kpi("Forecast MAPE",  _fmt(mape, "%"),       "target ≤ 10%",   _color_mape(mape)),       unsafe_allow_html=True)
k4.markdown(_kpi("Supplier OTD",   _fmt(otd_rate, "%"),   "target ≥ 90%",   _color_otif(otd_rate)),   unsafe_allow_html=True)
k5.markdown(_kpi("Avg DOS",        _fmt(avg_dos, "d"),    "target 30–60d",  "default"),                unsafe_allow_html=True)
k6.markdown(_kpi("Stockout Risk",  str(stockout_ct) if stockout_ct is not None else "—", "SKUs < 14d", "red" if stockout_ct else "green"), unsafe_allow_html=True)
k7.markdown(_kpi("Excess Stock",   str(excess_ct)   if excess_ct is not None else "—",   "SKUs > 90d", "amber" if excess_ct else "green"), unsafe_allow_html=True)
k8.markdown(_kpi("Open POs",       str(open_pos)    if open_pos is not None else "—",    "in flight",  "default"), unsafe_allow_html=True)


# ═══════════════════════════════════════════════════════════════════════════════
# MIDDLE ROW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="bs-divider"/>', unsafe_allow_html=True)
left_mid, right_mid = st.columns(2)

# ── MIDDLE LEFT: OTIF / Fill rate / Stockout trend ───────────────────────────
with left_mid:
    st.markdown('<div class="section-header">Service Level Trends</div>', unsafe_allow_html=True)

    if not otif_f.empty:
        weekly = (
            otif_f.groupby("week")
            .agg(
                otif_rate=("otif_flag", lambda x: round(x.mean() * 100, 2)),
                fill_rate=("in_full",   lambda x: round(x.mean() * 100, 2) if "in_full" in otif_f.columns else None),
                order_count=("otif_flag", "count"),
            )
            .reset_index()
            .sort_values("week")
            .tail(12)
        )

        # OTIF trend
        fig_otif = _fig()
        fig_otif.add_trace(go.Scatter(
            x=weekly["week"], y=weekly["otif_rate"],
            mode="lines+markers", name="OTIF %",
            line=dict(color=_ACCENT, width=2),
            marker=dict(size=5),
        ))
        if "fill_rate" in weekly.columns:
            fig_otif.add_trace(go.Scatter(
                x=weekly["week"], y=weekly["fill_rate"],
                mode="lines+markers", name="Fill Rate %",
                line=dict(color=_GREEN, width=2, dash="dot"),
                marker=dict(size=5),
            ))
        fig_otif.add_hline(y=95, line_dash="dash", line_color=_RED,
                           annotation_text="Target 95%", annotation_font_color=_RED,
                           annotation_font_size=9)
        fig_otif.update_layout(title=dict(text="OTIF & Fill Rate (weekly)", font=dict(size=12, color=_WHITE)),
                               yaxis_range=[0, 105])
        st.plotly_chart(fig_otif, use_container_width=True)

        # Stockout exposure trend — weekly count of orders NOT in-full
        if "in_full" in otif_f.columns:
            stockout_weekly = (
                otif_f[~otif_f["in_full"]]
                .groupby("week")
                .size()
                .reset_index(name="short_shipped_orders")
                .sort_values("week")
                .tail(12)
            )
            fig_so = _fig()
            fig_so.add_trace(go.Bar(
                x=stockout_weekly["week"],
                y=stockout_weekly["short_shipped_orders"],
                name="Short-shipped orders",
                marker_color=_RED,
            ))
            fig_so.update_layout(
                title=dict(text="Stockout / Short-Ship Exposure (weekly)", font=dict(size=12, color=_WHITE)),
                showlegend=False,
            )
            st.plotly_chart(fig_so, use_container_width=True)
    else:
        st.info("No OTIF data for selection.")


# ── MIDDLE RIGHT: Inventory turns / DOS / Excess trend ───────────────────────
with right_mid:
    st.markdown('<div class="section-header">Inventory Health</div>', unsafe_allow_html=True)

    if not inv_f.empty:
        # Turns by product type
        turns_grp = (
            inv_f.groupby("product_type")
            .agg(avg_turns=("inventory_turns", "mean"), sku_count=("sku", "count"))
            .reset_index()
            .sort_values("avg_turns")
        )
        fig_turns = _fig()
        fig_turns.add_trace(go.Bar(
            x=turns_grp["avg_turns"].round(1), y=turns_grp["product_type"],
            orientation="h", name="Avg turns/year",
            marker_color=_ACCENT,
            text=turns_grp["avg_turns"].round(1), textposition="outside",
            textfont=dict(size=10, color=_WHITE),
        ))
        fig_turns.add_vline(x=4, line_dash="dash", line_color=_AMBER,
                            annotation_text="Min 4x", annotation_font_color=_AMBER,
                            annotation_font_size=9)
        fig_turns.update_layout(
            title=dict(text="Inventory Turns by Category (turns/year)", font=dict(size=12, color=_WHITE)),
            showlegend=False, xaxis_title=None,
        )
        st.plotly_chart(fig_turns, use_container_width=True)

        # DOS distribution (stacked bar: Critical / Reorder / OK / Excess)
        def _dos_band(d):
            if d < 14:   return "Critical (<14d)"
            if d < 30:   return "Reorder (14–30d)"
            if d <= 90:  return "OK (30–90d)"
            return "Excess (>90d)"

        inv_f["dos_band"] = inv_f["days_of_cover"].apply(_dos_band)
        dos_grp = (
            inv_f.groupby(["product_type", "dos_band"])
            .size()
            .reset_index(name="sku_count")
        )
        band_order  = ["Critical (<14d)", "Reorder (14–30d)", "OK (30–90d)", "Excess (>90d)"]
        band_colors = {
            "Critical (<14d)":   _RED,
            "Reorder (14–30d)":  _AMBER,
            "OK (30–90d)":       _GREEN,
            "Excess (>90d)":     "#b794f4",
        }
        fig_dos = _fig()
        for band in band_order:
            sub = dos_grp[dos_grp["dos_band"] == band]
            fig_dos.add_trace(go.Bar(
                x=sub["product_type"], y=sub["sku_count"],
                name=band, marker_color=band_colors[band],
            ))
        fig_dos.update_layout(
            title=dict(text="Days of Supply by Category (SKU count)", font=dict(size=12, color=_WHITE)),
            barmode="stack",
            legend=dict(font=dict(size=9)),
        )
        st.plotly_chart(fig_dos, use_container_width=True)
    else:
        st.info("No inventory data for selection.")


# ═══════════════════════════════════════════════════════════════════════════════
# BOTTOM ROW
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="bs-divider"/>', unsafe_allow_html=True)
bot_left, bot_right = st.columns(2)

# ── BOTTOM LEFT: Forecast accuracy & bias by product family ──────────────────
with bot_left:
    st.markdown('<div class="section-header">Forecast Performance by Product Family</div>', unsafe_allow_html=True)

    if not fc_f.empty:
        fc_valid2 = fc_f[fc_f["actual_sales_qty"] != 0].copy()
        fc_valid2["ape"] = (fc_valid2["forecast_qty"] - fc_valid2["actual_sales_qty"]).abs() / fc_valid2["actual_sales_qty"] * 100
        fc_valid2["bias"] = fc_valid2["forecast_qty"] - fc_valid2["actual_sales_qty"]

        by_cat = (
            fc_valid2.groupby("product_type")
            .agg(mape=("ape", "mean"), bias=("bias", "mean"))
            .reset_index()
            .sort_values("mape", ascending=False)
        )

        # MAPE bar
        fig_mape = _fig()
        fig_mape.add_trace(go.Bar(
            x=by_cat["product_type"],
            y=by_cat["mape"].round(1),
            name="MAPE %",
            marker_color=[_RED if v > 20 else _AMBER if v > 10 else _GREEN for v in by_cat["mape"]],
            text=by_cat["mape"].round(1).astype(str) + "%",
            textposition="outside",
            textfont=dict(size=10, color=_WHITE),
        ))
        fig_mape.add_hline(y=10, line_dash="dash", line_color=_AMBER,
                           annotation_text="Target 10%", annotation_font_color=_AMBER,
                           annotation_font_size=9)
        fig_mape.update_layout(
            title=dict(text="Forecast MAPE by Product Family", font=dict(size=12, color=_WHITE)),
            showlegend=False, yaxis_title="MAPE %",
        )
        st.plotly_chart(fig_mape, use_container_width=True)

        # Bias bar (over vs under)
        fig_bias = _fig()
        fig_bias.add_trace(go.Bar(
            x=by_cat["product_type"],
            y=by_cat["bias"].round(1),
            name="Bias (units)",
            marker_color=[_RED if v > 0 else _GREEN for v in by_cat["bias"]],
            text=by_cat["bias"].round(1),
            textposition="outside",
            textfont=dict(size=10, color=_WHITE),
        ))
        fig_bias.add_hline(y=0, line_color=_MUTED, line_width=1)
        fig_bias.update_layout(
            title=dict(text="Forecast Bias by Product Family (+over / −under)", font=dict(size=12, color=_WHITE)),
            showlegend=False, yaxis_title="Bias (units)",
        )
        st.plotly_chart(fig_bias, use_container_width=True)
    else:
        st.info("No forecast data for selection.")


# ── BOTTOM RIGHT: Supplier OTD / lead time variability / freight ─────────────
with bot_right:
    st.markdown('<div class="section-header">Supplier Performance</div>', unsafe_allow_html=True)

    if not po_f.empty:
        # OTD by supplier
        otd_by_sup = (
            po_f.groupby("supplier_name")["days_late"]
            .apply(lambda x: round((x <= 0).mean() * 100, 1))
            .reset_index(name="otd_rate")
            .sort_values("otd_rate")
        )
        fig_otd = _fig()
        fig_otd.add_trace(go.Bar(
            x=otd_by_sup["otd_rate"], y=otd_by_sup["supplier_name"],
            orientation="h", name="OTD %",
            marker_color=[_RED if v < 80 else _AMBER if v < 90 else _GREEN for v in otd_by_sup["otd_rate"]],
            text=otd_by_sup["otd_rate"].astype(str) + "%", textposition="outside",
            textfont=dict(size=10, color=_WHITE),
        ))
        fig_otd.add_vline(x=90, line_dash="dash", line_color=_AMBER,
                          annotation_text="Target 90%", annotation_font_color=_AMBER,
                          annotation_font_size=9)
        fig_otd.update_layout(
            title=dict(text="Supplier On-Time Delivery Rate", font=dict(size=12, color=_WHITE)),
            showlegend=False, xaxis_range=[0, 115],
        )
        st.plotly_chart(fig_otd, use_container_width=True)

        # Lead time variability (std dev of days_late) + premium freight
        lt_var = (
            po_f.groupby("supplier_name")
            .agg(
                std_days=("days_late", "std"),
                total_freight=("total_freight", "sum"),
                po_count=("po_id", "count"),
            )
            .reset_index()
            .fillna(0)
            .sort_values("std_days", ascending=False)
        )

        fig_ltv = _fig()
        fig_ltv.add_trace(go.Bar(
            x=lt_var["supplier_name"],
            y=lt_var["std_days"].round(1),
            name="Delay std dev (days)",
            marker_color=[_RED if v > 7 else _AMBER if v > 3 else _GREEN for v in lt_var["std_days"]],
            yaxis="y1",
        ))
        fig_ltv.add_trace(go.Scatter(
            x=lt_var["supplier_name"],
            y=lt_var["total_freight"].round(0),
            mode="lines+markers",
            name="Total freight ($)",
            line=dict(color=_ACCENT, width=2),
            marker=dict(size=6),
            yaxis="y2",
        ))
        fig_ltv.update_layout(
            title=dict(text="Lead Time Variability & Premium Freight by Supplier", font=dict(size=12, color=_WHITE)),
            yaxis=dict(title="Delay std dev (days)", **_AXIS),
            yaxis2=dict(title="Total freight ($)", overlaying="y", side="right", **_AXIS),
            legend=dict(font=dict(size=9), bgcolor="rgba(0,0,0,0)"),
            height=240,
        )
        st.plotly_chart(fig_ltv, use_container_width=True)
    else:
        st.info("No purchase order data for selection.")


# ═══════════════════════════════════════════════════════════════════════════════
# FOOTER — TOP 10 EXCEPTIONS
# ═══════════════════════════════════════════════════════════════════════════════
st.markdown('<hr class="bs-divider"/>', unsafe_allow_html=True)
st.markdown('<div class="section-header">Top 10 Exceptions — SKU / Supplier</div>', unsafe_allow_html=True)

exceptions = []

# SKU-level: low OTIF
if not otif_f.empty and "sku" in otif_f.columns:
    sku_otif = (
        otif_f.groupby(["sku", "product_type"])["otif_flag"]
        .agg(otif_rate=lambda x: round(x.mean() * 100, 1), orders=("count"))
        .reset_index()
        .rename(columns={"otif_rate": "value", "orders": "volume"})
    )
    sku_otif["exception"] = "Low OTIF"
    sku_otif["metric"] = sku_otif["value"].astype(str) + "% OTIF"
    sku_otif["entity"] = sku_otif["sku"]
    sku_otif["severity"] = sku_otif["value"].apply(
        lambda v: "High" if v < 80 else "Medium" if v < 90 else "Low"
    )
    sku_otif = sku_otif[sku_otif["value"] < 90].sort_values("value").head(5)
    exceptions.append(sku_otif[["entity", "product_type", "exception", "metric", "volume", "severity"]])

# Inventory: critical DOS
if not inv_f.empty:
    crit = inv_f[inv_f["days_of_cover"] < 14].copy()
    crit["entity"] = crit["sku"]
    crit["exception"] = "Critical DOS"
    crit["metric"] = crit["days_of_cover"].round(1).astype(str) + "d cover"
    crit["volume"] = crit["stock_on_hand_units"].round(0).astype(int)
    crit["severity"] = "High"
    exceptions.append(crit[["entity", "product_type", "exception", "metric", "volume", "severity"]].head(5))

# Supplier: high delay
if not po_f.empty:
    sup_delay = (
        po_f[po_f["days_late"] > 0]
        .groupby("supplier_name")
        .agg(late_pos=("po_id", "count"), avg_delay=("days_late", "mean"))
        .reset_index()
        .sort_values("avg_delay", ascending=False)
        .head(5)
    )
    sup_delay["entity"] = sup_delay["supplier_name"]
    sup_delay["product_type"] = "—"
    sup_delay["exception"] = "Supplier Delay"
    sup_delay["metric"] = sup_delay["avg_delay"].round(1).astype(str) + "d avg late"
    sup_delay["volume"] = sup_delay["late_pos"]
    sup_delay["severity"] = sup_delay["avg_delay"].apply(
        lambda v: "High" if v > 10 else "Medium" if v > 5 else "Low"
    )
    exceptions.append(sup_delay[["entity", "product_type", "exception", "metric", "volume", "severity"]])

if exceptions:
    exc_df = pd.concat(exceptions, ignore_index=True).head(10)

    # Colour-code severity in the severity column label
    def _sev_icon(s):
        return {"High": "🔴 High", "Medium": "🟡 Medium", "Low": "🟢 Low"}.get(s, s)

    exc_df["severity"] = exc_df["severity"].apply(_sev_icon)
    exc_df.columns = ["Entity", "Category", "Exception Type", "Metric", "Volume", "Severity"]

    st.dataframe(
        exc_df,
        use_container_width=True,
        hide_index=True,
        column_config={
            "Severity":       st.column_config.TextColumn(width="small"),
            "Exception Type": st.column_config.TextColumn(width="medium"),
            "Metric":         st.column_config.TextColumn(width="medium"),
            "Volume":         st.column_config.NumberColumn(width="small"),
        },
    )
else:
    st.success("No exceptions detected for the current selection.")
