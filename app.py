import gzip
import re
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

warnings.filterwarnings("ignore", category=FutureWarning)

# ============================================================
# ATMOS | WEATHER INTELLIGENCE DASHBOARD
# Works with compressed_data (1).csv.gz (Global Weather Repository style)
# ============================================================

st.set_page_config(
    page_title="ATMOS | Weather Intelligence",
    page_icon="🌍",
    layout="wide",
    initial_sidebar_state="expanded",
)

# ----------------------------- DESIGN TOKENS -----------------------------
BG = "#07111f"
PANEL = "#0d1b2e"
BORDER = "#1d3550"
TEXT = "#e8eef7"
MUTED = "#8fa5bb"
GRID = "#1f3249"
ACCENT = "#38bdf8"
COLORWAY = ["#38bdf8", "#2dd4bf", "#fbbf24", "#f472b6", "#a78bfa", "#fb923c", "#4ade80", "#f87171"]
SEQ_WARM = ["#fde68a", "#fbbf24", "#fb923c", "#ef4444", "#991b1b"]
SEQ_COOL = ["#0b2a4a", "#155e92", "#38bdf8", "#a5f3fc"]
SEQ_PURPLE = ["#ede9fe", "#a78bfa", "#7c3aed", "#4c1d95"]

st.markdown(
    f"""
    <style>
    .stApp {{ background: {BG}; color: {TEXT}; }}
    [data-testid="stSidebar"] {{ background: #0a1526; border-right: 1px solid {BORDER}; }}
    .block-container {{ padding-top: 1.2rem; padding-bottom: 2rem; max-width: 1650px; }}
    h1, h2, h3, h4 {{ color: #f4f7fb !important; letter-spacing: -0.01em; }}
    .hero {{
        padding: 24px 30px; border: 1px solid {BORDER}; border-radius: 16px;
        background: linear-gradient(120deg, #10243b 0%, #0f2f3d 100%); margin-bottom: 18px;
    }}
    .hero h1 {{ margin: 0 0 6px 0; font-size: 1.8rem; }}
    .hero p {{ color: #b7c8d9; font-size: 1rem; margin: 0; }}
    .chips {{ margin-top: 12px; }}
    .chip {{
        display: inline-block; padding: 3px 11px; margin-right: 8px; border-radius: 999px;
        background: rgba(56,189,248,.12); border: 1px solid rgba(56,189,248,.35);
        color: #9bdcf8; font-size: .78rem;
    }}
    .metric {{
        background: {PANEL}; border: 1px solid {BORDER}; border-top: 3px solid {ACCENT};
        border-radius: 12px; padding: 14px 16px; min-height: 108px;
    }}
    .metric-label {{ color: {MUTED}; font-size: .78rem; text-transform: uppercase; letter-spacing: .06em; }}
    .metric-value {{ color: #f5f8fc; font-size: 1.65rem; font-weight: 700; margin-top: 6px; }}
    .metric-note {{ color: #55d6a2; font-size: .76rem; margin-top: 3px; }}
    .section-title {{ margin: 22px 0 2px 0; font-size: 1.15rem; font-weight: 650; color: #f4f7fb; }}
    .section-sub {{ color: {MUTED}; font-size: .86rem; margin-bottom: 10px; }}
    .insights {{
        background: {PANEL}; border: 1px solid {BORDER}; border-left: 3px solid #2dd4bf;
        border-radius: 12px; padding: 14px 20px; margin: 14px 0 6px 0;
    }}
    .insights li {{ margin-bottom: 5px; }}
    div[data-testid="stPlotlyChart"] {{
        background: {PANEL}; border: 1px solid {BORDER}; border-radius: 14px; padding: 6px;
    }}
    div[data-testid="stDataFrame"] {{ border: 1px solid {BORDER}; border-radius: 10px; }}
    button[data-baseweb="tab"] {{ color: {MUTED}; }}
    button[data-baseweb="tab"][aria-selected="true"] {{ color: #f4f7fb; }}
    </style>
    """,
    unsafe_allow_html=True,
)

PROJECT_DIR = Path(__file__).resolve().parent

# -------------------------- FIND DATASET --------------------------
PREFERRED_DATASETS = [
    PROJECT_DIR / "compressed_data (1).csv.gz",
    PROJECT_DIR / "compressed_data.csv.gz",
    PROJECT_DIR / "weather_data.csv",
    PROJECT_DIR / "weather.csv",
    PROJECT_DIR / "data" / "compressed_data (1).csv.gz",
    PROJECT_DIR / "data" / "compressed_data.csv.gz",
    PROJECT_DIR / "data" / "weather_data.csv",
    PROJECT_DIR / "data" / "weather.csv",
]


def find_dataset():
    for file_path in PREFERRED_DATASETS:
        if file_path.exists():
            return file_path

    ignored_dirs = {".venv", "venv", "env", "__pycache__", ".git", "node_modules"}
    matches = []
    for file_path in PROJECT_DIR.rglob("*"):
        if any(part in ignored_dirs for part in file_path.parts):
            continue
        name = file_path.name.lower()
        if file_path.is_file() and (name.endswith(".csv") or name.endswith(".csv.gz")):
            matches.append(file_path)

    matches.sort(key=lambda p: (0 if "compressed_data" in p.name.lower() else 1, len(str(p))))
    return matches[0] if matches else None


# --------------------------- LOAD DATA ---------------------------
def normalize_name(name) -> str:
    """'air_quality_PM2.5' -> 'air_quality_pm2_5', 'latitude+Y1C1:AG1' -> 'latitude_y1c1_ag1'."""
    return re.sub(r"[^0-9a-z]+", "_", str(name).strip().lower()).strip("_")


ALIASES = {
    "Country": ["country", "country_name"],
    "City": ["location_name", "city", "city_name", "location"],
    "Date": ["last_updated", "date", "datetime", "date_time", "time"],
    "Latitude": ["latitude", "lat", "latitude_deg"],
    "Longitude": ["longitude", "lon", "lng", "long"],
    "Temperature_C": ["temperature_celsius", "temperature_c", "temp_c", "temperature", "temp"],
    "Humidity_pct": ["humidity_pct", "humidity", "relative_humidity"],
    "Precipitation_mm": ["precip_mm", "precipitation_mm", "precipitation", "rainfall_mm", "rain_mm"],
    "Wind_Speed_kmh": ["wind_kph", "wind_speed_kmh", "wind_speed", "windspeed_kph"],
    "Weather_Condition": ["condition_text", "weather_condition", "condition", "weather"],
}

AIR_ALIASES = {
    "air_quality_pm2_5": ["air_quality_pm2_5", "air_quality_pm25", "pm2_5", "pm25"],
    "air_quality_pm10": ["air_quality_pm10", "pm10"],
    "air_quality_nitrogen_dioxide": ["air_quality_nitrogen_dioxide", "air_quality_no2", "no2"],
    "air_quality_ozone": ["air_quality_ozone", "air_quality_o3", "o3"],
    "air_quality_carbon_monoxide": ["air_quality_carbon_monoxide", "air_quality_co", "co"],
    "air_quality_sulphur_dioxide": ["air_quality_sulphur_dioxide", "air_quality_so2", "so2"],
}

EXTRA_NUMERIC = ["pressure_mb", "cloud", "uv_index", "visibility_km", "feels_like_celsius", "gust_kph"]


@st.cache_data(show_spinner="Loading weather dataset...")
def load_dataset(path_string: str) -> pd.DataFrame:
    file_path = Path(path_string)

    if file_path.name.lower().endswith(".gz"):
        with gzip.open(file_path, "rt", encoding="utf-8", errors="replace") as handle:
            data = pd.read_csv(handle, low_memory=False)
    else:
        data = pd.read_csv(file_path, low_memory=False)

    data.columns = [normalize_name(c) for c in data.columns]
    data = data.loc[:, ~data.columns.duplicated()].copy()

    # Some files have a mangled latitude header (e.g. "latitude+Y1C1:AG1").
    if "latitude" not in data.columns:
        for col in data.columns:
            if col.startswith("latitude"):
                data.rename(columns={col: "latitude"}, inplace=True)
                break

    for target, candidates in {**ALIASES, **AIR_ALIASES}.items():
        if target in data.columns:
            continue
        for candidate in candidates:
            if candidate in data.columns:
                data.rename(columns={candidate: target}, inplace=True)
                break

    for col in ALIASES:
        if col not in data.columns:
            data[col] = np.nan

    numeric_columns = [
        "Latitude", "Longitude", "Temperature_C", "Humidity_pct",
        "Precipitation_mm", "Wind_Speed_kmh",
    ] + EXTRA_NUMERIC + list(AIR_ALIASES)
    for col in numeric_columns:
        if col in data.columns:
            data[col] = pd.to_numeric(data[col], errors="coerce")

    data["Date"] = pd.to_datetime(data["Date"], errors="coerce")

    for col in ["Country", "City", "Weather_Condition"]:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip()
        data.loc[data[col].eq(""), col] = "Unknown"

    return data


dataset_path = find_dataset()
if dataset_path is None:
    st.error("Dataset not found. Put compressed_data (1).csv.gz in the same folder as app.py.")
    st.stop()

try:
    df = load_dataset(str(dataset_path))
except Exception as error:
    st.error(f"Dataset loading failed: {error}")
    st.stop()

if df.empty:
    st.error("The selected dataset contains no records.")
    st.stop()

if df["Temperature_C"].notna().sum() == 0:
    st.error("Temperature data could not be found or converted.")
    st.write("Dataset columns:", df.columns.tolist())
    st.stop()

# --------------------------- WEATHER STATUS ---------------------------
SEVERITY_LEVELS = ["Normal", "Warning", "Extreme"]
SEVERITY_STYLE = {
    "Normal": {"emoji": "🟢", "color": "#2ecc71", "size": 5},
    "Warning": {"emoji": "🟡", "color": "#f1c40f", "size": 7},
    "Extreme": {"emoji": "🔴", "color": "#e74c3c", "size": 9},
}
SEVERITY_COLORS = {k: v["color"] for k, v in SEVERITY_STYLE.items()}


def _col(data, name):
    if name in data.columns:
        return data[name]
    return pd.Series(np.nan, index=data.index)


def classify_weather(data):
    """Label each observation Normal / Warning / Extreme from its measurements."""
    temp = _col(data, "Temperature_C")
    wind = _col(data, "Wind_Speed_kmh")
    rain = _col(data, "Precipitation_mm")
    cond = data["Weather_Condition"].astype(str).str.lower()

    extreme = (
        (temp >= 40) | (temp <= -20) | (wind >= 60) | (rain >= 25)
        | cond.str.contains(r"blizzard|torrential|thunder|tornado|cyclone|hurricane", regex=True)
    )
    warning = (
        (temp >= 35) | (temp <= -10) | (wind >= 40) | (rain >= 10)
        | cond.str.contains(r"heavy|freezing|snow|sleet|ice pellets|fog|moderate rain", regex=True)
    )
    return pd.Series(
        np.select([extreme, warning], ["Extreme", "Warning"], default="Normal"),
        index=data.index,
    )


# --------------------------- SIDEBAR ---------------------------
st.sidebar.markdown("# 🌍 ATMOS")
st.sidebar.caption("WEATHER INTELLIGENCE PLATFORM")
st.sidebar.markdown("---")
st.sidebar.success(f"Dataset loaded: {dataset_path.name}")
st.sidebar.caption(f"{len(df):,} records")

country_values = sorted(v for v in df["Country"].dropna().unique().tolist() if v != "Unknown")
selected_countries = st.sidebar.multiselect(
    "Filter by country", country_values, default=[],
    help="Leave blank to include all countries.",
)

working = df.copy()
if selected_countries:
    working = working[working["Country"].isin(selected_countries)].copy()

city_values = sorted(v for v in working["City"].dropna().unique().tolist() if v != "Unknown")
selected_cities = st.sidebar.multiselect(
    "Filter by city", city_values, default=[],
    help="Leave blank to include all cities.",
)
if selected_cities:
    working = working[working["City"].isin(selected_cities)].copy()

if working["Date"].notna().any():
    date_min = working["Date"].min().date()
    date_max = working["Date"].max().date()
    chosen_dates = st.sidebar.date_input(
        "Date range", value=(date_min, date_max), min_value=date_min, max_value=date_max,
    )
    if isinstance(chosen_dates, (tuple, list)) and len(chosen_dates) == 2:
        start_date, end_date = chosen_dates
        working = working[
            working["Date"].isna() | working["Date"].dt.date.between(start_date, end_date)
        ].copy()

st.sidebar.markdown("---")
page = st.sidebar.radio(
    "DASHBOARD PAGES",
    [
        "Global Overview",
        "Temperature Analysis",
        "Rainfall & Climate",
        "Weather Conditions",
        "Air Quality",
        "Data Explorer",
    ],
)
st.sidebar.caption("ATMOS • Global weather analytics")

if working.empty:
    st.warning("No records match the selected filters. Clear the filters in the sidebar.")
    st.stop()

working["Severity"] = classify_weather(working)
is_filtered = bool(selected_countries or selected_cities)


# --------------------------- HELPERS ---------------------------
def fnum(value, decimals=1, unit=""):
    if value is None or pd.isna(value):
        return "N/A"
    return f"{value:,.{decimals}f}{unit}"


def metric_card(label, value, note="", color=ACCENT):
    st.markdown(
        f'<div class="metric" style="border-top-color:{color}">'
        f'<div class="metric-label">{label}</div>'
        f'<div class="metric-value">{value}</div>'
        f'<div class="metric-note">{note}</div></div>',
        unsafe_allow_html=True,
    )


def kpi_row(items):
    cols = st.columns(len(items))
    for col, item in zip(cols, items):
        with col:
            metric_card(*item)
    st.write("")


def section(title, subtitle=""):
    st.markdown(f'<div class="section-title">{title}</div>', unsafe_allow_html=True)
    if subtitle:
        st.markdown(f'<div class="section-sub">{subtitle}</div>', unsafe_allow_html=True)


def style_figure(fig, height=380):
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=COLORWAY,
        font=dict(color="#dce7f3", family="Inter, Segoe UI, sans-serif", size=12),
        title=dict(font=dict(size=15, color="#f4f7fb"), x=0.01, xanchor="left"),
        margin=dict(l=14, r=14, t=56, b=16),
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11)),
        hoverlabel=dict(bgcolor="#0f2237", bordercolor=BORDER, font=dict(color="#f1f5f9")),
        bargap=0.18,
    )
    fig.update_xaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(size=12))
    fig.update_yaxes(gridcolor=GRID, zerolinecolor=GRID, linecolor=GRID, title_font=dict(size=12))
    return fig


def show_chart(fig, height=380):
    st.plotly_chart(
        style_figure(fig, height), use_container_width=True, theme=None,
        config={"displaylogo": False, "modeBarButtonsToRemove": ["lasso2d", "select2d"]},
    )


def valid_numeric(column):
    if column in working.columns:
        return working[column].dropna()
    return pd.Series(dtype=float)


def sample_frame(data, limit=8000):
    return data.sample(limit, random_state=7) if len(data) > limit else data


def known(data, column):
    return data[data[column] != "Unknown"]


def add_trend_line(fig, x, y, name="Linear trend"):
    mask = x.notna() & y.notna()
    if mask.sum() > 2 and x[mask].nunique() > 1:
        coef = np.polyfit(x[mask], y[mask], 1)
        xs = np.linspace(x[mask].min(), x[mask].max(), 60)
        fig.add_trace(
            go.Scatter(
                x=xs, y=np.polyval(coef, xs), mode="lines", name=name,
                line=dict(color="#fbbf24", width=2.5, dash="dash"),
            )
        )
    return fig


def ranked_bar(data, category, value, title, label, top=15, ascending=False,
               scale=SEQ_WARM, horizontal=True, fmt=".1f"):
    ranked = (
        data.dropna(subset=[value]).groupby(category, as_index=False)[value].mean()
        .sort_values(value, ascending=ascending).head(top)
    )
    if ranked.empty:
        st.info("Not enough data for this chart.")
        return
    if horizontal:
        fig = px.bar(
            ranked, x=value, y=category, orientation="h", title=title,
            labels={value: label, category: ""}, color=value,
            color_continuous_scale=scale, text=value,
        )
        fig.update_layout(yaxis={"categoryorder": "total ascending"})
    else:
        fig = px.bar(
            ranked, x=category, y=value, title=title,
            labels={value: label, category: ""}, color=value,
            color_continuous_scale=scale, text=value,
        )
    fig.update_traces(texttemplate=f"%{{text:{fmt}}}", textposition="outside", cliponaxis=False)
    fig.update_coloraxes(showscale=False)
    show_chart(fig, 430 if horizontal else 380)


def correlation_heatmap(data, columns, labels, title):
    present = [c for c in columns if c in data.columns and data[c].notna().sum() > 2]
    if len(present) < 2:
        st.info("Not enough numeric columns for a correlation matrix.")
        return
    corr = data[present].corr()
    corr.index = [labels.get(c, c) for c in present]
    corr.columns = [labels.get(c, c) for c in present]
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
        color_continuous_scale=["#f87171", "#0d1b2e", "#38bdf8"], title=title,
    )
    fig.update_coloraxes(colorbar=dict(title="r", thickness=12))
    show_chart(fig, 430)


def make_globe(data, height=620):
    """Interactive 3D Earth (drag to rotate, scroll to zoom) with severity markers."""
    points = data.dropna(subset=["Latitude", "Longitude"]).copy()
    points = points[
        points["Latitude"].between(-90, 90) & points["Longitude"].between(-180, 180)
    ]

    if not points.empty:
        points = points.sort_values("Date", na_position="first").drop_duplicates(
            subset=["City", "Country", "Latitude", "Longitude"], keep="last"
        )
        if len(points) > 6000:
            points = points.sample(6000, random_state=7)
        points["Severity"] = classify_weather(points)

    fig = go.Figure()
    counts = {level: 0 for level in SEVERITY_LEVELS}

    for level in SEVERITY_LEVELS:
        style = SEVERITY_STYLE[level]
        subset = points[points["Severity"] == level] if not points.empty else points
        counts[level] = len(subset)
        if subset.empty:
            continue
        hover = (
            "<b>" + subset["City"].astype(str) + "</b>, " + subset["Country"].astype(str)
            + "<br>Status: " + style["emoji"] + " " + level
            + "<br>Condition: " + subset["Weather_Condition"].astype(str)
            + "<br>Temperature: " + subset["Temperature_C"].round(1).astype(str) + " °C"
            + "<br>Wind: " + _col(subset, "Wind_Speed_kmh").round(1).astype(str) + " km/h"
            + "<br>Rain: " + _col(subset, "Precipitation_mm").round(1).astype(str) + " mm"
        )
        fig.add_trace(
            go.Scattergeo(
                lon=subset["Longitude"], lat=subset["Latitude"], mode="markers",
                name=f"{style['emoji']} {level} ({len(subset):,})",
                text=hover.to_numpy(), hovertemplate="%{text}<extra></extra>",
                marker=dict(
                    size=style["size"], color=style["color"], opacity=0.92,
                    line=dict(width=0.6, color=BG),
                ),
            )
        )

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=78, lat=20, roll=0),
        showocean=True, oceancolor="#0a2540",
        showland=True, landcolor="#164563",
        showcountries=True, countrycolor="#2f6285",
        showcoastlines=True, coastlinecolor="#5b93b5",
        showlakes=False, showframe=False,
        lataxis=dict(showgrid=True, gridcolor="#1c3a57"),
        lonaxis=dict(showgrid=True, gridcolor="#1c3a57"),
        bgcolor="rgba(0,0,0,0)",
    )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=5, b=0),
        font=dict(color="#dce7f3"),
        legend=dict(
            x=0.01, y=0.99, bgcolor="rgba(11,23,40,0.75)",
            bordercolor=BORDER, borderwidth=1, itemsizing="constant",
        ),
        dragmode="pan",
    )
    return fig, counts


BOLD = re.compile(r"\*\*(.+?)\*\*")


def build_insights(data):
    items = []
    places = known(data, "Country")
    if not places.empty:
        by_country = places.groupby("Country")
        temp = by_country["Temperature_C"].mean().dropna()
        if len(temp) > 1:
            items.append(
                f"🌡️ **Warmest country:** {temp.idxmax()} ({temp.max():.1f} °C average) · "
                f"**coolest:** {temp.idxmin()} ({temp.min():.1f} °C)."
            )
        rain = by_country["Precipitation_mm"].mean().dropna()
        if len(rain) > 1 and rain.max() > 0:
            items.append(f"🌧️ **Wettest country:** {rain.idxmax()} ({rain.max():.2f} mm average per observation).")
        wind = by_country["Wind_Speed_kmh"].mean().dropna()
        if len(wind) > 1:
            items.append(f"💨 **Windiest country:** {wind.idxmax()} ({wind.max():.1f} km/h average).")
    share = data["Severity"].value_counts(normalize=True) * 100
    items.append(
        f"🚨 **{share.get('Extreme', 0):.1f}%** of observations are Extreme and "
        f"**{share.get('Warning', 0):.1f}%** are Warning-level."
    )
    pair = data[["Temperature_C", "Humidity_pct"]].dropna()
    if len(pair) > 30:
        r = pair["Temperature_C"].corr(pair["Humidity_pct"])
        if pd.notna(r):
            direction = "fall" if r < 0 else "rise"
            strength = "strongly" if abs(r) >= 0.5 else "moderately" if abs(r) >= 0.3 else "weakly"
            items.append(
                f"🔗 Humidity tends to {direction} {strength} as temperature rises (correlation {r:+.2f})."
            )
    return items


# ----------------------------- HEADER -----------------------------
if working["Date"].notna().any():
    period = f"{working['Date'].min():%d %b %Y} → {working['Date'].max():%d %b %Y}"
else:
    period = "Date range unavailable"
n_countries = working.loc[working["Country"] != "Unknown", "Country"].nunique()
n_cities = working.loc[working["City"] != "Unknown", "City"].nunique()

st.markdown(
    f"""
    <div class="hero">
      <h1>🌍 ATMOS Weather Intelligence</h1>
      <p>Global weather monitoring, climate analysis and interactive 3D geographical visualization.</p>
      <div class="chips">
        <span class="chip">{len(working):,} observations</span>
        <span class="chip">{n_countries:,} countries</span>
        <span class="chip">{n_cities:,} cities</span>
        <span class="chip">{period}</span>
      </div>
    </div>
    """,
    unsafe_allow_html=True,
)

temperature = valid_numeric("Temperature_C")
humidity = valid_numeric("Humidity_pct")
rainfall = valid_numeric("Precipitation_mm")
wind_speed = valid_numeric("Wind_Speed_kmh")
global_mean_temp = df["Temperature_C"].mean()

# =========================== OVERVIEW ===========================
if page == "Global Overview":
    extreme_share = (working["Severity"] == "Extreme").mean() * 100
    temp_note = "Mean recorded temperature"
    if is_filtered and pd.notna(global_mean_temp):
        temp_note = f"{temperature.mean() - global_mean_temp:+.1f} °C vs global average"

    kpi_row([
        ("Weather Records", f"{len(working):,}", "Filtered observations", "#38bdf8"),
        ("Countries", f"{n_countries:,}", "Countries covered", "#2dd4bf"),
        ("Cities", f"{n_cities:,}", "Locations covered", "#a78bfa"),
        ("Avg Temperature", fnum(temperature.mean(), 1, " °C"), temp_note, "#fb923c"),
        ("Avg Humidity", fnum(humidity.mean(), 1, "%"), "Relative humidity", "#38bdf8"),
        ("Extreme Share", f"{extreme_share:.1f}%", "Observations at Extreme level", "#f87171"),
    ])

    insight_html = "".join(
        "<li>" + BOLD.sub(r"<b>\1</b>", item) + "</li>" for item in build_insights(working)
    )
    st.markdown(
        '<div class="insights"><b>Key insights</b><ul>' + insight_html + "</ul></div>",
        unsafe_allow_html=True,
    )

    section("🌐 Interactive 3D Earth Globe",
            "Drag to rotate, scroll to zoom, hover a marker for details. One marker per location (latest observation).")
    shown_levels = st.multiselect(
        "Show weather status", SEVERITY_LEVELS, default=SEVERITY_LEVELS,
        format_func=lambda lvl: f"{SEVERITY_STYLE[lvl]['emoji']} {lvl}",
    )
    globe_data = working
    if shown_levels and len(shown_levels) < len(SEVERITY_LEVELS):
        globe_data = working[working["Severity"].isin(shown_levels)]

    globe, level_counts = make_globe(globe_data)
    st.plotly_chart(globe, use_container_width=True, theme=None, config={"displaylogo": False})
    kpi_row([
        ("🟢 Normal", f"{level_counts['Normal']:,}", "Locations", SEVERITY_COLORS["Normal"]),
        ("🟡 Warning", f"{level_counts['Warning']:,}", "Locations", SEVERITY_COLORS["Warning"]),
        ("🔴 Extreme", f"{level_counts['Extreme']:,}", "Locations", SEVERITY_COLORS["Extreme"]),
    ])
    with st.expander("How statuses are decided"):
        st.markdown(
            "- 🔴 **Extreme**: temperature ≥ 40 °C or ≤ −20 °C, wind ≥ 60 km/h, rain ≥ 25 mm, "
            "or a blizzard / thunderstorm / torrential / cyclone-type condition.\n"
            "- 🟡 **Warning**: temperature ≥ 35 °C or ≤ −10 °C, wind ≥ 40 km/h, rain ≥ 10 mm, "
            "or heavy rain / snow / sleet / freezing / fog conditions.\n"
            "- 🟢 **Normal**: everything else."
        )

    section("Temperature & status mix")
    left, right = st.columns([3, 2])
    with left:
        fig = px.histogram(
            working.dropna(subset=["Temperature_C"]), x="Temperature_C", nbins=40,
            title="Distribution of recorded temperatures",
            labels={"Temperature_C": "Temperature (°C)"}, color_discrete_sequence=[ACCENT],
        )
        fig.add_vline(
            x=temperature.median(), line_dash="dash", line_color="#fbbf24",
            annotation_text=f"Median {temperature.median():.1f} °C", annotation_font_color="#fbbf24",
        )
        fig.update_yaxes(title="Observations")
        show_chart(fig)
    with right:
        mix = (
            working["Severity"].value_counts().reindex(SEVERITY_LEVELS).fillna(0)
            .rename_axis("Status").reset_index(name="Records")
        )
        fig = px.pie(
            mix, names="Status", values="Records", hole=0.62, title="Weather status share",
            color="Status", color_discrete_map=SEVERITY_COLORS,
        )
        fig.update_traces(textinfo="percent", textfont_size=13, sort=False)
        show_chart(fig)

    section("Country & city rankings")
    left, right = st.columns(2)
    with left:
        ranked_bar(known(working, "Country"), "Country", "Temperature_C",
                   "Warmest countries (average °C)", "Average temperature (°C)", top=12)
    with right:
        ranked_bar(known(working, "Country"), "Country", "Temperature_C",
                   "Coolest countries (average °C)", "Average temperature (°C)", top=12,
                   ascending=True, scale=list(reversed(SEQ_COOL)))

    hottest = (
        known(working, "City").dropna(subset=["Temperature_C"])
        .groupby(["City", "Country"], as_index=False)["Temperature_C"].mean()
        .sort_values("Temperature_C", ascending=False).head(10)
    )
    if not hottest.empty:
        st.markdown("#### 🔥 Hottest cities")
        st.dataframe(
            hottest.rename(columns={"Temperature_C": "Average Temperature (°C)"}),
            use_container_width=True, hide_index=True,
            column_config={
                "Average Temperature (°C)": st.column_config.ProgressColumn(
                    "Average Temperature (°C)", format="%.1f",
                    min_value=float(hottest["Temperature_C"].min() - 5),
                    max_value=float(hottest["Temperature_C"].max()),
                )
            },
        )

# ======================= TEMPERATURE ANALYSIS =======================
elif page == "Temperature Analysis":
    st.subheader("🌡️ Temperature Analysis")
    kpi_row([
        ("Average", fnum(temperature.mean(), 2, " °C"), "Mean temperature", "#fb923c"),
        ("Median", fnum(temperature.median(), 2, " °C"), "Middle observation", "#38bdf8"),
        ("Maximum", fnum(temperature.max(), 2, " °C"), "Highest observation", "#f87171"),
        ("Minimum", fnum(temperature.min(), 2, " °C"), "Lowest observation", "#60a5fa"),
        ("Std. Deviation", fnum(temperature.std(), 2, " °C"), "Spread of values", "#a78bfa"),
        ("95th Percentile", fnum(temperature.quantile(0.95), 2, " °C"), "Hot-end threshold", "#fbbf24"),
    ])

    tab_dist, tab_rank, tab_trend, tab_geo = st.tabs(
        ["📊 Distribution", "🏙️ Rankings", "📈 Trends", "🧭 Geography"]
    )

    with tab_dist:
        fig = px.histogram(
            working.dropna(subset=["Temperature_C"]), x="Temperature_C", nbins=45, marginal="box",
            title="Temperature frequency distribution",
            labels={"Temperature_C": "Temperature (°C)"}, color_discrete_sequence=[ACCENT],
        )
        show_chart(fig, 460)

        box_data = known(working, "Country").dropna(subset=["Temperature_C"])
        if not box_data.empty:
            top_countries = box_data["Country"].value_counts().head(20).index
            subset = box_data[box_data["Country"].isin(top_countries)]
            order = subset.groupby("Country")["Temperature_C"].median().sort_values().index
            fig = px.box(
                subset, x="Country", y="Temperature_C", category_orders={"Country": list(order)},
                title="Temperature spread across the 20 most represented countries",
                labels={"Temperature_C": "Temperature (°C)", "Country": ""},
                color_discrete_sequence=[ACCENT],
            )
            show_chart(fig, 460)

    with tab_rank:
        cities = known(working, "City")
        left, right = st.columns(2)
        with left:
            ranked_bar(cities, "City", "Temperature_C", "15 warmest cities (average °C)",
                       "Average °C", top=15)
        with right:
            ranked_bar(cities, "City", "Temperature_C", "15 coolest cities (average °C)",
                       "Average °C", top=15, ascending=True, scale=list(reversed(SEQ_COOL)))

    with tab_trend:
        dated = working.dropna(subset=["Date", "Temperature_C"])
        daily = None
        if not dated.empty:
            daily = (
                dated.assign(Day=dated["Date"].dt.floor("D")).groupby("Day")["Temperature_C"]
                .agg(["mean", "min", "max"]).reset_index()
            )
        if daily is None or len(daily) < 2:
            st.info("Only one day of data is available, so a trend line cannot be drawn.")
        else:
            fig = go.Figure()
            fig.add_trace(go.Scatter(
                x=daily["Day"], y=daily["max"], mode="lines", name="Daily max",
                line=dict(width=0), hovertemplate="Max %{y:.1f} °C<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=daily["Day"], y=daily["min"], mode="lines", name="Min–max range",
                line=dict(width=0), fill="tonexty", fillcolor="rgba(56,189,248,0.16)",
                hovertemplate="Min %{y:.1f} °C<extra></extra>",
            ))
            fig.add_trace(go.Scatter(
                x=daily["Day"], y=daily["mean"], mode="lines", name="Daily mean",
                line=dict(color=ACCENT, width=2.2), hovertemplate="Mean %{y:.1f} °C<extra></extra>",
            ))
            if len(daily) >= 7:
                fig.add_trace(go.Scatter(
                    x=daily["Day"], y=daily["mean"].rolling(7, min_periods=3).mean(),
                    mode="lines", name="7-day average", line=dict(color="#fbbf24", width=2.5, dash="dot"),
                ))
            fig.update_layout(title="Daily temperature: mean with min–max band", hovermode="x unified")
            fig.update_yaxes(title="Temperature (°C)")
            show_chart(fig, 440)

        if not dated.empty and dated["Date"].dt.hour.nunique() > 1:
            by_hour = dated.groupby(dated["Date"].dt.hour)["Temperature_C"].mean().reset_index()
            by_hour.columns = ["Hour", "Temperature_C"]
            fig = px.area(
                by_hour, x="Hour", y="Temperature_C", title="Average temperature by hour of day",
                labels={"Hour": "Hour of day", "Temperature_C": "Average °C"},
                color_discrete_sequence=["#fb923c"],
            )
            fig.update_xaxes(dtick=2)
            show_chart(fig, 360)

    with tab_geo:
        geo = working.dropna(subset=["Latitude", "Temperature_C"])
        if geo.empty:
            st.info("Latitude values are unavailable.")
        else:
            fig = px.scatter(
                sample_frame(geo), x="Latitude", y="Temperature_C", color="Temperature_C",
                color_continuous_scale="Turbo", opacity=0.6, hover_data=["City", "Country"],
                title="Temperature vs latitude", labels={"Temperature_C": "Temperature (°C)"},
            )
            fig.update_traces(marker=dict(size=5))
            fig.update_coloraxes(colorbar=dict(title="°C", thickness=12))
            show_chart(fig, 440)

        if "feels_like_celsius" in working.columns:
            feels = working.dropna(subset=["feels_like_celsius", "Temperature_C"])
            if not feels.empty:
                feels = sample_frame(feels)
                fig = px.scatter(
                    feels, x="Temperature_C", y="feels_like_celsius", opacity=0.45,
                    title="Actual vs 'feels like' temperature",
                    labels={"Temperature_C": "Actual (°C)", "feels_like_celsius": "Feels like (°C)"},
                    color_discrete_sequence=["#2dd4bf"],
                )
                lo = float(min(feels["Temperature_C"].min(), feels["feels_like_celsius"].min()))
                hi = float(max(feels["Temperature_C"].max(), feels["feels_like_celsius"].max()))
                fig.add_trace(go.Scatter(x=[lo, hi], y=[lo, hi], mode="lines", name="Equal",
                                         line=dict(color="#fbbf24", dash="dash")))
                show_chart(fig, 420)

# ========================= RAINFALL & CLIMATE =========================
elif page == "Rainfall & Climate":
    st.subheader("🌧️ Rainfall & Climate")
    rainy_share = (rainfall > 0).mean() * 100 if not rainfall.empty else np.nan
    pressure = valid_numeric("pressure_mb")
    kpi_row([
        ("Avg Rainfall", fnum(rainfall.mean(), 2, " mm"), "Per observation", "#38bdf8"),
        ("Max Rainfall", fnum(rainfall.max(), 2, " mm"), "Highest observation", "#60a5fa"),
        ("Rainy Share", fnum(rainy_share, 1, "%"), "Observations with rain", "#2dd4bf"),
        ("Avg Humidity", fnum(humidity.mean(), 1, "%"), "Relative humidity", "#a78bfa"),
        ("Avg Wind", fnum(wind_speed.mean(), 1, " km/h"), "Wind speed", "#fbbf24"),
        ("Avg Pressure", fnum(pressure.mean(), 0, " mb"), "Sea-level pressure", "#fb923c"),
    ])

    tab_rain, tab_wind, tab_rel = st.tabs(["🌧️ Precipitation", "💨 Wind", "🔗 Relationships"])

    with tab_rain:
        left, right = st.columns(2)
        with left:
            if not rainfall.empty:
                fig = px.histogram(
                    working.dropna(subset=["Precipitation_mm"]), x="Precipitation_mm", nbins=40,
                    title="Rainfall distribution (log scale)", log_y=True,
                    labels={"Precipitation_mm": "Precipitation (mm)"}, color_discrete_sequence=[ACCENT],
                )
                fig.update_yaxes(title="Observations (log)")
                show_chart(fig)
            else:
                st.info("Rainfall column is not available.")
        with right:
            if not humidity.empty:
                fig = px.histogram(
                    working.dropna(subset=["Humidity_pct"]), x="Humidity_pct", nbins=30,
                    title="Humidity distribution", labels={"Humidity_pct": "Humidity (%)"},
                    color_discrete_sequence=["#a78bfa"],
                )
                fig.update_yaxes(title="Observations")
                show_chart(fig)
            else:
                st.info("Humidity column is not available.")

        if not rainfall.empty:
            places = known(working, "Country")
            left, right = st.columns(2)
            with left:
                ranked_bar(places, "Country", "Precipitation_mm", "Wettest countries (average mm)",
                           "Average precipitation (mm)", top=15, scale=list(reversed(SEQ_COOL)), fmt=".2f")
            with right:
                counts = places.groupby("Country")["Precipitation_mm"].agg(
                    rainy=lambda s: (s > 0).mean() * 100, n="count"
                ).reset_index()
                counts = counts[counts["n"] >= 5].sort_values("rainy", ascending=False).head(15)
                if not counts.empty:
                    fig = px.bar(
                        counts, x="rainy", y="Country", orientation="h", text="rainy",
                        title="Share of observations with rain (%)",
                        labels={"rainy": "Rainy observations (%)", "Country": ""},
                        color="rainy", color_continuous_scale=list(reversed(SEQ_COOL)),
                    )
                    fig.update_traces(texttemplate="%{text:.0f}%", textposition="outside", cliponaxis=False)
                    fig.update_layout(yaxis={"categoryorder": "total ascending"})
                    fig.update_coloraxes(showscale=False)
                    show_chart(fig, 430)

    with tab_wind:
        left, right = st.columns(2)
        with left:
            if not wind_speed.empty:
                fig = px.histogram(
                    working.dropna(subset=["Wind_Speed_kmh"]), x="Wind_Speed_kmh", nbins=40,
                    title="Wind speed distribution", labels={"Wind_Speed_kmh": "Wind speed (km/h)"},
                    color_discrete_sequence=["#fbbf24"],
                )
                fig.update_yaxes(title="Observations")
                show_chart(fig, 430)
            else:
                st.info("Wind speed is not available.")
        with right:
            compass = ["N", "NNE", "NE", "ENE", "E", "ESE", "SE", "SSE",
                       "S", "SSW", "SW", "WSW", "W", "WNW", "NW", "NNW"]
            rose = pd.DataFrame()
            if "wind_direction" in working.columns:
                rose = (
                    working[working["wind_direction"].isin(compass)]
                    .groupby("wind_direction")["Wind_Speed_kmh"].agg(["count", "mean"])
                    .reindex(compass).dropna().reset_index()
                )
            if rose.empty:
                st.info("Wind direction column is not available.")
            else:
                fig = go.Figure(go.Barpolar(
                    r=rose["count"], theta=rose["wind_direction"],
                    marker=dict(color=rose["mean"], colorscale="Viridis", showscale=True,
                                colorbar=dict(title="km/h", thickness=12)),
                    customdata=rose["mean"],
                    hovertemplate="%{theta}: %{r} obs<br>Avg %{customdata:.1f} km/h<extra></extra>",
                ))
                fig.update_layout(title="Wind rose (length = frequency, colour = avg speed)")
                fig.update_polars(
                    bgcolor="rgba(0,0,0,0)",
                    radialaxis=dict(gridcolor=GRID, linecolor=GRID),
                    angularaxis=dict(gridcolor=GRID, linecolor=GRID, direction="clockwise",
                                     rotation=90, categoryarray=compass),
                )
                show_chart(fig, 430)

        if not wind_speed.empty:
            ranked_bar(known(working, "Country"), "Country", "Wind_Speed_kmh",
                       "Windiest countries (average km/h)", "Average wind (km/h)", top=15,
                       scale=["#fef3c7", "#fbbf24", "#f59e0b", "#b45309"], horizontal=False)

    with tab_rel:
        pair = working.dropna(subset=["Humidity_pct", "Temperature_C"])
        if not pair.empty:
            pair = sample_frame(pair)
            fig = px.scatter(
                pair, x="Temperature_C", y="Humidity_pct",
                color="Precipitation_mm" if not rainfall.empty else None,
                color_continuous_scale="Blues", opacity=0.55, title="Temperature vs humidity",
                labels={"Temperature_C": "Temperature (°C)", "Humidity_pct": "Humidity (%)",
                        "Precipitation_mm": "Precip. (mm)"},
            )
            fig.update_traces(marker=dict(size=5))
            add_trend_line(fig, pair["Temperature_C"], pair["Humidity_pct"])
            show_chart(fig, 440)

        correlation_heatmap(
            working,
            ["Temperature_C", "Humidity_pct", "Precipitation_mm", "Wind_Speed_kmh",
             "pressure_mb", "cloud", "uv_index", "visibility_km"],
            {"Temperature_C": "Temperature", "Humidity_pct": "Humidity", "Precipitation_mm": "Rainfall",
             "Wind_Speed_kmh": "Wind", "pressure_mb": "Pressure", "cloud": "Cloud cover",
             "uv_index": "UV index", "visibility_km": "Visibility"},
            "Correlation between weather variables",
        )

# ======================== WEATHER CONDITIONS ========================
elif page == "Weather Conditions":
    st.subheader("⛅ Weather Conditions")
    conditions = working[working["Weather_Condition"] != "Unknown"]
    if conditions.empty:
        st.info("Weather condition descriptions are unavailable.")
    else:
        counts = (
            conditions["Weather_Condition"].value_counts().head(12)
            .rename_axis("Weather condition").reset_index(name="Records")
        )
        top = counts.iloc[0]
        kpi_row([
            ("Distinct Conditions", f"{conditions['Weather_Condition'].nunique():,}", "Reported descriptions", "#38bdf8"),
            ("Most Common", str(top["Weather condition"]), f"{top['Records']:,} records", "#2dd4bf"),
            ("Warning Level", f"{(working['Severity'] == 'Warning').mean() * 100:.1f}%", "Of observations", SEVERITY_COLORS["Warning"]),
            ("Extreme Level", f"{(working['Severity'] == 'Extreme').mean() * 100:.1f}%", "Of observations", SEVERITY_COLORS["Extreme"]),
        ])

        left, right = st.columns(2)
        with left:
            fig = px.pie(counts, names="Weather condition", values="Records", hole=0.55,
                         title="Weather condition composition (top 12)")
            fig.update_traces(textinfo="percent", textfont_size=11)
            show_chart(fig, 430)
        with right:
            fig = px.bar(counts, x="Records", y="Weather condition", orientation="h", text="Records",
                         title="Most frequently reported conditions", labels={"Weather condition": ""},
                         color="Records", color_continuous_scale=SEQ_COOL)
            fig.update_traces(textposition="outside", cliponaxis=False)
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            fig.update_coloraxes(showscale=False)
            show_chart(fig, 430)

        left, right = st.columns(2)
        with left:
            detail = (
                conditions.groupby("Weather_Condition")
                .agg(Records=("Weather_Condition", "size"), Temp=("Temperature_C", "mean"),
                     Humidity=("Humidity_pct", "mean"), Wind=("Wind_Speed_kmh", "mean"))
                .reset_index().sort_values("Records", ascending=False).head(15)
            )
            st.markdown("#### Typical profile per condition")
            st.dataframe(
                detail.rename(columns={
                    "Weather_Condition": "Condition", "Temp": "Avg temp (°C)",
                    "Humidity": "Avg humidity (%)", "Wind": "Avg wind (km/h)",
                }).round(1),
                use_container_width=True, hide_index=True, height=420,
            )
        with right:
            places = known(working, "Country")
            if not places.empty:
                top_c = places["Country"].value_counts().head(12).index
                mix = (places[places["Country"].isin(top_c)]
                       .groupby(["Country", "Severity"]).size().reset_index(name="Records"))
                fig = px.bar(
                    mix, x="Country", y="Records", color="Severity", barmode="relative",
                    title="Weather status mix by country (12 most represented)",
                    color_discrete_map=SEVERITY_COLORS, category_orders={"Severity": SEVERITY_LEVELS},
                    labels={"Country": ""},
                )
                fig.update_layout(barnorm="percent")
                fig.update_yaxes(title="Share of observations (%)")
                show_chart(fig, 470)

    extremes = working.dropna(subset=["Temperature_C"])
    cols_show = ["City", "Country", "Temperature_C", "Weather_Condition"]
    left, right = st.columns(2)
    with left:
        st.markdown("#### 🔥 Highest temperature observations")
        st.dataframe(extremes.nlargest(10, "Temperature_C")[cols_show], use_container_width=True, hide_index=True)
    with right:
        st.markdown("#### ❄️ Lowest temperature observations")
        st.dataframe(extremes.nsmallest(10, "Temperature_C")[cols_show], use_container_width=True, hide_index=True)

# ============================= AIR QUALITY =============================
elif page == "Air Quality":
    st.subheader("🌫️ Air Quality Analysis")
    air_labels = {
        "air_quality_pm2_5": "PM2.5",
        "air_quality_pm10": "PM10",
        "air_quality_nitrogen_dioxide": "Nitrogen dioxide",
        "air_quality_ozone": "Ozone",
        "air_quality_carbon_monoxide": "Carbon monoxide",
        "air_quality_sulphur_dioxide": "Sulphur dioxide",
    }
    available_air = {
        col: label for col, label in air_labels.items()
        if col in working.columns and working[col].notna().any()
    }

    if not available_air:
        st.info("No air-quality measurements are available in this dataset.")
    else:
        chosen_air = st.selectbox(
            "Select air-quality measure", options=list(available_air),
            format_func=lambda x: available_air[x],
        )
        air_label = available_air[chosen_air]
        vals = working[chosen_air].dropna()
        kpi_row([
            ("Average", fnum(vals.mean(), 2), f"{air_label} (µg/m³)", "#38bdf8"),
            ("Median", fnum(vals.median(), 2), "Middle observation", "#2dd4bf"),
            ("95th Percentile", fnum(vals.quantile(0.95), 2), "High-end level", "#fbbf24"),
            ("Maximum", fnum(vals.max(), 2), "Worst observation", "#f87171"),
        ])

        left, right = st.columns(2)
        with left:
            fig = px.histogram(
                working.dropna(subset=[chosen_air]), x=chosen_air, nbins=45,
                title=f"{air_label} distribution", labels={chosen_air: air_label},
                color_discrete_sequence=["#a78bfa"],
            )
            fig.update_yaxes(title="Observations")
            show_chart(fig, 410)
        with right:
            if chosen_air == "air_quality_pm2_5":
                bins = [-np.inf, 12, 35.4, 55.4, 150.4, 250.4, np.inf]
                names = ["Good", "Moderate", "Unhealthy (sensitive)", "Unhealthy", "Very unhealthy", "Hazardous"]
                colors = ["#22c55e", "#facc15", "#fb923c", "#ef4444", "#a855f7", "#7f1d1d"]
                cat = pd.cut(vals, bins=bins, labels=names).value_counts().reindex(names).fillna(0)
                cat_df = cat.rename_axis("Category").reset_index(name="Records")
                fig = px.bar(
                    cat_df, x="Category", y="Records", color="Category",
                    color_discrete_map=dict(zip(names, colors)), text="Records",
                    title="PM2.5 health categories (US EPA breakpoints)", labels={"Category": ""},
                )
                fig.update_traces(textposition="outside", cliponaxis=False)
                fig.update_layout(showlegend=False)
                show_chart(fig, 410)
            else:
                ranked_bar(known(working, "Country"), "Country", chosen_air,
                           f"Countries with the highest {air_label}", f"Average {air_label}",
                           top=12, scale=SEQ_PURPLE)

        ranked_bar(known(working, "City"), "City", chosen_air,
                   f"15 cities with the highest average {air_label}", f"Average {air_label}",
                   top=15, scale=SEQ_PURPLE)

        if len(available_air) > 1:
            correlation_heatmap(working, list(available_air), available_air, "Correlation between pollutants")

        pair = working.dropna(subset=[chosen_air, "Temperature_C"])
        if len(pair) > 3:
            pair = sample_frame(pair)
            fig = px.scatter(
                pair, x="Temperature_C", y=chosen_air, opacity=0.5,
                title=f"{air_label} vs temperature",
                labels={"Temperature_C": "Temperature (°C)", chosen_air: air_label},
                color_discrete_sequence=["#a78bfa"],
            )
            fig.update_traces(marker=dict(size=5))
            add_trend_line(fig, pair["Temperature_C"], pair[chosen_air])
            show_chart(fig, 420)

# ============================= DATA EXPLORER =============================
elif page == "Data Explorer":
    st.subheader("🗃️ Weather Data Explorer")
    st.write(
        f"Showing **{len(working):,}** records and **{len(working.columns)}** columns "
        f"from `{dataset_path.name}`."
    )

    c1, c2 = st.columns([2, 3])
    with c1:
        search_text = st.text_input("Search (city, country or condition)")
    with c2:
        visible = st.multiselect(
            "Columns to display", list(working.columns),
            default=[c for c in ["City", "Country", "Date", "Temperature_C", "Humidity_pct",
                                 "Precipitation_mm", "Wind_Speed_kmh", "Weather_Condition", "Severity"]
                     if c in working.columns],
        )

    display_data = working
    if search_text.strip():
        needle = search_text.strip().lower()
        mask = (
            working["City"].str.lower().str.contains(needle, na=False, regex=False)
            | working["Country"].str.lower().str.contains(needle, na=False, regex=False)
            | working["Weather_Condition"].str.lower().str.contains(needle, na=False, regex=False)
        )
        display_data = working[mask]
    shown = display_data[visible] if visible else display_data

    st.dataframe(shown, use_container_width=True, height=520)
    st.download_button(
        "⬇️ Download displayed data as CSV",
        data=shown.to_csv(index=False).encode("utf-8"),
        file_name="atmos_weather_data.csv", mime="text/csv",
    )

    numeric_view = display_data.select_dtypes(include=np.number)
    if not numeric_view.empty:
        st.markdown("#### Summary statistics")
        st.dataframe(numeric_view.describe().T.round(2), use_container_width=True)

    left, right = st.columns(2)
    with left:
        st.markdown("#### Dataset information")
        info_df = pd.DataFrame({
            "Column": working.columns,
            "Data type": [str(dtype) for dtype in working.dtypes],
            "Non-null values": [int(working[col].notna().sum()) for col in working.columns],
            "Missing values": [int(working[col].isna().sum()) for col in working.columns],
        })
        st.dataframe(info_df, use_container_width=True, height=400)
    with right:
        missing = working.isna().sum().sort_values(ascending=False).head(20)
        missing = missing[missing > 0]
        if missing.empty:
            st.success("No missing values in the displayed dataset.")
        else:
            missing_df = missing.rename_axis("Column").reset_index(name="Missing values")
            fig = px.bar(missing_df, x="Missing values", y="Column", orientation="h",
                         title="Columns with the most missing values", labels={"Column": ""},
                         color_discrete_sequence=["#fb923c"])
            fig.update_layout(yaxis={"categoryorder": "total ascending"})
            show_chart(fig, 430)

# ----------------------------- FOOTER -----------------------------
st.markdown("---")
st.markdown(
    f"<div style='text-align:center;color:{MUTED};padding:8px;'>"
    "<b>ATMOS | WEATHER INTELLIGENCE SYSTEM</b><br>"
    "Global Weather Analytics and Visualization<br>"
    "Built with Python • Streamlit • Pandas • Plotly"
    "</div>",
    unsafe_allow_html=True,
)
