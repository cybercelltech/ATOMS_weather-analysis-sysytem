import gzip
import re
import warnings
from datetime import datetime
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
# weather_data.csv comes first: if an older compressed_data*.csv.gz is still in the
# folder it must not win over the CSV you are actually adding cities to.
PREFERRED_DATASETS = [
    PROJECT_DIR / "weather_data.csv",
    PROJECT_DIR / "weather.csv",
    PROJECT_DIR / "data" / "weather_data.csv",
    PROJECT_DIR / "data" / "weather.csv",
    PROJECT_DIR / "compressed_data (1).csv.gz",
    PROJECT_DIR / "compressed_data.csv.gz",
    PROJECT_DIR / "data" / "compressed_data (1).csv.gz",
    PROJECT_DIR / "data" / "compressed_data.csv.gz",
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


def parse_mixed_dates(values: pd.Series) -> pd.Series:
    """Parse 'last_updated' when the file mixes formats.

    Rows with a day above 12 look like '16-05-2024 14:15'; rows that were opened and
    re-saved in Excel look like '1/6/2024 0:15'. Both are day-first. Parsing with one
    inferred format silently turns every row of the other format into a blank date.
    """
    text = values.astype(str).str.strip()
    parsed = pd.to_datetime(text, format="%d-%m-%Y %H:%M", errors="coerce")
    for fmt in ("%d/%m/%Y %H:%M", "%Y-%m-%d %H:%M", "%d-%m-%Y %H:%M:%S", "%Y-%m-%d %H:%M:%S"):
        missing = parsed.isna()
        if not missing.any():
            break
        parsed[missing] = pd.to_datetime(text[missing], format=fmt, errors="coerce")
    missing = parsed.isna()
    if missing.any():
        try:  # last resort for anything unusual (needs pandas >= 2.0)
            parsed[missing] = pd.to_datetime(text[missing], format="mixed", dayfirst=True, errors="coerce")
        except (TypeError, ValueError):
            pass
    return parsed


# file_mtime_ns / file_size are not used inside the function. They are part of the cache
# key on purpose, so editing weather_data.csv makes Streamlit reload it instead of
# serving the copy it cached earlier (the cache used to be keyed on the path alone).
@st.cache_data(show_spinner="Loading weather dataset...")
def load_dataset(path_string: str, file_mtime_ns: int = 0, file_size: int = 0) -> pd.DataFrame:
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

    data["Date"] = parse_mixed_dates(data["Date"])

    for col in ["Country", "City", "Weather_Condition"]:
        data[col] = data[col].fillna("Unknown").astype(str).str.strip()
        data.loc[data[col].eq(""), col] = "Unknown"

    return data


dataset_path = find_dataset()
if dataset_path is None:
    st.error("Dataset not found. Put compressed_data (1).csv.gz in the same folder as app.py.")
    st.stop()

try:
    _stat = dataset_path.stat()
    df = load_dataset(str(dataset_path), _stat.st_mtime_ns, _stat.st_size)
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
_n_places = (df["City"] + " | " + df["Country"]).nunique()
st.sidebar.caption(
    f"{len(df):,} records · {_n_places:,} places · "
    f"file saved {datetime.fromtimestamp(_stat.st_mtime):%d %b %Y, %H:%M}"
)

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


GLOBE_MODES = {
    "all": "Every observation (one marker per record)",
    "latest": "Latest observation per location",
    "worst": "Worst status in period per location",
}


def _prepare_places(data):
    """Clean coordinates and give each recorded place one identity and one position.

    Returns (points, place_coords, missing_places, total_rows, n_locations).
    """
    points = data.copy()
    if "Severity" not in points.columns:
        points["Severity"] = classify_weather(points)

    # Blank out impossible coordinates instead of dropping the whole row, so the
    # city can still be placed from its other readings.
    points.loc[~points["Latitude"].between(-90, 90), "Latitude"] = np.nan
    points.loc[~points["Longitude"].between(-180, 180), "Longitude"] = np.nan

    # One identity per recorded place. Rows without a city name are told apart by position.
    unknown_city = points["City"].eq("Unknown")
    place_id = points["City"].astype(str) + " | " + points["Country"].astype(str)
    if unknown_city.any():
        place_id = place_id.where(
            ~unknown_city,
            "Unknown @ " + points["Latitude"].round(2).astype(str) + "," + points["Longitude"].round(2).astype(str),
        )
    points["_place"] = place_id

    # Every place gets its median coordinates from whichever of its rows have them.
    place_coords = points.groupby("_place")[["Latitude", "Longitude"]].median()
    missing_places = place_coords[place_coords["Latitude"].isna() | place_coords["Longitude"].isna()].index
    return points, place_coords, missing_places, len(points), points["_place"].nunique()


def _sphere_xyz(lat_deg, lon_deg, radius):
    lat = np.radians(np.asarray(lat_deg, dtype=float))
    lon = np.radians(np.asarray(lon_deg, dtype=float))
    return (
        radius * np.cos(lat) * np.cos(lon),
        radius * np.cos(lat) * np.sin(lon),
        radius * np.sin(lat),
    )



_NE = "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/"
GEO_URLS = {
    "coast": [_NE + "ne_50m_coastline.geojson", _NE + "ne_110m_coastline.geojson"],
    "borders": [_NE + "ne_50m_admin_0_boundary_lines_land.geojson",
                _NE + "ne_110m_admin_0_boundary_lines_land.geojson"],
}
LAND_URLS = [_NE + "ne_50m_land.geojson", _NE + "ne_110m_land.geojson"]


def _download_json(urls):
    """Return the first GeoJSON that downloads; raise with every failure if none does."""
    import json
    import urllib.request

    problems = []
    for url in urls:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                return json.loads(resp.read().decode("utf-8"))
        except Exception as error:
            problems.append(f"{url.split('/')[2]}: {error}")
    raise RuntimeError(" | ".join(problems))


@st.cache_data(show_spinner="Loading world outlines...")
def _world_lines_cached(kind: str):
    """Coastline / country-border lines as lon,lat paths. Raises when offline (so it is not cached)."""
    import json

    cache_file = PROJECT_DIR / f".geo_{kind}_v2.json"
    if cache_file.exists():
        try:
            saved = json.loads(cache_file.read_text())
            if saved:
                return saved
        except Exception:
            pass
    gj = _download_json(GEO_URLS[kind])
    paths = []
    for feat in gj["features"]:
        for line in _geometry_lines(feat.get("geometry")):
            pts = [(float(pt[0]), float(pt[1])) for pt in line]
            if len(pts) >= 2:
                paths.append(_decimate(pts, 0.05))
    try:
        cache_file.write_text(json.dumps(paths))
    except Exception:
        pass
    return paths


def load_world_lines(kind: str):
    try:
        return _world_lines_cached(kind)
    except Exception:
        return []


@st.cache_data(show_spinner="Loading land outlines...")
def _land_polygons():
    """Land polygons (list of polygons, each a list of lon/lat rings). Raises when offline."""
    import json

    cache_file = PROJECT_DIR / ".geo_land_v2.json"
    if cache_file.exists():
        try:
            saved = json.loads(cache_file.read_text())
            if saved:
                return saved
        except Exception:
            pass
    gj = _download_json(LAND_URLS)
    polygons = []
    for feat in gj.get("features", []):
        geom = feat.get("geometry") or {}
        if geom.get("type") == "Polygon":
            parts = [geom["coordinates"]]
        elif geom.get("type") == "MultiPolygon":
            parts = geom["coordinates"]
        else:
            continue
        for part in parts:
            polygons.append([[(float(pt[0]), float(pt[1])) for pt in ring] for ring in part])
    if not polygons:
        raise ValueError("no land polygons in the downloaded file")
    try:
        cache_file.write_text(json.dumps(polygons))
    except Exception:
        pass
    return polygons


def _land_mask(polygons, n_lat, n_lon, supersample=4):
    """Rasterise land polygons to a (n_lat x n_lon) 0..1 mask, row 0 = south pole."""
    from PIL import Image, ImageDraw

    w, h = n_lon * supersample, n_lat * supersample
    img = Image.new("L", (w, h), 0)
    draw = ImageDraw.Draw(img)
    sx, sy = (w - 1) / 360.0, (h - 1) / 180.0
    for polygon in polygons:
        for i, ring in enumerate(polygon):
            pts = [((pt[0] + 180.0) * sx, (90.0 - pt[1]) * sy) for pt in ring]
            if len(pts) >= 3:
                draw.polygon(pts, fill=255 if i == 0 else 0)
    img = img.resize((n_lon, n_lat), getattr(Image, "Resampling", Image).BOX)
    return np.flipud(np.asarray(img, dtype=float) / 255.0)


# Scalar 0..1 -> colour. 0-0.30 is ocean (deep navy -> turquoise shallows),
# 0.31-1.0 is land (rainforest -> savanna -> desert -> taiga -> tundra -> ice).
EARTH_COLORSCALE = [
    [0.00, "#030d24"], [0.08, "#06214a"], [0.18, "#0b4a8a"], [0.26, "#1e90c8"], [0.30, "#47c9dc"],
    [0.31, "#14532d"], [0.40, "#2f7d32"], [0.47, "#6fae45"], [0.55, "#c8b45a"], [0.62, "#c98f4d"],
    [0.72, "#7d8f55"], [0.82, "#8a9a86"], [0.90, "#cfd9e3"], [1.00, "#ffffff"],
]


@st.cache_data(show_spinner="Building Earth surface...")
def _build_surface(has_land: bool, step: float = 0.75):
    lats = np.arange(-90.0, 90.0 + step / 2, step)
    lons = np.arange(-180.0, 180.0 + step / 2, step)
    lon_g, lat_g = np.meshgrid(lons, lats)
    abs_lat = np.abs(lat_g)

    if has_land:
        from PIL import Image, ImageFilter

        land = _land_mask(_land_polygons(), len(lats), len(lons))
        shelf = np.asarray(
            Image.fromarray(np.uint8(land * 255)).filter(ImageFilter.GaussianBlur(2.5)), dtype=float
        ) / 255.0
    else:
        land = np.zeros_like(lat_g)
        shelf = land
    is_land = land > 0.5

    ocean = 0.03 + 0.15 * np.cos(np.radians(lat_g)) ** 2 + 0.11 * np.clip(shelf * 1.6, 0, 1)
    ocean = np.clip(ocean, 0.0, 0.29)
    base = np.interp(
        abs_lat, [0, 10, 20, 28, 36, 46, 56, 64, 70, 90],
        [0.34, 0.45, 0.58, 0.62, 0.48, 0.46, 0.78, 0.88, 1.0, 1.0],
    )
    texture = 0.035 * np.sin(np.radians(lat_g) * 9) * np.cos(np.radians(lon_g) * 7)
    land_val = np.clip(base + texture, 0.32, 1.0)
    value = np.where(is_land, land_val, ocean)

    x, y, z = _sphere_xyz(lat_g, lon_g, 1.0)
    return {"x": np.round(x, 4), "y": np.round(y, 4), "z": np.round(z, 4), "c": np.round(value, 3)}


def earth_surface():
    """Return (surface_arrays, land_error). Falls back to a plain ocean ball when offline."""
    try:
        _land_polygons()
        return _build_surface(True), None
    except Exception as error:
        return _build_surface(False), str(error)


@st.cache_data
def _space_dust(seed: int = 11):
    """Stars and coloured nebula clouds placed in the cube around the globe."""
    rng = np.random.default_rng(seed)
    pts = rng.uniform(-1.5, 1.5, size=(4000, 3))
    pts = pts[np.linalg.norm(pts, axis=1) > 1.15][:1200]
    palette = np.array(["#ffffff", "#ffffff", "#cfe8ff", "#9fc5ff", "#fff1c1", "#ffd2a1", "#ffb3d9", "#c4b5fd"])
    colors = palette[rng.integers(0, len(palette), len(pts))]
    sizes = rng.choice([1.2, 1.6, 2.0, 2.6, 3.6], size=len(pts), p=[0.35, 0.3, 0.2, 0.1, 0.05])
    nebulae = []
    for color in ["#7c3aed", "#2563eb", "#db2777", "#0ea5e9", "#a855f7", "#14b8a6"]:
        centre = rng.normal(size=3)
        centre = centre / np.linalg.norm(centre) * rng.uniform(1.25, 1.4)
        cloud = np.clip(centre + rng.normal(scale=0.22, size=(22, 3)), -1.48, 1.48)
        nebulae.append((color, cloud[np.linalg.norm(cloud, axis=1) > 1.12]))
    return {"stars": pts, "colors": colors, "sizes": sizes, "nebulae": nebulae}


def _lines_trace(paths, radius, color, width, name):
    xs, ys, zs = [], [], []
    for path in paths:
        arr = np.asarray(path, dtype=float)
        x, y, z = _sphere_xyz(arr[:, 1], arr[:, 0], radius)
        xs += list(np.round(x, 4)) + [None]
        ys += list(np.round(y, 4)) + [None]
        zs += list(np.round(z, 4)) + [None]
    return go.Scatter3d(x=xs, y=ys, z=zs, mode="lines", line=dict(color=color, width=width),
                        hoverinfo="skip", name=name, showlegend=False)


# --------------------------- INDIAN STATE BORDERS ---------------------------
# Source order: india_states.geojson next to app.py (your own file wins) -> the copy
# saved after the first successful download (.geo_india_states.json) -> the URLs below.
# Boundaries are drawn exactly as the source file draws them, so a different file gives
# a different depiction (for example of Jammu & Kashmir and Ladakh).
LOCAL_STATE_FILE = PROJECT_DIR / "india_states.geojson"
STATE_CACHE_FILE = PROJECT_DIR / ".geo_india_states.json"
INDIA_STATE_SOURCES = [
    "https://gist.githubusercontent.com/jbrobst/56c13bbbf9d97d187fea01ca62ea5112/raw/e388c4cae20aa53cb5090210a42ebb9b765c0a36/india_states.geojson",
    "https://raw.githubusercontent.com/geohacker/india/master/state/india_state.geojson",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces_lines.geojson",
    "https://cdn.jsdelivr.net/gh/nvkelso/natural-earth-vector@master/geojson/ne_10m_admin_1_states_provinces_lines.geojson",
]
STATE_BORDER_COLOR = "#e0b050"
_COUNTRY_KEYS = ("adm0_a3", "iso_a3", "sov_a3", "admin", "adm0_name", "country", "name_0")


def _geometry_lines(geometry):
    """Yield every line / polygon ring of a GeoJSON geometry as a list of [lon, lat, ...]."""
    if not geometry:
        return
    kind, coords = geometry.get("type"), geometry.get("coordinates")
    if kind == "LineString":
        yield coords
    elif kind in ("MultiLineString", "Polygon"):
        yield from coords
    elif kind == "MultiPolygon":
        for polygon in coords:
            yield from polygon
    elif kind == "GeometryCollection":
        for part in geometry.get("geometries", []):
            yield from _geometry_lines(part)


def _country_tag(properties):
    """Return the lower-cased country label of a feature, or None if it has none."""
    for key, value in (properties or {}).items():
        if str(key).lower() in _COUNTRY_KEYS:
            return str(value).strip().lower()
    return None


def _decimate(path, tolerance=0.02):
    """Drop vertices within `tolerance` degrees of the last kept one (~2 km), keep both ends."""
    kept = [path[0]]
    for lon, lat in path[1:-1]:
        if abs(lon - kept[-1][0]) + abs(lat - kept[-1][1]) >= tolerance:
            kept.append((lon, lat))
    kept.append(path[-1])
    return kept


def india_paths_from_geojson(geojson):
    """Turn a states GeoJSON (polygons or lines) into [(lon, lat), ...] paths.

    A world-wide file is reduced to India; an India-only file is used as it is.
    """
    features = geojson.get("features", [])
    if any(_country_tag(f.get("properties")) is not None for f in features):
        features = [f for f in features if _country_tag(f.get("properties")) in ("ind", "india")]
    paths = []
    for feature in features:
        for line in _geometry_lines(feature.get("geometry")):
            points = [(float(pt[0]), float(pt[1])) for pt in line]
            if len(points) >= 2:
                paths.append(_decimate(points))
    if not paths:
        raise ValueError("no Indian state outlines found in this file")
    return paths


@st.cache_data(show_spinner="Loading Indian state borders...")
def _india_state_paths(local_stamp: float):
    """Local file > saved copy > download. Raises when nothing works, so a failure is never cached."""
    import json
    import urllib.request

    if LOCAL_STATE_FILE.exists():
        return india_paths_from_geojson(json.loads(LOCAL_STATE_FILE.read_text(encoding="utf-8")))

    if STATE_CACHE_FILE.exists():
        try:
            saved = json.loads(STATE_CACHE_FILE.read_text())
            if saved:
                return saved
        except Exception:
            pass

    problems = []
    for url in INDIA_STATE_SOURCES:
        try:
            with urllib.request.urlopen(url, timeout=30) as resp:
                paths = india_paths_from_geojson(json.loads(resp.read().decode("utf-8")))
        except Exception as error:
            problems.append(f"{url.split('/')[2]}: {error}")
            continue
        try:
            STATE_CACHE_FILE.write_text(json.dumps(paths))
        except Exception:
            pass
        return paths
    raise RuntimeError(" | ".join(problems))


def load_india_state_lines():
    """Return (paths, error_message). paths is [] when the borders could not be loaded."""
    stamp = LOCAL_STATE_FILE.stat().st_mtime if LOCAL_STATE_FILE.exists() else 0.0
    try:
        return _india_state_paths(stamp), None
    except Exception as error:
        return [], str(error)


def _geo_lines_trace(paths, color, width, name):
    """Lon/lat line trace for the flat-projection Scattergeo globe."""
    lons, lats = [], []
    for path in paths:
        arr = np.asarray(path, dtype=float)
        lons += list(np.round(arr[:, 0], 3)) + [None]
        lats += list(np.round(arr[:, 1], 3)) + [None]
    return go.Scattergeo(lon=lons, lat=lats, mode="lines", line=dict(color=color, width=width),
                         hoverinfo="skip", name=name, showlegend=False)


WORLD_STATE_SOURCES = [
    _NE + "ne_10m_admin_1_states_provinces_lines.geojson",
    "https://raw.githubusercontent.com/nvkelso/natural-earth-vector/master/geojson/ne_10m_admin_1_states_provinces_lines.geojson",
]
WORLD_STATE_CACHE = PROJECT_DIR / ".geo_world_states_v1.json"
COUNTRY_BORDER_COLOR = "#e2e8f0"


@st.cache_data(show_spinner="Loading state / province borders for every country...")
def _world_state_paths():
    """Admin-1 (state / province) boundary lines of every country. Raises when nothing works."""
    import json

    if WORLD_STATE_CACHE.exists():
        try:
            saved = json.loads(WORLD_STATE_CACHE.read_text())
            if saved.get("others"):
                return saved
        except Exception:
            pass
    gj = _download_json(WORLD_STATE_SOURCES)
    others, india = [], []
    for feature in gj.get("features", []):
        bucket = india if _country_tag(feature.get("properties")) in ("ind", "india") else others
        for line in _geometry_lines(feature.get("geometry")):
            pts = [(float(pt[0]), float(pt[1])) for pt in line]
            if len(pts) >= 2:
                bucket.append(_decimate(pts, 0.05))
    if not others:
        raise ValueError("no state / province lines found in the downloaded file")
    result = {"others": others, "india": india}
    try:
        WORLD_STATE_CACHE.write_text(json.dumps(result))
    except Exception:
        pass
    return result


def load_all_state_lines():
    """Return (paths, error). India uses your india_states.geojson / its own source when available."""
    india_paths, india_error = load_india_state_lines()
    try:
        world = _world_state_paths()
        error = None
    except Exception as exc:
        world, error = {"others": [], "india": []}, str(exc)
    paths = list(world["others"]) + list(india_paths or world["india"])
    if not paths:
        return [], error or india_error
    return paths, error


def _hover_text(subset):
    return (
        "<b>" + subset["City"].astype(str) + "</b>, " + subset["Country"].astype(str)
        + "<br>Status: " + subset["Severity"].astype(str)
        + "<br>Condition: " + subset["Weather_Condition"].astype(str)
        + "<br>Temperature: " + subset["Temperature_C"].round(1).astype(str) + " °C"
        + "<br>Wind: " + _col(subset, "Wind_Speed_kmh").round(1).astype(str) + " km/h"
        + "<br>Rain: " + _col(subset, "Precipitation_mm").round(1).astype(str) + " mm"
    )


def make_globe(data, mode="latest", levels=None, height=720, show_states=True, focus=None):
    """Interactive 3D Earth (drag to rotate, scroll to zoom) with severity markers.

    mode: 'latest' = newest row per place, 'worst' = most severe row per place,
          'all' = one marker for every record.
    All three modes share the same globe, stars and borders.
    """
    points, place_coords, missing_places, total_rows, n_locations = _prepare_places(data)

    if mode == "worst":
        points["_rank"] = points["Severity"].map({"Normal": 0, "Warning": 1, "Extreme": 2})
        points = points.sort_values(["_rank", "Date"], na_position="first").drop(columns="_rank")
        points = points.drop_duplicates(subset="_place", keep="last")
    elif mode == "latest":
        points = points.sort_values("Date", na_position="first").drop_duplicates(subset="_place", keep="last")
    points = points.drop(columns=["Latitude", "Longitude"]).join(place_coords, on="_place")
    points = points.dropna(subset=["Latitude", "Longitude"])
    if levels:
        points = points[points["Severity"].isin(levels)]

    fig = go.Figure()

    # ---- space: stars + coloured nebula glow ----
    dust = _space_dust()
    for color, cloud in dust["nebulae"]:
        if len(cloud):
            fig.add_trace(go.Scatter3d(
                x=cloud[:, 0], y=cloud[:, 1], z=cloud[:, 2], mode="markers", hoverinfo="skip",
                showlegend=False, marker=dict(size=58, color=color, opacity=0.07, line=dict(width=0)),
            ))
    st_pts = dust["stars"]
    fig.add_trace(go.Scatter3d(
        x=st_pts[:, 0], y=st_pts[:, 1], z=st_pts[:, 2], mode="markers", hoverinfo="skip", showlegend=False,
        marker=dict(size=dust["sizes"].tolist(), color=dust["colors"].tolist(), opacity=0.9, line=dict(width=0)),
    ))

    # ---- Earth: ocean depth + land biomes + polar ice ----
    surf, land_error = earth_surface()
    st.session_state["_land_error"] = land_error
    fig.add_trace(go.Surface(
        x=surf["x"], y=surf["y"], z=surf["z"], surfacecolor=surf["c"], cmin=0, cmax=1,
        colorscale=EARTH_COLORSCALE, showscale=False, hoverinfo="skip",
        lighting=dict(ambient=0.62, diffuse=0.85, specular=0.28, roughness=0.55, fresnel=0.35),
        lightposition=dict(x=6000, y=-3000, z=4000),
    ))

    # ---- thin glowing atmosphere ----
    a_lat, a_lon = np.meshgrid(np.arange(-90, 91, 3.0), np.arange(-180, 181, 3.0), indexing="ij")
    ax, ay, az = _sphere_xyz(a_lat, a_lon, 1.035)
    fig.add_trace(go.Surface(
        x=ax, y=ay, z=az, surfacecolor=np.zeros_like(ax), showscale=False, hoverinfo="skip",
        colorscale=[[0, "#6cc4ff"], [1, "#6cc4ff"]], opacity=0.10,
        lighting=dict(ambient=1.0, diffuse=0.0, specular=0.0),
    ))

    # ---- faint graticule ----
    gx, gy, gz = [], [], []
    for lat in range(-60, 61, 30):
        lons = np.linspace(-180, 180, 120)
        x, y, z = _sphere_xyz(np.full_like(lons, lat), lons, 1.001)
        gx += list(x) + [None]; gy += list(y) + [None]; gz += list(z) + [None]
    for lon in range(-180, 180, 30):
        lats = np.linspace(-90, 90, 90)
        x, y, z = _sphere_xyz(lats, np.full_like(lats, lon), 1.001)
        gx += list(x) + [None]; gy += list(y) + [None]; gz += list(z) + [None]
    fig.add_trace(go.Scatter3d(
        x=gx, y=gy, z=gz, mode="lines", line=dict(color="rgba(150,210,255,0.25)", width=1),
        hoverinfo="skip", showlegend=False,
    ))

    # ---- coastlines, country borders, state / province borders ----
    coast = load_world_lines("coast")
    if coast:
        fig.add_trace(_lines_trace(coast, 1.002, "#bdf1ff", 2, "Coastlines"))
    borders = load_world_lines("borders")
    if borders:
        fig.add_trace(_lines_trace(borders, 1.003, COUNTRY_BORDER_COLOR, 2, "Country borders"))
    state_error = None
    if show_states:
        state_paths, state_error = load_all_state_lines()
        if state_paths:
            fig.add_trace(_lines_trace(state_paths, 1.0025, STATE_BORDER_COLOR, 1, "State / province borders"))
    st.session_state["_state_error"] = state_error

    # ---- severity markers ----
    counts = {level: 0 for level in SEVERITY_LEVELS}
    px_size = {"Normal": 4, "Warning": 5.5, "Extreme": 7}
    if mode == "all":
        px_size = {"Normal": 2.2, "Warning": 3.0, "Extreme": 3.0}

    for level in SEVERITY_LEVELS:
        style = SEVERITY_STYLE[level]
        subset = points[points["Severity"] == level]
        counts[level] = len(subset)
        if subset.empty:
            continue
        x, y, z = _sphere_xyz(subset["Latitude"], subset["Longitude"], 1.006)
        x, y, z = np.round(x, 4), np.round(y, 4), np.round(z, 4)
        if level != "Normal" and len(subset) <= 5000:  # soft glow ring behind alert markers
            fig.add_trace(go.Scatter3d(
                x=x, y=y, z=z, mode="markers", hoverinfo="skip", showlegend=False,
                marker=dict(size=px_size[level] * 2.6, color=style["color"], opacity=0.22, line=dict(width=0)),
            ))
        marker = dict(size=px_size[level], color=style["color"], opacity=0.95)
        if mode != "all":
            marker["line"] = dict(width=1, color="rgba(255,255,255,0.65)")
        trace = dict(
            x=x, y=y, z=z, mode="markers", marker=marker,
            name=f"{style['emoji']} {level} ({len(subset):,})",
        )
        if mode == "all":
            trace["hoverinfo"] = "skip"
        else:
            trace["text"] = _hover_text(subset).to_numpy()
            trace["hovertemplate"] = "%{text}<extra></extra>"
        fig.add_trace(go.Scatter3d(**trace))

    # In 'all' mode hover details come from one invisible marker per city (keeps the page light).
    if mode == "all" and not points.empty:
        latest = points.sort_values("Date", na_position="first").drop_duplicates("_place", keep="last")
        x, y, z = _sphere_xyz(latest["Latitude"], latest["Longitude"], 1.008)
        hover = (
            "<b>" + latest["City"].astype(str) + "</b>, " + latest["Country"].astype(str)
            + "<br>Latest status: " + latest["Severity"].astype(str)
            + "<br>Records in selection: " + latest["_place"].map(points["_place"].value_counts()).astype(str)
            + "<br>Condition: " + latest["Weather_Condition"].astype(str)
            + "<br>Temperature: " + latest["Temperature_C"].round(1).astype(str) + " °C"
        )
        fig.add_trace(go.Scatter3d(
            x=x, y=y, z=z, mode="markers", name="City details (hover)", showlegend=True,
            text=hover.to_numpy(), hovertemplate="%{text}<extra></extra>",
            marker=dict(size=5, color="rgba(255,255,255,0.0)", line=dict(width=0)),
        ))

    if focus:
        f_lat, f_lon, f_span = focus
        dist = float(np.interp(f_span, [3, 30, 120, 200], [0.72, 0.95, 1.2, 1.45]))
        ex, ey, ez = _sphere_xyz(f_lat, f_lon, dist)
    else:
        ex, ey, ez = _sphere_xyz(20, 78, 1.45)
    space = "#02030b"
    axis = dict(visible=False, range=[-1.5, 1.5])
    fig.update_layout(
        height=height,
        paper_bgcolor=space,
        uirevision=str(focus),
        margin=dict(l=0, r=0, t=5, b=0),
        font=dict(color="#dce7f3"),
        scene=dict(
            bgcolor=space, xaxis=axis, yaxis=dict(axis), zaxis=dict(axis),
            aspectmode="cube",
            camera=dict(eye=dict(x=float(ex), y=float(ey), z=float(ez))),
        ),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(11,23,40,0.75)", bordercolor=BORDER, borderwidth=1),
    )
    return fig, counts, total_rows, n_locations, sorted(str(p) for p in missing_places)


VIEW_MODES = {
    "globe": "🌍 Globe",
    "satellite": "🛰️ Satellite map",
    "street": "🌙 Dark street map",
}
SATELLITE_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/World_Imagery/MapServer/tile/{z}/{y}/{x}"
LABEL_TILES = "https://server.arcgisonline.com/ArcGIS/rest/services/Reference/World_Boundaries_and_Places/MapServer/tile/{z}/{y}/{x}"


def _focus_view(data, choice):
    """(lat, lon, span_degrees) centred on a country, or None for the whole world."""
    if choice == "World":
        return None
    sub = data[data["Country"] == choice][["Latitude", "Longitude"]].apply(pd.to_numeric, errors="coerce").dropna()
    sub = sub[sub["Latitude"].between(-90, 90) & sub["Longitude"].between(-180, 180)]
    if sub.empty:
        return None
    lat_med = float(sub["Latitude"].median())
    span = max(
        float(sub["Latitude"].max() - sub["Latitude"].min()),
        float(sub["Longitude"].max() - sub["Longitude"].min()) * float(np.cos(np.radians(lat_med))),
        1.0,
    )
    return lat_med, float(sub["Longitude"].median()), span


def _globe_points(data, mode, levels):
    """Same point selection as make_globe, for the flat map views."""
    points, place_coords, missing_places, total_rows, n_locations = _prepare_places(data)
    if mode == "worst":
        points["_rank"] = points["Severity"].map({"Normal": 0, "Warning": 1, "Extreme": 2})
        points = points.sort_values(["_rank", "Date"], na_position="first").drop(columns="_rank")
        points = points.drop_duplicates(subset="_place", keep="last")
    elif mode == "latest":
        points = points.sort_values("Date", na_position="first").drop_duplicates(subset="_place", keep="last")
    points = points.drop(columns=["Latitude", "Longitude"]).join(place_coords, on="_place")
    points = points.dropna(subset=["Latitude", "Longitude"])
    if levels:
        points = points[points["Severity"].isin(levels)]
    return points, total_rows, n_locations, sorted(str(p) for p in missing_places)


def make_map(data, mode="latest", levels=None, height=720, basemap="satellite", focus=None, show_labels=False):
    """Flat interactive map: scroll zooms toward the cursor, drag pans. Same markers as the globe."""
    points, total_rows, n_locations, missing = _globe_points(data, mode, levels)
    st.session_state["_state_error"] = None
    st.session_state["_land_error"] = None

    new_api = hasattr(go, "Scattermap")
    Trace = go.Scattermap if new_api else go.Scattermapbox
    fig = go.Figure()
    counts = {level: 0 for level in SEVERITY_LEVELS}
    sizes = {"Normal": 7, "Warning": 10, "Extreme": 13}
    if mode == "all":
        sizes = {"Normal": 5, "Warning": 7, "Extreme": 8}

    # Legend shows real record counts (same as the globe legend and the KPI cards),
    # not just the number of markers drawn.
    record_counts = data["Severity"].value_counts() if "Severity" in data.columns else None
    for level in SEVERITY_LEVELS:
        style = SEVERITY_STYLE[level]
        subset = points[points["Severity"] == level]
        counts[level] = len(subset)
        if subset.empty:
            continue
        n_records = int(record_counts.get(level, 0)) if record_counts is not None else len(subset)
        label = f"{n_records:,} records"
        if len(subset) != n_records:
            label += f" · {len(subset):,} markers"
        trace = dict(
            lat=subset["Latitude"], lon=subset["Longitude"], mode="markers",
            name=f"{style['emoji']} {level} ({label})",
            marker=dict(size=sizes[level], color=style["color"], opacity=0.92),
        )
        if mode == "all":
            trace["hoverinfo"] = "skip"
        else:
            trace.update(_hover_and_labels(subset, show_labels and (len(points) <= 800 or level != "Normal")))
        fig.add_trace(Trace(**trace))

    if mode == "all" and not points.empty:
        latest = points.sort_values("Date", na_position="first").drop_duplicates("_place", keep="last")
        hover = (
            "<b>" + latest["City"].astype(str) + "</b>, " + latest["Country"].astype(str)
            + "<br>Latest status: " + latest["Severity"].astype(str)
            + "<br>Records in selection: " + latest["_place"].map(points["_place"].value_counts()).astype(str)
            + "<br>Condition: " + latest["Weather_Condition"].astype(str)
            + "<br>Temperature: " + latest["Temperature_C"].round(1).astype(str) + " °C"
        )
        city_trace = dict(
            lat=latest["Latitude"], lon=latest["Longitude"], mode="markers", name="City details (hover)",
            hovertext=hover.to_numpy(), hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(size=9, color="#ffffff", opacity=0.02),
        )
        if show_labels and len(latest) <= 800:
            city_trace.update(
                mode="markers+text", text=latest["City"].astype(str).to_numpy(),
                textposition="top center", textfont=dict(size=10, color="#ffffff"),
            )
        fig.add_trace(Trace(**city_trace))

    if focus:
        lat0, lon0, span = focus
        zoom = float(np.clip(np.log2(360.0 / max(span, 0.5)) - 0.3, 1.0, 10.0))
    else:
        lat0, lon0, zoom = 20.0, 10.0, 1.2

    cfg = dict(center=dict(lat=lat0, lon=lon0), zoom=zoom)
    if basemap == "satellite":
        cfg["style"] = "white-bg"
        cfg["layers"] = [
            dict(below="traces", sourcetype="raster", source=[SATELLITE_TILES],
                 sourceattribution="Esri World Imagery"),
            dict(below="traces", sourcetype="raster", source=[LABEL_TILES],
                 sourceattribution="Esri Boundaries & Places"),
        ]
    else:
        cfg["style"] = "carto-darkmatter"

    fig.update_layout(
        **{"map" if new_api else "mapbox": cfg},
        height=height,
        paper_bgcolor="#02030b",
        margin=dict(l=0, r=0, t=5, b=0),
        font=dict(color="#dce7f3"),
        uirevision=f"{basemap}-{focus}",
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(11,23,40,0.75)", bordercolor=BORDER, borderwidth=1),
    )
    return fig, counts, total_rows, n_locations, missing


def _hover_and_labels(subset, labels=False):
    """Hover card for every marker, plus permanent city-name labels when asked."""
    args = dict(hovertext=_hover_text(subset).to_numpy(), hovertemplate="%{hovertext}<extra></extra>")
    if labels:
        args.update(
            mode="markers+text", text=subset["City"].astype(str).to_numpy(),
            textposition="top center", textfont=dict(size=10, color="#ffffff"),
        )
    return args


@st.cache_data(show_spinner="Preparing state borders...")
def _coarse_state_paths(tolerance: float = 0.35):
    """State borders thinned for the rotating globe (keeps dragging smooth). Raises when incomplete."""
    paths, error = load_all_state_lines()
    if not paths or error:
        raise RuntimeError(error or "no state borders available")
    out = []
    for path in paths:
        pts = [(float(pt[0]), float(pt[1])) for pt in path]
        if len(pts) >= 2:
            out.append(_decimate(pts, tolerance))
    return out


def load_coarse_state_lines():
    try:
        return _coarse_state_paths(), None
    except Exception as error:
        return [], str(error)


SPACE_KEY = "atmos_space"
TEMP_SCALE = [[0.0, "#2b6cb0"], [0.2, "#38bdf8"], [0.4, "#4ade80"], [0.6, "#fbbf24"], [0.8, "#f97316"], [1.0, "#ef4444"]]


def space_background_css():
    """CSS starfield + coloured nebula glow behind the globe."""
    import random

    rnd = random.Random(5)
    layers = [
        "radial-gradient(ellipse at 14% 22%, rgba(124,58,237,0.38), transparent 42%)",
        "radial-gradient(ellipse at 86% 18%, rgba(37,99,235,0.34), transparent 40%)",
        "radial-gradient(ellipse at 78% 84%, rgba(219,39,119,0.30), transparent 42%)",
        "radial-gradient(ellipse at 18% 88%, rgba(20,184,166,0.26), transparent 38%)",
    ]
    colors = ["#ffffff", "#ffffff", "#cfe8ff", "#fff1c1", "#ffd2a1", "#ffb3d9", "#c4b5fd"]
    for _ in range(110):
        x, y = rnd.uniform(0, 100), rnd.uniform(0, 100)
        r = rnd.choice([0.7, 0.9, 1.1, 1.4, 1.9])
        c = rnd.choice(colors)
        layers.append(f"radial-gradient(circle at {x:.1f}% {y:.1f}%, {c} 0, {c} {r}px, transparent {r + 0.8}px)")
    return (
        f'<style>.st-key-{SPACE_KEY} div[data-testid="stPlotlyChart"] {{ '
        f'background-color: #02030b !important; background-image: {", ".join(layers)} !important; }}</style>'
    )


def make_geo_globe(data, mode="latest", levels=None, height=720, show_states=True, focus=None,
                   show_labels=False, color_countries=True):
    """Rotating Earth globe: drag to rotate, scroll to zoom. It cannot be zoomed through, and every
    marker shows its city card on hover (optionally permanent city-name labels)."""
    points, total_rows, n_locations, missing = _globe_points(data, mode, levels)
    st.session_state["_land_error"] = None
    note = None
    if mode == "all" and len(points) > 4000:
        alerts = points[points["Severity"] != "Normal"].head(3000)
        normal = points[points["Severity"] == "Normal"]
        points = pd.concat([alerts, normal.sample(min(1000, len(normal)), random_state=7)])
        note = ("The globe draws at most about 4,000 markers (all Warning / Extreme first plus a sample "
                "of Normal) to stay smooth. Pick 'Latest' or 'Worst' mode, or a map view, for full detail.")
    st.session_state["_globe_note"] = note
    fig = go.Figure()

    if color_countries:
        by_country = known(data, "Country").dropna(subset=["Temperature_C"]).groupby("Country")["Temperature_C"].mean()
        if len(by_country):
            fig.add_trace(go.Choropleth(
                locations=by_country.index.tolist(), locationmode="country names",
                z=by_country.round(1).tolist(), zmin=-10, zmax=35, colorscale=TEMP_SCALE,
                marker=dict(opacity=0.62, line=dict(width=0)),
                colorbar=dict(title=dict(text="Avg °C", font=dict(size=11)), thickness=10, len=0.45,
                              x=0.99, tickfont=dict(size=10)),
                hovertemplate="%{location}<br>Average %{z} °C<extra></extra>", name="Country average",
            ))

    state_error = None
    if show_states:
        state_paths, state_error = load_coarse_state_lines()
        if state_paths:
            fig.add_trace(_geo_lines_trace(state_paths, STATE_BORDER_COLOR, 0.6, "State / province borders"))
    st.session_state["_state_error"] = state_error

    counts = {level: 0 for level in SEVERITY_LEVELS}
    sizes = {"Normal": 6, "Warning": 8, "Extreme": 10}
    if mode == "all":
        sizes = {"Normal": 4, "Warning": 5, "Extreme": 6}
    for level in SEVERITY_LEVELS:
        style = SEVERITY_STYLE[level]
        subset = points[points["Severity"] == level]
        counts[level] = len(subset)
        if subset.empty:
            continue
        trace = dict(
            lon=subset["Longitude"], lat=subset["Latitude"], mode="markers",
            name=f"{style['emoji']} {level} ({len(subset):,})",
            marker=dict(size=sizes[level], color=style["color"], opacity=0.95,
                        line=dict(width=0.8, color="rgba(255,255,255,0.85)")),
        )
        if mode == "all":
            trace["hoverinfo"] = "skip"
        else:
            trace.update(_hover_and_labels(subset, show_labels and (len(points) <= 800 or level != "Normal")))
        fig.add_trace(go.Scattergeo(**trace))

    if mode == "all" and not points.empty:
        latest = points.sort_values("Date", na_position="first").drop_duplicates("_place", keep="last")
        hover = (
            "<b>" + latest["City"].astype(str) + "</b>, " + latest["Country"].astype(str)
            + "<br>Latest status: " + latest["Severity"].astype(str)
            + "<br>Records in selection: " + latest["_place"].map(points["_place"].value_counts()).astype(str)
            + "<br>Condition: " + latest["Weather_Condition"].astype(str)
            + "<br>Temperature: " + latest["Temperature_C"].round(1).astype(str) + " °C"
        )
        city_trace = dict(
            lon=latest["Longitude"], lat=latest["Latitude"], mode="markers", name="City details (hover)",
            hovertext=hover.to_numpy(), hovertemplate="%{hovertext}<extra></extra>",
            marker=dict(size=9, color="#ffffff", opacity=0.02),
        )
        if show_labels and len(latest) <= 800:
            city_trace.update(
                mode="markers+text", text=latest["City"].astype(str).to_numpy(),
                textposition="top center", textfont=dict(size=10, color="#ffffff"),
            )
        fig.add_trace(go.Scattergeo(**city_trace))

    if focus:
        lat0, lon0, span = focus
        scale = float(np.interp(span, [3, 30, 120, 200], [7.0, 3.0, 1.5, 1.0]))
    else:
        lat0, lon0, scale = 20.0, 78.0, 1.0

    fig.update_geos(
        projection_type="orthographic",
        projection_rotation=dict(lon=lon0, lat=lat0, roll=0),
        projection_scale=scale,
        resolution=110,
        showocean=True, oceancolor="#0b4f8f",
        showland=True, landcolor="#3f9a52",
        showlakes=True, lakecolor="#2aa8e0",
        showcountries=True, countrycolor="#f1f5f9", countrywidth=0.8,
        showcoastlines=True, coastlinecolor="#bdf1ff", coastlinewidth=1,
        showframe=False, bgcolor="rgba(0,0,0,0)",
        lataxis=dict(showgrid=True, gridcolor="rgba(150,210,255,0.20)", dtick=30),
        lonaxis=dict(showgrid=True, gridcolor="rgba(150,210,255,0.20)", dtick=30),
    )
    fig.update_layout(
        height=height,
        paper_bgcolor="rgba(0,0,0,0)",
        margin=dict(l=0, r=0, t=5, b=0),
        font=dict(color="#dce7f3"),
        uirevision=str(focus),
        legend=dict(x=0.01, y=0.99, bgcolor="rgba(11,23,40,0.75)", bordercolor=BORDER, borderwidth=1,
                    itemsizing="constant"),
    )
    return fig, counts, total_rows, n_locations, missing


EARTH_TEXTURES = {
    "Blue Marble (satellite)": "earth-blue-marble.jpg",
    "Daytime map": "earth-day.jpg",
    "Night lights": "earth-night.jpg",
}
GLOBE_HEIGHT = 720
COUNTRY_LINES_URL = "https://raw.githubusercontent.com/vasturiano/globe.gl/master/example/datasets/ne_110m_admin_0_countries.geojson"

GLOBE_HTML = r"""<!doctype html><html><head><meta charset="utf-8">
<style>
html,body{margin:0;background:#02030b;overflow:hidden;font-family:Inter,Segoe UI,sans-serif}
#g{width:100vw;height:__HEIGHT__px}
#err{display:none;color:#fca5a5;padding:24px;font-size:14px}
#legend{position:absolute;left:12px;top:12px;background:rgba(11,23,40,.78);border:1px solid #1d3550;
border-radius:10px;padding:8px 12px;color:#dce7f3;font-size:12px;line-height:1.7}
#legend i{display:inline-block;width:9px;height:9px;border-radius:50%;margin-right:7px}
</style>
<script src="https://unpkg.com/globe.gl"></script></head><body>
<div id="g"></div>
<div id="legend">__LEGEND__</div>
<div id="err">The globe library could not be loaded. This view needs an internet connection
(it loads globe.gl and the Earth textures from unpkg.com).</div>
<script>
const D = __DATA__;
const IMG = 'https://unpkg.com/three-globe/example/img/';
if (typeof Globe === 'undefined') {
  document.getElementById('err').style.display = 'block';
  document.getElementById('legend').style.display = 'none';
} else {
  let K = 1;  // marker / label scale: shrinks as you zoom in so dots stay small next to states
  const world = Globe({animateIn: false})(document.getElementById('g'))
    .width(window.innerWidth).height(__HEIGHT__)
    .globeImageUrl(IMG + D.texture)
    .bumpImageUrl(IMG + 'earth-topology.png')
    .backgroundImageUrl(IMG + 'night-sky.png')
    .atmosphereColor('#7cc4ff').atmosphereAltitude(0.22)
    .pointsData(D.points).pointLat('lat').pointLng('lng').pointColor('color')
    .pointAltitude('alt').pointRadius(d => d.r * K).pointResolution(10).pointLabel('tip')
    .labelsData(D.labels).labelLat('lat').labelLng('lng').labelText('text')
    .labelSize(() => 0.4 * K).labelDotRadius(() => 0.08 * K).labelColor(() => 'rgba(255,255,255,0.95)')
    .labelResolution(2).labelAltitude(0.012);
  world.pointOfView({lat: D.pov.lat, lng: D.pov.lng, altitude: D.pov.alt}, 0);
  const c = world.controls();
  c.minDistance = 118; c.maxDistance = 600; c.zoomSpeed = 0.7; c.rotateSpeed = 0.7;
  world.renderer().setPixelRatio(Math.min(window.devicePixelRatio || 1, 1.25));
  let lastK = -1;
  const refreshSize = () => {
    const alt = Math.max(world.camera().position.length() / 100 - 1, 0.05);
    const k = Math.round(Math.min(Math.max(alt / 2.2, 0.4), 1.2) * 8) / 8;  // coarse steps: rebuild rarely
    if (k === lastK) return;
    lastK = k; K = k;
    world.pointRadius(d => d.r * K);
    world.labelSize(() => 0.4 * K).labelDotRadius(() => 0.08 * K);
  };
  c.addEventListener('change', refreshSize);
  refreshSize();
  window.addEventListener('resize', () => world.width(window.innerWidth));
  // Borders are painted straight onto the Earth texture: one image instead of thousands of line objects.
  const toPaths = g => {
    const out = [];
    g.features.forEach(f => {
      const geom = f.geometry;
      const polys = geom.type === 'Polygon' ? [geom.coordinates] : geom.coordinates;
      polys.forEach(poly => poly.forEach(ring => out.push(ring.map(q => [q[1], q[0]]))));
    });
    return out;
  };
  fetch('__BORDERS__').then(r => r.json()).then(toPaths).catch(() => []).then(countryPaths => {
    const img = new Image();
    img.crossOrigin = 'anonymous';
    img.onload = () => {
      const maxT = world.renderer().capabilities.maxTextureSize || 4096;
      const W = Math.min(6144, maxT), H = W / 2;
      const cv = document.createElement('canvas');
      cv.width = W; cv.height = H;
      const g = cv.getContext('2d');
      g.drawImage(img, 0, 0, W, H);
      g.lineJoin = 'round';
      const stroke = (paths, color, width) => {
        g.strokeStyle = color; g.lineWidth = width; g.beginPath();
        paths.forEach(pts => {
          let prev = null;
          pts.forEach(q => {
            const x = (q[1] + 180) / 360 * W, y = (90 - q[0]) / 180 * H;
            if (prev === null || Math.abs(q[1] - prev) > 180) g.moveTo(x, y); else g.lineTo(x, y);
            prev = q[1];
          });
        });
        g.stroke();
      };
      stroke(D.paths.map(p => p.pts), 'rgba(170,205,255,0.45)', 1.2);
      stroke(countryPaths, 'rgba(235,242,255,0.62)', 1.8);
      cv.toBlob(b => { if (b) world.globeImageUrl(URL.createObjectURL(b)); }, 'image/jpeg', 0.92);
    };
    img.src = IMG + D.texture;
  });
}
</script></body></html>"""


def make_webgl_globe(data, mode="latest", levels=None, height=GLOBE_HEIGHT, show_states=True, focus=None,
                     show_labels=False, earth_style="Blue Marble (satellite)"):
    """Realistic WebGL globe (globe.gl): satellite texture, terrain relief, atmosphere, starry sky.
    Returns (html, counts, total_rows, n_locations, missing_places)."""
    import html as html_lib
    import json

    points, total_rows, n_locations, missing = _globe_points(data, mode, levels)
    st.session_state["_land_error"] = None

    note = None
    if mode == "all":
        # Readings of one city share the same spot, so draw one dot per city (its worst status)
        # instead of a pile of overlapping dots, and show how many records it stands for.
        per_place = points["_place"].value_counts()
        ranked = points.assign(_rank=points["Severity"].map({"Normal": 0, "Warning": 1, "Extreme": 2}))
        points = ranked.sort_values(["_rank", "Date"], na_position="first").drop_duplicates("_place", keep="last")
        points = points.drop(columns="_rank")
        points["_n"] = points["_place"].map(per_place)
        note = ("Records from the same city share one spot, so the globe draws one dot per city "
                "(its worst status) and the hover card shows how many records it covers.")
    if len(points) > 1200:
        alerts = points[points["Severity"] != "Normal"].head(800)
        normal = points[points["Severity"] == "Normal"]
        points = pd.concat([alerts, normal.sample(min(400, len(normal)), random_state=7)])
        note = ("The globe draws at most about 1,200 markers (all Warning / Extreme first plus a sample "
                "of Normal) to stay smooth.")
    st.session_state["_globe_note"] = note

    def esc(value):
        return html_lib.escape(str(value))

    def fmt(value):
        return "N/A" if value is None or pd.isna(value) else f"{value:.1f}"

    alt = {"Normal": 0.001, "Warning": 0.001, "Extreme": 0.001}  # flat 2D dots
    rad = {"Normal": 0.26, "Warning": 0.34, "Extreme": 0.42}
    if mode == "all":
        rad = {"Normal": 0.26, "Warning": 0.34, "Extreme": 0.42}

    counts = {level: 0 for level in SEVERITY_LEVELS}
    pts = []
    for level in SEVERITY_LEVELS:
        sub = points[points["Severity"] == level]
        counts[level] = len(sub)
        color = SEVERITY_STYLE[level]["color"]
        n_rec = sub["_n"] if "_n" in sub.columns else pd.Series(np.nan, index=sub.index)
        for lat, lon, city, country, cond, temp, wind, rain, when, n in zip(
            sub["Latitude"], sub["Longitude"], sub["City"], sub["Country"], sub["Weather_Condition"],
            sub["Temperature_C"], _col(sub, "Wind_Speed_kmh"), _col(sub, "Precipitation_mm"), sub["Date"], n_rec,
        ):
            seen = "" if pd.isna(when) else f"<br>Observed: {when:%d %b %Y %H:%M}"
            if not pd.isna(n):
                seen += f"<br>Records here: {int(n):,}"
            tip = (
                f"<div style='font:12px sans-serif;line-height:1.5'><b>{esc(city)}</b>, {esc(country)}"
                f"<br>Status: <b style='color:{color}'>{level}</b><br>Condition: {esc(cond)}"
                f"<br>Temperature: {fmt(temp)} °C<br>Wind: {fmt(wind)} km/h<br>Rain: {fmt(rain)} mm{seen}</div>"
            )
            pts.append({"lat": round(float(lat), 3), "lng": round(float(lon), 3), "color": color,
                        "alt": alt[level], "r": rad[level], "tip": tip})

    labels = []
    if show_labels and not points.empty:
        src = points if len(points) <= 300 else points[points["Severity"] != "Normal"]
        src = src.drop_duplicates("_place").head(300)
        labels = [
            {"lat": round(float(la), 3), "lng": round(float(lo), 3), "text": str(city)}
            for la, lo, city in zip(src["Latitude"], src["Longitude"], src["City"])
        ]

    paths, state_error = [], None
    if show_states:
        state_paths, state_error = load_coarse_state_lines()
        paths = [
            {"pts": [[round(la, 3), round(lo, 3)] for lo, la in path], "color": "rgba(251,191,36,0.8)"}
            for path in state_paths
        ]
    st.session_state["_state_error"] = state_error

    if focus:
        f_lat, f_lon, span = focus
        pov = {"lat": f_lat, "lng": f_lon, "alt": float(np.interp(span, [3, 30, 120, 200], [0.35, 0.8, 1.6, 2.4]))}
    else:
        pov = {"lat": 20, "lng": 78, "alt": 2.2}

    records = data["Severity"].value_counts() if "Severity" in data.columns else counts
    legend = "<b>Weather status (records)</b><br>" + "<br>".join(
        f"<i style='background:{SEVERITY_STYLE[lv]['color']}'></i>{lv}: {int(records.get(lv, 0)):,}"
        for lv in SEVERITY_LEVELS
    ) + f"<br><span style='opacity:.65'>{sum(counts.values()):,} markers drawn</span>"
    legend += "<br><span style='opacity:.65'>Hover a dot for city details</span>"
    payload = {"points": pts, "labels": labels, "paths": paths, "pov": pov,
               "texture": EARTH_TEXTURES.get(earth_style, "earth-blue-marble.jpg")}
    data_json = json.dumps(payload, separators=(",", ":")).replace("</", "<\\/")
    html_doc = (
        GLOBE_HTML.replace("__HEIGHT__", str(int(height))).replace("__LEGEND__", legend)
        .replace("__BORDERS__", COUNTRY_LINES_URL).replace("__DATA__", data_json)
    )
    return html_doc, counts, total_rows, n_locations, missing


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
            "Drag to rotate, scroll to zoom, hover a marker for details. Choose what each marker shows with 'Marker mode'.")
    ctl_left, ctl_right, ctl_extra = st.columns([3, 2, 1.4])
    with ctl_left:
        shown_levels = st.multiselect(
            "Show weather status", SEVERITY_LEVELS, default=SEVERITY_LEVELS,
            format_func=lambda lvl: f"{SEVERITY_STYLE[lvl]['emoji']} {lvl}",
        )
    with ctl_right:
        globe_mode = st.selectbox(
            "Marker mode", list(GLOBE_MODES), index=1, format_func=lambda m: GLOBE_MODES[m],
            help="'Latest' and 'Worst' draw one marker per city. 'Every observation' draws "
                 "individual records, so repeated readings of one city overlap on the same spot.",
        )
    with ctl_extra:
        st.markdown("<div style='height:1.7rem'></div>", unsafe_allow_html=True)
        show_states = st.checkbox(
            "State borders (all countries)", value=True,
            help="Draws state / province borders for every country on the globe.",
        )

    view_col, focus_col, opt_col = st.columns([3, 2, 2])
    with view_col:
        view_mode = st.radio(
            "View mode", list(VIEW_MODES), horizontal=True, format_func=lambda v: VIEW_MODES[v],
            help="Globe: rotate the whole Earth. Satellite / dark street map: flat maps that zoom "
                 "in exactly where your cursor is.",
        )
    with focus_col:
        focus_options = ["World"] + sorted(c for c in working["Country"].unique() if c != "Unknown")
        focus_choice = st.selectbox(
            "Zoom to country", focus_options,
            help="Re-centres the view on this country (works in every view mode).",
        )
    with opt_col:
        show_labels = st.checkbox(
            "Show city names", value=False,
            help="Writes each city's name next to its marker (when there are many cities, only "
                 "Warning / Extreme ones are labelled). Hover any marker for the full details.",
        )
        earth_style = st.selectbox(
            "Earth style (globe)", list(EARTH_TEXTURES),
            help="Satellite photo, daytime map, or city lights at night.",
        )
    focus = _focus_view(working, focus_choice)

    try:
        holder = st.container(key=SPACE_KEY)
    except TypeError:  # older Streamlit without keyed containers: plain dark background instead
        holder = st.container()

    with st.spinner("Drawing map..."):
        if view_mode == "globe":
            globe, level_counts, globe_rows, globe_locations, missing_cities = make_webgl_globe(
                working, mode=globe_mode, levels=shown_levels or None, show_states=show_states,
                focus=focus, show_labels=show_labels, earth_style=earth_style,
            )
        else:
            globe, level_counts, globe_rows, globe_locations, missing_cities = make_map(
                working, mode=globe_mode, levels=shown_levels or None, basemap=view_mode,
                focus=focus, show_labels=show_labels,
            )
    with holder:
        if view_mode == "globe":
            import streamlit.components.v1 as components

            components.html(globe, height=GLOBE_HEIGHT)
        else:
            st.plotly_chart(
                globe, use_container_width=True, theme=None,
                key=f"globe_{view_mode}_{globe_mode}_{focus_choice}",
                config={"displaylogo": False, "scrollZoom": True},
            )
    if view_mode == "globe":
        st.caption(
            "Drag to spin the Earth, scroll to zoom (it stops before the surface). Hover a marker for "
            "its city card. To zoom exactly to your cursor, use a map view or 'Zoom to country'."
        )
        if st.session_state.get("_globe_note"):
            st.info(st.session_state["_globe_note"])
    if view_mode == "globe" and show_states and st.session_state.get("_state_error"):
        st.warning(
            "State / province borders could not be fully loaded (the download failed). Check your "
            "internet connection and reload the page. India's borders can also come from an "
            "`india_states.geojson` file next to app.py."
        )
        st.caption(st.session_state["_state_error"])
    if view_mode == "globe" and st.session_state.get("_land_error"):
        st.warning(
            "The land outlines could not be downloaded, so the Earth is drawn as plain ocean. "
            "Reload the page once you are online."
        )
        st.caption(st.session_state["_land_error"])
    drawn = sum(level_counts.values())
    st.caption(
        f"Your selection has **{globe_rows:,}** records covering **{globe_locations:,}** "
        f"recorded places. Drawing **{drawn:,}** markers ({GLOBE_MODES[globe_mode].lower()})."
    )
    if globe_mode == "all" and globe_locations < drawn:
        st.info(
            f"All {drawn:,} records are drawn, but they come from only {globe_locations:,} places, "
            "so readings from the same city sit on exactly the same spot and look like one dot. "
            "That is accurate: the map cannot show more distinct positions than your data has."
        )
    if missing_cities:
        st.warning(
            f"{len(missing_cities):,} recorded place(s) have no valid latitude/longitude anywhere "
            "in the data, so they cannot be placed on the globe."
        )
        with st.expander("Show places without coordinates"):
            st.write(", ".join(missing_cities[:500]))
    rec_counts = working["Severity"].value_counts()
    kpi_row([
        ("🟢 Normal", f"{int(rec_counts.get('Normal', 0)):,}", "Records in selection", SEVERITY_COLORS["Normal"]),
        ("🟡 Warning", f"{int(rec_counts.get('Warning', 0)):,}", "Records in selection", SEVERITY_COLORS["Warning"]),
        ("🔴 Extreme", f"{int(rec_counts.get('Extreme', 0)):,}", "Records in selection", SEVERITY_COLORS["Extreme"]),
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
