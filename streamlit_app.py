"""
Market Access Dashboard — Prior Authorization (PsO) Extraction Pipeline
=======================================================================

A highly interactive Streamlit dashboard for the `result` table produced by the
PA extraction pipeline. It is resilient to column-naming differences (spaces vs
underscores), safely cleans 'NA' strings to real NaNs, coerces the numeric
columns with `pd.to_numeric(errors='coerce')`, and visualizes the data from as
many angles as the dataset allows — all rendered in a refined dark theme.

Run with:
    pip install streamlit pandas plotly openpyxl
    streamlit run market_access_dashboard.py

Use the uploader at the top of the page to load a CSV or Excel file
(.csv / .xlsx / .xls). If you don't upload anything, the app falls back to a
local `result.csv` (or `result.xlsx`) in the working directory.
(`openpyxl` is only required if you upload/read Excel files.)

The loader accepts BOTH header styles, e.g. either
    "Number_of_Steps_through_Brands"   (underscore form)
    "Number of Steps through Brands"   (spaced form, as in result.xlsx)
and maps them to a single canonical schema before analysis.
"""

from __future__ import annotations

import io
import re

import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import plotly.io as pio
import streamlit as st

# ─────────────────────────────────────────────────────────────────────────────
# Column groups & schema (canonical names)
# ─────────────────────────────────────────────────────────────────────────────
NUMERIC_COLS = [
    "Number_of_Steps_through_Generic",
    "Number_of_Steps_through_Brands",
    "Initial_Authorization_Duration_months",
    "Reauthorization_Duration_months",
    "Access Score",
]
# Yes / No / NA flag columns
YESNO_COLS = [
    "Step_through_Phototherapy",
    "TB_Test_required",
    "Quantity_Limits",
    "Reauthorization_Required",
]
TEXT_COLS = ["Filename", "Brand", "Age", "Specialist_Types"]

EXPECTED_COLS = TEXT_COLS + NUMERIC_COLS + YESNO_COLS

# Friendly labels for the flag columns
YESNO_LABELS = {
    "Step_through_Phototherapy": "Phototherapy Step",
    "TB_Test_required": "TB Test Required",
    "Quantity_Limits": "Quantity Limits",
    "Reauthorization_Required": "Reauthorization Required",
}

# Values that should be treated as missing
NA_LIKE = {"na", "n/a", "nan", "none", "null", "", "-", "--", "n.a.",
           "not specified", "unspecified"}

# ─────────────────────────────────────────────────────────────────────────────
# Column canonicalization
# ─────────────────────────────────────────────────────────────────────────────
def _norm_key(s) -> str:
    """Lower-case and strip everything that isn't a letter or digit."""
    return re.sub(r"[^a-z0-9]+", "", str(s).strip().lower())


_CANON_VARIANTS = {
    "Filename": ["filename", "file", "policyfile", "document", "documentname"],
    "Brand": ["brand", "drug", "product", "medication", "agent"],
    "Age": ["age", "agerestriction", "agelimit", "agecriteria"],
    "Specialist_Types": ["specialisttypes", "specialisttype", "specialist",
                         "specialists", "prescriber", "prescriberspecialty",
                         "prescribingspecialist", "specialtytypes"],
    "Number_of_Steps_through_Generic": ["numberofstepsthroughgeneric",
                                        "stepsthroughgeneric", "genericsteps",
                                        "numberofgenericsteps", "generic"],
    "Number_of_Steps_through_Brands": ["numberofstepsthroughbrands",
                                       "numberofstepsthroughbrand",
                                       "stepsthroughbrands", "stepsthroughbrand",
                                       "brandsteps", "numberofbrandsteps", "branded"],
    "Initial_Authorization_Duration_months": ["initialauthorizationdurationmonths",
                                              "initialauthorizationdurationinmonths",
                                              "initialauthorizationduration",
                                              "initialauthduration",
                                              "initialauthorization"],
    "Reauthorization_Duration_months": ["reauthorizationdurationmonths",
                                        "reauthorizationdurationinmonths",
                                        "reauthorizationduration", "reauthduration",
                                        "reauthorisationduration"],
    "Access Score": ["accessscore", "score", "accessindex"],
    "Step_through_Phototherapy": ["stepthroughphototherapy", "phototherapystep",
                                  "phototherapy", "stepsthroughphototherapy"],
    "TB_Test_required": ["tbtestrequired", "tbtest", "tbrequired",
                         "tuberculosistest", "tbscreening"],
    "Quantity_Limits": ["quantitylimits", "quantitylimit", "qtylimits",
                        "quantitylevellimit", "quantitylimitations"],
    "Reauthorization_Required": ["reauthorizationrequired", "reauthrequired",
                                 "reauthorisationrequired"],
}

ALIAS_TO_CANON: dict[str, str] = {}
for _canon, _variants in _CANON_VARIANTS.items():
    for _v in [_norm_key(_canon)] + _variants:
        ALIAS_TO_CANON[_v] = _canon


def canonicalize_columns(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df.columns = [str(c).strip() for c in df.columns]

    out = pd.DataFrame(index=df.index)
    for col in df.columns:
        canon = ALIAS_TO_CANON.get(_norm_key(col), col)
        if canon in out.columns:
            if df[col].notna().sum() > out[canon].notna().sum():
                out[canon] = df[col].values
        else:
            out[canon] = df[col].values
    return out


# ─────────────────────────────────────────────────────────────────────────────
# Aesthetic system  —  "after-dark editorial analytics"
# ─────────────────────────────────────────────────────────────────────────────
INK = "#ECEAE3"          # warm light primary text
SUBTLE = "#9C988C"       # muted warm grey label text
FAINT = "#6E6A60"        # very muted (footnotes, rules)
ACCENT = "#34D8C6"       # luminous teal (primary)
ACCENT_SOFT = "#6FE9DA"  # lighter teal
ACCENT2 = "#F0A93B"      # warm amber (secondary)
PAPER = "#0C0E12"        # app background base
PANEL = "#13171E"        # sidebar / raised panel
CARD = "#161B23"         # chart & metric card
CARD_HI = "#1C222C"      # inputs / hovered surfaces
LINE = "rgba(236,234,227,0.10)"  # hairline borders
GRID = "rgba(236,234,227,0.07)"  # plot gridlines
INK_ON_ACCENT = "#08110F"        # dark text used on accent fills

PALETTE = [
    "#34D8C6", "#F0A93B", "#6FA8FF", "#B79CFF", "#FF8FB3",
    "#57D6A0", "#FF9F5A", "#8AA0FF", "#3FD0E6", "#FF7E7E",
]

FONT_BODY = "IBM Plex Sans, -apple-system, Segoe UI, sans-serif"
FONT_MONO = "IBM Plex Mono, ui-monospace, monospace"

DONUT_COLORS = {"Yes": ACCENT, "No": "#46505E", "Not specified": "#2A313B"}


def install_plotly_theme() -> None:
    """Register and activate a cohesive dark Plotly template with responsive margins."""
    tpl = go.layout.Template()
    tpl.layout = go.Layout(
        font=dict(family=FONT_BODY, color=INK, size=13),
        paper_bgcolor="rgba(0,0,0,0)",
        plot_bgcolor="rgba(0,0,0,0)",
        colorway=PALETTE,
        title=dict(font=dict(family="Fraunces, Georgia, serif", size=18, color=INK),
                   x=0.0, xanchor="left"),
        # Increased native margins to provide internal breathing room (replacing CSS padding)
        margin=dict(l=24, r=24, t=64, b=24, autoexpand=True),
        xaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE, tickcolor=LINE,
                   title_font=dict(size=12, color=SUBTLE), tickfont=dict(size=11, color=SUBTLE),
                   automargin=True), # Safely scales long X-axis labels
        yaxis=dict(gridcolor=GRID, zerolinecolor=GRID, linecolor=LINE,
                   title_font=dict(size=12, color=SUBTLE), tickfont=dict(size=11, color=SUBTLE),
                   automargin=True), # Safely scales long Y-axis labels
        legend=dict(bgcolor="rgba(0,0,0,0)", font=dict(size=11, color=SUBTLE)),
        hoverlabel=dict(bgcolor=CARD_HI, bordercolor=LINE,
                        font=dict(family=FONT_BODY, size=12, color=INK)),
        colorscale=dict(sequential="Viridis"),
    )
    pio.templates["macc_dark"] = tpl
    pio.templates.default = "macc_dark"


# Dynamic CSS variables --------------------------------------------------------
_CSS_VARS = f"""
@import url('https://fonts.googleapis.com/css2?family=Fraunces:opsz,wght@9..144,400;9..144,500;9..144,600;9..144,700&family=IBM+Plex+Mono:wght@400;500;600&family=IBM+Plex+Sans:wght@300;400;500;600;700&display=swap');
:root {{
  --ink:{INK}; --subtle:{SUBTLE}; --faint:{FAINT};
  --accent:{ACCENT}; --accent-soft:{ACCENT_SOFT}; --accent2:{ACCENT2};
  --paper:{PAPER}; --panel:{PANEL}; --card:{CARD}; --card-hi:{CARD_HI};
  --line:{LINE}; --grid:{GRID}; --ink-on-accent:{INK_ON_ACCENT};
  --font-body:{FONT_BODY}; --font-mono:{FONT_MONO};
  color-scheme: dark;
}}
"""

# Static CSS rules -------------------------------------------------------------
_CSS_RULES = r"""
@keyframes maccRise { from { opacity:0; transform:translateY(10px);} to {opacity:1; transform:none;} }

html, body, [class*="css"] { -webkit-font-smoothing:antialiased; }

.stApp {
  background:
    radial-gradient(1100px 560px at 14% -10%, rgba(52,216,198,.10), rgba(52,216,198,0) 60%),
    radial-gradient(900px 520px at 92% 4%, rgba(240,169,59,.06), rgba(240,169,59,0) 55%),
    linear-gradient(180deg, var(--paper) 0%, #0A0C10 100%);
  background-attachment: fixed;
  color: var(--ink);
  font-family: var(--font-body);
}

.stApp, .stApp p, .stApp li, .stApp span, .stApp label, .stApp small,
[data-testid="stMarkdownContainer"], [data-testid="stMarkdownContainer"] p,
[data-testid="stWidgetLabel"], [data-testid="stWidgetLabel"] p,
[data-testid="stHeadingWithActionElements"] { color: var(--ink); }

.block-container { padding-top: 1.6rem; padding-bottom: 4rem; max-width: 1500px; }

h1, h2, h3, h4 { font-family:'Fraunces', Georgia, serif !important; color:var(--ink); letter-spacing:-0.01em; }

/* Hero */
.macc-hero {
  border:1px solid var(--line); border-radius:20px; padding:28px 32px;
  background:
    radial-gradient(700px 240px at 88% -40%, rgba(52,216,198,.16), rgba(52,216,198,0) 70%),
    linear-gradient(135deg, #181D26 0%, #11151C 100%);
  box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 30px 60px -34px rgba(0,0,0,.75);
  position:relative; overflow:hidden; animation: maccRise .5s ease both;
}
.macc-hero:before { content:""; position:absolute; right:-70px; top:-70px;
  width:280px; height:280px; background: radial-gradient(circle, rgba(52,216,198,.20), rgba(52,216,198,0) 70%); }
.macc-hero h1 { font-size:2.45rem; font-weight:600; margin:6px 0 8px 0; line-height:1.04;
  background: linear-gradient(180deg, #FFFFFF 10%, #CFE9E4 95%);
  -webkit-background-clip:text; background-clip:text; -webkit-text-fill-color:transparent; }
.macc-hero .sub { color:var(--subtle); font-size:1.03rem; max-width:64ch; line-height:1.55; }
.macc-eyebrow { font-family:var(--font-mono); text-transform:uppercase; letter-spacing:.24em; font-size:.72rem; color:var(--accent); font-weight:600; }
.macc-tags { margin-top:16px; display:flex; gap:8px; flex-wrap:wrap; }
.macc-tag { font-family:var(--font-mono); font-size:.72rem; color:var(--subtle);
  border:1px solid var(--line); border-radius:999px; padding:5px 12px;
  background:rgba(255,255,255,.03); backdrop-filter:blur(4px); }

/* Section headers */
.macc-section { margin:34px 0 8px 0; }
.macc-section .macc-eyebrow { color:var(--accent2); }
.macc-section h2 { font-size:1.55rem; font-weight:600; margin:3px 0 2px 0; }
.macc-section .desc { color:var(--subtle); font-size:.93rem; margin-bottom:4px; line-height:1.5; }
.macc-rule { height:2px; width:60px; background:linear-gradient(90deg, var(--accent), rgba(52,216,198,0)); border-radius:2px; margin-top:9px; }

/* Enforce strict box-sizing globally for these elements to prevent layout shift */
[data-testid="stMetric"], [data-testid="stPlotlyChart"], [data-testid="stExpander"] { box-sizing: border-box; }

/* Metric cards */
[data-testid="stMetric"] {
  background: linear-gradient(180deg, var(--card) 0%, #12161D 100%);
  border:1px solid var(--line); border-radius:16px; padding:16px 18px 14px 18px;
  box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 22px 40px -30px rgba(0,0,0,.8);
  transition: transform .16s ease, box-shadow .16s ease, border-color .16s ease;
}
[data-testid="stMetric"]:hover { transform:translateY(-3px); border-color:rgba(52,216,198,.45);
  box-shadow: 0 26px 46px -28px rgba(52,216,198,.35); }
[data-testid="stMetricLabel"], [data-testid="stMetricLabel"] p { color:var(--subtle) !important; font-weight:500; }
[data-testid="stMetricLabel"] p { font-size:.82rem; letter-spacing:.01em; }
[data-testid="stMetricValue"] { font-family:var(--font-mono); font-weight:600; color:var(--ink); font-size:1.95rem; }
[data-testid="stMetricDelta"] { font-family:var(--font-mono); font-size:.78rem; }

/* Expanders */
[data-testid="stExpander"] { border:1px solid var(--line); border-radius:16px; background:rgba(255,255,255,.02); overflow:hidden; }
[data-testid="stExpander"] summary { font-family:'Fraunces', serif; font-size:1.05rem; color:var(--ink); }
[data-testid="stExpander"] summary:hover { color:var(--accent); }
[data-testid="stExpander"] svg { fill:var(--subtle); }

/* Plotly card framing - OVERFLOW HIDDEN KILLS SCROLLBARS */
[data-testid="stPlotlyChart"] {
  background: linear-gradient(180deg, var(--card) 0%, #13171E 100%);
  border:1px solid var(--line); border-radius:16px;
  box-shadow: 0 1px 0 rgba(255,255,255,.03) inset, 0 22px 40px -32px rgba(0,0,0,.8);
  overflow: hidden; 
}
.macc-caption { color:var(--subtle); font-size:.8rem; font-style:italic; margin:-2px 0 8px 2px; }

/* Sidebar */
[data-testid="stSidebar"] { background: linear-gradient(180deg, var(--panel) 0%, #0F131A 100%); border-right:1px solid var(--line); }
[data-testid="stSidebar"] * { color:var(--ink); }
[data-testid="stSidebar"] h2, [data-testid="stSidebar"] h3 { font-size:1.08rem; color:var(--ink); }
[data-testid="stSidebar"] [data-testid="stWidgetLabel"] p, [data-testid="stSidebar"] label { color:var(--ink) !important; font-weight:500; }
[data-testid="stSidebar"] hr { border-color:var(--line); }

/* Inputs */
[data-baseweb="select"] > div, [data-baseweb="input"] > div, [data-testid="stSidebar"] [data-baseweb="select"] > div {
  background: var(--card-hi) !important; border-color: var(--line) !important; color: var(--ink) !important; border-radius:10px !important; }
[data-baseweb="select"] input, [data-baseweb="input"] input, [data-testid="stSidebar"] input { color: var(--ink) !important; }
[data-baseweb="select"] svg { fill: var(--subtle) !important; }
[data-baseweb="select"] [data-baseweb="select"] div { color: var(--ink) !important; }
::placeholder { color: var(--faint) !important; }

/* Tags / Menus / Sliders */
[data-baseweb="tag"] { background: var(--accent) !important; border:none !important; border-radius:8px !important; }
[data-baseweb="tag"] span, [data-baseweb="tag"] div { color: var(--ink-on-accent) !important; }
[data-baseweb="tag"] [role="presentation"] svg, [data-baseweb="tag"] svg { fill: var(--ink-on-accent) !important; }
[data-baseweb="popover"] [role="listbox"], ul[role="listbox"], [data-baseweb="menu"], [data-baseweb="menu"] ul {
  background: var(--card-hi) !important; border:1px solid var(--line) !important; border-radius:12px !important; box-shadow:0 18px 40px -18px rgba(0,0,0,.85) !important; }
[role="listbox"] li, [role="option"], [data-baseweb="menu"] li { background: transparent !important; color: var(--ink) !important; }
[role="option"]:hover, [role="option"][aria-selected="true"], [data-baseweb="menu"] li:hover { background: rgba(52,216,198,.16) !important; color: var(--accent-soft) !important; }
[data-baseweb="slider"] [role="slider"] { background: var(--accent) !important; border-color: var(--accent) !important; box-shadow:0 0 0 4px rgba(52,216,198,.18) !important; }
[data-testid="stSliderThumbValue"], [data-testid="stThumbValue"] { color: var(--accent-soft) !important; font-family:var(--font-mono); }
[data-testid="stTickBarMin"], [data-testid="stTickBarMax"] { color: var(--subtle) !important; }
[data-baseweb="slider"] div[role="progressbar"] ~ div { background: var(--accent) !important; }

/* Uploader & Buttons */
[data-testid="stFileUploaderDropzone"] { background: var(--card) !important; border:1.5px dashed var(--line) !important; border-radius:14px !important; transition:border-color .15s ease; }
[data-testid="stFileUploaderDropzone"]:hover { border-color: rgba(52,216,198,.5) !important; }
[data-testid="stFileUploaderDropzone"] *, [data-testid="stFileUploaderDropzoneInstructions"] * { color: var(--ink) !important; }
[data-testid="stFileUploaderDropzoneInstructions"] small { color: var(--subtle) !important; }
[data-testid="stFileUploader"] button { background: var(--card-hi) !important; color: var(--ink) !important; border:1px solid var(--line) !important; border-radius:10px !important; }
[data-testid="stFileUploader"] button:hover { border-color: var(--accent) !important; color: var(--accent) !important; }
[data-testid="stVerticalBlockBorderWrapper"] { border-radius:16px; }
.stButton > button, [data-testid="stBaseButton-secondary"] { background: var(--card-hi); color: var(--ink); border:1px solid var(--line); border-radius:11px; font-weight:500; transition: all .15s ease; }
.stButton > button:hover { border-color: var(--accent); color: var(--accent); box-shadow:0 0 0 3px rgba(52,216,198,.12); }
[data-testid="stDownloadButton"] button { background: rgba(52,216,198,.12); color: var(--accent-soft); border:1px solid rgba(52,216,198,.4); }
[data-testid="stDownloadButton"] button:hover { background: rgba(52,216,198,.2); }
[data-testid="stAlert"] { border-radius:13px; border:1px solid var(--line); background: rgba(255,255,255,.03); }
[data-testid="stAlert"] * { color: var(--ink) !important; }
[data-testid="stDataFrame"], [data-testid="stDataFrameResizable"] { border:1px solid var(--line); border-radius:13px; overflow:hidden; }
code { color: var(--accent-soft); background: rgba(52,216,198,.1); border-radius:5px; padding:1px 5px; }

#MainMenu, footer { visibility:hidden; }
[data-testid="stHeader"] { background: rgba(0,0,0,0); }
::-webkit-scrollbar { width:11px; height:11px; }
::-webkit-scrollbar-thumb { background: rgba(236,234,227,.14); border-radius:8px; border:2px solid transparent; background-clip:content-box; }
::-webkit-scrollbar-thumb:hover { background: rgba(52,216,198,.4); background-clip:content-box; }
::-webkit-scrollbar-track { background: transparent; }
"""

CSS = "<style>\n" + _CSS_VARS + "\n" + _CSS_RULES + "\n</style>"


# ─────────────────────────────────────────────────────────────────────────────
# Small formatting helpers
# ─────────────────────────────────────────────────────────────────────────────
def safe_mean(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.mean()) if len(v) else np.nan


def safe_median(s: pd.Series) -> float:
    v = pd.to_numeric(s, errors="coerce").dropna()
    return float(v.median()) if len(v) else np.nan


def fmt_num(x: float, dec: int = 1) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.{dec}f}"


def fmt_pct(x: float) -> str:
    return "—" if x is None or (isinstance(x, float) and np.isnan(x)) else f"{x:.1f}%"


def delta_str(filtered: float, overall: float, pct: bool = False) -> str | None:
    if any(v is None or (isinstance(v, float) and np.isnan(v)) for v in (filtered, overall)):
        return None
    d = filtered - overall
    return f"{d:+.1f}%" if pct else f"{d:+.1f}"


def req_share(series: pd.Series) -> float:
    n = int(len(series))
    if n == 0:
        return np.nan
    return float((series == "Yes").sum()) / n * 100


# ─────────────────────────────────────────────────────────────────────────────
# Data loading & cleaning
# ─────────────────────────────────────────────────────────────────────────────
def normalize_yesno(series: pd.Series) -> pd.Series:
    yes = {"yes", "y", "true", "1", "required", "positive"}
    no = {"no", "n", "false", "0", "not required", "negative"}

    def _map(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip().lower()
        if t in NA_LIKE:
            return np.nan
        if t in yes:
            return "Yes"
        if t in no:
            return "No"
        return str(v).strip().title()

    return series.map(_map)


def normalize_quantity_limits(series: pd.Series) -> pd.Series:
    def _map(v):
        if pd.isna(v):
            return np.nan
        t = str(v).strip()
        if t.lower() in NA_LIKE:
            return np.nan
        if t.lower() in {"yes", "y", "true", "required"}:
            return "Yes"
        if t.lower() in {"no", "n", "false", "not required"}:
            return "No"
        return "Yes"

    return series.map(_map)


def clean_data(df: pd.DataFrame) -> pd.DataFrame:
    df = canonicalize_columns(df)
    for col in EXPECTED_COLS:
        if col not in df.columns:
            df[col] = np.nan

    for c in df.columns:
        df[c] = df[c].map(
            lambda v: np.nan if (isinstance(v, str) and v.strip().lower() in NA_LIKE) else v
        )

    for c in NUMERIC_COLS:
        df[c] = pd.to_numeric(df[c], errors="coerce")

    for c in YESNO_COLS:
        if c == "Quantity_Limits":
            df[c] = normalize_quantity_limits(df[c])
        else:
            df[c] = normalize_yesno(df[c])

    for c in ["Filename", "Brand", "Age", "Specialist_Types"]:
        df[c] = df[c].map(lambda v: str(v).strip() if isinstance(v, str) else v)
    df["Brand"] = df["Brand"].map(lambda v: v.upper() if isinstance(v, str) else v)

    df["Total_Steps"] = (
        df["Number_of_Steps_through_Generic"].fillna(0)
        + df["Number_of_Steps_through_Brands"].fillna(0)
    )
    both_missing = (
        df["Number_of_Steps_through_Generic"].isna()
        & df["Number_of_Steps_through_Brands"].isna()
    )
    df.loc[both_missing, "Total_Steps"] = np.nan

    return df


def _read_table_bytes(data: bytes, filename: str) -> pd.DataFrame:
    name = (filename or "").lower()
    if name.endswith((".xlsx", ".xlsm", ".xls")):
        try:
            return pd.read_excel(io.BytesIO(data))
        except ImportError as exc:
            raise ImportError("Please install openpyxl to read Excel files.") from exc
    return pd.read_csv(io.BytesIO(data))


def _read_table_path(path: str) -> pd.DataFrame:
    p = path.lower()
    if p.endswith((".xlsx", ".xlsm", ".xls")):
        return pd.read_excel(path)
    return pd.read_csv(path)


@st.cache_data(show_spinner=False)
def load_and_clean_from_bytes(data: bytes, filename: str) -> pd.DataFrame:
    return clean_data(_read_table_bytes(data, filename))


@st.cache_data(show_spinner=False)
def load_and_clean_from_path(path: str) -> pd.DataFrame:
    return clean_data(_read_table_path(path))


def find_local_data() -> str | None:
    import os
    for candidate in ("result.csv", "result.xlsx", "result.xlsm", "result.xls"):
        if os.path.exists(candidate):
            return candidate
    return None


SPLIT_RE = re.compile(r"\s*(?:[,;/|]|\band\b|\bor\b|&|\+)\s*", flags=re.IGNORECASE)


def row_specialists(value) -> list[str]:
    if pd.isna(value):
        return []
    parts = SPLIT_RE.split(str(value))
    out = []
    for p in parts:
        t = p.strip().title()
        if t and t.lower() not in NA_LIKE:
            out.append(t)
    return out


def specialist_long(df: pd.DataFrame) -> pd.DataFrame:
    rows = []
    for _, r in df.iterrows():
        for sp in row_specialists(r.get("Specialist_Types")):
            rows.append({
                "Filename": r.get("Filename"),
                "Brand": r.get("Brand"),
                "Specialist": sp,
                "Access Score": r.get("Access Score"),
            })
    return pd.DataFrame(rows, columns=["Filename", "Brand", "Specialist", "Access Score"])


# ─────────────────────────────────────────────────────────────────────────────
# Figure builders  (each returns a Plotly figure; all are empty-data safe)
# ─────────────────────────────────────────────────────────────────────────────
def _empty_fig(msg: str = "No data for the current filters") -> go.Figure:
    fig = go.Figure()
    fig.add_annotation(text=msg, showarrow=False, font=dict(size=14, color=SUBTLE))
    fig.update_layout(height=320, xaxis=dict(visible=False), yaxis=dict(visible=False))
    return fig


def _style(fig: go.Figure, height: int = 380, legend_bottom: bool = False) -> go.Figure:
    fig.update_layout(height=height)
    if legend_bottom:
        fig.update_layout(
            # Pushed safely below the X-axis labels to prevent overlap
            legend=dict(orientation="h", yanchor="bottom", y=-0.25, xanchor="center", x=0.5)
        )
    return fig


def build_score_by_brand(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["Access Score"])
    if d.empty:
        return _empty_fig()
    g = (
        d.groupby("Brand")
        .agg(avg_score=("Access Score", "mean"), n=("Access Score", "size"))
        .reset_index()
        .sort_values("avg_score", ascending=False)
    )
    fig = px.bar(
        g, x="avg_score", y="Brand", orientation="h",
        color="avg_score", color_continuous_scale="Viridis",
        custom_data=["n"], title="Average Access Score by Brand",
        labels={"avg_score": "Avg Access Score", "Brand": ""},
    )
    fig.update_traces(
        hovertemplate="<b>%{y}</b><br>Avg Access Score: %{x:.2f}<br>Policies: %{customdata[0]}<extra></extra>"
    )
    fig.update_layout(yaxis=dict(categoryorder="total ascending"), coloraxis_colorbar=dict(title="Score"))
    return _style(fig, height=max(360, 26 * len(g) + 120))


def build_score_distribution(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["Access Score"])
    if d.empty:
        return _empty_fig()
    fig = px.histogram(
        d, x="Access Score", nbins=24, marginal="box",
        color_discrete_sequence=[ACCENT], title="Distribution of Access Scores",
        labels={"Access Score": "Access Score"},
    )
    mean_v = d["Access Score"].mean()
    fig.add_vline(x=mean_v, line_dash="dash", line_color=ACCENT2,
                  annotation_text=f"mean {mean_v:.1f}", annotation_position="top")
    fig.update_layout(bargap=0.06, yaxis_title="Policies")
    return _style(fig, height=380)


def build_score_box_by_brand(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    d = df.dropna(subset=["Access Score"])
    if d.empty:
        return _empty_fig()
    top = d["Brand"].value_counts().head(top_n).index
    d = d[d["Brand"].isin(top)]
    order = d.groupby("Brand")["Access Score"].median().sort_values(ascending=False).index
    fig = px.box(
        d, x="Brand", y="Access Score", color="Brand", points="all",
        category_orders={"Brand": list(order)}, hover_data=["Filename"], 
        title=f"Access Score Spread by Brand (top {len(top)})", color_discrete_sequence=PALETTE,
    )
    fig.update_layout(showlegend=False, xaxis_title="", xaxis_tickangle=-30)
    return _style(fig, height=420)


def build_friction_scatter(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["Number_of_Steps_through_Generic", "Number_of_Steps_through_Brands"]).copy()
    if d.empty:
        return _empty_fig()
    rng = np.random.default_rng(7)
    d["gx"] = d["Number_of_Steps_through_Generic"] + rng.uniform(-0.16, 0.16, len(d))
    d["by"] = d["Number_of_Steps_through_Brands"] + rng.uniform(-0.16, 0.16, len(d))
    d["size"] = d["Total_Steps"].fillna(1).clip(lower=1)
    fig = px.scatter(
        d, x="gx", y="by", color="Access Score", size="size",
        color_continuous_scale="Plasma", size_max=20,
        custom_data=["Filename", "Brand", "Number_of_Steps_through_Generic",
                     "Number_of_Steps_through_Brands", "Access Score"],
        title="Step-Therapy Friction — Generic vs Brand Steps",
        labels={"gx": "Generic Steps", "by": "Brand Steps"},
    )
    fig.update_traces(
        marker=dict(line=dict(width=0.5, color="rgba(255,255,255,.35)")),
        hovertemplate=("<b>%{customdata[1]}</b> — %{customdata[0]}<br>"
                       "Generic steps: %{customdata[2]}<br>Brand steps: %{customdata[3]}<br>"
                       "Access Score: %{customdata[4]:.1f}<extra></extra>"),
    )
    lim = float(np.nanmax([d["Number_of_Steps_through_Generic"].max(),
                           d["Number_of_Steps_through_Brands"].max(), 1])) + 0.6
    fig.add_shape(type="line", x0=0, y0=0, x1=lim, y1=lim, line=dict(color=LINE, width=1, dash="dot"))
    fig.add_annotation(x=lim * 0.82, y=lim * 0.9, text="balanced", showarrow=False, font=dict(color=SUBTLE, size=10))
    fig.update_layout(coloraxis_colorbar=dict(title="Score"))
    return _style(fig, height=420)


def build_friction_density(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["Number_of_Steps_through_Generic", "Number_of_Steps_through_Brands"])
    if d.empty:
        return _empty_fig()
    fig = px.density_heatmap(
        d, x="Number_of_Steps_through_Generic", y="Number_of_Steps_through_Brands",
        color_continuous_scale="Viridis", text_auto=True, title="Friction Density — Policy Counts",
        labels={"Number_of_Steps_through_Generic": "Generic Steps", "Number_of_Steps_through_Brands": "Brand Steps"},
    )
    fig.update_layout(coloraxis_colorbar=dict(title="Policies"))
    return _style(fig, height=420)


def build_steps_by_brand(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    if df.empty:
        return _empty_fig()
    top = df["Brand"].value_counts().head(top_n).index
    d = df[df["Brand"].isin(top)]
    g = d.groupby("Brand").agg(
        Generic=("Number_of_Steps_through_Generic", "mean"),
        Branded=("Number_of_Steps_through_Brands", "mean"),
    ).reset_index()
    g["__tot"] = g["Generic"].fillna(0) + g["Branded"].fillna(0)
    g = g.sort_values("__tot", ascending=False).drop(columns="__tot")
    mlt = g.melt(id_vars="Brand", value_vars=["Generic", "Branded"],
                 var_name="Step Type", value_name="Avg Steps")
    mlt["Step Type"] = mlt["Step Type"].map({"Generic": "Generic", "Branded": "Brand"})
    fig = px.bar(
        mlt, x="Brand", y="Avg Steps", color="Step Type", barmode="group",
        color_discrete_map={"Generic": ACCENT2, "Brand": ACCENT},
        title=f"Average Step Count by Brand (top {len(top)})",
    )
    fig.update_layout(xaxis_title="", xaxis_tickangle=-30, legend_title_text="")
    return _style(fig, height=420, legend_bottom=True)


def build_requirement_donut(df: pd.DataFrame, col: str) -> go.Figure:
    s = df[col].copy()
    s = s.where(s.isin(["Yes", "No"]), other=np.nan)
    counts = s.value_counts(dropna=False)
    counts.index = [("Not specified" if (pd.isna(i)) else i) for i in counts.index]
    counts = counts.groupby(level=0).sum()
    if counts.sum() == 0:
        return _empty_fig()
    dfd = counts.reset_index()
    dfd.columns = ["Response", "Count"]
    fig = px.pie(
        dfd, names="Response", values="Count", hole=0.62,
        color="Response", color_discrete_map=DONUT_COLORS, title=YESNO_LABELS.get(col, col),
    )
    fig.update_traces(textinfo="percent", sort=False, marker=dict(line=dict(color=CARD, width=2)),
                      hovertemplate="<b>%{label}</b><br>%{value} policies (%{percent})<extra></extra>")
    share = req_share(df[col])
    center = "—" if (isinstance(share, float) and np.isnan(share)) else f"{share:.0f}%"
    fig.add_annotation(text=f"<b>{center}</b><br><span style='font-size:11px'>Yes</span>",
                       showarrow=False, font=dict(size=22, color=INK, family="Fraunces, serif"))
    fig.update_layout(showlegend=True, legend=dict(orientation="h", y=-0.25)) # Pushed legend safely below donut
    return _style(fig, height=330)


def build_specialist_bar(long_df: pd.DataFrame) -> go.Figure:
    if long_df.empty:
        return _empty_fig("No specialist data")
    g = long_df["Specialist"].value_counts().reset_index()
    g.columns = ["Specialist", "Count"]
    g = g.sort_values("Count")
    fig = px.bar(
        g, x="Count", y="Specialist", orientation="h",
        color="Count", color_continuous_scale="Tealgrn", title="Required Specialist Types (mentions)",
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x} policies<extra></extra>")
    fig.update_layout(yaxis_title="", coloraxis_showscale=False)
    return _style(fig, height=max(330, 30 * len(g) + 110))


def build_specialist_treemap(long_df: pd.DataFrame) -> go.Figure:
    if long_df.empty:
        return _empty_fig("No specialist data")
    g = long_df["Specialist"].value_counts().reset_index()
    g.columns = ["Specialist", "Count"]
    fig = px.treemap(
        g, path=[px.Constant("All Specialists"), "Specialist"], values="Count",
        color="Count", color_continuous_scale="Viridis", title="Specialist Mix (treemap)",
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value} policies<extra></extra>",
                      marker=dict(line=dict(color=PAPER, width=2)))
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=64, l=16, r=16, b=16))
    return _style(fig, height=400)


def build_brand_specialist_sunburst(long_df: pd.DataFrame, top_n: int = 10) -> go.Figure:
    if long_df.empty:
        return _empty_fig("No specialist data")
    top = long_df["Brand"].value_counts().head(top_n).index
    d = long_df[long_df["Brand"].isin(top)]
    g = d.groupby(["Brand", "Specialist"]).size().reset_index(name="Count")
    if g.empty:
        return _empty_fig("No specialist data")
    fig = px.sunburst(
        g, path=["Brand", "Specialist"], values="Count",
        color="Count", color_continuous_scale="Tealgrn",
        title=f"Brand → Specialist (top {len(top)} brands)",
    )
    fig.update_traces(hovertemplate="<b>%{label}</b><br>%{value} policies<extra></extra>",
                      marker=dict(line=dict(color=PAPER, width=1.5)))
    fig.update_layout(coloraxis_showscale=False, margin=dict(t=64, l=16, r=16, b=16))
    return _style(fig, height=420)


def build_duration_hist(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["Initial_Authorization_Duration_months"])
    if d.empty:
        return _empty_fig()
    fig = px.histogram(
        d, x="Initial_Authorization_Duration_months", nbins=18, marginal="rug",
        color_discrete_sequence=[ACCENT], title="Initial Authorization Duration (months)",
        labels={"Initial_Authorization_Duration_months": "Months"},
    )
    med = d["Initial_Authorization_Duration_months"].median()
    fig.add_vline(x=med, line_dash="dash", line_color=ACCENT2,
                  annotation_text=f"median {med:.0f} mo", annotation_position="top")
    fig.update_layout(bargap=0.06, yaxis_title="Policies")
    return _style(fig, height=380)


def build_duration_box_by_brand(df: pd.DataFrame, top_n: int = 12) -> go.Figure:
    d = df.dropna(subset=["Initial_Authorization_Duration_months"])
    if d.empty:
        return _empty_fig()
    top = d["Brand"].value_counts().head(top_n).index
    d = d[d["Brand"].isin(top)]
    order = (d.groupby("Brand")["Initial_Authorization_Duration_months"]
             .median().sort_values(ascending=False).index)
    fig = px.box(
        d, x="Brand", y="Initial_Authorization_Duration_months", color="Brand",
        category_orders={"Brand": list(order)}, hover_data=["Filename"],
        color_discrete_sequence=PALETTE, title=f"Auth Duration by Brand (top {len(top)})",
        labels={"Initial_Authorization_Duration_months": "Months"},
    )
    fig.update_layout(showlegend=False, xaxis_title="", xaxis_tickangle=-30)
    return _style(fig, height=420)


def build_duration_vs_score(df: pd.DataFrame) -> go.Figure:
    d = df.dropna(subset=["Initial_Authorization_Duration_months", "Access Score"]).copy()
    if d.empty:
        return _empty_fig()
    fig = px.scatter(
        d, x="Initial_Authorization_Duration_months", y="Access Score",
        color="Brand", custom_data=["Filename", "Brand"],
        color_discrete_sequence=PALETTE, title="Auth Duration vs Access Score",
        labels={"Initial_Authorization_Duration_months": "Initial Auth (months)"},
    )
    fig.update_traces(marker=dict(size=10, line=dict(width=0.5, color="rgba(255,255,255,.35)")),
                      hovertemplate=("<b>%{customdata[1]}</b> — %{customdata[0]}<br>"
                                     "Duration: %{x} mo<br>Score: %{y:.1f}<extra></extra>"))
    fig.update_layout(showlegend=False)
    return _style(fig, height=420)


def build_age_breakdown(df: pd.DataFrame) -> go.Figure:
    s = df["Age"].dropna()
    if s.empty:
        return _empty_fig("No age data")
    g = s.value_counts().reset_index()
    g.columns = ["Age Restriction", "Count"]
    g = g.sort_values("Count")
    fig = px.bar(
        g, x="Count", y="Age Restriction", orientation="h",
        color="Count", color_continuous_scale="Tealgrn", title="Age Restrictions",
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x} policies<extra></extra>")
    fig.update_layout(yaxis_title="", coloraxis_showscale=False)
    return _style(fig, height=max(300, 34 * len(g) + 110))


def build_corr_heatmap(df: pd.DataFrame) -> go.Figure:
    cols = NUMERIC_COLS + ["Total_Steps"]
    sub = df[cols].apply(pd.to_numeric, errors="coerce")
    sub = sub.loc[:, sub.notna().sum() > 1]
    sub = sub.loc[:, sub.std(numeric_only=True) > 0]
    if sub.shape[1] < 2:
        return _empty_fig("Not enough numeric variation for correlation")
    corr = sub.corr()
    nice = {
        "Number_of_Steps_through_Generic": "Generic Steps",
        "Number_of_Steps_through_Brands": "Brand Steps",
        "Initial_Authorization_Duration_months": "Init. Auth (mo)",
        "Reauthorization_Duration_months": "Reauth (mo)",
        "Access Score": "Access Score",
        "Total_Steps": "Total Steps",
    }
    corr = corr.rename(index=nice, columns=nice)
    fig = px.imshow(
        corr, text_auto=".2f", aspect="auto", zmin=-1, zmax=1,
        color_continuous_scale="RdBu_r", title="Correlation of Numeric Parameters",
    )
    fig.update_layout(coloraxis_colorbar=dict(title="r"))
    return _style(fig, height=440)


def build_parcats(df: pd.DataFrame) -> go.Figure:
    dims = [c for c in YESNO_COLS if df[c].notna().any()]
    if not dims:
        return _empty_fig("No categorical requirement data")
    d = df.copy()
    for c in dims:
        d[c] = d[c].where(d[c].isin(["Yes", "No"]), other="Not specified")
    color_vals = pd.to_numeric(d["Access Score"], errors="coerce")
    if color_vals.notna().any():
        d["__color"] = color_vals.fillna(color_vals.median())
    else:
        d["__color"] = 0
    dimensions = [go.parcats.Dimension(values=d[c], label=YESNO_LABELS.get(c, c)) for c in dims]
    fig = go.Figure(
        go.Parcats(
            dimensions=dimensions,
            line=dict(color=d["__color"], colorscale="Viridis",
                      colorbar=dict(title="Access Score"), shape="hspline"),
            hoveron="color", arrangement="freeform",
            labelfont=dict(color=INK, size=12, family=FONT_BODY),
            tickfont=dict(color=SUBTLE, size=11, family=FONT_BODY),
        )
    )
    # Extra side padding specifically for Parcats to stop labels from clipping
    fig.update_layout(title="Requirement Combinations (flows colored by Access Score)", margin=dict(l=64, r=64, t=64, b=40))
    return _style(fig, height=440)


def build_completeness(df: pd.DataFrame) -> go.Figure:
    cols = [c for c in EXPECTED_COLS]
    pct = [(c, df[c].notna().mean() * 100) for c in cols]
    g = pd.DataFrame(pct, columns=["Field", "Completeness"]).sort_values("Completeness")
    fig = px.bar(
        g, x="Completeness", y="Field", orientation="h",
        color="Completeness", color_continuous_scale="Tealgrn",
        range_color=[0, 100], title="Field Completeness (% non-missing)",
    )
    fig.update_traces(hovertemplate="<b>%{y}</b><br>%{x:.0f}% populated<extra></extra>")
    fig.update_layout(yaxis_title="", coloraxis_showscale=False, xaxis_ticksuffix="%")
    return _style(fig, height=max(330, 26 * len(g) + 110))


def build_radar(df: pd.DataFrame, brand: str) -> go.Figure:
    bdf = df[df["Brand"] == brand]
    if bdf.empty:
        return _empty_fig("No rows for this brand")

    def metrics(frame):
        return {
            "Access Score": safe_mean(frame["Access Score"]),
            "Generic Steps": safe_mean(frame["Number_of_Steps_through_Generic"]),
            "Brand Steps": safe_mean(frame["Number_of_Steps_through_Brands"]),
            "Init. Auth (mo)": safe_mean(frame["Initial_Authorization_Duration_months"]),
            "TB Test": req_share(frame["TB_Test_required"]),
            "Reauth": req_share(frame["Reauthorization_Required"]),
            "Phototherapy": req_share(frame["Step_through_Phototherapy"]),
            "Qty Limits": req_share(frame["Quantity_Limits"]),
        }

    cats = ["Access Score", "Generic Steps", "Brand Steps", "Init. Auth (mo)",
            "TB Test", "Reauth", "Phototherapy", "Qty Limits"]

    def _den(series) -> float:
        s = pd.to_numeric(series, errors="coerce")
        m = s.max()
        return float(m) if pd.notna(m) and m > 0 else 1.0

    denom = {
        "Access Score": _den(df["Access Score"]),
        "Generic Steps": _den(df["Number_of_Steps_through_Generic"]),
        "Brand Steps": _den(df["Number_of_Steps_through_Brands"]),
        "Init. Auth (mo)": _den(df["Initial_Authorization_Duration_months"]),
        "TB Test": 100, "Reauth": 100, "Phototherapy": 100, "Qty Limits": 100,
    }

    def norm(vals):
        out = []
        for c in cats:
            v = vals[c]
            v = 0 if (v is None or (isinstance(v, float) and np.isnan(v))) else v
            out.append(min(v / denom[c], 1.0) if denom[c] else 0)
        return out

    bm, om = metrics(bdf), metrics(df)
    fig = go.Figure()
    fig.add_trace(go.Scatterpolar(r=norm(om) + [norm(om)[0]], theta=cats + [cats[0]],
                                  fill="toself", name="All brands (filtered)",
                                  line=dict(color="#7E8794"), fillcolor="rgba(126,135,148,.20)"))
    fig.add_trace(go.Scatterpolar(r=norm(bm) + [norm(bm)[0]], theta=cats + [cats[0]],
                                  fill="toself", name=brand, line=dict(color=ACCENT),
                                  fillcolor="rgba(52,216,198,.28)"))
    fig.update_layout(
        title=f"Restrictiveness Fingerprint — {brand}",
        polar=dict(bgcolor="rgba(0,0,0,0)", radialaxis=dict(visible=True, range=[0, 1], showticklabels=False, gridcolor=GRID),
                   angularaxis=dict(gridcolor=GRID, tickfont=dict(color=SUBTLE, size=11))),
        legend=dict(orientation="h", y=-0.25),
    )
    return _style(fig, height=440)


# ─────────────────────────────────────────────────────────────────────────────
# Layout helpers
# ─────────────────────────────────────────────────────────────────────────────
def section(eyebrow: str, title: str, desc: str = "") -> None:
    st.markdown(
        f"""<div class='macc-section'>
        <div class='macc-eyebrow'>{eyebrow}</div>
        <h2>{title}</h2>
        <div class='desc'>{desc}</div>
        <div class='macc-rule'></div></div>""",
        unsafe_allow_html=True,
    )

def caption(text: str) -> None:
    st.markdown(f"<div class='macc-caption'>{text}</div>", unsafe_allow_html=True)


# ─────────────────────────────────────────────────────────────────────────────
# Main app
# ─────────────────────────────────────────────────────────────────────────────
def main() -> None:
    st.set_page_config(
        page_title="Market Access Dashboard — PA Pipeline",
        page_icon="🧬", layout="wide", initial_sidebar_state="expanded",
    )
    st.markdown(CSS, unsafe_allow_html=True)
    install_plotly_theme()

    st.markdown("<div class='macc-eyebrow' style='margin-bottom:6px'>Data Source</div>", unsafe_allow_html=True)
    with st.container(border=True):
        up_l, up_r = st.columns([3, 1])
        with up_l:
            uploaded = st.file_uploader(
                "Upload your prior-authorization results — CSV or Excel",
                type=["csv", "xlsx", "xlsm", "xls"],
                help="Drag a .csv or .xlsx file here, or click Browse. Column headers may use spaces or underscores.",
            )
        with up_r:
            st.markdown(
                f"<div style='color:{SUBTLE};font-size:.82rem;padding-top:.4rem'>"
                "Accepted: <b>.csv</b>, <b>.xlsx</b>, <b>.xls</b>.<br>"
                "If nothing is uploaded, a local <code>result.csv</code>/"
                "<code>result.xlsx</code> is used automatically.</div>",
                unsafe_allow_html=True,
            )

    try:
        if uploaded is not None:
            df_all = load_and_clean_from_bytes(uploaded.getvalue(), uploaded.name)
            source_kind, source_msg = "success", f"Loaded **{uploaded.name}** — {len(df_all):,} rows."
        else:
            local = find_local_data()
            if local is None:
                raise FileNotFoundError
            df_all = load_and_clean_from_path(local)
            source_kind = "info"
            source_msg = f"Using local **{local}** — {len(df_all):,} rows. Upload a file above to override."
    except FileNotFoundError:
        st.info("⬆️ Upload a **CSV** or **Excel** file above to get started.")
        st.stop()
    except ImportError as e:
        st.error(str(e))
        st.stop()
    except Exception as e:
        st.error(f"Could not read the file: {e}")
        st.stop()

    getattr(st, source_kind)(source_msg)

    st.markdown(
        f"""<div class='macc-hero'>
            <div class='macc-eyebrow'>Prior Authorization · Plaque Psoriasis</div>
            <h1>Market Access Dashboard</h1>
            <div class='sub'>An exhaustive, interactive view of payer prior-authorization
            requirements extracted by the PA pipeline — access scores, step-therapy friction,
            specialist gating, safety screening, and authorization durations across brands.</div>
            <div class='macc-tags'>
                <span class='macc-tag'>{df_all['Filename'].nunique():,} policies</span>
                <span class='macc-tag'>{df_all['Brand'].nunique():,} brands</span>
                <span class='macc-tag'>PA requirement schema</span>
                <span class='macc-tag'>NA-safe numeric coercion</span>
            </div>
        </div>""",
        unsafe_allow_html=True,
    )

    st.sidebar.markdown("### 🎛️ Filters")
    brands = sorted([b for b in df_all["Brand"].dropna().unique()])
    sel_brands = st.sidebar.multiselect("Brand", brands, default=[], key="flt_brands", help="Empty = all brands")

    all_specs = sorted(specialist_long(df_all)["Specialist"].dropna().unique().tolist())
    sel_specs = st.sidebar.multiselect("Specialist type", all_specs, default=[], key="flt_specs", help="Empty = all. Keeps policies mentioning any selected specialist.")

    score_series = df_all["Access Score"].dropna()
    if not score_series.empty:
        s_min, s_max = float(score_series.min()), float(score_series.max())
        if s_min == s_max:
            s_max = s_min + 1.0
        sel_score = st.sidebar.slider("Access Score range", s_min, s_max, (s_min, s_max), key="flt_score")
    else:
        sel_score = None

    def flag_filter(label, col):
        return st.sidebar.selectbox(label, ["All", "Yes", "No"], index=0, key=f"flt_{col}")

    f_tb = flag_filter("TB test required", "TB_Test_required")
    f_re = flag_filter("Reauthorization required", "Reauthorization_Required")
    f_ph = flag_filter("Phototherapy step", "Step_through_Phototherapy")
    f_ql = flag_filter("Quantity limits", "Quantity_Limits")

    st.sidebar.markdown("---")
    top_n = st.sidebar.slider("Max brands in per-brand charts", 5, 30, 12)
    if st.sidebar.button("↺ Reset filters", width="stretch"):
        for k in list(st.session_state.keys()):
            if k.startswith("flt_"):
                del st.session_state[k]
        st.rerun()

    df = df_all.copy()
    if sel_brands:
        df = df[df["Brand"].isin(sel_brands)]
    if sel_specs:
        sel_set = set(sel_specs)
        mask = df["Specialist_Types"].map(lambda v: bool(set(row_specialists(v)) & sel_set))
        df = df[mask]
    if sel_score is not None:
        in_range = df["Access Score"].between(sel_score[0], sel_score[1])
        df = df[in_range | df["Access Score"].isna()]
    for col, choice in [("TB_Test_required", f_tb), ("Reauthorization_Required", f_re),
                        ("Step_through_Phototherapy", f_ph), ("Quantity_Limits", f_ql)]:
        if choice != "All":
            df = df[df[col] == choice]

    if df.empty:
        st.warning("No policies match the current filters. Widen your selection in the sidebar.")
        st.stop()

    st.sidebar.markdown(
        f"<div style='font-family:{FONT_MONO};font-size:.8rem;color:{SUBTLE}'>"
        f"Showing <b style='color:{ACCENT}'>{len(df):,}</b> of {len(df_all):,} policies</div>",
        unsafe_allow_html=True,
    )

    section("01 · Overview", "Executive Summary", "Headline metrics for the current selection. Deltas compare the filtered selection against the full dataset average.")

    o_score, o_med = safe_mean(df_all["Access Score"]), safe_median(df_all["Access Score"])
    o_gen, o_brd = safe_mean(df_all["Number_of_Steps_through_Generic"]), safe_mean(df_all["Number_of_Steps_through_Brands"])
    o_tb = req_share(df_all["TB_Test_required"])
    o_re = req_share(df_all["Reauthorization_Required"])
    o_ph = req_share(df_all["Step_through_Phototherapy"])
    o_ql = req_share(df_all["Quantity_Limits"])
    o_dur = safe_mean(df_all["Initial_Authorization_Duration_months"])
    o_redur = safe_mean(df_all["Reauthorization_Duration_months"])

    f_score, f_med = safe_mean(df["Access Score"]), safe_median(df["Access Score"])
    f_gen, f_brd = safe_mean(df["Number_of_Steps_through_Generic"]), safe_mean(df["Number_of_Steps_through_Brands"])
    f_tb_pct = req_share(df["TB_Test_required"])
    f_re_pct = req_share(df["Reauthorization_Required"])
    f_ph_pct = req_share(df["Step_through_Phototherapy"])
    f_ql_pct = req_share(df["Quantity_Limits"])
    f_dur = safe_mean(df["Initial_Authorization_Duration_months"])
    f_redur = safe_mean(df["Reauthorization_Duration_months"])

    r1 = st.columns(4)
    r1[0].metric("Policies in view", f"{len(df):,}")
    r1[1].metric("Unique brands", f"{df['Brand'].nunique():,}")
    r1[2].metric("Avg Access Score", fmt_num(f_score, 1), delta=delta_str(f_score, o_score), delta_color="off")
    r1[3].metric("Median Access Score", fmt_num(f_med, 1), delta=delta_str(f_med, o_med), delta_color="off")

    r2 = st.columns(4)
    r2[0].metric("Avg Generic Steps", fmt_num(f_gen, 1), delta=delta_str(f_gen, o_gen), delta_color="off")
    r2[1].metric("Avg Brand Steps", fmt_num(f_brd, 1), delta=delta_str(f_brd, o_brd), delta_color="off")
    r2[2].metric("TB Test Required", fmt_pct(f_tb_pct), delta=delta_str(f_tb_pct, o_tb, pct=True), delta_color="off")
    r2[3].metric("Reauth Required", fmt_pct(f_re_pct), delta=delta_str(f_re_pct, o_re, pct=True), delta_color="off")

    r3 = st.columns(4)
    r3[0].metric("Avg Init. Auth (mo)", fmt_num(f_dur, 1), delta=delta_str(f_dur, o_dur), delta_color="off")
    r3[1].metric("Avg Reauth (mo)", fmt_num(f_redur, 1), delta=delta_str(f_redur, o_redur), delta_color="off")
    r3[2].metric("Phototherapy Step", fmt_pct(f_ph_pct), delta=delta_str(f_ph_pct, o_ph, pct=True), delta_color="off")
    r3[3].metric("Quantity Limits", fmt_pct(f_ql_pct), delta=delta_str(f_ql_pct, o_ql, pct=True), delta_color="off")
    caption("Requirement percentages are the share of all policies in view that explicitly answer “Yes”; missing or “No” count as not required.")

    section("02 · Access Score", "The Access Score Landscape", "Which brands face the steepest access barriers, and how scores are distributed.")
    c = st.columns([1.15, 1])
    with c[0]:
        st.plotly_chart(build_score_by_brand(df), use_container_width=True, key="score_by_brand")
    with c[1]:
        st.plotly_chart(build_score_distribution(df), use_container_width=True, key="score_dist")
    st.plotly_chart(build_score_box_by_brand(df, top_n), use_container_width=True, key="score_box_brand")
    caption("Box plots show median, IQR and individual policies (jittered) per brand.")

    section("03 · Step Therapy", "Step-Therapy Friction", "How many generic and branded agents a patient must fail before approval.")
    c = st.columns(2)
    with c[0]:
        st.plotly_chart(build_friction_scatter(df), use_container_width=True, key="friction_scatter")
        caption("Bubble size = total required steps · color = Access Score. Points jittered to reduce overlap.")
    with c[1]:
        st.plotly_chart(build_friction_density(df), use_container_width=True, key="friction_density")
        caption("Counts of policies at each (generic, brand) step combination.")
    st.plotly_chart(build_steps_by_brand(df, top_n), use_container_width=True, key="steps_by_brand")

    section("04 · Requirements", "Clinical & Administrative Gates", "Share of policies imposing each categorical requirement (NA shown as 'Not specified').")
    c = st.columns(4)
    for col, container in zip(YESNO_COLS, c):
        with container:
            st.plotly_chart(build_requirement_donut(df, col), use_container_width=True, key=f"req_donut_{col}")

    section("05 · Specialists", "Who Can Prescribe", "Specialist gating extracted from the policy text (multi-valued cells are split).")
    long_df = specialist_long(df)
    c = st.columns([1, 1])
    with c[0]:
        st.plotly_chart(build_specialist_bar(long_df), use_container_width=True, key="spec_bar")
    with c[1]:
        st.plotly_chart(build_specialist_treemap(long_df), use_container_width=True, key="spec_treemap")
    st.plotly_chart(build_brand_specialist_sunburst(long_df, top_n), use_container_width=True, key="spec_sunburst")

    section("06 · Duration & Age", "Authorization Windows & Age Limits", "How long initial approvals last, and the age restrictions encoded in policy.")
    c = st.columns(2)
    with c[0]:
        st.plotly_chart(build_duration_hist(df), use_container_width=True, key="dur_hist")
    with c[1]:
        st.plotly_chart(build_age_breakdown(df), use_container_width=True, key="age_breakdown")
    c = st.columns(2)
    with c[0]:
        st.plotly_chart(build_duration_box_by_brand(df, top_n), use_container_width=True, key="dur_box_brand")
    with c[1]:
        st.plotly_chart(build_duration_vs_score(df), use_container_width=True, key="dur_vs_score")

    section("07 · Multivariate", "Cross-Parameter Patterns", "Correlations between numeric parameters.")
    c = st.columns([1, 1.25])
    with c[0]:
        st.plotly_chart(build_corr_heatmap(df), use_container_width=True, key="corr_heatmap")

    section("08 · Drill-down", "Brand Fingerprint", "Compare a single brand's restrictiveness profile against the filtered cohort.")
    drill_brands = sorted(df["Brand"].dropna().unique().tolist())
    if drill_brands:
        dcols = st.columns([1, 2])
        with dcols[0]:
            pick = st.selectbox("Select a brand", drill_brands, key="drill_brand")
            bsub = df[df["Brand"] == pick]
            st.metric("Policies", f"{len(bsub):,}")
            st.metric("Avg Access Score", fmt_num(safe_mean(bsub["Access Score"]), 1))
            st.metric("Avg Total Steps", fmt_num(safe_mean(bsub["Total_Steps"]), 1))
            st.metric("Avg Init. Auth (mo)", fmt_num(safe_mean(bsub["Initial_Authorization_Duration_months"]), 1))
        with dcols[1]:
            st.plotly_chart(build_radar(df, pick), use_container_width=True, key="radar_fingerprint")

    section("09 · Appendix", "Data Quality & Records", "")
    with st.expander("📐 Field completeness", expanded=False):
        st.plotly_chart(build_completeness(df), use_container_width=True, key="completeness")

    with st.expander("🗂️ Filtered records", expanded=False):
        show = df.drop(columns=[c for c in ["Total_Steps"] if c in df.columns]).reset_index(drop=True)
        st.dataframe(show, width="stretch", height=420)
        st.download_button(
            "⬇️ Download filtered CSV",
            data=show.to_csv(index=False).encode("utf-8"),
            file_name="market_access_filtered.csv", mime="text/csv",
            use_container_width=True,
        )

    with st.expander("ℹ️ Methodology & column notes", expanded=False):
        st.markdown(
            """
**Cleaning.** Every `NA`-like token (`NA`, `N/A`, `none`, blank, `-`, …) is converted to a real
`NaN`. Numeric columns — *Number_of_Steps_through_Generic*, *Number_of_Steps_through_Brands*,
*Initial_Authorization_Duration_months*, *Access Score* — are coerced with
`pd.to_numeric(errors="coerce")`. Yes/No flag columns are normalized to `{Yes, No, NaN}`.

**Percentages.** Requirement percentages (e.g. *TB Test Required*) are computed over policies with a
non-missing Yes/No response; *Not specified* slices in the donuts represent missing data.

**Specialists.** *Specialist_Types* cells may list several specialties; they are split on commas,
semicolons, slashes and the words *and*/*or* before counting.

**Deltas.** KPI deltas compare the current filtered selection against the full-dataset average and are
shown in a neutral color — they describe difference, not "good" or "bad."
            """
        )
    st.markdown(
        f"<div style='text-align:center;color:{FAINT};font-family:{FONT_MONO};"
        f"font-size:.75rem;margin-top:28px'>Market Access Dashboard · built with Streamlit + Plotly</div>",
        unsafe_allow_html=True,
    )

if __name__ == "__main__":
    main()
