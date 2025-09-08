#  Import packages

import streamlit as st
import pickle
import datetime
import pandas as pd
import plotly.graph_objects as go
from streamlit_option_menu import option_menu
import io
import plotly.express as px
import altair as alt
from matplotlib_venn import venn2
import matplotlib.pyplot as plt
import numpy as np
import seaborn as sns
# import scikit-learn as sklearn

# clustering import
import plotly.figure_factory as ff
from scipy.cluster import hierarchy
from scipy.spatial.distance import pdist
from sklearn.preprocessing import StandardScaler
import json
import plotly.io as pio
# from sklearn.cluster import KMeans
from sklearn.cluster import AgglomerativeClustering
from sklearn.metrics import silhouette_score
from sklearn.ensemble import RandomForestClassifier
import shap
from lime.lime_tabular import LimeTabularExplainer
import hdbscan
from shap.plots._waterfall import waterfall_legacy
import streamlit.components.v1 as components
from sklearn.linear_model import LogisticRegression
from streamlit_scroll_to_top import scroll_to_here
from io import BytesIO
import base64

# deployment check
# import streamlit as st, sklearn, numpy
# st.sidebar.write("▶︎ sklearn.__version__:", sklearn.__version__)
# st.sidebar.write("▶︎ numpy.__version__:  ", numpy.__version__)

# -----------------------------------------------


# Scroll Up Functionality CODE
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------

def goto(page_name: str):
    st.session_state.page = page_name
    st.session_state._scroll_top = True
    st.rerun()

# Initialize page once
if "page" not in st.session_state:
    st.session_state.page = "Home"
    

# Handle nav-triggered jump to top (ONE pop in the whole app)
if st.session_state.pop("_scroll_top", False):
    scroll_to_here(0, key="top")

# Helper functions

## Plot functions---------------------------------
## -----------------------------------------------
## -----------------------------------------------

# function that plots daily line graph
def daily_line_altair(df):
    # Detect indexing (1-based vs 0-based)
    zero_indexed = df["days_in_year"].min() == 0

    # Quarter boundaries for a 365-day year (your convention: 91, 182, 273, 364)
    q_ends_1based = [91, 182, 273]
    q_ends = [x - 1 for x in q_ends_1based] if zero_indexed else q_ends_1based

    # Limit to be within 365 (omit day 365)
    df = df[df["days_in_year"] < 365]

    # 1) Aggregate raw counts
    day_counts = (
        df.groupby("days_in_year", as_index=False)["y"]
          .sum()
          .rename(columns={"y": "Contacts Made"})
    )

    # 2) Create a full index of days (1..364 or 0..364 depending on indexing)
    start_day = 0 if zero_indexed else 1
    full = pd.DataFrame({"days_in_year": range(start_day, 365)})

    # 3) Left-merge and fill missing with 0
    merged = full.merge(day_counts, on="days_in_year", how="left")
    merged["Contacts Made"] = merged["Contacts Made"].fillna(0)

    # --- compute quarter starts/ends and midpoints ---
    q_starts = [start_day] + [e + 1 for e in q_ends]         # start of each quarter
    q_ends_all = q_ends + [364]                               # end of each quarter
    q_mids = [ (s + e) / 2 for s, e in zip(q_starts, q_ends_all) ]
    q_labels = ["Q1", "Q2", "Q3", "Q4"]
    qlabel_df = pd.DataFrame({"days_in_year": q_mids, "quarter": q_labels})

    # 4) Base line chart
    line = (
        alt.Chart(merged)
        .mark_line(interpolate="linear", point=True)
        .encode(
            x=alt.X(
                "days_in_year:Q",
                title="Day of Year",
                scale=alt.Scale(domain=[start_day, 364], nice=False, clamp=True),
                axis=alt.Axis(tickMinStep=1),
            ),
            y=alt.Y("Contacts Made:Q", title="Contacts Made"),
            tooltip=["days_in_year:Q", "Contacts Made:Q"],
        )
    )

    # 5) Quarter boundary rules (vertical lines)
    q_rules_df = pd.DataFrame({"days_in_year": q_ends})
    quarter_rules = (
        alt.Chart(q_rules_df)
        .mark_rule(size=2, color="rgba(255,255,255,0.55)", strokeDash=[4, 4])
        .encode(x=alt.X("days_in_year:Q"))
    )

    # 6) Quarter labels at exact midpoints (works for both 0- and 1-indexed)
    quarter_labels = (
        alt.Chart(qlabel_df)
        .mark_text(
            fontSize=14, fontWeight="bold", color="white",
            # optional halo for readability:
            # stroke="black", strokeWidth=3
        )
        .encode(
            x=alt.X("days_in_year:Q", scale=alt.Scale(domain=[start_day, 364])),
            y=alt.value(18),     # pin near top of plot; adjust if height changes
            text="quarter:N",
        )
    )

    # 7) Combine
    chart = (line + quarter_rules + quarter_labels).properties(
        width="container",
        height=300,
        title="Daily Contacts Made Over a Year",
        padding={"top": 20, "right": 0, "bottom": 0, "left": 0},
    ).configure_title(fontSize=18, anchor="start")

    return chart


# plot monthly trend_line graph
def monthly_line_altair(df):
    # 1) Aggregate by month
    month_counts = (
        df
        .groupby("month", as_index=False)["y"]
        .sum()
        .rename(columns={"y": "Contacts Made"})
    )
    # 2) Ensure all 12 months appear
    full_months = pd.DataFrame({"month": list(range(1,13))})
    merged = (
        full_months
        .merge(month_counts, on="month", how="left")
        .fillna(0)
    )
    # 3) Map to month names & ordered categorical
    names = ["Jan","Feb","Mar","Apr","May","Jun","Jul","Aug","Sep","Oct","Nov","Dec"]
    merged["month_name"] = pd.Categorical(
        [names[m-1] for m in merged["month"]],
        categories=names,
        ordered=True
    )

    # 4a) base chart transforms
    base = alt.Chart(merged)

    # 4b) line + points layer
    line = base.mark_line(interpolate="linear", point=True).encode(
        x=alt.X("month_name:O", title="Month", sort=names, axis=alt.Axis(labelAngle=0)),
        y=alt.Y("Contacts Made:Q", title="Contacts Made"),
        tooltip=[
            alt.Tooltip("month_name:O", title="Month"),
            alt.Tooltip("Contacts Made:Q", title="Contacts Made")
        ]
    )

    # 4c) text‐label layer, nudged above each point
    labels = base.mark_text(
        dy=-10,              # move labels 10px above points
        fontSize=12,
        color="white"
    ).encode(
        x=alt.X("month_name:O", sort=names),
        y="Contacts Made:Q",
        text=alt.Text("Contacts Made:Q")
    )

    # 4c) quarter boundary rules at end of Mar/Jun/Sep/Dec
    #    Built from `base` so it already has data; works in Altair v4.
    quarter_rules = (
        base
        .transform_filter(
            alt.FieldOneOfPredicate(field="month_name",
                                    oneOf=["Mar", "Jun", "Sep", "Dec"])
        )
        .mark_rule(size=2, color="rgba(255,255,255,0.25)", strokeDash=[4, 4])
        .encode(
            x=alt.X("month_name:O", sort=names)
            # NOTE: In Altair v4 this draws at the CENTER of the month band.
            # If you later upgrade to Altair 5, add: bandPosition=1 to place it
            # at the END of each month band (true quarter boundary).
        )
    )

    # --- quarter labels (Q1..Q4) placed in the middle month of each partition ---
    qlabel_df = pd.DataFrame({
        "month_name": ["Feb", "May", "Aug", "Nov"],  # middle of each quarter
        "quarter":    ["Q1",  "Q2",  "Q3",  "Q4"]
    })

    quarter_labels = (
        alt.Chart(qlabel_df)
        .mark_text(fontSize=14, fontWeight="bold", color="white", dy=0, dx=-28)
        .encode(
            x=alt.X("month_name:O", sort=names),
            # Use a fixed screen position so the label sits near the top of the plot.
            # Adjust this value if you change chart height.
            y=alt.value(18),
            text="quarter:N"
        )
        # .properties(title=None)
    )

    chart = (line + labels + quarter_rules + quarter_labels).properties(
        width="container", height=300, title="Monthly Contacts Made",
        padding={"top": 20, "right": 0, "bottom": 0, "left": 0},
    ).configure_title(fontSize=18, anchor="start")

    return chart

# plot monthly success rate chart
def monthly_success_altair(df):
    months = ["Jan","Feb","Mar","Apr","May","Jun",
              "Jul","Aug","Sep","Oct","Nov","Dec"]

    # 1) Transformations: month name, mean rate, percent
    base = (
        alt.Chart(df)
        .transform_calculate(
            month_name=f"['" + "','".join(months) + f"'][datum.month - 1]"
        )
        .transform_aggregate(
            rate="mean(y)",
            groupby=["month_name"]
        )
        .transform_calculate(
            rate_pct="datum.rate * 100"
        )
    )

    # 2) Line + points
    line = base.mark_line(interpolate="linear", point=True).encode(
        x=alt.X(
            "month_name:O",
            sort=months,
            title="Month",
            axis=alt.Axis(labelAngle=0)
        ),
        y=alt.Y("rate_pct:Q", title="Success Rate (%)", scale=alt.Scale(domain=[0, 60])),
        tooltip=[
            alt.Tooltip("month_name:O", title="Month"),
            alt.Tooltip("rate_pct:Q", title="Success Rate", format=".1f")
        ]
    )

    # 3) Data labels over points
    labels = base.mark_text(dy=-10, fontSize=12, color="white").encode(
        x=alt.X("month_name:O", sort=months),
        y="rate_pct:Q",
        text=alt.Text("rate_pct:Q", format=".1f")
    )

    # 4) Quarter boundary rules at Mar/Jun/Sep/Dec (center of band in Altair v4)
    quarter_rules = (
        base
        .transform_filter(
            alt.FieldOneOfPredicate(field="month_name", oneOf=["Mar", "Jun", "Sep", "Dec"])
        )
        .mark_rule(size=2, color="rgba(255,255,255,0.25)", strokeDash=[4, 4])
        .encode(x=alt.X("month_name:O", sort=months))
    )

    # 5) Quarter labels (Q1..Q4) at Feb/May/Aug/Nov, nudged left to sit between months
    qlabel_df = pd.DataFrame({
        "month_name": ["Feb", "May", "Aug", "Nov"],   # middle month of each quarter
        "quarter":    ["Q1",  "Q2",  "Q3",  "Q4"]
    })

    quarter_labels = (
        alt.Chart(qlabel_df)
        .mark_text(
            fontSize=14, fontWeight="bold", color="white",
            dx=-28,          # ← nudge left (negative = left). Tweak to taste.
            # Optional halo for readability on busy backgrounds:
            # stroke="black", strokeWidth=3
        )
        .encode(
            x=alt.X("month_name:O", sort=months),
            y=alt.value(18),     # fixed screen y near top; adjust if height changes
            text="quarter:N"
        )
    )

    # 6) Combine everything
    chart = (line + labels + quarter_rules + quarter_labels).properties(
        width="container",
        height=300,
        title="Monthly Win Rate",
        padding={"top": 20, "right": 0, "bottom": 0, "left": 0},
    ).configure_title(fontSize=18, anchor="start")

    return chart

# Plot pie chart for contact channel
def contact_channel_pie(df, width, height, filter_col="y", filter_val=1):
    """
    Given a DataFrame `df`, filters for df[filter_col] == filter_val,
    then builds & returns a Plotly Pie chart of contact_cellular vs contact_telephone.
    """
    # 1) Filter to “wins”
    wins = df[df[filter_col] == filter_val]

    # 2) Sum up the two channels
    sums = wins[["contact_cellular", "contact_telephone"]].sum().astype(int)
    counts = [sums["contact_cellular"], sums["contact_telephone"]]

    # 3) Build the Pie
    labels = ["Cellular", "Telephone"]
    fig = go.Figure(go.Pie(
        labels=labels,
        values=counts,
        textinfo="label+value+percent",
        textposition="inside",  
        insidetextorientation="radial",
        marker=dict(
            colors=["#EF553B", "#636EFA"],    # same two colors
            line=dict(width=1, color="white")
        ),
        textfont=dict(
            color=["white", "white"],         # white text on both slices
            size=14
        )
    ))
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(
        title=dict(
            text="Wins Per Contact Channel ",
            pad=dict(t=-10),  # reduce top padding
            font=dict(size=16)
        ),
        margin=dict(t=40, b=0, l=0, r=0),  # shrink figure top margin
        showlegend=False,
        width=width,
        height=height
    )
    return fig

st.markdown(
    """
    <style>
    /* Reduce space between tab headers and their content */
    div[data-testid="stTabs"] div[role="tabpanel"] {
        padding-top: 0.25rem;
    }
    </style>
    """,
    unsafe_allow_html=True
)

# plot venn diagram for success cases over loan types
def plot_loan_venn(
    df,
    width_px,
    height_px,
    filter_col="y",
    filter_val=1,
    scale=0.34,
    dpi=100,
):

    # get different proportions
    wins = df[df[filter_col] == filter_val]
    both     = int(((wins["housing"] == 1) & (wins["loan"] == 1)).sum())
    housing  = int(wins["housing"].sum()) - both
    personal = int(wins["loan"].sum())    - both
    no_loans = int(((wins["housing"] == 0) & (wins["loan"] == 0)).sum())

    w_in, h_in = width_px / dpi, height_px / dpi
    fig, ax = plt.subplots(figsize=(w_in, h_in), dpi=dpi, facecolor="none")
    ax.set_facecolor("none")

    # build venn diagram
    v = venn2(
        subsets=(housing, personal, both),
        set_labels=("Housing Loan", "Personal Loan"),
        ax=ax,
        normalize_to=scale
    )

    v.get_patch_by_id('10').set_color('#636EFA')
    v.get_patch_by_id('01').set_color('#EF553B')
    v.get_patch_by_id('11').set_color('#00CC96')

    # build frame
    fscale = max(width_px, height_px) / 300.0
    for txt in v.set_labels:
        txt.set_color("white"); txt.set_fontweight("bold"); txt.set_alpha(1); txt.set_fontsize(7 * fscale)
    for txt in v.subset_labels:
        if txt is not None:
            txt.set_color("white"); txt.set_fontweight("bold"); txt.set_alpha(1); txt.set_fontsize(7 * fscale)

    ax.text(0.78, 0.86, f"No Loans: {no_loans}",
            ha="center", va="center",
            fontsize=7 * fscale, color="white", fontweight="bold", alpha=1,
            transform=ax.transAxes)

    ax.set_title("Wins Per Loan Ownership",
                 color="white", fontweight="bold", fontsize=8 * fscale, pad=2)

    for spine in ax.spines.values(): spine.set_visible(False)
    ax.margins(0.01, 0.01)

    # Move the diagram UP by shrinking bottom and enlarging top margin
    plt.subplots_adjust(left=0.00, right=0.98, top=0.52, bottom=0.00)

    return fig


# Function that displays KDE plot
def kde_age_distribution(df, field="age", filter_col="y", filter_val=1, bandwidth=5):
    """
    Render a 1D KDE plot (density estimate) of `field` for rows where df[filter_col] == filter_val.
    """
    # Filter wins
    wins = df[df[filter_col] == filter_val]
    
    # Build KDE via Vega-Lite transform_density
    chart = (
        alt.Chart(wins)
        .transform_density(
            field,
            as_=[field, "density"],
            extent=[18, 100],    # compute KDE from 18 → 100
            bandwidth=bandwidth
        )
        .mark_area(opacity=0.5, interpolate="monotone")
        .encode(
            x=alt.X(
                f"{field}:Q",
                title="Age",
                scale=alt.Scale(domain=[18, 100]),   # ⟵ clamp the axis here
                axis=alt.Axis(tickMinStep=5)
            ),
            y=alt.Y("density:Q", title="Density"),
            tooltip=[
                alt.Tooltip(field, title="Age"),
                alt.Tooltip("density:Q", title="Density", format=".3f")
            ]
        )
        .properties(
            width="container",
            height=350,
            title="Age Distribution over Wins",
            padding={"top": 20, "right": 0, "bottom": 0, "left": 0},  # ← add top padding
        )
        .configure_title(fontSize=18, anchor="start")
        .configure_axis(labelFontSize=12, titleFontSize=14)
    )
    return chart

# Function that plot the plot x age duration heatmap
def plot_age_duration_heatmap(df, 
                             age_start=18, age_end=66, age_step=5,
                             dur_start=0, dur_end=1200, dur_step=60):
    """
    Returns an Altair heatmap of conversion rate (y) by 5-year age bins vs. duration bins.
    """
    heatmap_data = df.copy()
    # 1) Age bins
    age_bins   = list(range(age_start, age_end + age_step, age_step))
    age_labels = [f"{b}-{b+age_step-1}" for b in age_bins[:-1]]
    heatmap_data['age_bin'] = pd.cut(
        heatmap_data['age'],
        bins=age_bins,
        labels=age_labels,
        right=False
    )

    # 2) Duration bins
    dur_bins   = list(range(dur_start, dur_end + dur_step, dur_step))
    # dur_labels = [f"{b}-{b+dur_step-1}" for b in dur_bins[:-1]]
    dur_labels = [
        f"{int(b/60)}–{int((b + dur_step)/60)}"
        for b in dur_bins[:-1]
    ]
    heatmap_data['duration_bin'] = pd.cut(
        heatmap_data['duration'],
        bins=dur_bins,
        labels=dur_labels,
        right=False
    )

    # 3) Pivot + fill
    heatmap_df = (
        heatmap_data
          .pivot_table(
             index='age_bin',
             columns='duration_bin',
             values='y',
             aggfunc='mean'
          )
          .fillna(0)
    )

    # 4) Melt back to long form
    hm_long = heatmap_df.reset_index().melt(
        id_vars='age_bin',
        var_name='duration_bin',
        value_name='conversion_rate'
    )

    # 5) Altair heatmap
    chart = (
        alt.Chart(hm_long)
          .mark_rect()
          .encode(
            x=alt.X("duration_bin:N", title="Minutes", sort=dur_labels, axis=alt.Axis(labelAngle=0, labelAlign="center", labelFontSize=9)),
            y=alt.Y("age_bin:O",      title="Age Bin",      sort=age_labels[::-1]),
            color=alt.Color("conversion_rate:Q", 
                            title="Conversion Rate", 
                            scale=alt.Scale(scheme="blues")),
            tooltip=[
              alt.Tooltip("age_bin:O",        title="Age Bin"),
              alt.Tooltip("duration_bin:N",   title="Minutes"),
              alt.Tooltip("conversion_rate:Q",title="Conversion Rate", format=".1%")
            ]
          )
          .properties(
            width="container", height=400,
            padding={"top": 20, "right": 0, "bottom": 0, "left": 0},  # ← add top padding
            title="Conversion Rate Heatmap (Age × Duration in Minutes)"
          )
          .configure_title(fontSize=18, anchor="start")
          .configure_axis(labelFontSize=12, titleFontSize=14)
    )
    return chart

# plot the loan type X duration heatmap
def plot_loans_duration_heatmap(df, dur_start=0, dur_end=1200, dur_step=60):
    """
    Returns an Altair heatmap of conversion rate (y) by loan category vs. duration bins.
    """
    heatmap_data = df.copy()

    # 1) Duration bins
    dur_bins   = list(range(dur_start, dur_end + dur_step, dur_step))
    # dur_labels = [f"{b}-{b+dur_step-1}" for b in dur_bins[:-1]]
    dur_labels = [
        f"{int(b/60)}–{int((b + dur_step)/60)}"
        for b in dur_bins[:-1]
    ]
    heatmap_data['duration_bin'] = pd.cut(
        heatmap_data['duration'],
        bins=dur_bins,
        labels=dur_labels,
        right=False
    )

    # 2) Loan category
    conditions = [
      (heatmap_data["housing"] == 1) & (heatmap_data["loan"] == 1),  # both
      (heatmap_data["housing"] == 0) & (heatmap_data["loan"] == 0),  # none
      (heatmap_data["housing"] == 1) & (heatmap_data["loan"] == 0),  # housing only
      (heatmap_data["housing"] == 0) & (heatmap_data["loan"] == 1)   # personal only
    ]
    choices = ["Both Loans","No Loans","Housing Loans","Personal Loans"]
    heatmap_data["loans?"] = np.select(conditions, choices, default="unknown")

    # 3) Pivot + fill
    heatmap_df = (
        heatmap_data
          .pivot_table(
             index='loans?',
             columns='duration_bin',
             values='y',
             aggfunc='mean'
          )
          .fillna(0)
    )

    # 4) Melt
    hm_long = heatmap_df.reset_index().melt(
        id_vars='loans?',
        var_name='duration_bin',
        value_name='conversion_rate'
    )

    # 5) Altair heatmap
    chart = (
        alt.Chart(hm_long)
          .mark_rect()
          .encode(
            x=alt.X("duration_bin:N", title="Minutes", sort=dur_labels, axis=alt.Axis(labelAngle=0, labelAlign="center", labelFontSize=9)),
            y=alt.Y("loans?:O",      title="Loan Type",     sort=choices),
            color=alt.Color("conversion_rate:Q", 
                            title="Conversion Rate", 
                            scale=alt.Scale(scheme="blues")),
            tooltip=[
              alt.Tooltip("loans?:O",        title="Loan Type"),
              alt.Tooltip("duration_bin:N",  title="Minutes"),
              alt.Tooltip("conversion_rate:Q",title="Conversion Rate", format=".1%")
            ]
          )
          .properties(
            width="container", height=400,
            padding={"top": 20, "right": 0, "bottom": 0, "left": 0},  # ← add top padding
            title="Conversion Rate Heatmap (Loans × Duration in Minutes)"
          )
          .configure_title(fontSize=18, anchor="start")
          .configure_axis(labelFontSize=12, titleFontSize=14)
    )
    return chart

# plot the donut chart for proportion of each previous outcome 
def previous_donut(df, width, height, filter_col="poutcome", filter_val=1):
    """
    Filters df[filter_col] == filter_val, then builds a Plotly donut chart
    showing the proportion of Wins vs Losses (column 'y').
    """
    # 1) Filter
    wins = df[df[filter_col] == filter_val]

    # 2) Count outcomes
    counts = wins["y"].value_counts().sort_index()
    zero_ct = int(counts.get(0, 0))
    one_ct  = int(counts.get(1, 0))

    # 3) Donut labels/values
    labels = ["Losses", "Wins"]
    values = [zero_ct, one_ct]

    # 4) Build donut
    fig = go.Figure(go.Pie(
        labels=labels,
        values=values,
        hole=0.45,
        textinfo="label+value+percent",
        textposition="inside",
        insidetextorientation="horizontal",
        marker=dict(
            colors=["#EF553B", "#636EFA"],   # same palette as your pie
            line=dict(width=1, color="white")
        ),
        textfont=dict(color=["white", "white"], size=14)
    ))

    fig_title = "Proportion of Campaign Outcome"

    # Title by filter value
    # if filter_val == 0:
    #     fig_title = "Proportion of Campaign Outcome"
    # elif filter_val == 0.5:
    #     fig_title = "Proportion of Campaign Outcome"
    # elif filter_val == 1:
    #     fig_title = "Proportion of Campaign Outcome"
    # else:
    #     fig_title = "Proportion of Campaign Outcome"

    # Styling to match your pie helper (tight top gap + fixed size)
    fig.update_traces(marker=dict(line=dict(width=1, color="white")))
    fig.update_layout(
        title=dict(text=fig_title, pad=dict(t=10), font=dict(size=16)),
        margin=dict(t=40, b=0, l=0, r=0),
        showlegend=False,
        width=width,
        height=height
    )

    return fig

## Clustering-related Functions
## -----------------------------------------------
## -----------------------------------------------

@st.cache_data(show_spinner=False)
def get_scaled(data: pd.DataFrame, cols: list):
    scaler = StandardScaler()
    Xs = scaler.fit_transform(data[cols])
    return scaler, Xs

@st.cache_data(show_spinner=False)
def get_labels(Xs: np.ndarray):
    clusterer = hdbscan.HDBSCAN(
        min_cluster_size=max(2, int(round(Xs.shape[0] / 100))),
        cluster_selection_method='eom'
    )
    return clusterer.fit_predict(Xs)

@st.cache_resource(show_spinner=False)
def get_surrogate(labels: np.ndarray, Xs: np.ndarray):
    rf_dict = {}
    for cl in np.unique(labels):
        y_bin = (labels == cl).astype(int)
        rf = RandomForestClassifier(n_estimators=5, random_state=42)
        rf.fit(Xs, y_bin)
        rf_dict[int(cl)] = rf
    return rf_dict

@st.cache_resource(show_spinner=False)
def get_multi_surrogate(Xs: np.ndarray, labels: np.ndarray, max_samples: 50):
    """Train one RandomForest to predict all cluster labels at once."""
    n = min(max_samples, len(labels))
    # reproducible sampling
    idx = np.random.RandomState(42).choice(len(labels), size=n, replace=False)
    X_small = Xs[idx]
    y_small = labels[idx]

    rf_small = RandomForestClassifier(
        n_estimators=5,   # fewer trees
        max_depth=5,       # shallower trees
        random_state=42,
        n_jobs=-1
    )
    rf_small.fit(X_small, y_small)
    return rf_small


# Function to perform HDBScan clustering algorithm
def auto_hdbscan(X, min_size=10):
    """HDBSCAN clusters automatically—no k needed (noise = -1)."""
    return hdbscan.HDBSCAN(min_cluster_size=min_size).fit_predict(X)

# Function to show feature example and descriptions table
def show_example_table(data, selected_cols, special=0):

    DESC = {
        "age":               "Age (whole number)",
        "education":         "Education level (1=elementatry, 2=secondary, 3=post-secondary)",
        "default":           "Previously default? (1=yes, 0=no)",
        "balance":           "Bank balance (decimal number)",
        "contact_cellular":  "Contact channel is cellular phone? (1=yes,0=no)",
        "contact_telephone": "Contact channel is landline phone? (1=yes,0=no)",
        "housing":           "Housing loan? (1=yes, 0=no)",
        "loan":              "Personal loan? (1=yes, 0=no)",
        "day":               "Day of month in the most recent marketing campaign (whole number: 1–31)",
        "month":             "Month of year in the most recent marketing campaign (whole number: 1–12)",
        "duration":          "Call duration in seconds (whole number)",   # default
        "campaign":          "Times contacted during this marketing campaign, including this time (whole number)",
        "pdays":             "Days since last marketing campaign (whole number, -1=first time)",
        "previous":          "Times contacted before this marketing campaign (whole number)",
        "poutcome":          "Outcome of previous marketing campaign (0=failure / 0.5=nonexistent / 1=success)",
        "marital_divorced":  "Divorced? (1=yes, 0=no)",
        "marital_married":   "Married? (1=yes, 0=no)",
        "marital_single":    "Single? (1=yes, 0=no)",
        "job_admin.":        "Working an administrative job? (1=yes, 0=no)",
        "job_blue_collar":   "Working a blue collar job? (1=yes, 0=no)",
        "job_entrepreneur":  "Working as a entrepreneur? (1=yes, 0=no)",
        "job_housemaid":     "Working as a housemaid? (1=yes, 0=no)",
        "job_management":    "Working in management? (1=yes, 0=no)",
        "job_retired":       "Retired? (1=yes, 0=no)",
        "job_self_employed": "Self-employed? (1=yes, 0=no)",
        "job_services":      "Work in the services industry? (1=yes, 0=no)",
        "job_student":       "Being a student? (1=yes, 0=no)",
        "job_technician":    "Work as a technician? (1=yes, 0=no)",
        "job_unemployed":    "Not working now? (1=yes, 0=no)",
        "job_unknown":       "Unknown job? (1=yes, 0=no)",
        "days_in_year":      "Current day of year (whole number: 1–366)",
        "y":                 "Did the client subscribed the product (term deposit)? (1=yes, 0=no) (TARGET VARIABLE)",
    }

    EX = {
        "age":               "42",
        "education":         "2",
        "default":           "0",
        "balance":           "10000.00",
        "contact_cellular":  "0",
        "contact_telephone": "1",
        "housing":           "0",
        "loan":              "1",
        "day":               "15",
        "month":             "7",
        "duration":          "900",   # default in seconds
        "campaign":          "3",
        "pdays":             "5",
        "previous":          "2",
        "poutcome":          "success",
        "marital_divorced":  "1",
        "marital_married":   "0",
        "marital_single":    "1",
        "job_admin.":        "0",
        "job_blue_collar":   "1",
        "job_entrepreneur":  "0",
        "job_housemaid":     "1",
        "job_management":    "0",
        "job_retired":       "1",
        "job_self_employed": "0",
        "job_services":      "1",
        "job_student":       "0",
        "job_technician":    "1",
        "job_unemployed":    "0",
        "job_unknown":       "1",
        "days_in_year":      "59",
        "y":                 "1",
    }

    # 👉 Override duration based on `special`
    if special != 0:
        DESC["duration"] = "Call duration in minutes (whole number)"
        EX["duration"]   = "15"  # example in minutes

    # --- Build the DataFrame as you already do ---
    ordered_feats = [f for f in selected_cols if f != "y"]
    if "y" in selected_cols:
        ordered_feats.append("y")

    feature_info = pd.DataFrame(
        index=["Description", "Example"],
        columns=ordered_feats,
        data=""
    )

    feature_info.loc["Description"] = [DESC.get(f, "") for f in ordered_feats]
    feature_info.loc["Example"]     = [EX.get(f,   "") for f in ordered_feats]

    fi = feature_info.T.reset_index().rename(columns={"index": "Feature"})
    fi.index = range(1, len(fi) + 1)

    st.dataframe(
        fi,
        use_container_width=True,
        column_config={
            "Feature":      st.column_config.TextColumn(width="medium"),
            "Description":  st.column_config.TextColumn(width="large"),
            "Example":      st.column_config.TextColumn(width="small"),
        },
    )





#  Function that computes and plots raw Cluster Feature Means
def show_cluster_feature_means_raw(data, selected_cols):

    # ─── Cluster Means & Δ-Means Tables ───────────────────────────────
    st.markdown(
        f"""<h4>{"<span style='color:#00BCD4;'>Feature Averages Among Groups</span>"}</h4>""",
        unsafe_allow_html=True
    )
    # st.subheader("For Regular Means:")

    data.rename(columns={"Cluster": "Groups"}, inplace=True)

    cluster_means = data.groupby("Groups")[selected_cols].mean().round(2)
    overall_mean  = data[selected_cols].mean()
    delta_means   = (cluster_means.subtract(overall_mean, axis=1)).round(2)

    # rename index: -1 → "Noise", others → "Index X"
    def make_label(idx):
        return "Outliers" if idx == -1 else f"Customer Group {1+idx}"
    cluster_means.index = cluster_means.index.map(make_label)
    # now move "Outliers" to the end
    if "Outliers" in cluster_means.index:
        # build a new index order: everything except Outliers, then Outliers
        new_order = [lab for lab in cluster_means.index if lab != "Outliers"] + ["Outliers"]
        cluster_means = cluster_means.loc[new_order]

    # delta_means.index   = delta_means.index.map(make_label)
    # if "Outliers" in delta_means.index:
    #     # build a new index order: everything except Outliers, then Outliers
    #     new_order = [lab for lab in delta_means.index if lab != "Outliers"] + ["Outliers"]
    #     delta_means = delta_means.loc[new_order]

    # Define which columns should be int vs float
    int_cols = ["age", "day", "duration", "pdays", "days_in_year", "campaign", "contact_telephone"]
    float_cols = [c for c in cluster_means.columns if c not in int_cols]

    styled_means = (
        cluster_means
        .style
        .background_gradient(cmap="vlag")
        .format({col: "{:.0f}" for col in int_cols} | {col: "{:.2f}" for col in float_cols})
    )
    st.dataframe(styled_means)

    # Delta table (uncomment to use)
    # st.subheader("For Δ-means (Customer Group Mean - Overall Mean):")
    # # st.write("Δ-means (Customer Group Mean - Overall Mean):")
    # styled_delta = (
    #     delta_means
    #     .style
    #     .background_gradient(cmap="vlag")
    #     .format("{:.2f}")
    # )
    # st.dataframe(styled_delta)



# Function that build Violin Plots on Raw Data
def plot_violin_top_features_raw(data, selected_cols, top_n=3):
    # st.subheader(f"2. Violin Plots (Original Scale) for Top {top_n} Features (TBA)")

    # 1) Figure out the top-n features by variance of raw cluster means
    cluster_means = data.groupby("Cluster")[selected_cols].mean()
    top_feats = cluster_means.var().sort_values(ascending=False).index[:top_n].tolist()

    # Codebase for printing the violin plots (uncomment to reuse)

    # # 2) Build a label mapping: -1 → Noise, else → Cluster {i}
    # unique_idxs = sorted(data["Cluster"].unique())
    # label_map = {idx: ("Noise" if idx == -1 else f"Cluster {1+idx}") for idx in unique_idxs}

    # # 3) Copy data and add a human-readable cluster column
    # df = data.copy()
    # df["Cluster_label"] = df["Cluster"].map(label_map)

    # # 4) Create columns for side-by-side plots
    # cols = st.columns(len(top_feats))
    # for i, feat in enumerate(top_feats):
    #     with cols[i]:
    #         fig, ax = plt.subplots()
    #         sns.violinplot(
    #             x="Cluster_label",
    #             y=feat,
    #             data=df,
    #             inner="quartile",
    #             order=[label_map[idx] for idx in unique_idxs],  # preserve ordering
    #             ax=ax
    #         )
    #         ax.set_title(f"{feat} distribution by cluster")
    #         ax.set_xlabel("")  # optional: remove repeated x-axis labels
    #         ax.tick_params(axis='x', rotation=45)
    #         st.pyplot(fig)
    
    return top_feats


# Function that plots Tree-Based Importance to show importance of each factor
def plot_tree_feature_importance(data, X_scaled, selected_cols, top_n=5):
    st.markdown(
        f"""<h4>{'<span style="color:#00BCD4;">Crucial Factors Forming the Groups</span>'}</h4>""",
        unsafe_allow_html=True
    )
    # st.header("Important Factors That Formed the Customer Groups (& Outliers)")

    # build tabs for users to traverse
    cluster_labels = sorted(set(data["Cluster"]))
    if -1 in cluster_labels:
        cluster_labels = [cl for cl in cluster_labels if cl != -1] + [-1]

    # 2) Build the tab‐names in exactly the same order
    tab_labels = [
        ("Outliers" if cl == -1 else f"Customer Group {cl+1}")
        for cl in cluster_labels
    ]

    rf_models = {}
    tabs = st.tabs(tab_labels)
    for tab, cl in zip(tabs, cluster_labels):
        with tab:
            # 1) Train surrogate
            y = (data["Cluster"] == cl).astype(int)
            rf = RandomForestClassifier(n_estimators=100, random_state=42)
            rf.fit(X_scaled, y)
            rf_models[cl] = rf

            # 2) Get top_n importances
            imps = pd.Series(rf.feature_importances_, index=selected_cols).nlargest(top_n)

            # 3) Create a smaller figure on a light-grey background
            fig, ax = plt.subplots(
                figsize=(8, 5),            # smaller width x height
            )
            light_bg = "#f5f5f5"
            fig.patch.set_facecolor(light_bg)
            ax.set_facecolor(light_bg)

            # 4) Plot bars
            bars = ax.bar(imps.index, imps.values, color="#5a9bd4", alpha=0.85)

            # 5) Annotate each bar
            for bar in bars:
                h = bar.get_height()
                ax.text(
                    bar.get_x() + bar.get_width() / 2,
                    h,
                    f"{h:.2f}",
                    ha="center", va="bottom",
                    fontsize=10,
                    color="#333"
                )

            # 6) Style axes & title with dark text
            title = ("Outliers" if cl == -1 else f"Customer Group {cl+1}")
            ax.set_title(f"Top {top_n} Crucial Features for {title} & their Importance Score", color="#333", fontsize=8)
            ax.set_xlabel("Features", color="#333", fontsize=8)
            ax.set_ylabel("Importance Score", color="#333", fontsize=8)
            ax.tick_params(colors="#333", size=4)
            plt.setp(ax.get_xticklabels(), rotation=0, color="#333", fontsize=8)
            plt.setp(ax.get_yticklabels(), color="#333", fontsize=8)

            plt.tight_layout()

            # 7) Render with fixed width
            # st.pyplot(fig, use_container_width=False)
            buf = BytesIO()
            fig.savefig(buf, format="png", bbox_inches="tight")
            buf.seek(0)
            img_bytes = buf.getvalue()
            b64 = base64.b64encode(img_bytes).decode()

            st.markdown(
                f"""
                <div style="display:flex; justify-content:center;">
                    <img src="data:image/png;base64,{b64}" style="max-width:100%; height:auto;">
                </div>
                """,
                unsafe_allow_html=True
            )
            

    return rf_models

# Plot SHAP Explanation for Custom Point (via Top-Feature Sliders)
def show_shap_explanation_custom(
    rf_model,
    scaler,
    data,
    selected_cols,
    top_n: int = 5
):
    """
    Renders sliders for the top_n most important features, predicts the cluster
    for the custom point via rf_model, and shows a SHAP waterfall plot explaining it.
    """
    st.subheader("SHAP Explanation for Added Custom Customer")

    # 1) Identify top-n features by RF importance
    importances = pd.Series(rf_model.feature_importances_, index=selected_cols)
    top_feats = importances.nlargest(top_n).index.tolist()

    # 2) Get global means for defaults
    global_means = data[selected_cols].mean()

    # print(top_feats)

    # 3) Build sliders
    st.write(f"Adjust values for top {top_n} features:")
    raw_vals = {}
    for feat in top_feats:
        lo = data[feat].min()
        hi = data[feat].max()
        mean = global_means[feat]

        if feat == "balance":
            # continuous slider
            raw_vals[feat] = st.slider(
                label=feat,
                min_value=0,
                # max_value=float(hi),
                # value=float(mean),
                value=10000.00,
                step=0.01,
                format="%.2f",
                key=f"slider_{feat}"
            )
        elif feat == "duration":
            # UI in minutes, convert back to seconds for the model
            min_m = int(np.floor(lo / 60.0))
            max_m = int(np.ceil(hi / 60.0))
            default_m = int(round(mean / 60.0))

            minutes = st.slider(
                label="duration (minutes)",
                min_value=min_m,
                max_value=max_m,
                value=default_m,
                step=1,
                format="%d",
                key=f"slider_{feat}_min"
            )
            raw_vals[feat] = minutes * 60  # store back in seconds
        else:
            # **integer** slider
            raw_vals[feat] = st.slider(
                label=feat,
                min_value=int(lo),
                max_value=int(hi),
                value=int(round(mean)),
                step=1,
                format="%d",              # <-- force integer formatting
                key=f"slider_{feat}"
            )

    # 4) Build raw point
    raw_point  = np.array([ raw_vals.get(f, global_means[f]) 
                            for f in selected_cols ]).reshape(1, -1)
    scaled_pt  = scaler.transform(raw_point)

    # # 5) Predict cluster
    pred = rf_model.predict(scaled_pt)[0]
    label = "Outliers" if pred == -1 else f"Customer Group{pred}"
    proba = rf_model.predict_proba(scaled_pt)[0][pred]
    # st.write(f"**Predicted Customer Group: ** {label} (probability={proba*100:.2f}%)")

    st.markdown(
        f"""<p><strong>Predicted Customer Group:</strong> 
        <span style='color:#FFC107 !important;'><u>{label}</u></span> 
        (probability = <span style='color:#FFC107 !important;'><u>{proba*100:.2f}%</u></span>)</p>""",
        unsafe_allow_html=True
    )

    # Build a single‐output explainer for the “pred” class probability:
    explainer = shap.Explainer(
        lambda d: rf_model.predict_proba(d)[:, pred], 
        data[selected_cols]    # or X you trained on
    )

    # Compute the Explanation object (shape: 1×n_features)
    full_exp = explainer(scaled_pt)

    # Extract the first (and only) explanation
    single_exp = full_exp[0]

    # Now plot exactly one waterfall
    shap.initjs()
    # draw into an Axes
    ax = shap.plots.waterfall(single_exp, show=False)

    # extract the parent Figure
    fig = ax.figure

    # render that
    st.pyplot(fig)

# get function for LIME
@st.cache_data(show_spinner=False)
def get_lime_explainer(
    sample: np.ndarray,
    feature_names: list,
    class_names: list
) -> LimeTabularExplainer:
    """
    Build and return a LimeTabularExplainer on a small sample.
    This will be cached so we only do the expensive setup once per (sample,features,classes).
    """
    return LimeTabularExplainer(
        training_data=sample,
        feature_names=feature_names,
        class_names=class_names,
        discretize_continuous=True
    )

# show the lime explanation from the lime explainer
def show_lime_explanation_custom(
    rf_model,
    scaler,
    data: pd.DataFrame,
    selected_cols: list,
    top_n: int = 5
):
    # --- session state for resetting the form ---
    if "lime_form_id" not in st.session_state:
        st.session_state.lime_form_id = 0
    form_id = st.session_state.lime_form_id

    st.markdown("<br>", unsafe_allow_html=True)
    # st.markdown(
    #     "<h4><span style='color:#00BCD4;'>Adjust the Values for Your Client. "
    #     "(Use the Table Above to Refer to the Right Features)</span></h4>",
    #     unsafe_allow_html=True
    # )

    # 1) Identify top-n features by RF importance
    importances = pd.Series(rf_model.feature_importances_, index=selected_cols)
    top_feats = importances.nlargest(top_n).index.tolist()
    global_means = data[selected_cols].mean()

    # 2) Collect slider inputs inside a FORM
    with st.form(key=f"lime_form_{form_id}"):
        raw_vals = {}
        for feat in top_feats:
            lo, hi = float(data[feat].min()), float(data[feat].max())
            default = float(global_means[feat])
            input_key = f"lime_{feat}_{form_id}"

            if feat == "balance":
                # Allow any value, not capped by min/max
                raw_vals[feat] = st.number_input(
                    feat,
                    min_value=0.00,
                    max_value=100000000.00,
                    value=10000.00,
                    step=0.01,
                    format="%.2f",
                    key=input_key
                )

            elif feat == "duration":
                # Let user choose in minutes, store back in seconds
                min_m = int(np.floor(lo / 60.0))
                max_m = int(np.ceil(hi / 60.0))
                default_m = int(round(default / 60.0))

                minutes = st.slider(
                    "duration (minutes)",
                    min_value=min_m,
                    max_value=60,
                    value=default_m,
                    step=1,
                    format="%d",
                    key=input_key
                )
                raw_vals[feat] = minutes * 60  # convert to seconds

            else:
                # Integer slider for other features
                raw_vals[feat] = st.slider(
                    feat,
                    int(lo), int(hi), int(default),
                    step=1, format="%d", key=input_key
                )

        st.markdown("<br>", unsafe_allow_html=True)
        submit = st.form_submit_button("🔍 Run Customer Group Assignment Prediction", use_container_width=True)
        
    # 3) Only on submit do you build & show result
    if not submit:
        return

    with st.spinner("Finding the best group for your added customer..."):
        # 3) Build the raw point & scale it
        raw_point = np.array([raw_vals.get(f, global_means[f]) for f in selected_cols]).reshape(1, -1)
        scaled_pt = scaler.transform(raw_point)

        # 4) Predict & pull out class names
        pred_label = int(rf_model.predict(scaled_pt)[0])
        sk_classes = list(rf_model.classes_)  # keeps the order used by predict_proba
        pred_index = sk_classes.index(pred_label)
        label_map = {cid: ("Outliers" if cid == -1 else f"Customer Group {cid+1}") for cid in sk_classes}

        # 5) Probabilities for each class
        proba_row = rf_model.predict_proba(scaled_pt)[0]  # same order as sk_classes

        # Build a tidy DataFrame
        prob_df = (
            pd.DataFrame({
                "Group ID": sk_classes,
                "Group": [label_map[cid] for cid in sk_classes],
                "Probability": proba_row
            })
            .sort_values("Probability", ascending=False)
            .reset_index(drop=True)
        )
        prob_df["Probability (%)"] = (prob_df["Probability"] * 100).round(2)
        prob_df.index = prob_df.index + 1

        output_col1, output_col2 = st.columns([7, 3], gap="large")
        with output_col1:
            msg = (
                "Predicted Outcome: "
                f"<span style='color:#FFC107;'>{label_map[pred_label]}</span>, "
                "(Probability = "
                f"<span style='color:#FFC107;'>{proba_row[pred_index]*100:.0f}%</span>)"
            )
            st.markdown(f"<h2>{msg}</h2>", unsafe_allow_html=True)

            # Show probability table under the headline
            # st.markdown("#### Probability by group")
            st.dataframe(
                prob_df[["Group", "Probability (%)"]],
                use_container_width=True
            )

        with output_col2:
            st.markdown("""
            <style>
            /* Push down all buttons inside the second output column */
            div[data-testid="column"]:nth-of-type(2) div.stButton > button {
                margin-top: 2rem;   /* increase until it looks good */
            }
            </style>
            """, unsafe_allow_html=True)
            # 5) Add a reset button to start fresh
            if st.button("🔁 Try a new prediction", use_container_width=True):
                st.session_state.lime_form_id += 1  # bump the form id to reset widget state
                st.rerun()  # reload page so defaults take effect

    # Note: If the user does NOT click reset, they can keep tweaking the sliders and re-submitting.

    # if insist to display XAI for customer segmentation
    # # # 5) Prepare a small sample for LIME (cap at 200 rows)
    # sample = data[selected_cols].values
    # if len(sample) > 50:
    #     idx = np.linspace(0, len(sample)-1, 50, dtype=int)
    #     sample = sample[idx]

    # # 6) Get the cached explainer and run it *inside* a spinner
    # # explainer = get_lime_explainer(sample, selected_cols, class_names)
    # with st.spinner("Fitting XAI local model…"):
    #     explainer = LimeTabularExplainer(
    #     training_data      = sample,
    #     feature_names      = selected_cols,
    #     class_names        = class_names,
    #     discretize_continuous=False
    #     )
    #     exp = explainer.explain_instance(
    #         raw_point[0],
    #         lambda x: rf_model.predict_proba(scaler.transform(x)),
    #         labels=(pred_index,),
    #         num_features=top_n,
    #         num_samples=50 
    #     )

    # # 9) Render LIME’s HTML
    # html = exp.as_html()
    # wrapper = f"""
    # <div style="
    #     background-color: white;
    #     padding: 16px;
    #     border-radius: 8px;
    #     box-shadow: 0 2px 8px rgba(0,0,0,0.1);
    #     display: flex;
    #     flex-wrap: wrap;
    #     justify-content: space-between;
    #     gap: 200px;          /* ↑ doubled the gap */
    # ">
    # {html}
    # </div>
    # """
    # components.html(wrapper, height=500, scrolling=True)

# Function that plots 3D Scatter on Raw
def plot_3d_clusters_raw(data, selected_cols, top_features):
    st.markdown(
        f"""<h4>{'<span style="color:#00BCD4;">3D View of Customer Groups (& Outliers)</span>'}</h4>""",
        unsafe_allow_html=True
    )
    # st.header("3D Cluster Visualization")
    
    # pick the top 3 features
    top3 = top_features[:3]

    # compute raw counts per cluster
    counts = data["Cluster"].value_counts().sort_index()

    # build label map with counts baked in
    ordered_labels = []
    label_map = {}
    # first real clusters
    non_outliers = [c for c in counts.index if c != -1]
    for i, cl in enumerate(non_outliers):
        label = f"Customer Group {i+1} ({counts[cl]})"
        label_map[cl] = label
        ordered_labels.append(label)
    # then outliers last, if any
    if -1 in counts.index:
        label = f"Outliers ({counts[-1]})"
        label_map[-1] = label
        ordered_labels.append(label)

    # map each row’s numeric cluster to its new label
    df = data.copy()
    df["Cluster Label"] = df["Cluster"].map(label_map)

    # pick a palette
    palette = px.colors.qualitative.Plotly
    color_map = {lbl: palette[i % len(palette)] for i,lbl in enumerate(ordered_labels)}

    # now scatter_3d — legend entries will be your labels-with-counts
    fig3d = px.scatter_3d(
        df,
        x=top3[0], y=top3[1], z=top3[2],
        color="Cluster Label",
        category_orders={"Cluster Label": ordered_labels},
        color_discrete_map=color_map,
        # title="3D view of Customer Groups & Outliers",
        width=200, height=550
    )
    st.plotly_chart(fig3d)

# XAI-related Functions
#-----------------------------------------------
#-----------------------------------------------
@st.cache_data

# Function that load XAI (LIME & SHAP) explainers
@st.cache_resource
def load_explainers(_model, _df: pd.DataFrame, feature_names: tuple):
    """
    Builds two SHAP explainers (for P(Yes) and P(No)) + one LIME explainer
    all on exactly the same feature set.
    """
    X = _df.loc[:, list(feature_names)]
    # SHAP: single‐output explainer for P(Yes)
    shap_explainer = shap.Explainer(
        lambda data: _model.predict_proba(data)[:, 1],
        X
    )
    # LIME: full classifier explainer (we’ll ask for label=1 later)
    lime_explainer = LimeTabularExplainer(
        X.values,
        feature_names   = list(feature_names),
        class_names     = ['Lose','Win'],
        discretize_continuous=True
    )
    return shap_explainer, lime_explainer

# Function that displays the explanations
def show_explanations(model, inputs, shap_explainer, lime_explainer, max_lime_features: int = 10):
    # ─── normalize inputs to 1×n_features DataFrame ───
    if isinstance(inputs, dict):
        X = pd.DataFrame([inputs])
    elif isinstance(inputs, (list, np.ndarray)):
        arr = np.array(inputs).reshape(1, -1)
        cols = lime_explainer.feature_names
        X = pd.DataFrame(arr, columns=cols)
    else:
        X = inputs.copy()
    assert X.shape[0] == 1, "Need exactly one row of inputs"

    st.markdown("<br>", unsafe_allow_html=True)
    # st.header("Through Explainable AI (XAI):")


    # ─── Display LIME output for label=1 (“Yes”) ───
    # st.markdown("**1. LIME: Explains your model’s prediction by creating a simple model just around your input, showing which features had the biggest influence on the result!**")
    st.markdown(
        """
        <strong>
        1. <span style="color:#FFC107;">LIME</span> (Local Interpretable Model-Agnostic Explanations): <br></br>
        Explains your model’s prediction by  
        <span style="color:#00BCD4;">creating a simple model just around your input</span>, showing which  
        <span style="color:#00BCD4;">features had the biggest influence</span> 
        on the result.<br>
        </strong>
        """,
        unsafe_allow_html=True
    )
    lime_exp = lime_explainer.explain_instance(
        X.values.flatten(),
        model.predict_proba,
        labels=(1,),
        # labels=(1,0),
        num_features=min(max_lime_features, X.shape[1])
    )

    lime_html = lime_exp.as_html()

    # wrap it in a white box with some padding & rounded corners
    wrapper = f"""
    <div style="
    background:#cfd1d4;
    padding:26px 28px;
    border-radius:10px;
    box-shadow:0 2px 12px rgba(0,0,0,.18);
    ">
    <style>
        /* Lay the 3 sections out in a row with equal gaps */
        .lime {{
        display: flex !important;
        align-items: flex-start;
        justify-content: flex-start;
        gap: 192px;                     /* equal distance between the 3 blocks */
        }}
        .lime > table {{                 /* remove default table margins */
        margin: 0 !important;
        background: rgba(255,255,255,.85) !important;  /* soften white inside */
        border-collapse: separate !important;
        border-spacing: 0 !important;
        }}

        /* Reasonable widths so the center plot has room */
        .lime > table:nth-of-type(1) {{ flex: 0 0 260px; }}   /* probabilities */
        .lime > table:nth-of-type(2) {{ flex: 1 1 560px; }}   /* contribution bars */
        .lime > table:nth-of-type(3) {{ flex: 0 0 260px; }}   /* feature values  */

        /* Extra breathing room between the 3 main sections (fallback if flex gaps ignored) */
        .lime .lime-top-table {{ margin-bottom: 0 !important; }}
        .lime .lime-table {{ margin: 0 !important; }}

        /* Make middle bar plot more sparse vertically */
        .lime > table:nth-of-type(2) td {{
        padding-top: 8px !important;
        padding-bottom: 8px !important;
        }}
        .lime > table:nth-of-type(2) svg g rect {{
        transform: translateY(6px);    /* separate bars visually */
        }}
        

        /* Mobile fallback: stack vertically */
        @media (max-width: 1100px) {{
        .lime {{ flex-direction: column; gap: 24px; }}
        .lime > table:nth-of-type(1),
        .lime > table:nth-of-type(2),
        .lime > table:nth-of-type(3) {{ flex: 1 1 auto; }}
        }}
    </style>
    {lime_html}
    </div>
    """
    components.html(wrapper, height=300)


    # components.html(lime_exp.as_html(), height=350)

    # ─── SHAP force plot for P(Yes) ───
    # st.markdown("**2. SHAP: Shows how much each of your inputs helps push the prediction higher or lower!**")
    st.markdown(
        """
        <br></br>
        <strong>
        2. <span style="color:#FFC107;">SHAP</span> (SHapley Additive exPlanations):<br></br>
        Explains your model’s prediction by showing how much each of your inputs 
        <span style="color:#00BCD4;">helps push the prediction higher or lower.</span>
        </strong><br>
        """,
        unsafe_allow_html=True
    )
    expl = shap_explainer(X)     # Explanation with shape (1, n_features)
    single_exp = expl[0]          # pick the one row
    shap.initjs()
    fig = shap.plots.force(single_exp, matplotlib=True, show=False)
    # Set figure + axes background
    fig.patch.set_facecolor("#cfd1d4")
    ax = plt.gca()
    ax.set_facecolor("#cfd1d4")

    # Loop through all text objects to fix the white boxes
    for text in ax.texts:
        bbox = text.get_bbox_patch()
        if bbox is not None:
            bbox.set_facecolor("#cfd1d4")  # match grey background
            bbox.set_edgecolor("none")     # remove border

    # Hide vertical lines
    for line in ax.lines:
        line.set_color("#cfd1d4")  # match background color
        line.set_alpha(1)          # fully opaque to cover them

    st.pyplot(fig, bbox_inches="tight")

# export-related functions
#-----------------------------------------------
#-----------------------------------------------

@st.cache_data
def get_csv_bytes(df: pd.DataFrame) -> bytes:
    return df.to_csv(index=False).encode("utf-8")

@st.cache_data
def get_excel_buffer(df: pd.DataFrame) -> io.BytesIO:
    buf = io.BytesIO()
    df.to_excel(buf, index=False, sheet_name="sheet", engine="openpyxl")
    buf.seek(0)
    return buf



# MAIN CODE
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------
st.set_page_config(
    page_title="FinML-Studio", 
    layout="wide",
    initial_sidebar_state="expanded"
)

# customer css styling
st.markdown(
    """
    <style>
        /* make any st.metric container use the sidebar bg */
        [data-testid="stMetric"] {
            background-color: var(--sidebar-background) !important;
            padding: 1rem 1.5rem !important;
            border-radius: 0.75rem !important;
        }

        [data-testid="stMetric"] {
            background-color: #393939;
            text-align: center;
            padding: 15px 0;
        }

        [data-testid="stMetricLabel"] {
            display: flex;
            justify-content: center;
            align-items: center;
        }

        [data-testid="stMetricDeltaIcon-Up"] {
            position: relative;
            left: 38%;
            -webkit-transform: translateX(-50%);
            -ms-transform: translateX(-50%);
            transform: translateX(-50%);
        }
        [data-testid="stMetricDeltaIcon-Down"] {
            position: relative;
            left: 38%;
            -webkit-transform: translateX(-50%);
            -ms-transform: translateX(-50%);
            transform: translateX(-50%);
        }

        div[data-testid="stPlotlyChart"] {
            background-color: var(--sidebar-background) !important;
            padding: 1rem !important;
            border-radius: 0.75rem !important;
        }
        .venn-container {
        background-color: var(--sidebar-background) !important;
        padding: 1rem !important;
        border-radius: 0.75rem !important;
        }
    </style>
    """,
    unsafe_allow_html=True,
)


# --- CACHED RESOURCE LOADING ---
# @st.experimental_memo
@st.cache_data
# function that load the ML predictive models
def load_models():
    # Load Decision Tree (or Resampled Model)
    # with open('../../Model/DT_Model_Deploy.pkl', 'rb') as f:
    # from home
    # with open('../../Model/DT_Resampled_Model_Deploy.pkl', 'rb') as f:
    # for deployment

    # cloud:
    with open('Models/Resampled_Models/Decision_Tree/DT_Resampled_Model_Deploy.pkl', 'rb') as f:
    # local:
    # with open('Model/DT_Resampled_Model_Deploy(OLD).pkl', 'rb') as f:
        dt_pipeline = pickle.load(f)
        dt_model = dt_pipeline.named_steps['classifier']
    # Load Random Forest (or Resampled Model)
    # with open('../../Model/RF_Model_Deploy.pkl', 'rb') as f:

    # cloud:
    with open('Models/Resampled_Models/Random_Forest/RF_Resampled_Model_Deploy.pkl', 'rb') as f:
    # local:
    # with open('Model/RF_Resampled_Model_Deploy(OLD).pkl', 'rb') as f:
        rf_pipeline = pickle.load(f)
        rf_model = rf_pipeline.named_steps['classifier']
    # Load XGBoost (or Resampled Model)
    # with open('../../Model/XGB_Model_Deploy.pkl', 'rb') as f:

    # cloud:
    with open('Models/Resampled_Models/XGBoost/XGB_Resampled_Model_Deploy.pkl', 'rb') as f:
    # local
    # with open('Model/XGB_Resampled_Model_Deploy(OLD).pkl', 'rb') as f:
        xgb_pipeline = pickle.load(f)
        xgb_model = xgb_pipeline.named_steps['classifier']
    return {
        'Decision Tree': dt_model,
        'Random Forest': rf_model,
        'XGBoost': xgb_model
    }

# @st.experimental_singleton
@st.cache_resource
# functuion that load the data for this app
def load_data():
    # Load processed data for dashboard
    url = (
        "https://raw.githubusercontent.com/Alex-Mak-MCW/FinML-Studio/refs/heads/main/Data/processed_Input.csv"
    )
    return pd.read_csv(url)

# --- REUSABLE UTILS ---
# Function that makes model prediction based on model input
def make_prediction(model, user_input):
    return model.predict(user_input)

# Takes user input for decisiion tree model
def user_input_form_decision_tree():
    st.markdown("<br>", unsafe_allow_html=True)
    # st.subheader('"A Tree-based Model that Makes Decisions by Splitting Data Repeatedly on Feature Values"')
    st.markdown(
        '<h3 style="color:#FFC107;">"A Tree-based AI Model that Makes Decisions by <u>Splitting Data Repeatedly on Feature Values</u>"</h3>',
        unsafe_allow_html=True
    )

    # 1) Build the pros/cons table
    dt_pros_cons_df = pd.DataFrame({
        "Strengths": [
            "🔍  Highly Interpretable",
            "🧩  Can Handle Mixed Data Types",
            "🛡️  Robust to Outliers"
        ],
        "Weaknesses": [
            "⚠️  Prone to Overfitting",
            "📉  High Variance",
            "🌪️  Instability"
        ]
    })
    dt_pros_cons_df.index = [1, 2, 3]
    
    # dt_pros_cons_df.index = [''] * len(dt_pros_cons_df)
    # 2) Display it at the top

    # st.table(dt_pros_cons_df)            # static table :contentReference[oaicite:12]{index=12}
    st.dataframe(
        dt_pros_cons_df,
        column_config={
            "Strengths": st.column_config.TextColumn(width="medium"),
            "Weaknesses": st.column_config.TextColumn(width="medium"),
        },
        hide_index=True,
        use_container_width=True
    )
    # st.dataframe(pros_cons_df, use_container_width=True)  # interactive alternative :contentReference[oaicite:13]{index=13}
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")

    # st.markdown(
    #     "<h3 style='color:#FFC107;'>Fill in Your Customer's Values:</h3>",
    #     unsafe_allow_html=True
    # )
    st.subheader("Fill in Your Client's Values:")


    # Calculate the current day of the year for days_in_year
    day_of_year = datetime.datetime.now().timetuple().tm_yday

    # 3) Show input
    # IF USING REGULAR DT MODEL
    # # Assign inputs
    # age = st.slider("What is your Age (18-65)?", min_value=18, max_value=65, value=42, key=0)
    # balance = st.number_input("What is you bank account balance (0-100000000)?", min_value=0, max_value=100000000, value=10000, key=1)
    # duration = st.slider("How long was the Duration of your last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=2)
    # campaign = st.slider("How many times did our bank contact you?", min_value=0, max_value=15, value=0, key=3)
    # pdays = st.slider("How many days ago when we last contacted you?", min_value=0, max_value=1000, value=5, key=4)
    # poutcome = st.selectbox("What is the outcome of your last campaign?", ["Unknown", "Failure", "Success"], index=0, key=5)  # Default value is 0
    # days_in_year = st.slider("What is the number of Days in a year you are currently at?", min_value=0, max_value=365, value=day_of_year, key=6)

    # IF USING Resampled DT MODEL
    # Assign inputs
    age = st.slider("What is your client's age (18-65)?", min_value=18, max_value=65, value=42, key=0)
    balance = st.number_input("What is your client's bank account balance?", min_value=0, max_value=100000000, value=10000, key=1)
    housing = st.selectbox("Does your client have any housing loans?", ["No", "Yes"], index=0, key=2)  # Default value is 0
    duration = st.slider("How long was your client contacted (in minutes)?", min_value=0, max_value=300, value=15, key=3)
    # campaign = st.slider("How many times did our bank contact you?", min_value=0, max_value=15, value=0, key=3)
    # pdays = st.slider("How many days ago when we last contacted you?", min_value=0, max_value=1000, value=5, key=4)
    poutcome = st.selectbox("What was the outcome of your client was previosuly approached?", ["Unknown", "Failure", "Success"], index=0, key=4)  # Default value is 0
    # days_in_year = st.slider("What is the number of Days in a year did you contact your client?", min_value=0, max_value=365, value=day_of_year, key=5)


    year = datetime.date.today().year
    # let the user pick a date
    chosen_date = st.date_input(
        "Pick the date your client was last contacted:",
        value=datetime.date(year, 1, 1),  # default = Jan 1
        min_value=datetime.date(year, 1, 1),
        max_value=datetime.date(year, 12, 31),
        key=5
    )

    # convert date → day-of-year number
    days_in_year = (chosen_date - datetime.date(year, 1, 1)).days
    print(days_in_year)

    
    # 4) Input Handling
    age          = float(age)
    balance      = float(balance)
    if housing == "Yes":
        housing = 1
    else:
        housing = 0
    duration     = float(duration) * 60   # convert minutes → seconds
    # campaign     = float(campaign)
    # pdays        = float(pdays)
    days_in_year = float(days_in_year)
    if poutcome == "Failure":
        poutcome = 0
    elif poutcome == "Unknown":
        poutcome = 0.5
    else:
        poutcome = 1

    # 5) Return output as dict
    # **NEW**: return a dict instead of a list
    # return {
    #     "age":          age,
    #     "balance":      balance,
    #     "duration":     duration,
    #     "campaign":     campaign,
    #     "pdays":        pdays,
    #     "poutcome":     poutcome,
    #     "days_in_year": days_in_year,
    # }


    return {
        "age":          age,
        "balance":      balance,
        "housing":     housing,
        "duration":     duration,
        "poutcome":     poutcome,
        "days_in_year": days_in_year,
    }

# Takes user input for random forest (resampled) model
def user_input_form_random_forest():
    
    st.markdown("<br>", unsafe_allow_html=True)
    # st.subheader('"Combining many Decision Trees and their Predictions into 1 Outcome"')
    st.markdown(
        '<h3 style="color:#FFC107;">"<u>Combining many Decision Trees</u> and their Predictions into 1 Overall Outcome"</h3>',
        unsafe_allow_html=True
    )

    # 1) Build the pros/cons table
    rf_pros_cons_df = pd.DataFrame({
        "Strengths": [
            "🌲  Less Likely to Overfit",
            "📊  Handles High-dimensional Data Well",
            "🛡️  Robust to Noise & Outliers"
        ],
        "Weaknesses": [
            "🔍  Less Interpretable than Decision Tree",
            "💾  Higher Memory Usage",
            "🐢  Slower Predictions"
        ]
    })
    # rf_pros_cons_df.index = [''] * len(rf_pros_cons_df)
    rf_pros_cons_df.index = [1, 2, 3]
    # 2) Display it at the top
    # st.table(rf_pros_cons_df)            # static table :contentReference[oaicite:12]{index=12}
    st.dataframe(
        rf_pros_cons_df,
        column_config={
            "Strengths": st.column_config.TextColumn(width="medium"),
            "Weaknesses": st.column_config.TextColumn(width="medium"),
        },
        hide_index=True,
        use_container_width=True
    )
    # st.dataframe(pros_cons_df, use_container_width=True)  # interactive alternative :contentReference[oaicite:13]{index=13}
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Fill in Your Client's Values:")
    # st.markdown(
    #     "<h3 style='color:#FFC107;'>Fill in Your Customer's Values:</h3>",
    #     unsafe_allow_html=True
    # )


    # Get current date
    day_of_year = datetime.datetime.now().timetuple().tm_yday

    # 3) take user input
    # Regular RF Model
    # Assign inputs
    # age = st.slider("What is your Age (18-65)?", min_value=18, max_value=65, value=42, key=10)
    # balance = st.number_input("What is your bank account balance (0-100000000)?", min_value=0, max_value=100000000, value=10000, key=11)
    # duration = st.slider("How long was the duration of your client's last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=12)
    # campaign = st.slider("How many times did we contact your client?", min_value=0, max_value=15, value=0, key=13)
    # pdays = st.slider("How many days ago when we last contacted your client?", min_value=0, max_value=1000, value=5, key=14)
    # poutcome = st.selectbox("What is the outcome of your client's last campaign?", ["Unknown", "Failure", "Success"], index=0, key=15)  # Default value is 0
    # days_in_year = st.slider("What is the number of Days in a year did you contact your client?", min_value=0, max_value=365, value=day_of_year, key=16)

    age = st.slider("What is your client's age (18-65)?", min_value=18, max_value=65, value=42, key=10)
    balance = st.number_input("What is your client's bank account balance?", min_value=0, max_value=100000000, value=10000, key=11)
    housing = st.selectbox("Does your client have any housing loans?", ["No", "Yes"], index=0, key=12)  # Default value is 0
    duration = st.slider("How long was your client contacted (in minutes)?", min_value=0, max_value=300, value=15, key=13)
    pdays = st.slider("How many days ago when your client was contacted?", min_value=0, max_value=1000, value=5, key=14)
    poutcome = st.selectbox("What was the outcome of your client was previosuly approached?", ["Unknown", "Failure", "Success"], index=0, key=15)  # Default value is 0
    marital_married = st.selectbox("Is your client married?", ["No", "Yes"], index=0, key=16)  # Default value is 0
    job_blue_collar = st.selectbox("Does your client have a blue collor job?", ["No", "Yes"], index=0, key=17)  # Default value is 0
    # days_in_year = st.slider("What is the number of Days in a year did you contact your client?", min_value=0, max_value=365, value=day_of_year, key=18)
    year = datetime.date.today().year
    # let the user pick a date
    chosen_date = st.date_input(
        "Pick the date your client was last contacted:",
        value=datetime.date(year, 1, 1),  # default = Jan 1
        min_value=datetime.date(year, 1, 1),
        max_value=datetime.date(year, 12, 31),
        key=18
    )

    # convert date → day-of-year number
    days_in_year = (chosen_date - datetime.date(year, 1, 1)).days
    print(days_in_year)
    
    # 4) input handling
    age = float(age)
    balance = float(balance)
    if housing == "Yes":
        housing = 1
    else:
        housing = 0
    duration = float(duration) *60
    pdays        = float(pdays)
    if marital_married == "Yes":
        marital_married = 1
    else:
        marital_married = 0
    if job_blue_collar == "Yes":
        job_blue_collar = 1
    else:
        job_blue_collar = 0

    days_in_year = float(days_in_year)

    # for poutcome
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1

    # 5) return input values in dict
    #  IF original model
    # return {
    #     "age":          age,
    #     "balance":      balance,
    #     "duration":     duration,
    #     "campaign":     campaign,
    #     "pdays":        pdays,
    #     "poutcome":     poutcome,
    #     "days_in_year": days_in_year,
    # }

    return {
        "age":          age,
        "balance":      balance,
        "housing":      housing,
        "duration":     duration,
        "pdays":        pdays,
        "poutcome":     poutcome,
        "marital_married": marital_married,
        "job_blue_collar": job_blue_collar,
        "days_in_year": days_in_year,
    }

# take user input for XGBoost
def user_input_form_xgboost():

    st.markdown("<br>", unsafe_allow_html=True)
    # st.subheader('"An Intelligent Tree that Learns from its 100 Ancestors to Make Educated Decisions."')
    st.markdown(
        '<h3 style="color:#FFC107;">"An Intelligent Tree that <u>Learns from its 100 Ancestors</u> to Make Educated Decisions."</h3>',
        unsafe_allow_html=True
    )

    # 1) Build the pros/cons table
    xgb_pros_cons_df = pd.DataFrame({
        "Strengths": [
            "⚡  Most Powerful",
            "🔧  Self Correcting & Tuning",
            "☁️  Handles Missing Values Natively"
        ],
        "Weaknesses": [
            "👓  Least Interpretable",
            "⏳  Longer Training Ttimes",
            "🛠️  Harder to Optimize"
        ]
    })
    # xgb_pros_cons_df.index = [''] * len(xgb_pros_cons_df)
    xgb_pros_cons_df.index = [1, 2, 3]
    # 2) Display it at the top
    # st.table(xgb_pros_cons_df)            # static table :contentReference[oaicite:12]{index=12}
    st.dataframe(
        xgb_pros_cons_df,
        column_config={
            "Strengths": st.column_config.TextColumn(width="medium"),
            "Weaknesses": st.column_config.TextColumn(width="medium"),
        },
        hide_index=True,
        use_container_width=True
    )
    # st.dataframe(pros_cons_df, use_container_width=True)  # interactive alternative :contentReference[oaicite:13]{index=13}
    # st.markdown("<br><br>", unsafe_allow_html=True)
    st.markdown("---")
    st.subheader("Fill in Your Client's Values:")
    # st.markdown(
    #     "<h3 style='color:#FFC107;'>Fill in Your Customer's Values:</h3>",
    #     unsafe_allow_html=True
    # )


    # Get the current date
    day_of_year = datetime.datetime.now().timetuple().tm_yday

    # Mapping dictionary for Yes/No to 1/0
    yes_no_mapping = {"Yes": 1, "No": 0}

    # 3) Take user input
    # IF Regular XGB Model
    # input handling
    # housing = yes_no_mapping[st.selectbox("Do you have any housing loans?", ["No", "Yes"], index=0, key=20)]
    # loan = yes_no_mapping[st.selectbox("Do you have any personal loans?", ["No", "Yes"], index=0, key=21)]
    # duration = st.slider("Duration of your last campaign (in minutes)?", min_value=0, max_value=300, value=15, key=22)
    # pdays = st.slider("How many days ago was your last contacted by us?", min_value=0, max_value=1000, value=5, key=23)
    # poutcome = st.selectbox("Outcome of your last campaign?", ["Unknown", "Failure", "Success"], index=0, key=24)  # Default value is 0
    # marital_married = yes_no_mapping[st.selectbox("Are you married?", ["No", "Yes"], index=0, key=25)]
    # job_blue_collar = yes_no_mapping[st.selectbox("Do you work as a blue collar job?", ["No", "Yes"], index=0, key=26)]
    # job_housemaid = yes_no_mapping[st.selectbox("Do you work as a housemaid?", ["No", "Yes"], index=0, key=27)]
    # days_in_year = st.slider("What is the number of Days in a year you are currently at?", min_value=0, max_value=365, value=day_of_year, key=28)

    housing = yes_no_mapping[st.selectbox("Does your client have any housing loans?", ["No", "Yes"], index=0, key=20)]
    loan = yes_no_mapping[st.selectbox("Do your client have any personal loans?", ["No", "Yes"], index=0, key=21)]
    duration = st.slider("How long was your client contacted (in minutes)?", min_value=0, max_value=300, value=15, key=22)
    poutcome = st.selectbox("What was the outcome of your client was previosuly approached?", ["Unknown", "Failure", "Success"], index=0, key=23)  # Default value is 0
    contact_cellular = yes_no_mapping[st.selectbox("Do you contact your client based on his/her cellphone?", ["No", "Yes"], index=0, key=24)]
    # marital_single = yes_no_mapping[st.selectbox("Is your client single?", ["No", "Yes"], index=0, key=25)]
    # marital_married = yes_no_mapping[st.selectbox("Is your client married?", ["No", "Yes"], index=0, key=25)]
    # marital_divorced = yes_no_mapping[st.selectbox("Is your client divorced?", ["No", "Yes"], index=0, key=25)]

    marital_overall = st.selectbox("What is your client's marital status?", ["Single", "Married", "Divorced"], index=0, key=25)  # Default value is 0
    job_overall = st.selectbox("What is your client's job?", ["Unknown", "Blue Collar", "Management", "Services", "Technician"], index=0, key=26)  # Default value is 0

    # job_blue_collar = yes_no_mapping[st.selectbox("Does your client have a blue collar job?", ["No", "Yes"], index=0, key=26)]
    # job_management = yes_no_mapping[st.selectbox("Does your client have a management job?", ["No", "Yes"], index=0, key=26)]
    # job_services = yes_no_mapping[st.selectbox("Does your client work in ?", ["No", "Yes"], index=0, key=26)]
    # job_technician = yes_no_mapping[st.selectbox("Does your client have a blue collar job?", ["No", "Yes"], index=0, key=26)]

    
    # 4) input handling
    housing = float(housing)
    loan = float(loan)
    duration = float(duration) * 60
    # days_in_year = float(days_in_year)
    if contact_cellular=="No":
        contact_cellular=0
    else:
        contact_cellular=1

    # for poutcome
    if poutcome=="Failure":
        poutcome=0
    elif poutcome=="Unknown":
        poutcome=0.5
    elif poutcome=="Success":
        poutcome=1

    single=0
    married=0
    divorced=0

    # marital
    if marital_overall== "Single":
        single=1
    elif marital_overall =="Married":
        married=1
    elif marital_overall =="Divorced":
        divorced=1

    # "Unknown", "Blue Collar", "Management", "Services", "Technician"
    blue_collar=0
    management=0
    services=0
    technician=0

    if job_overall== "Blue Collar":
        blue_collar=1
    elif job_overall =="Management":
        management=1
    elif job_overall =="Services":
        services=1
    elif job_overall =="Technician":
        technician=1

    # 5) return output in dict
    # IF Original XGB Model
    # return input values in dict
    # return {
    #     "housing":          housing,
    #     "loan":             loan,
    #     "duration":         duration,
    #     "pdays":            float(pdays),
    #     "poutcome":         poutcome,
    #     "marital_married":  marital_married,
    #     "job_blue_collar":  job_blue_collar,
    #     "job_housemaid":    job_housemaid,
    #     "days_in_year":     days_in_year,
    # }


    return {
        "housing":          housing,
        "loan":             loan,
        "duration":         duration,
        "poutcome":         poutcome,
        "contact_cellular": contact_cellular,
        "marital_single":   single,
        "marital_married":  married,
        "marital_divorced": divorced,
        "job_blue_collar":  blue_collar,
        "job_management":   management,
        "job_services":     services,
        "job_technician":   technician
    }

# Function that displays success/failed prediction
def display_prediction(prediction):
    col1, col2 = st.columns([0.1, 0.9])
    # print(prediction) # for testing

    # Predict success case:
    if prediction[0] == 1:
        with col1:
            st.image("App_Visualizations/Result_Icons/success_icon.png", width=50)  # Use an icon for success
        with col2:
            # st.write("### The Marketing Campaign will Succeed!")
            st.markdown(
                f"""<h3>Your Client Will 
                <span style="color:#22C55E"><u>Subscribe</u></span> to the Term Deposit Product.</h3>""",
                unsafe_allow_html=True
            )
    # Predict failure case:
    elif prediction[0] == 0:
        with col1:
            st.image("App_Visualizations/Result_Icons/failure_icon.png", width=50)  # Use an icon for failure
        with col2:
            # st.write("### The Marketing Campaign will Fail.")
            st.markdown(
                f"""<h3>Your Client Will 
                <span style="color:#EF4444"><u>Not Subscribe</u></span> to the Term Deposit Product.</h3>""",
                unsafe_allow_html=True
            )


# --- PAGE FUNCTIONS ---
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------

# Function convert image to base64
def _img_to_base64(path):
    """Read a local image file and return a base64 data-URI string."""
    with open(path, "rb") as f:
        data = f.read()
    b64 = base64.b64encode(data).decode()
    return f"data:image/png;base64,{b64}"

# function that builds the homepage layout
def home_page(models, data, raw_data):

    # CSS
    st.markdown(
        """
        <style>
        /* Match card titles with sidebar nav‐link font: */
        .card-title {
        font-size: 1.25rem !important;  /* e.g. 18px if nav‐link is 18px */
        margin: 0 0 8px 0;
        color: #fff !important;
        }
        /* Match card desc with sidebar nav‐link font: */
        .card-desc {
        font-size: 1rem !important;      /* 16px if nav‐link is 16px */
        margin: 0 0 12px 0;
        color: #fff !important;
        }
        /* 1) Give the whole card a gray frame instead of white */
        .card-container {
            border: 1px solid #777 !important;    /* darker grey */
            border-radius: 8px;
            padding:16px 16px 32px 16px !important;  /* top right bottom left */
            background-color: transparent;
        }

        /* 1 & 3) Push text in from the left edge, and keep font sizes you already set */
        .card-content {
            padding-left: 12px;
        }

        /* 3) Dim the images slightly */
        .card-img img {
            filter: brightness(0.85);             /* 75% brightness */
        }
        </style>
        """,
        unsafe_allow_html=True,
    )

    st.markdown(
        """
        <style>
        /* move the whole page content up */
        .block-container { padding-top: 0rem; } /* tweak 0–1rem to taste */

        /* give page titles a predictable gap from whatever is above them */
        h1.page-title { margin-top: -3.5rem; margin-bottom: 0rem; font-size: 7.5em }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
        """
        <h1 class='page-title'>
            Welcome to 
            <span style="color:#FFC107;">Fin</span><span style="color:#00BCD4;">ML</span> Studio!
        </h1>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<br>", unsafe_allow_html=True)

    # st.markdown("---")



    # st.markdown('<p class="card-desc">This app uses data science and machine learning methodologies to improve the performance of a bank financial product, especially in the areas of: </p>', unsafe_allow_html=True)
    st.markdown(
        """
        <p class="card-desc">
            This app uses 
            <span style="color:#00BCD4;">data science</span> 
            & 
            <span style="color:#00BCD4;">machine learning</span> 
            methodologies to improve the performance of a financial product <span style="color:#00BCD4;">(term deposit)</span>, especially in the areas of:
        </p>
        """,
        unsafe_allow_html=True
    )
    st.markdown(
        """
        <ol class="card-desc" style="margin-left:1.5rem; padding-left:0;">
        <li>Product Analytics</li>
        <li>Demand Forecasting</li>
        <li>Business Intelligence</li>
        </ol>
        """,
        unsafe_allow_html=True
    )

    # st.markdown("<br>", unsafe_allow_html=True)
    st.markdown('<p class="card-desc">Come try the functionalities below!</p>', unsafe_allow_html=True)
    st.markdown("<br>", unsafe_allow_html=True)
    # for local deployment

    cloud_deployment=""
    # cloud_deployment="Code/Model_Deployment/"

    cards = [
        (
            "Deposit Prediction",
            "Predict client subscriptions with our fine-tuned AI/ML models!",
            "Deposit Prediction",
            f"{cloud_deployment}App_Visualizations/Homepage_Icons/predictive-icon.jpg"
        ),
        (
            "Interactive Dashboard",
            "Discover hidden trends & insights via exploratory data analysis (EDA)!",
            "Interactive Dashboard",
            f"{cloud_deployment}App_Visualizations/Homepage_Icons/dashboard-icon2.jpg"
        ),
        (
            "Customer Segmentation",
            "Assign customers into groups using our clustering algorithm!",
            "Customer Segmentation",
            f"{cloud_deployment}App_Visualizations/Homepage_Icons/cluster-analysis-icon.jpg"
        ),
        (
            "Data Overview & Export",
            "Download & use our original / cleaned data prepared for you!",
            "Data Overview & Export",
            f"{cloud_deployment}App_Visualizations/Homepage_Icons/export-data-icon.jpg"
        ),
    ]

    # displays 4 boxes for the app's functionalities
    cols = st.columns(4, gap="large")

    st.markdown("""
        <style>
        :root{ --card-blue:#FFC107; }        /* pick your blue */
        .card-title{
        color: var(--card-blue) !important; 
        font-weight: 700;
        margin: 0 0 6px 0;
        }
        /* optional: lighter on hover */
        .card-container:hover .card-title{ color:#60A5FA; } 
        /* Make all Streamlit buttons full-width */
        div.stButton > button {
            width: 100% !important;    /* stretch to fill the container */
            display: block;
            text-align: center;        /* keep text centered */
        }
        </style>
        """, unsafe_allow_html=True)

    # for each box, show image, title, and button
    for col, (title, desc, page_key, img_path) in zip(cols, cards):
        with col:
            uri = _img_to_base64(img_path)
            card_html = f"""
            <div class="card-container">
            <div class="card-content">
                <h4 class="card-title">{title}</h4>
                <p class="card-desc">{desc}</p>
            </div>
            <div class="card-img" style="text-align:center; margin-top:16px;">
                <img src="{uri}"
                    style="max-width:100%; max-height:120px; object-fit:contain;"/>
            </div>
            </div>
            """
            st.markdown(card_html, unsafe_allow_html=True)
            
            # st.markdown("<br>", unsafe_allow_html=True)

            if col.button("Try it out!", key=f"btn_{page_key}"):
                st.session_state.page = page_key
                st.rerun()


    st.markdown("<br><br>", unsafe_allow_html=True)

# Prediction Page
def prediction_page(models, data):
    st.markdown(
        """
        <style>
        /* move the whole page content up */
        .block-container { padding-top: 0rem; } /* tweak 0–1rem to taste */

        /* give page titles a predictable gap from whatever is above them */
        h1.page-title { margin-top: -4rem; margin-bottom: -3rem; }
        </style>
        """,
        unsafe_allow_html=True
    ) 

    st.markdown(
        "<h1 class='page-title' style='color:#FFC107; font-size:7.5rem;'>Deposit Prediction </h1>",
        # "<span style='color:white;'> - With our Tuned AI/ML Model</span></h1>",
        unsafe_allow_html=True
    )
    # st.markdown("<hr>",unsafe_allow_html=True) 

    model_names = list(models.keys())

    # --- set a default active model once ---
    if "active_model" not in st.session_state:
        st.session_state.active_model = model_names[0]   # pick the first by default
    if "prev_active_model" not in st.session_state:
        st.session_state.prev_active_model = st.session_state.active_model

    # --- selector (segmented control with radio fallback) ---
    try:
        active = st.segmented_control(" ", model_names, key="active_model")
    except Exception:
        active = st.radio("Model", model_names, index=model_names.index(st.session_state.active_model), key="active_model")

    # --- clear per-model predictions when switching models ---
    if st.session_state.active_model != st.session_state.prev_active_model:
        for n in model_names:
            st.session_state[f"{n}_has_pred"] = False
            st.session_state[f"{n}_pred"] = None
            st.session_state[f"{n}_input_snapshot"] = None
        st.session_state.prev_active_model = st.session_state.active_model

    # now you can safely use:
    name  = st.session_state.active_model
    model = models[name]

    # --- collect inputs (your existing per-model forms) ---
    if name == 'Decision Tree':
        inputs_dict = user_input_form_decision_tree()
    elif name == 'Random Forest':
        inputs_dict = user_input_form_random_forest()
    else:
        inputs_dict = user_input_form_xgboost()

    inputs = pd.DataFrame([inputs_dict])
    feature_names = tuple(inputs_dict.keys()) 

    # per-model keys
    run_key  = f"{name}_has_pred"
    pred_key = f"{name}_pred"
    snap_key = f"{name}_input_snapshot"

    st.session_state.setdefault(run_key, False)
    st.session_state.setdefault(pred_key, None)
    st.session_state.setdefault(snap_key, None)

    # auto-cancel when inputs change
    snapshot_now = tuple((k, inputs_dict[k]) for k in sorted(inputs_dict))
    if st.session_state[snap_key] is not None and st.session_state[snap_key] != snapshot_now:
        st.session_state[run_key]  = False
        st.session_state[pred_key] = None

    # st.markdown("""
    # <style>
    # div.stButton > button {
    #     width: 100% !important;   /* make button fill its container */
    # }
    # </style>
    # """, unsafe_allow_html=True)

    # buttons
    c1, c2 = st.columns([1, 1])
    with c1:
        do_predict = st.button(f"Predict with {name}  🔍", key=f"predict_{name}", use_container_width=True)
    with c2:
        do_cancel  = st.button("Try Another Prediction 🔄", key=f"cancel_{name}", use_container_width=True)

    if do_cancel:
        st.session_state[run_key]  = False
        st.session_state[pred_key] = None

    if do_predict:
        st.session_state[pred_key] = make_prediction(model, inputs)
        st.session_state[snap_key] = snapshot_now
        st.session_state[run_key]  = True

    if st.session_state[run_key] and st.session_state[pred_key] is not None:
        st.markdown("---", unsafe_allow_html=True)
        display_prediction(st.session_state[pred_key])
        shap_exp, lime_exp = load_explainers(model, data, feature_names)
        st.markdown("---")
        with st.expander("🧠 Check how the model makes its decision through explainable AI (XAI)", expanded=True):
            show_explanations(model, inputs, shap_exp, lime_exp)


# Function displaying the interactive dashboard page
def dashboard_page(data):
    # ─── Inject CSS for the card shape ─────────────────────────────────
    st.markdown(
        """
        <style>
        .kpi-card {
        background-color: rgba(255, 255, 255, 0.5);
        border-radius: 12px;
        padding: 0.25rem 0;       /* less vertical padding */
        margin-bottom: -2rem;         /* remove extra gap below */
        box-shadow: 0 4px 8px rgba(0, 0, 0, 0.15);
        text-align: center;
        }
        .kpi-card [data-testid="stPlotlyChart"] {
        padding: 0 !important;
        margin: 0 !important;
        }
        /* If you use Plotly inside the card, make its paper transparent */
        .kpi-card .js-plotly-plot .plotly {
          background-color: transparent !important;
        }
        .kpi-card section[data-testid="stPlotlyChart"] {
        margin: 0 auto !important;
        }

        /* 1) Reset the surrounding Streamlit container */
        section[data-testid="stPlotlyChart"] {
        background: none !important;
        padding: 0 !important;
        box-shadow: none !important;
        margin-bottom: 1rem !important;
        }

        /* 2) Apply a semi-transparent white “card” just behind the plot area */
        section[data-testid="stPlotlyChart"] .js-plotly-plot .plotly {
        background-color: rgba(255,255,255,0.5) !important;
        border-radius: 8px !important;
        padding: 1rem !important;
        }

        [data-testid="stMarkdownContainer"] h4 {
        background-color: #393939;
        color: white;
        display: inline-block;
        padding: 0.25rem 0.5rem;
        border-radius: 6px;
        margin-bottom: 1rem;
        margin-top: 0rem;
        }

        /* 2) Style each of your 2×2 boxes */
        .box-card {
        background: #ffffff;
        border-radius: 8px;
        padding: 1rem;
        box-shadow: 0 4px 12px rgba(0,0,0,0.1);
        margin-bottom: 1rem;
        }
        .rec-card {
        background-color: #393939;
        color: white;
        padding: 1rem;
        border-radius: 10px;
        margin-bottom: -10.5rem;
        margin-top: -0.5rem;
        }
        .rec-card h2, .rec-card li {
        color: white; 
        margin: 0.5rem 0;
        }
        .rec-card ul {
        padding-left: 1.2rem;
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # more CSS styling
    st.markdown(
        """
        <style>
        /* lift page content */
        .block-container { padding-top: 0rem; }

        /* align rows that contain page titles + controls */
        .page-title-row {
            display: flex;
            align-items: flex-end;   /* aligns the selectbox baseline with title bottom */
            margin-top: -10rem;
            margin-bottom: 0rem;
        }

        h1.page-title {
            margin: 0;   /* reset margins so row spacing takes effect */
            color: #9966FF;
            margin-top:-6.5rem
        }
        div[data-baseweb="select"] {
            margin-top: -2.5rem;   /* tweak until it aligns */
        }
        div[data-baseweb="select"] > div {
            font-size: 1.0rem !important;   /* increase the selected option text */
        }

        /* Target the dropdown menu items */
        ul[role="listbox"] li {
            font-size: 1.0rem !important;   /* increase the options in the dropdown */
        }
        .persona-label {
            margin: 0; 
            padding: 0rem;
            font-size: 0.9rem;        /* tweak size */
            font-weight: 600;
            color: white;             /* or #FFC107 if you want accent */
            line-height: 1;           /* tight line spacing */
            margin-bottom: -4.25rem;  /* 👈 lift it closer to the selectbox */
            transform: translateY(-3rem);
        }
        </style>
        """,
        unsafe_allow_html=True
    )

    # display introductuin
    # st.header("Interactive Dashboard - Choose Your Persona & Explore Key Metrics and Visualizations")
    
    with st.container():
        st.markdown("<div class='page-title-row'>", unsafe_allow_html=True)
        col1, col2 = st.columns([6, 1])  # 3:1 width ratio

        with col1:
            st.markdown(
                "<h1 class='page-title' style='color:#FFC107; font-size:7.5rem;'>Interactive Dashboard</h1>",
                # "<span style='color:white;'> - Explore Key Metrics & Visualizations</span></h1>",
                unsafe_allow_html=True
            )
        with col2:
            st.markdown(
                "<p class='persona-label'>User Persona:</p>",
                unsafe_allow_html=True
            )
            persona = st.selectbox(
                "User Persona:",  # empty label so it doesn't show above
                ["Sales Representative", "Marketing Manager"],
                label_visibility="collapsed"   # 👈 hides the label
            )
        st.markdown("</div>", unsafe_allow_html=True)

    # st.markdown(
    #     "<h1 style='color:#9966FF;'>Interactive Dashboard - Explore Key Metrics & Visualizations</h1>",
    #     unsafe_allow_html=True
    # )
    # st.markdown("---")
    # # st.subheader("Choose Your Persona & Explore Key Metrics and Visualizations:")
    # persona = st.selectbox("User Persona:", ["Salesperson", "Marketing Manager"])
    # st.markdown("---")
    # st.markdown('<br>', unsafe_allow_html=True)
    # st.markdown(
    #     """
    #     <div style="border-top: 1px solid white; margin: 0; padding: 0;"></div>
    #     """,
    #     unsafe_allow_html=True
    # )
    # st.markdown('<br>', unsafe_allow_html=True)

    # sub function: visually shows KPIs
    def kpi_indicator(label, value, suffix="", color="#000000"):
        # If suffix is "$", switch to prefix
        if suffix == "$":
            prefix = "$ "
            suffix = ""
        else:
            prefix = ""

        fig = go.Figure(go.Indicator(
            mode="number",
            value=value,
            title={
                "text": label,
                "font": {"size": 20, "color": "#FFFFFF"}
            },
            number={
                "font": {"size": 48, "color": color},
                "prefix": prefix,
                "suffix": suffix
            },
            domain={"x": [0.3, 0.6], "y": [0.3, 0.4]}
        ))

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=90
        )
        return fig

    # show most important factor as a KPI card
    def kpi_indicator_text(label, text_value, color="#000000"):
        fig = go.Figure()

        # Add title
        fig.add_annotation(
            text=f"<span style='color:white;font-size:20px'>{label}</span>",
            x=0.5, y=1.075, xref="paper", yref="paper",
            showarrow=False
        )

        # Add text value (centered)
        fig.add_annotation(
            text=f"<span style='color:{color};font-size:48px'>{text_value}</span>",
            x=0.475, y=0.75, xref="paper", yref="paper",
            showarrow=False
        )

        fig.update_layout(
            paper_bgcolor="rgba(0,0,0,0)",
            plot_bgcolor="rgba(0,0,0,0)",
            margin=dict(l=0, r=0, t=0, b=0),
            height=90,
            xaxis=dict(visible=False),
            yaxis=dict(visible=False)
        )
        return fig

    # ─── Layout: 5 equal columns ────────────────────────────────────────

    k1, k2, k3, k4, k5 = st.columns(5, gap="small")

    # CARD_START = '<div class="kpi-card" style="background:#fff;border-radius:12px;padding:1rem;box-shadow:0 4px 12px rgba(0,0,0,0.15);margin:1rem 0;">'
    # CARD_END   = '</div>'

    # Col 1: persona selector + metric
    with k1:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        # persona = st.selectbox("User Persona:", ["Salesperson", "Marketing Manager"])
        # st.metric("Most Important Factor:", "Call Duration")
        # st.markdown("""
        #     <style>
        #     [data-testid="stMetricValue"] {
        #         color: #f83464;
        #         font-size: 2.5rem;      /* increase value font size */
        #     }
        #     """, unsafe_allow_html=True)
        # st.metric("Most Important Factor:", "Call Duration")
        fig = kpi_indicator_text("Most Important Factor:", "Call Duration", color="#f83464")
        st.plotly_chart(fig, use_container_width=True)

        st.markdown('</div>', unsafe_allow_html=True)

    # Col 2: Conversion Rate
    with k2: 
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        fig = kpi_indicator("Conversion Rate", round(data['y'].mean()*100,2), "%", color="#FFC107") #e28743
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Col 3: Persona-specific KPI
    with k3:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        if persona == 'Marketing Manager':
            fig = kpi_indicator(
                "First-Time Conversion Rate",
                round(data[data['previous']==0]['y'].mean()*100,2),
                "%", color="#FFC107"
            )#eab676
        else:
            fig = kpi_indicator(
                "Avg. Success Duation (mins)",
                round(data[data['y']==1]['duration'].mean()/60,2),
                "", color="#FFC107"
            )#76b5c5
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Col 4: First Contact %
    with k4:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        fig = kpi_indicator("First Contact Proportion", round((data['previous']==0).mean()*100,2), "%", color="#FFC107")#FFB6C1
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # Col 5: Persona-specific KPI
    with k5:
        st.markdown('<div class="kpi-card">', unsafe_allow_html=True)
        if persona == 'Marketing Manager':
            fig = kpi_indicator(
                "Avg. Balance for Success",
                round(data[data['y']==1]['balance'].mean(),2), "$", color="#FFC107"
            )#abdbe3
        else:
            fig = kpi_indicator(
                "Avg. Past Success Rate",
                round(data[data['y']==1]['poutcome'].mean()*100,2),
                "%", color="#FFC107"
            )#1e81b0
        st.plotly_chart(fig, use_container_width=True)
        st.markdown('</div>', unsafe_allow_html=True)

    # st.markdown("---")
    # st.markdown('<hr>', unsafe_allow_html=True)
    # st.markdown('<br>', unsafe_allow_html=True)
    # st.markdown(
    #     """
    #     <div style="border-top: 1px solid white; margin: 0; padding: 0;"></div>
    #     """,
    #     unsafe_allow_html=True
    # )
    st.markdown(
        "<hr style='margin: -20px 0 0px; border:0; height:2px; background: rgba(255,255,255,1);'>",
        unsafe_allow_html=True
    )
    # st.markdown('<br>', unsafe_allow_html=True)

    # different display outpuet for each persona
    if persona =="Marketing Manager":

        # --- Create our 2×2 grid ---
        row1_col1, row1_col2 = st.columns([2.85, 5], gap="medium")
        row2_col1, row2_col2 = st.columns([2.85, 5], gap="medium")

        # Top-left box: Marketing-based recommendation
        with row1_col1:
            st.markdown("""
                <style>
                .rec-card h3 {
                    color: #00BCD4; /* Teal */
                }
                </style>
                """, unsafe_allow_html=True)

            recommendations_html = """
            <div class="rec-card">
            <h3>Data-Driven Recommendations</h3>
            <ul>
                <li>Target campaigns in <u>March, August, November, and December</u>. </li>
                <li><u>Focus on customers in their 30's</u>, they are the most responsive group.</li>
                <li>Boost conversion by <u>increasing call duration</u>, this is the most impactful method.</li>
                <li><u>Customers without loans</u> are more likely to <u>convert in shorter calls</u>.</li>
                <li>Since <u>most use cellular phones</u>, get their attention quickly and increase engagement time.</li>
                <li><u>Re-engage past subscribers</u>. Over 50% converted again when contacted.</li>
            </ul>
            </div>
            """
            st.markdown(recommendations_html, unsafe_allow_html=True)

        # top-right box: display plots for both Sales and Marketing
        with row1_col2:
            st.markdown('<br>', unsafe_allow_html=True)
            # st.markdown('<div class="box-card">', unsafe_allow_html=True)
            st.markdown(
                "<h3 style='color:#00BCD4;'>Marketing Campaign Trend</h3>",
                unsafe_allow_html=True
            )
            # st.subheader("Marketing Campaign Trend Over Time")
            ts_tab, ms_tab = st.tabs(["Monthly Contacts","Monthly Win Rate"])
            with ts_tab:
                # daily number of success over time plot
                # st.markdown('<div class="box-card">', unsafe_allow_html=True)
                st.altair_chart(monthly_line_altair(data), use_container_width=True)
                # st.markdown('</div>', unsafe_allow_html=True)
                # st.altair_chart(daily_line_altair(data), use_container_width=True)
            with ms_tab:
                # monthly succeess rate
                # st.markdown('<div class="box-card">', unsafe_allow_html=True)
                # st.markdown('</div>', unsafe_allow_html=True)
                st.altair_chart(monthly_success_altair(data), use_container_width=True)
    
        # bottom-left box: works both Sales and Marketing
        with row2_col1:
            st.markdown(
                "<h3 style='color:#00BCD4;'>Proportion of Wins</h3>",
                unsafe_allow_html=True
            )
            # st.subheader("Wins by Channel & Loans")
            contact_tab, loan_tab = st.tabs(["Contact Channel","Loan Ownership"])
            with contact_tab:
                # Plot 3: contact type pie (Plotly)
                contact_fig= contact_channel_pie(data, 325, 325)    
                st.plotly_chart(contact_fig, use_container_width=True)
            with loan_tab:
                # Plot 4: loan Venn (matplotlib)
                venn_fig = plot_loan_venn(data, width_px=300, height_px=200, scale=0.5)
                st.markdown('<div class="venn-container">', unsafe_allow_html=True)
                st.pyplot(venn_fig)  
                st.markdown('</div>', unsafe_allow_html=True)

        # Bottom-right box: distributions & heatmaps (Plots 5, 6 & 7)
        with row2_col2:
            st.markdown(
                "<h3 style='color:#00BCD4;'>Distribution & Heatmaps Over Wins</h3>",
                unsafe_allow_html=True
            )
            # st.subheader("Distributions & Heatmaps Over Wins")
            dist_tab, heat_tab, loan_heat_tab = st.tabs([
                "Age Distribution Over Wins","Age × Duration Heatmap","Loan × Duration Heatmap"
            ])

            with dist_tab:
                # Plot 5: age KDE (Altair)
                st.altair_chart(kde_age_distribution(data), use_container_width=True)

            with heat_tab:
                # Plot 6: conversion heatmap by age & duration (Altair)
                st.altair_chart(plot_age_duration_heatmap(data), use_container_width=True)

            with loan_heat_tab:
                # Plot 7: conversion heatmap by loan type & duration (Altair)
                st.altair_chart(plot_loans_duration_heatmap(data), use_container_width=True)

    # if persona is salesperson
    elif persona =="Sales Representative":

        # --- Create our 2×2 grid ---
        row1_col1, row2_col2, row2_col1 = st.columns([1.15,1,1], gap="medium")
        # row2_col2 = st.columns(1, gap="medium")

        # Top-left box: Sales-based recommendation
        with row1_col1:
            st.markdown("""
                <style>
                .rec-card h3 {
                    color: #00BCD4; /* Teal */
                }
                </style>
                """, unsafe_allow_html=True)

            recommendations_html = """
                <div class="rec-card">
                <h3>Data-Driven Recommendations</h3>
                <ul>
                    <li><u>Prior subscribers are likely to convert again</u>, prioritize follow-ups with them.</li>
                    <li>Focus outreach during <u>summer / Christmas</u> when conversion rates are the highest.</li>
                    <li>Don’t overlook clients with existing loans —> <u>40% of them still convert</u>.</li>
                    <li><u>Duration is crucial for success</u> —> try to aim for longer conversations (<u> > 9 minutes</u>).</li>
                    <li><u>Most clients use mobile phones</u> — grab their attention quickly & make every second count.</li>
                    <li>Most days have made <u>fewer than 50 calls</u>. Increase outreach volume to improve campaign success.</li>
                </ul>
                </div>
                """

            st.markdown(recommendations_html, unsafe_allow_html=True)
    
        # Bottom-left box: display plots for both Sales and Marketing
        with row2_col1:
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(
                "<h3 style='color:#00BCD4;'>Proportion of Wins</h3>",
                unsafe_allow_html=True
            )
            # st.subheader("Campaign Outcome by Channel & Loans")
            contact_tab, loan_tab = st.tabs(["Contact Channel","Loan Ownership"])
            with contact_tab:
                # Plot 3: contact type pie (Plotly)
                contact_fig= contact_channel_pie(data, 325, 325)    
                st.plotly_chart(contact_fig, use_container_width=True)
            with loan_tab:
                # Plot 4: loan Venn (matplotlib)
                venn_fig = plot_loan_venn(data, width_px=300, height_px=225, scale=0.4) 
                st.markdown('<div class="venn-container">', unsafe_allow_html=True)
                st.pyplot(venn_fig)  
                st.markdown('</div>', unsafe_allow_html=True)

        # Bottom-right box: displays plot for both Sales and Marketing
        # Box (2,2): distributions & heatmaps (Plots 5, 6 & 7)
        with row2_col2:
        # with st.columns(1):
            st.markdown('<br>', unsafe_allow_html=True)
            st.markdown(
                "<h3 style='color:#00BCD4;'>Campaign Outcome Based on Past</h3>",
                unsafe_allow_html=True
            )
            # st.subheader("Campaign Outcome Based on Past Campaign's Outcomes")
            no_past_tab, past_tab, inconclusive_tab= st.tabs([
                "No Past","Successful Past", "Inconclusive Past"
            ])

            width=325
            height=325

            with no_past_tab:
                # Plot 5: age donut for past failed scenarios
                st.plotly_chart(previous_donut(df=data, width=width, height=height, filter_val=0), use_container_width=True)

            with past_tab:
                # Plot 6: age donut for past success scenarios
                st.plotly_chart(previous_donut(df=data, width=width, height=height, filter_val=1), use_container_width=True)

            with inconclusive_tab:
                # Plot 7: age donut for past inconclusive scenarios
                st.plotly_chart(previous_donut(df=data, width=width, height=height, filter_val=0.5), use_container_width=True)

        # Top-right box: display plots for both Sales and Marketing
        # with row1_col2:
        # st.markdown('<br>', unsafe_allow_html=True)
        # st.markdown('<div class="box-card">', unsafe_allow_html=True)
        st.markdown(
            "<h3 style='color:#00BCD4;'>Marketing Campaign Trend</h3>",
            unsafe_allow_html=True
        )
        # st.subheader("Marketing Campaign Trend Over Time")
        ts_tab, ms_tab = st.tabs(["Daily Contacts","Monthly Win Rate"])
        with ts_tab:
            # daily number of success over time plot
            st.altair_chart(daily_line_altair(data), use_container_width=True)
        with ms_tab:
            # monthly succeess rate
            st.altair_chart(monthly_success_altair(data), use_container_width=True)
        # st.markdown('</div>', unsafe_allow_html=True)

# Displays clustering page
def clustering_page(data):
    # CSS on top of each page 
    st.markdown(
        """
        <style>
        /* move the whole page content up */
        .block-container { padding-top: 0rem; } /* tweak 0–1rem to taste */

        /* give page titles a predictable gap from whatever is above them */
        h1.page-title { margin-top: -4rem; margin-bottom: 0rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    st.markdown(
                "<h1 class='page-title' style='color:#FFC107; font-size:7.5rem;'>Customer Segmentation</h1>",
                # "<span style='color:white;'> - Group the Customers Using AI</span></h1>",
                unsafe_allow_html=True
            )
    # st.markdown("<hr>",unsafe_allow_html=True)

    # ─── 0) Setup expander state & callback ────────────────────────────
    # if "cluster_expanded" not in st.session_state: 
    #     st.session_state.cluster_expanded = False

    def _expand_cluster():
        st.session_state.cluster_expanded = True

    
    # let users to choose the features they want to use for clustering
    # st.subheader('Feature Selection')

    # 1) Define your groups here:
    FEATURE_GROUPS = {
        "Personal Information": ["age", "education", "balance", "contact_telephone"],
        "Loans": ["housing", "loan", "default"],
        "Campaign Metrics": ["day", "month", "duration", "campaign", "pdays", "previous", "poutcome", "days_in_year"]
    }

    # --- new wording (DESCS/EXS), plus emojis ---
    DESCS = {
        "Personal Information": "Demographics & account context: age, education level, balance, contact method.",
        "Loans": "Existing credit obligations: housing/mortgage, personal loan, default history.",
        "Campaign Metrics": "Campaign data: contact time/duration, # of contacts, prior contacts & outcome, etc.",
    }
    EXS = {
        "Personal Information": "e.g., age=42, balance=10000.00",
        "Loans": "e.g., housing=1, loan=0",
        "Campaign Metrics": "e.g., duration=900, previous=2",
    }
    EMOJI = {
        "Personal Information": "👤",
        "Loans": "🏦",
        "Campaign Metrics": "📣",
    }

    # --- unchanged multiselect behavior ---
    chosen = st.multiselect("Feature Selection - Please Select Your Feature Groups:", list(FEATURE_GROUPS.keys()))
    if not chosen:
        st.warning("Select at least one group to proceed.")
        return  # ⬅️ keep the original early-return behavior

    # --- same columns as old code, but append emoji to Feature Group ---
    sel_rows = [
        {"Feature Group": f"{g} {EMOJI.get(g, '')}".strip(), "Description": DESCS[g], "Examples": EXS[g]}
        for g in chosen
    ]
    df = pd.DataFrame(sel_rows)

    # --- show as dataframe (with graceful fallback) ---
    try:
        st.dataframe(
            df,
            hide_index=True,
            use_container_width=True,
            column_config={
                "Feature Group": st.column_config.TextColumn(width="medium"),
                "Description":   st.column_config.TextColumn(width="large"),
                "Examples":      st.column_config.TextColumn(width="medium"),
            },
        )
    except Exception:
        # Older Streamlit: fall back to table and hide index if possible
        try:
            st.table(df.style.hide(axis="index"))
        except Exception:
            st.table(df)


    cols = []
    for g in chosen:
        cols += FEATURE_GROUPS[g]
    cols = list(dict.fromkeys(cols))
    # st.write(f"{len(cols)} features selected.")

    
    st.markdown(
        f"""<p>{'<span style="color:#FFC107;"><u>'+str(len(cols))+'</u></span>'} features are selected.</p>""",
        unsafe_allow_html=True
    )
    # if len(cols) < 2:
    #     st.error("Pick at least two features.")
    #     return

    # invalidate on change
    if st.session_state.get("last_cols") != cols:
        for k in ("labels", "Xs", "rf_multi", "scaler"): st.session_state.pop(k, None)
        st.session_state["last_cols"] = cols
    
    st.markdown("<br>",unsafe_allow_html=True)

    # cluster when clicked
    # Note: we attach on_click so that after clicking, cluster_expanded=True
    run = st.button(
        "Run HDBSCAN Clustering Algorithm to Categorize Clients",
        on_click=_expand_cluster,
        use_container_width=True
    )

    if run:
        with st.spinner("Clustering Customers…"):
            scaler, Xs = get_scaled(data, cols)
            labels = get_labels(Xs)
            rf_multi = get_multi_surrogate(Xs, labels, 500)
            st.session_state.update({
                "labels":   labels,
                "Xs":       Xs,
                "rf_multi": rf_multi,
                "scaler":   scaler,
            })

    if "labels" not in st.session_state:
        return  # nothing to show yet

    # display results
    if "labels" in st.session_state:
        scaler   = st.session_state.scaler
        labels   = st.session_state.labels
        rf_multi = st.session_state.rf_multi

        clustered = data.assign(Cluster=labels)


        unique = np.unique(labels)

        # compute count excluding outliers
        has_outliers = -1 in unique
        num_clusters = len(unique) - (1 if has_outliers else 0)

        st.markdown("<hr>", unsafe_allow_html=True)
        st.markdown("<br>", unsafe_allow_html=True)

        # build your message
        if has_outliers:
            msg = f"There are <span style='color: #FFC107;'>{num_clusters} Customer Groups</span> (and <span style='color:#FFC107;'>1 Outliers Group</span>)."
        else:
            msg = f"There are <span style='color:#FFC107;'>{num_clusters} Customer Groups</span>."

        st.markdown(f"<h2>{msg}</h2>", unsafe_allow_html=True)
        # if has_outliers:
        #     msg = f"There are {num_clusters} customer groups (plus an outliers group)."
        # else:
        #     msg = f"There are {num_clusters} customer groups."
        # st.header(msg)

        st.markdown("<hr>", unsafe_allow_html=True)
        # st.markdown("<br></br>", unsafe_allow_html=True)

        st.markdown(
            "<h3 class='page-title' style='color:#FFC107;'>Visualizations:</h3>",
            unsafe_allow_html=True
        )
        with st.expander(" ", expanded=True):


            clustered = data.assign(Cluster=st.session_state["labels"])
            selected_cols=cols
            
            # ─────────────────────────────────────────────────────────────────
            # 1) show feature descriptions & examples
            # st.markdown("---")
            # show_example_table(clustered, selected_cols)
            
            # 2) violin plots
            # st.markdown("---")
            top_features = plot_violin_top_features_raw(clustered, selected_cols, top_n=3)

            testcol1, testcol2 = st.columns(2, gap="medium")

            with testcol1:
                # 3) 3D scatter
                plot_3d_clusters_raw(clustered, selected_cols, top_features)

            with testcol2:
                # note: if you trained one RF per cluster in rf_dict, pick rf_dict[cl] in the plotting fn
                plot_tree_feature_importance(
                    clustered,
                    st.session_state["scaler"].transform(clustered[selected_cols]),
                    selected_cols
                )
            
            
            
            # 4) cluster means table
            st.markdown("---")
            show_cluster_feature_means_raw(clustered, selected_cols)
            
            # 5) tree-based importance
            # st.markdown("---")
            # # note: if you trained one RF per cluster in rf_dict, pick rf_dict[cl] in the plotting fn
            # plot_tree_feature_importance(
            #     clustered,
            #     st.session_state["scaler"].transform(clustered[selected_cols]),
            #     selected_cols
            # )
            
            # 6) LIME explainer in an expander
            # st.markdown("---")
            # st.subheader("Try enter a new customer to see which group does he/she belong!")
        st.markdown(
            "<h3 class='page-title' style='color:#FFC107;'>Try Enter a New Customer to See Which Group he/she Belongs!</h3>",
            unsafe_allow_html=True
        )
        with st.expander(" ", expanded=True):
            # st.markdown("---")
            show_example_table(clustered, selected_cols, special=1)

            show_lime_explanation_custom(
                st.session_state["rf_multi"],
                st.session_state["scaler"],
                clustered,
                cols,
                top_n=5
            )
                
            # Shoe LIME Function for XAI
            # show_lime_explanation_custom(
            #     # if you kept my rf_dict, you’ll need to pluck the right cluster’s RF:
            #     # st.session_state["rf_dict"][1],
            #     st.session_state["rf_multi"],
            #     st.session_state["scaler"],
            #     clustered,
            #     selected_cols,
            #     top_n=5
            # )


# Showing the data overview & export page
def overview_page(data, preprocessed):
    # Inject CSS on top
    st.markdown(
        """
        <style>
        /* move the whole page content up */
        .block-container { padding-top: 0rem; } /* tweak 0–1rem to taste */

        /* give page titles a predictable gap from whatever is above them */
        h1.page-title { margin-top: -5rem; margin-bottom: 0rem; }
        </style>
        """,
        unsafe_allow_html=True
    ) 
    st.markdown("<h1 class='page-title' style='color:#FFC107; font-size:7.5rem;'>Data Overview & Export</h1>",unsafe_allow_html=True) 

    # st.header("Data Overview & Export")
    # st.markdown("---")
    # st.write("This page lets you download the dataset used for this app, either the original “raWWWw” dataset or the cleaned & feature-engineered version.")
        
    # st.write("This dataset captures information from direct marketing campaigns from a Portuguese banking institution. Its goal is to predict whether its clients will subscribe a term deposit or not.")
    
    # st.markdown("---")
    st.markdown("<br>", unsafe_allow_html=True)

    # ————— Way 2: Brief narrative description —————
    st.markdown(
        f"""
        <p>
        This dataset collected the results of a Portuguese bank’s telemarketing campaign between May 2008 and November 2010, it is later shared to the UCI ML repository and it can be accessed here. <strong><a href="https://archive.ics.uci.edu/dataset/222/bank+marketing">(Link)</a></strong>
        </p>

        <p>
        It has a classification objective by aiming to predict whether a client will subscribe to a term deposit based on:
        </p>

        <ul>
            <li><span style="color: #00BCD4;">Demographics</span> (age, job, marital status, etc.)</li>
            <li><span style="color: #00BCD4;">Past campaign interactions</span> (call duration, number of contacts)</li>
            <li><span style="color: #00BCD4;">Economic indicators</span> (balance, housing/loan status)</li>
        </ul>
        """,
        unsafe_allow_html=True
    )

    st.markdown("<hr>", unsafe_allow_html=True)
    
    st.write("This page lets you download the dataset used for this app, including the original “raw” dataset or the “processed” data by utilizng the following techniques:") 
    st.markdown(
        """
        <ol>
            <li>Transform null data through <span style="color: #00BCD4;">data imputation</span></li>
            <li>Applied different <span style="color: #00BCD4;">encoding techniques</span> (label encoding and one-hot encoding) to encode different categorical features</li>
            <li><span style="color: #00BCD4;">Outlier detection and removal</span></li>
            <li>Create new features with <span style="color: #00BCD4;">feature engineering</span> techniques</li>
        </ol>
        """,
        unsafe_allow_html=True
    )


    st.markdown("---")

    # ————— Way 3: Feature data dictionary —————
    with st.expander("ℹ️ Feature Descriptions"):
        show_example_table(data, selected_cols=preprocessed.columns)
    st.markdown("---")

    # ————— Way 4: Interactive data preview & filtering —————
    st.markdown("<h2 style='color:#00BCD4;'>Data Preview</h2>",unsafe_allow_html=True)
    # st.subheader("Data Preview")
    cols_to_show = st.multiselect(
        "Pick features to preview:",
        options=preprocessed.columns.tolist(),
        default=preprocessed.columns.tolist()[:5]
    )
    if cols_to_show:
        df_to_show = preprocessed[cols_to_show].head(10).copy()
        df_to_show.index = df_to_show.index + 1  # make it start at 1
        st.dataframe(df_to_show, use_container_width=True)
    st.markdown("---")

    st.markdown("<h2 style='color:#00BCD4;'>Data Export</h2>",unsafe_allow_html=True)
    # st.subheader("Data Export")
    # st.markdown("<br>", unsafe_allow_html=True)
    # 2 box with a divider separating them
    col1, col_div, col2 = st.columns([1, 0.02, 1])

    st.markdown("---")



    # Raw data box
    # ─── Raw Data ──────────────────────────────────────────────
    with col1:
        st.subheader("Raw Data")
        # st.markdown(f"- Rows **(Number of Entries):** <u>{data.shape[0]:,}</u>  \n- Columns **(Number of Features):** <u>{data.shape[1]:,}</u>", unsafe_allow_html=True)
        st.markdown(
            f"- Rows **(Number of Entries):** <span style='color:#00BCD4;'><u>{data.shape[0]:,}</u></span>  \n"
            f"- Columns **(Number of Features):** <span style='color:#00BCD4;'><u>{data.shape[1]:,}</u></span>",
            unsafe_allow_html=True
        )
        csv_bytes = get_csv_bytes(data)
        # st.download_button("Download CSV 📥", csv_bytes, "raw_data.csv", "text/csv")
        xlsx_buf   = get_excel_buffer(data)
        # st.download_button("Download XLSX 📥", xlsx_buf, "raw_data.xlsx",
        #                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")

        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download CSV 📥", csv_bytes, "raw_data.csv", "text/csv", use_container_width=True)
        with c2:
            st.download_button("Download XLSX 📥", xlsx_buf, "raw_data.xlsx",
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)

    # ─── Divider ───────────────────────────────────────────────
    with col_div:
        st.markdown(
        """
        <div style="
            border-left:2px solid #ccc;
            height:100%;
            min-height:200px;   /* adjust to match your content */
            margin:0 auto;
        "></div> 
        """,
        unsafe_allow_html=True
        )
    
    # Preprocessed data box
    # ─── Preprocessed Data ─────────────────────────────────────
    with col2:
        st.subheader("Processed / Cleaned Data")
        st.markdown(
            f"- Rows **(Number of Entries):** <span style='color:#00BCD4;'><u>{preprocessed.shape[0]:,}</u></span>  \n"
            f"- Columns **(Number of Features):** <span style='color:#00BCD4;'><u>{preprocessed.shape[1]:,}</u></span> (+15)",
            unsafe_allow_html=True
        )
        # st.markdown(f"- Rows **(Number of Entries):** <u>{preprocessed.shape[0]:,}</u>  \n- Columns **(Number of Features):** <u>{preprocessed.shape[1]:,} **(+15)**</u>", unsafe_allow_html=True)
        csv2 = get_csv_bytes(preprocessed)
        xlsx2 = get_excel_buffer(preprocessed)
        # st.download_button("Download CSV 📥", csv2, "Preprocessed_Data.csv", "text/csv")
        # st.download_button("Download XLSX 📥", xlsx2, "Preprocessed_Data.xlsx",
        #                    "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        c1, c2 = st.columns(2)
        with c1:
            st.download_button("Download CSV 📥", csv2, "Preprocessed_Data.csv", "text/csv", use_container_width=True)
        with c2:
            st.download_button("Download XLSX 📥", xlsx2, "Preprocessed_Data.xlsx",
                           "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet", use_container_width=True)
    
# Displays the final acknowledgement page
def acknowledgement_page(data):

    st.markdown(
        """
        <style>
        /* move the whole page content up */
        .block-container { padding-top: 0rem; } /* tweak 0–1rem to taste */

        /* give page titles a predictable gap from whatever is above them */
        h1.page-title { margin-top: 0rem; margin-bottom: 0rem; }
        </style>
        """,
        unsafe_allow_html=True
    ) 

    # display text
    # st.header("Acknowledgements")
    st.markdown("<h1 class='page-title' style='color:#FFC107;'>Acknowledgements</h1>",unsafe_allow_html=True)
    st.markdown("---")

    #FFC107
    # <span style='color: #FFC107;'><u>{num_clusters} customer groups</u></span>


    ack_html = """
    First of all, this entire application was developed starts from a graduate data science course project. I further tuned the built models, operationalized them under a streamlit application and added more features. 
    
    I want to sincerely thank my project teammates: <span style='color: #FFC107;'>**Zheng En Than**</span> and <span style='color: #FFC107;'>**Emily Au**</span>. We cleaned the collected data, 
    performed exploratory data analysis, developed the machine learning models, and wrote a scientific report together. I would also like to thank my course instructor <span style='color: #FFC107;'>**Dr. Jay Newby**</span> for his guidance and mentorship. If you are interested to learn more about the scientific details of our work, please visit the <a href="https://github.com/Alex-Mak-MCW/Deposit_Subcriptions_Predictions_Project" target="_blank"><strong>User Guide</strong></a>!
    <br><br>
    Additionally, I want to acknowledge <span style='color: #FFC107;'>**Sérgio Moro**</span>, <span style='color: #FFC107;'>**P. Cortez**</span>, and <span style='color: #FFC107;'>**P. Rita**</span> for sharing the UCI ML Bank Telemarketing Dataset, which is the fundamental backbone of this project.
    <br><br>
    Last but not least, shout out to the user test group (<span style='color: #FFC107;'>**Steven Ge**, **Ruike Xu**, **Tek Chan**, **Jerry Chan**, **David Lee**</span>). Their opinions and feedback further elevate this project to a whole new level.

    """
    st.markdown(ack_html, unsafe_allow_html=True)

    st.markdown("<br>", unsafe_allow_html=True)

    st.image("App_Visualizations/title_icon_temp.gif", width=300, caption="Me vibin' when I am creating this project :)")

# --- MAIN APP ---
#-----------------------------------------------
#-----------------------------------------------
#-----------------------------------------------


# global dictionary to link page and index
PAGE_TO_INDEX = {
    "Home":                              0,
    "Deposit Prediction":  1,
    "Interactive Dashboard":            2,
    "Customer Segmentation":            3,
    "Data Overview & Export":           4,
    "Acknowledgements":                 5,
}

# The main function that puts everything together
def main():
    # st.title("Bank Term Deposit App")
    st.markdown(
        """
        <style>
        /* move the whole page content up */
        .block-container { padding-top: 0rem; } /* tweak 0–1rem to taste */

        /* give page titles a predictable gap from whatever is above them */
        h1.page-title { margin-top: 0rem; margin-bottom: 0rem; }
        </style>
        """,
        unsafe_allow_html=True
    )

    # only update page, sidebar will be sync iin a different way
    if "page" not in st.session_state:
        st.session_state.page = "Home"
    # if "sidebar_choice" not in st.session_state:
    #     st.session_state.sidebar_choice = "Home"

    # define a sidebar callback
    def sidebar_navigate():
        st.session_state.page = st.session_state.sidebar_choice

    # Sync sidebar here
    def _sync_page_with_sidebar(new_choice):
        # ignore new_choice, just copy over the widget state
        st.session_state.page = st.session_state.sidebar_choice

    SIDEBAR_TITLE_STYLE = """
        <style>
        .sidebar-h1 {
            font-size: 2.5rem !important;   /* match main h1 */
            font-weight: bold;
            color: #A569FF;
            margin-top:0.25rem;
        }
        </style>
    """
    st.markdown(SIDEBAR_TITLE_STYLE, unsafe_allow_html=True)

    # display sidebar
    with st.sidebar:
        # title
        # st.title("FinML Studio")
        # st.markdown(
        #     "<br>",
        #     unsafe_allow_html=True
        # )

        st.markdown(
            """
            <style>
            [data-testid="stSidebar"] img {
                margin-top: -100px;   /* adjust negative value as needed */
                margin-bottom:-75px;
            }
            </style>
            """,
            unsafe_allow_html=True
        )

        st.image("App_Visualizations/Homepage_Icons/sidebar-icon.png", width=290)  # Use an icon for success
        
        
        # If decided to show app title in text (just uncomment)
        # st.markdown(
        #     "<h1 style='color:#A569FF;'>FinML Studio</h1>",
        #     unsafe_allow_html=True
        # )
        # st.markdown("<h1 class='sidebar-h1'>FinML Studio</h1>", unsafe_allow_html=True)
        # st.markdown(
        #     "<h1 style='color:#A569FF; font-size:2.5rem; font-weight:bold; margin-top:-5rem;'>FinML Studio</h1>",
        #     unsafe_allow_html=True
        # )


        today = datetime.date.today().strftime("%Y-%m-%d")
        # st.caption(f"v1.1.0 • Last Updated On: {today}")
        st.markdown(
            f"<h5 style='text-align:center;margin:-1rem 0;'>"
            f"v1.1.0 • Last Updated On: {today}"
            f"</h5>",
            unsafe_allow_html=True,
        )

        st.markdown("---")

        current = st.session_state.page
        idx     = PAGE_TO_INDEX.get(current, 0)

        # option menu for users to select pages to navigate
        choice = option_menu(
            menu_title=None,
            # options=["Home", "Deposit Subscription Prediction", "Interactive Dashboard", "Customer Segmentation", "Data Overview & Export", "Acknowledgements"],
            options=list(PAGE_TO_INDEX.keys()),
            icons=["house", "bank", "bar-chart-line", "pie-chart-fill", "cloud-download", "award"],
            menu_icon="app-indicator",
            # default_index=0,
            default_index=idx,
            orientation="vertical",
            # key="sidebar_choice",
            # on_change=lambda: st.session_state.update(page=st.session_state.sidebar_choice)
            # on_change=lambda *args: st.session_state.update(page=st.session_state.sidebar_choice)
            # on_change=_sync_page_with_sidebar
            styles={
                "container": {
                    "padding": "0!important",
                    "background-color": "transparent",    # ← key line
                },
                "nav-link": {
                    "--hover-color": "rgba(255, 255, 255, 0.1)",   # lighter, subtle white
                },
                "nav-link-selected": {
                    "background-color": "#f63366",
                    "color": "white",
                },
            }
        )
        # When moving poage
        if choice != current:
            # # goto(choice)

            st.session_state.page = choice # only chaning page
            st.session_state._scroll_top = True # adding the part of scrolling back to top
            st.rerun()

        # --- Help & feedback Expander---
        with st.expander("❓ Help & Docs"):
            st.write("- [User Guide](https://github.com/Alex-Mak-MCW/FinML-Studio/blob/main/README.md)")
            st.write("- [Source Code](https://github.com/Alex-Mak-MCW/FinML-Studio/tree/main)")
            # st.write("- [Contact Me!](https://www.linkedin.com/in/alex-mak-/824187247)")
            st.write("- [Contact Me!](https://www.linkedin.com/in/alex-mak-mcw)")
        
        current_year=datetime.date.today().year

        st.caption(f"© {current_year} Alex Mak, All Rights Reserved")

    # choice = st.sidebar.radio("Go to", ["Prediction", "Dashboard"] )

    # init functions: load models and data
    models = load_models()
    data   = load_data()

    raw_data=pd.read_csv("https://raw.githubusercontent.com/Alex-Mak-MCW/FinML-Studio/refs/heads/main/Data/input.csv", sep=";")

    # get current page
    page = st.session_state.page


    # back page button at the top of every page except home page
    if st.session_state.page != "Home":
        if st.button("🏠 Back to Home"):
            st.session_state.page = "Home"
            # st.experimental_rerun()
            st.rerun()
    # a tiny spacer so the title doesn't crowd the button
    st.markdown("<div style='height:0.20rem'></div>", unsafe_allow_html=True)


    # page navigation based on user's selection
    if page == "Home":
        home_page(models, data, raw_data)
    elif page == "Deposit Prediction":
        prediction_page(models, data)
    elif page == "Interactive Dashboard":
        dashboard_page(data)
    elif page == "Customer Segmentation":
        clustering_page(data)
    elif page == "Data Overview & Export":
        overview_page(raw_data, data)
    elif page == "Acknowledgements":
        acknowledgement_page(data)

if __name__ == "__main__":
    main()
