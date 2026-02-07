import streamlit as st
import pandas as pd
import sqlite3
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import PolynomialFeatures
from sklearn.linear_model import LinearRegression

# -----------------------
# Page configuration
# -----------------------
st.set_page_config(
    page_title="Global CO₂ Emissions Dashboard",
    layout="wide"
)

st.title("🌍 Global CO₂ Emissions Dashboard (1750–2020)")
st.markdown("""
This dashboard explores **long-term global carbon dioxide emissions**,  
highlighting historical trends, growth dynamics, structural shifts, and
business-as-usual projections.
""")

# -----------------------
# Load data
# -----------------------
@st.cache_data
def load_data():
    conn = sqlite3.connect("data/processed/global_co2_emissions.db")
    df = pd.read_sql(
        "SELECT year, emissions FROM global_emissions ORDER BY year",
        conn
    )
    return df

df = load_data()

# -----------------------
# Sidebar controls
# -----------------------
st.sidebar.header("Controls")

min_year, max_year = int(df.year.min()), int(df.year.max())
year_range = st.sidebar.slider(
    "Select year range",
    min_year,
    max_year,
    (min_year, max_year)
)

filtered_df = df[
    (df.year >= year_range[0]) &
    (df.year <= year_range[1])
]

# -----------------------
# KPI metrics
# -----------------------
st.subheader("📊 Key Indicators")

col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Latest Emissions (GtCO₂)",
        f"{filtered_df.iloc[-1]['emissions']:.2f}"
    )

with col2:
    growth = (
        (filtered_df.iloc[-1]["emissions"] /
         filtered_df.iloc[0]["emissions"]) - 1
    ) * 100
    st.metric(
        "Growth Since Start (%)",
        f"{growth:.1f}%"
    )

with col3:
    avg_growth = filtered_df["emissions"].pct_change().mean() * 100
    st.metric(
        "Avg Annual Growth (%)",
        f"{avg_growth:.2f}%"
    )

# -----------------------
# Time series plot
# -----------------------
st.subheader("📈 Emissions Over Time")

fig, ax = plt.subplots()
ax.plot(filtered_df["year"], filtered_df["emissions"])
ax.set_xlabel("Year")
ax.set_ylabel("Emissions (GtCO₂)")
ax.set_title("Global CO₂ Emissions")
st.pyplot(fig)

# -----------------------
# Log-scale comparison
# -----------------------
st.subheader("📐 Linear vs Log Scale")

col1, col2 = st.columns(2)

with col1:
    fig, ax = plt.subplots()
    ax.plot(filtered_df["year"], filtered_df["emissions"])
    ax.set_title("Linear Scale")
    st.pyplot(fig)

with col2:
    fig, ax = plt.subplots()
    ax.plot(filtered_df["year"], filtered_df["emissions"])
    ax.set_yscale("log")
    ax.set_title("Log Scale")
    st.pyplot(fig)

# -----------------------
# Structural regimes
# -----------------------
st.subheader("🏭 Emissions Regimes")

def classify_era(year):
    if year < 1850:
        return "Pre-Industrial"
    elif year < 1950:
        return "Industrialization"
    else:
        return "Post-1950 Acceleration"

df["era"] = df["year"].apply(classify_era)
era_summary = df.groupby("era")["emissions"].mean().reset_index()

st.dataframe(era_summary)

# -----------------------
# Forecasting
# -----------------------
st.subheader("🔮 Business-as-Usual Forecast")

forecast_horizon = st.slider("Forecast until year", 2030, 2100, 2050)

X = df[["year"]]
y = df["emissions"]

poly = PolynomialFeatures(degree=2)
X_poly = poly.fit_transform(X)

model = LinearRegression()
model.fit(X_poly, y)

future_years = np.arange(df.year.max(), forecast_horizon + 1).reshape(-1, 1)
future_poly = poly.transform(future_years)
future_emissions = model.predict(future_poly)

fig, ax = plt.subplots()
ax.plot(df["year"], df["emissions"], label="Historical")
ax.plot(
    future_years.flatten(),
    future_emissions,
    linestyle="--",
    label="BAU Projection"
)
ax.legend()
ax.set_xlabel("Year")
ax.set_ylabel("Emissions (GtCO₂)")
ax.set_title("Projected Emissions Trajectory")
st.pyplot(fig)

# -----------------------
# Carbon Budget Analysis
# -----------------------
st.subheader("🔥 Carbon Budget & Cumulative Emissions")

# IPCC-style carbon budget assumption
CARBON_BUDGET_GT = 400  # GtCO2 remaining for ~1.5°C (approximate)

# Compute cumulative emissions
df["cumulative_emissions"] = df["emissions"].cumsum()

latest_cumulative = df["cumulative_emissions"].iloc[-1]

remaining_budget = CARBON_BUDGET_GT - (
    df[df["year"] >= df["year"].max() - 1]["emissions"].sum()
)

# KPIs
col1, col2, col3 = st.columns(3)

with col1:
    st.metric(
        "Cumulative Emissions (GtCO₂)",
        f"{latest_cumulative:.0f}"
    )

with col2:
    st.metric(
        "Remaining Carbon Budget (GtCO₂)",
        f"{remaining_budget:.0f}",
        delta=f"-{df.iloc[-1]['emissions']:.1f} / year"
    )

with col3:
    years_left = remaining_budget / df.iloc[-1]["emissions"]
    st.metric(
        "Years Until Budget Exhausted",
        f"{max(years_left, 0):.1f}"
    )

# Cumulative emissions plot
fig, ax = plt.subplots()
ax.plot(df["year"], df["cumulative_emissions"], label="Cumulative Emissions")
ax.axhline(
    latest_cumulative + remaining_budget,
    linestyle="--",
    color="red",
    label="Carbon Budget Limit"
)
ax.set_xlabel("Year")
ax.set_ylabel("Cumulative Emissions (GtCO₂)")
ax.set_title("Cumulative CO₂ Emissions vs Carbon Budget")
ax.legend()
st.pyplot(fig)

# Forecast carbon budget exhaustion
future_cumulative = latest_cumulative + np.cumsum(future_emissions)

exceed_idx = np.where(future_cumulative >= latest_cumulative + remaining_budget)[0]

if len(exceed_idx) > 0:
    exhaustion_year = int(future_years[exceed_idx[0]])
    st.warning(
        f"⚠️ Under Business-as-Usual assumptions, the remaining carbon budget "
        f"would be exhausted around **{exhaustion_year}**."
    )
else:
    st.success("Carbon budget not exhausted within forecast horizon.")


# -----------------------
# Interpretation
# -----------------------
st.subheader("🧠 Interpretation")

st.markdown("""
**Key takeaways:**
- Global CO₂ emissions were negligible prior to industrialization
- Post-1950 emissions dominate cumulative atmospheric loading
- Recent years show slowing growth, not absolute decline
- Business-as-usual projections exceed climate-safe thresholds

This reinforces that **structural decarbonization** is required,
not incremental efficiency gains alone.
""")

st.subheader("🧠 Carbon Budget Interpretation")

st.markdown("""
**What this means:**

- Cumulative emissions matter more than annual emissions for climate outcomes
- At current emission rates, the remaining 1.5 °C carbon budget is measured in **years, not decades**
- Business-as-usual trajectories are incompatible with climate stabilization goals

**Policy implication:**  
Avoiding budget exhaustion requires **absolute emissions reductions**, not merely slowing growth.
""")

