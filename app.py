import datetime
from pathlib import Path
import numpy as np
import pandas as pd
import plotly.express as px
import plotly.graph_objects as go
import streamlit as st

# Optional Groq SDK import with graceful handling
try:
  from groq import Groq
except ImportError:
  Groq = None

# ==========================================
# CONFIGURATION & CONSTANTS
# ==========================================
ST_MODELS = [
    "openai/gpt-oss-120b",
    "llama-3.3-70b-versatile",
    "llama-3.1-8b-instant",
]

SEGMENT_COLORS = {
    "Promising": "#1f77b4",  # Blue
    "Growth": "#2ca02c",  # Green
    "Stable": "#7f7f7f",  # Gray
    "Decline": "#d62728",  # Red
    "Unlabeled": "#ff7f0e",  # Orange
}

CATEGORY_MAPPING = {
    "Electronics": "Electronics",
    "Gaming": "Electronics",
    "Home Appliances": "Home & Furniture",
    "Furniture": "Home & Furniture",
    "Furniture & Decor": "Home & Furniture",
    "Home Decor": "Home & Furniture",
    "Home & Garden": "Home & Furniture",
    "Home Improvement": "Home & Furniture",
    "Clothing": "Clothing & Fashion",
    "Fashion": "Clothing & Fashion",
    "Fashion & Apparel": "Clothing & Fashion",
    "Children's Clothing": "Clothing & Fashion",
    "Fashion Accessories": "Clothing & Fashion",
    "Sportswear": "Clothing & Fashion",
    "Groceries": "Groceries & Food",
    "Food & Beverages": "Groceries & Food",
    "Grocery Items": "Groceries & Food",
    "Grocery": "Groceries & Food",
    "Books": "Books",
    "Books & Magazines": "Books",
    "Health & Wellness": "Health & Beauty",
    "Beauty Products": "Health & Beauty",
    "Health & Beauty": "Health & Beauty",
    "Health Supplements": "Health & Beauty",
    "Beauty & Personal Care": "Health & Beauty",
    "Cosmetics": "Health & Beauty",
    "Sporting Goods": "Sports & Outdoors",
    "Sports & Outdoors": "Sports & Outdoors",
    "Sports Equipment": "Sports & Outdoors",
    "Outdoor Equipment": "Sports & Outdoors",
    "Gardening Tools": "Sports & Outdoors",
    "Toys": "Toys & Gaming",
    "Toys & Games": "Toys & Gaming",
    "Office Supplies": "Other",
    "Automotive": "Other",
}

st.set_page_config(
    page_title="NovaRetail Customer Intelligence",
    page_icon="🛍️",
    layout="wide",
    initial_sidebar_state="expanded",
)

# Custom Professional CSS Styling
st.markdown(
    """
    <style>
        .main-header { font-size: 2.2rem; font-weight: 700; color: #1e3a8a; margin-bottom: 0rem; }
        .sub-header { font-size: 1.1rem; color: #4b5563; margin-bottom: 1.5rem; }
        .metric-card { background-color: #f8fafc; border: 1px solid #e2e8f0; padding: 15px; border-radius: 8px; text-align: center; }
        .stAlert { border-radius: 6px; }
    </style>
""",
    unsafe_allow_html=True,
)


# ==========================================
# DATA LOADING & CLEANING
# ==========================================
@st.cache_data
def load_and_clean_data():
  file_path = Path(__file__).parent / "NR_dataset.xlsx"
  if not file_path.exists():
    # Fallback for local execution if file is in working directory
    file_path = Path("NR_dataset.xlsx")

  df = pd.read_excel(file_path, sheet_name="data")

  # Data Quality Fixes
  if "idx" in df.columns:
    df = df.drop(columns=["idx"])

  # 1. Fill missing label
  df["label"] = df["label"].fillna("Unlabeled")

  # 2. ProductCategory mapping
  df["ProductCategory_raw"] = df["ProductCategory"]
  df["ProductCategory"] = (
      df["ProductCategory"].map(CATEGORY_MAPPING).fillna("Other")
  )

  # 3. CustomerAgeGroup merging and logical ordering
  df["CustomerAgeGroup"] = df["CustomerAgeGroup"].replace({"55-64": "55+"})

  # 4. Ensure correct datatypes
  df["TransactionDate"] = pd.to_datetime(df["TransactionDate"])
  df["PurchaseAmount"] = pd.to_numeric(df["PurchaseAmount"])
  df["CustomerSatisfaction"] = pd.to_numeric(df["CustomerSatisfaction"])

  return df


try:
  df_raw = load_and_clean_data()
except Exception as e:
  st.error(
      f"Error loading dataset `NR_dataset.xlsx`: {e}. Please ensure the file is"
      " in the repository root."
  )
  st.stop()

# ==========================================
# SIDEBAR FILTERS
# ==========================================
st.sidebar.markdown("### 🔍 Dashboard Controls")

if st.sidebar.button("🔄 Reset All Filters", use_container_width=True):
  st.rerun()

st.sidebar.markdown("---")

# Segment filter
all_segments = ["Promising", "Growth", "Stable", "Decline", "Unlabeled"]
selected_segments = st.sidebar.multiselect(
    "Customer Segments",
    options=all_segments,
    default=["Promising", "Growth", "Stable", "Decline"],
)

# Region filter
all_regions = sorted(df_raw["CustomerRegion"].unique().tolist())
selected_regions = st.sidebar.multiselect(
    "Customer Regions", options=all_regions, default=all_regions
)

# Product Category filter
all_categories = sorted(df_raw["ProductCategory"].unique().tolist())
selected_categories = st.sidebar.multiselect(
    "Product Categories", options=all_categories, default=all_categories
)

# Retail Channel filter
all_channels = sorted(df_raw["RetailChannel"].unique().tolist())
selected_channels = st.sidebar.multiselect(
    "Retail Channels", options=all_channels, default=all_channels
)

# Age Group filter
age_order = ["18-24", "25-34", "35-44", "45-54", "55+"]
available_ages = [a for a in age_order if a in df_raw["CustomerAgeGroup"].unique()]
selected_ages = st.sidebar.multiselect(
    "Age Groups", options=available_ages, default=available_ages
)

# Gender filter
all_genders = sorted(df_raw["CustomerGender"].unique().tolist())
selected_genders = st.sidebar.multiselect(
    "Customer Gender", options=all_genders, default=all_genders
)

# Satisfaction Slider
sat_min, sat_max = int(df_raw["CustomerSatisfaction"].min()), int(
    df_raw["CustomerSatisfaction"].max()
)
selected_satisfaction = st.sidebar.slider(
    "Customer Satisfaction Range",
    min_value=sat_min,
    max_value=sat_max,
    value=(sat_min, sat_max),
)

# Date Range Filter
min_date = df_raw["TransactionDate"].min().date()
max_date = df_raw["TransactionDate"].max().date()
selected_dates = st.sidebar.date_input(
    "Transaction Date Range", value=(min_date, max_date), min_value=min_date, max_value=max_date
)

# Apply Filters
df_filtered = df_raw[
    df_raw["label"].isin(selected_segments)
    & df_raw["CustomerRegion"].isin(selected_regions)
    & df_raw["ProductCategory"].isin(selected_categories)
    & df_raw["RetailChannel"].isin(selected_channels)
    & df_raw["CustomerAgeGroup"].isin(selected_ages)
    & df_raw["CustomerGender"].isin(selected_genders)
    & df_raw["CustomerSatisfaction"].between(
        selected_satisfaction[0], selected_satisfaction[1]
    )
    & df_raw["TransactionDate"]
    .dt.date
    .between(
        selected_dates[0] if isinstance(selected_dates, tuple) else min_date,
        selected_dates[1]
        if isinstance(selected_dates, tuple) and len(selected_dates) > 1
        else max_date,
    )
]

total_rows_initial = len(df_raw)
filtered_rows_count = len(df_filtered)

st.sidebar.markdown("---")
st.sidebar.markdown(
    f"**Filtered:** `{filtered_rows_count}` of `{total_rows_initial}` rows"
)

# ==========================================
# MAIN HEADER & DATA HEALTH
# ==========================================
st.markdown(
    '<p class="main-header">NovaRetail Customer Intelligence</p>',
    unsafe_allow_html=True,
)
st.markdown(
    '<p class="sub-header">Strategic Insights & Growth Dashboard for Sophia'
    " Martinez, Director of Customer Intelligence</p>",
    unsafe_allow_html=True,
)

with st.expander("ℹ️ Data Notes & Health Check"):
  col_dh1, col_dh2 = st.columns(2)
  with col_dh1:
    st.markdown(
        "**Dataset Structure & Metrics:**"
        f"\n- **Total Raw Line Items:** {total_rows_initial}"
        f"\n- **Unique Customers:** {df_raw['CustomerID'].nunique()}"
        f"\n- **Unique Transactions:** {df_raw['TransactionID'].nunique()}"
        "\n- **Date Range:** Jan 15 – Feb 20, 2023 (Weekly aggregation"
        " recommended)"
        "\n- **Unlabeled Rows Handled:**"
        f" {(df_raw['label']=='Unlabeled').sum()} row(s) filled with 'Unlabeled'"
        " and excluded from segment charts by default."
    )
  with col_dh2:
    st.markdown(
        "**Key Analytical Caveats:**"
        "\n1. **Row-Level Segments:** A single CustomerID may appear with"
        " different behavioral labels across rows. Segments are analyzed at"
        " the row level."
        "\n2. **Small Sample Size:** $n=100$. Treat findings as directional"
        " guidance rather than definitive population metrics."
        "\n3. **Category Standardization:** 35 raw categories mapped into 8 clean"
        " commercial groups."
    )

# Zero-row check
if filtered_rows_count == 0:
  st.warning(
      "⚠️ The current filter selection returned zero rows. Please adjust your"
      " filters in the sidebar or click 'Reset All Filters'."
  )
  st.stop()

# ==========================================
# TOP KPI ROW
# ==========================================
total_revenue = df_filtered["PurchaseAmount"].sum()
unique_customers = df_filtered["CustomerID"].nunique()
revenue_per_customer = (
    total_revenue / unique_customers if unique_customers > 0 else 0
)
avg_line_item = df_filtered["PurchaseAmount"].mean()
avg_satisfaction = df_filtered["CustomerSatisfaction"].mean()

online_rev = df_filtered[df_filtered["RetailChannel"] == "Online"][
    "PurchaseAmount"
].sum()
pct_online = (
    (online_rev / total_revenue * 100) if total_revenue > 0 else 0
)

decline_rev = df_filtered[df_filtered["label"] == "Decline"][
    "PurchaseAmount"
].sum()
pct_decline = (
    (decline_rev / total_revenue * 100) if total_revenue > 0 else 0
)

kpi1, kpi2, kpi3, kpi4, kpi5, kpi6, kpi7 = st.columns(7)
kpi1.metric("Total Revenue", f"${total_revenue:,.2f}")
kpi2.metric("Unique Customers", f"{unique_customers:,}")
kpi3.metric("Rev / Customer", f"${revenue_per_customer:,.2f}")
kpi4.metric("Avg Line Value", f"${avg_line_item:,.2f}")
kpi5.metric("Avg Satisfaction", f"{avg_satisfaction:.2f} / 5.0")
kpi6.metric("Online Revenue", f"{pct_online:.1f}%")
kpi7.metric("Decline Risk Rev", f"{pct_decline:.1f}%")

st.markdown("---")

# ==========================================
# TABS SETUP
# ==========================================
tab1, tab2, tab3, tab4, tab5, tab6 = st.tabs([
    "📊 Executive Overview",
    "🚀 Growth Opportunities",
    "⚠️ Risk & Early Warnings",
    "👥 Customer Demographics",
    "💡 Strategic Recommendations",
    "💬 Ask the Data (AI)",
])


# ==========================================
# TAB 1: EXECUTIVE OVERVIEW
# ==========================================
with tab1:
  st.markdown("### Executive Overview & Revenue Performance")
  st.caption(
      "High-level breakdown of revenue generation across customer segments,"
      " geographic regions, and weekly transaction trends."
  )

  col_ex1, col_ex2 = st.columns(2)

  with col_ex1:
    # Revenue by segment (bar)
    seg_rev = (
        df_filtered.groupby("label")["PurchaseAmount"]
        .sum()
        .reset_index()
        .sort_values(by="PurchaseAmount", ascending=False)
    )
    fig_seg_bar = px.bar(
        seg_rev,
        x="label",
        y="PurchaseAmount",
        color="label",
        color_discrete_map=SEGMENT_COLORS,
        title="Revenue by Customer Segment",
        labels={
            "label": "Customer Segment",
            "PurchaseAmount": "Total Revenue (USD)",
        },
        text_auto="$#,##0",
    )
    fig_seg_bar.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_seg_bar, use_container_width=True)
    st.caption(
        "Small sample ($n=100$): treat findings as directional. Compares"
        " total spend contributions per segment."
    )

  with col_ex2:
    # Share of revenue by segment (donut)
    fig_seg_donut = px.pie(
        seg_rev,
        names="label",
        values="PurchaseAmount",
        title="Revenue Share by Customer Segment",
        hole=0.4,
        color="label",
        color_discrete_map=SEGMENT_COLORS,
    )
    fig_seg_donut.update_traces(textinfo="percent+label")
    fig_seg_donut.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_seg_donut, use_container_width=True)
    st.caption("Proportional share of total filtered revenue by segment.")

  col_ex3, col_ex4 = st.columns(2)

  with col_ex3:
    # Revenue by region (bar)
    reg_rev = (
        df_filtered.groupby("CustomerRegion")["PurchaseAmount"]
        .sum()
        .reset_index()
        .sort_values(by="PurchaseAmount", ascending=False)
    )
    fig_reg = px.bar(
        reg_rev,
        x="CustomerRegion",
        y="PurchaseAmount",
        title="Revenue by Geographic Region",
        color="CustomerRegion",
        labels={
            "CustomerRegion": "Region",
            "PurchaseAmount": "Total Revenue (USD)",
        },
        text_auto="$#,##0",
    )
    fig_reg.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_reg, use_container_width=True)
    st.caption("Regional sales performance across North, South, East, and West.")

  with col_ex4:
    # Weekly revenue trend (line)
    df_weekly = (
        df_filtered.set_index("TransactionDate")
        .resample("W")["PurchaseAmount"]
        .sum()
        .reset_index()
    )
    df_weekly["WeekLabel"] = (
        df_weekly["TransactionDate"].dt.strftime("%b %d, %Y")
    )
    fig_trend = px.line(
        df_weekly,
        x="TransactionDate",
        y="PurchaseAmount",
        markers=True,
        title="Weekly Revenue Trend (Jan – Feb 2023)",
        labels={
            "TransactionDate": "Week",
            "PurchaseAmount": "Revenue (USD)",
        },
    )
    fig_trend.update_traces(line_color="#1e3a8a", line_width=3)
    fig_trend.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_trend, use_container_width=True)
    st.caption(
        "Weekly aggregated trend. Note: dataset spans ~5 weeks; avoid"
        " over-interpreting macro seasonality."
    )

  st.markdown("#### 📌 Key Takeaways (Computed Dynamically)")
  top_seg_row = seg_rev.iloc[0] if not seg_rev.empty else None
  top_reg_row = reg_rev.iloc[0] if not reg_rev.empty else None

  t1_text = (
      f"• **Dominant Segment:** `{top_seg_row['label']}` leads revenue"
      f" generation with `${top_seg_row['PurchaseAmount']:,.2f}` under current"
      " filters."
      if top_seg_row is not None
      else "• No segment data available."
  )
  t2_text = (
      f"• **Top Region:** `{top_reg_row['CustomerRegion']}` generates the"
      f" highest regional volume at `${top_reg_row['PurchaseAmount']:,.2f}`."
      if top_reg_row is not None
      else "• No regional data available."
  )
  t3_text = (
      f"• **Channel Distribution:** Online channel accounts for"
      f" `{pct_online:.1f}%` of revenue, while Decline segment risk exposure"
      f" stands at `{pct_decline:.1f}%`."
  )

  st.info(f"{t1_text}\n\n{t2_text}\n\n{t3_text}")


# ==========================================
# TAB 2: GROWTH OPPORTUNITIES
# ==========================================
with tab2:
  st.markdown("### Growth Opportunities & Expansion Analysis (Goal A)")
  st.caption(
      "Identifying high-potential product categories, regional white spaces,"
      " and channel preferences for Growth & Promising segments."
  )

  # Focus on Growth & Promising
  df_gp = df_filtered[df_filtered["label"].isin(["Growth", "Promising"])]

  col_g1, col_g2 = st.columns(2)

  with col_g1:
    st.markdown("#### Category x Segment Performance (Growth & Promising)")
    if not df_gp.empty:
      cat_seg = (
          df_gp.groupby(["ProductCategory", "label"])["PurchaseAmount"]
          .sum()
          .reset_index()
      )
      fig_cat_seg = px.bar(
          cat_seg,
          x="ProductCategory",
          y="PurchaseAmount",
          color="label",
          barmode="group",
          color_discrete_map=SEGMENT_COLORS,
          title="Revenue by Category for Growth & Promising Segments",
          labels={
              "ProductCategory": "Product Category",
              "PurchaseAmount": "Revenue (USD)",
          },
      )
      fig_cat_seg.update_layout(
          margin=dict(t=40, b=20, l=20, r=20), xaxis_tickangle=-30
      )
      st.plotly_chart(fig_cat_seg, use_container_width=True)
    else:
      st.info(
          "No data found for Growth and Promising segments under current"
          " filters."
      )

  with col_g2:
    st.markdown("#### Region x Category Revenue Heatmap")
    reg_cat = (
        df_filtered.pivot_table(
            index="CustomerRegion",
            columns="ProductCategory",
            values="PurchaseAmount",
            aggfunc="sum",
            fill_value=0,
        )
    )
    if not reg_cat.empty:
      fig_heat = px.imshow(
          reg_cat,
          text_auto="$#,##0",
          aspect="auto",
          title="Revenue Heatmap: Region vs Product Category",
          color_continuous_scale="Blues",
      )
      fig_heat.update_layout(margin=dict(t=40, b=20, l=20, r=20))
      st.plotly_chart(fig_heat, use_container_width=True)
    else:
      st.info("Insufficient data for region/category heatmap.")

  st.markdown("---")
  col_g3, col_g4 = st.columns(2)

  with col_g3:
    st.markdown("#### Retail Channel Mix by Segment")
    chan_seg = (
        df_filtered.groupby(["label", "RetailChannel"])["PurchaseAmount"]
        .sum()
        .reset_index()
    )
    fig_chan = px.bar(
        chan_seg,
        x="label",
        y="PurchaseAmount",
        color="RetailChannel",
        barmode="stack",
        title="Channel Mix (Online vs Physical) Across Segments",
        labels={
            "label": "Segment",
            "PurchaseAmount": "Revenue (USD)",
            "RetailChannel": "Channel",
        },
    )
    fig_chan.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_chan, use_container_width=True)

  with col_g4:
    st.markdown("#### Top Growth & Promising Opportunities")
    if not df_gp.empty:
      top_opps = (
          df_gp.groupby(["ProductCategory", "CustomerRegion", "RetailChannel"])[
              "PurchaseAmount"
          ]
          .agg(["sum", "count"])
          .reset_index()
          .rename(columns={"sum": "TotalRevenue", "count": "LineItems"})
          .sort_values(by="TotalRevenue", ascending=False)
          .head(3)
      )
      for idx, row in top_opps.iterrows():
        st.success(
            f"**{row['ProductCategory']}** in `{row['CustomerRegion']}` via"
            f" `{row['RetailChannel']}` ➔ **${row['TotalRevenue']:,.2f}**"
            f" (`{row['LineItems']}` items)"
        )
    else:
      st.info("No growth/promising opportunities available under current filters.")

  st.markdown("#### Top Customers Table (Sortable)")
  cust_summary = (
      df_filtered.groupby(
          ["CustomerID", "label", "CustomerRegion", "RetailChannel"]
      )
      .agg(
          TotalSpend=("PurchaseAmount", "sum"),
          Transactions=("TransactionID", "count"),
          AvgSatisfaction=("CustomerSatisfaction", "mean"),
      )
      .reset_index()
      .sort_values(by="TotalSpend", ascending=False)
  )
  st.dataframe(
      cust_summary.style.format({
          "TotalSpend": "${:,.2f}",
          "AvgSatisfaction": "{:.1f}",
      }),
      use_container_width=True,
      hide_index=True,
  )
  st.caption("Small sample ($n=100$): treat findings as directional.")


# ==========================================
# TAB 3: RISK & EARLY WARNINGS
# ==========================================
with tab3:
  st.markdown("### Risk Detection & Early Warning Signals (Goal B)")
  st.caption(
      "Monitoring customer satisfaction distributions, profiling the Decline"
      " segment, and identifying at-risk accounts."
  )

  # Risk KPI row
  col_r1, col_r2, col_r3 = st.columns(3)
  col_r1.metric(
      "Revenue at Risk (Decline)",
      f"${decline_rev:,.2f}",
      help="Total revenue generated by customers in the Decline segment.",
  )

  low_sat_df = df_filtered[df_filtered["CustomerSatisfaction"] <= 2]
  low_sat_rev = low_sat_df["PurchaseAmount"].sum()
  low_sat_pct = (
      (low_sat_rev / total_revenue * 100) if total_revenue > 0 else 0
  )
  col_r2.metric(
      "Low Satisfaction Rev (<=2)",
      f"${low_sat_rev:,.2f} ({low_sat_pct:.1f}%)",
      help="Revenue associated with satisfaction scores of 1 or 2.",
  )

  at_risk_count = (
      df_filtered[
          (df_filtered["CustomerSatisfaction"] <= 2)
          | (df_filtered["label"] == "Decline")
      ]["CustomerID"]
      .nunique()
  )
  col_r3.metric(
      "At-Risk Unique Customers",
      f"{at_risk_count}",
      help=(
          "Distinct customers flagged either as Decline or having satisfaction"
          " <= 2."
      ),
  )

  st.markdown("---")
  col_r4, col_r5 = st.columns(2)

  with col_r4:
    st.markdown("#### Satisfaction Distribution by Segment")
    fig_box = px.box(
        df_filtered,
        x="label",
        y="CustomerSatisfaction",
        color="label",
        color_discrete_map=SEGMENT_COLORS,
        title="Customer Satisfaction Spread by Segment",
        points="all",
        labels={"label": "Segment", "CustomerSatisfaction": "Satisfaction (1-5)"},
    )
    fig_box.add_hline(
        y=3.0,
        line_dash="dash",
        line_color="red",
        annotation_text="Satisfaction Threshold (3.0)",
    )
    fig_box.update_layout(showlegend=False, margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_box, use_container_width=True)

  with col_r5:
    st.markdown("#### Decline Segment Profile")
    df_decline = df_filtered[df_filtered["label"] == "Decline"]
    if not df_decline.empty:
      dec_cat = (
          df_decline.groupby("ProductCategory")["PurchaseAmount"]
          .sum()
          .reset_index()
      )
      fig_dec = px.bar(
          dec_cat,
          x="ProductCategory",
          y="PurchaseAmount",
          title="Decline Segment Revenue by Product Category",
          color_discrete_sequence=["#d62728"],
          labels={
              "ProductCategory": "Category",
              "PurchaseAmount": "Revenue (USD)",
          },
          text_auto="$#,##0",
      )
      fig_dec.update_layout(
          margin=dict(t=40, b=20, l=20, r=20), xaxis_tickangle=-30
      )
      st.plotly_chart(fig_dec, use_container_width=True)
    else:
      st.info(
          "No Decline segment records found under current filter selection."
      )

  st.markdown("#### 🚨 At-Risk Customers Table")
  st.caption(
      "Customers meeting early warning criteria: Satisfaction $\\le 2$ or"
      " labeled 'Decline'."
  )

  df_at_risk_table = (
      df_filtered[
          (df_filtered["CustomerSatisfaction"] <= 2)
          | (df_filtered["label"] == "Decline")
      ]
      .groupby(["CustomerID", "label", "CustomerRegion", "RetailChannel"])
      .agg(
          TotalRevenue=("PurchaseAmount", "sum"),
          LastPurchase=("TransactionDate", "max"),
          AvgSatisfaction=("CustomerSatisfaction", "mean"),
      )
      .reset_index()
      .sort_values(by="TotalRevenue", ascending=False)
  )

  st.dataframe(
      df_at_risk_table.style.format({
          "TotalRevenue": "${:,.2f}",
          "AvgSatisfaction": "{:.1f}",
          "LastPurchase": lambda x: x.strftime("%Y-%m-%d"),
      }),
      use_container_width=True,
      hide_index=True,
  )

  # CSV Download button
  csv_data = df_at_risk_table.to_csv(index=False).encode("utf-8")
  st.download_button(
      label="📥 Download At-Risk Customers CSV",
      data=csv_data,
      file_name="nova_retail_at_risk_customers.csv",
      mime="text/csv",
  )

  st.markdown(
      "**Warning Signal Definitions:**\n"
      "1. **Decline Label:** Row-level classifier indicating historical churn"
      " risk or diminishing basket size.\n"
      "2. **Low Satisfaction ($\le 2$):** Indicates immediate service or"
      " product quality friction requiring proactive customer success"
      " outreach."
  )


# ==========================================
# TAB 4: CUSTOMER DEMOGRAPHICS
# ==========================================
with tab4:
  st.markdown("### Customer Demographics & Behavioral Mix")
  st.caption(
      "Analyzing revenue performance and satisfaction across age groups and"
      " gender."
  )

  col_d1, col_d2 = st.columns(2)

  with col_d1:
    st.markdown("#### Revenue by Age Group & Gender")
    age_gender = (
        df_filtered.groupby(["CustomerAgeGroup", "CustomerGender"])[
            "PurchaseAmount"
        ]
        .sum()
        .reset_index()
    )
    fig_age_gen = px.bar(
        age_gender,
        x="CustomerAgeGroup",
        y="PurchaseAmount",
        color="CustomerGender",
        barmode="group",
        title="Revenue by Age Group and Gender",
        labels={
            "CustomerAgeGroup": "Age Group",
            "PurchaseAmount": "Revenue (USD)",
            "CustomerGender": "Gender",
        },
        text_auto="$#,##0",
    )
    fig_age_gen.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_age_gen, use_container_width=True)

  with col_d2:
    st.markdown("#### Segment Mix by Age Group (100% Stacked)")
    age_seg = (
        df_filtered.groupby(["CustomerAgeGroup", "label"])["PurchaseAmount"]
        .sum()
        .reset_index()
    )
    # Calculate percentage for 100% stacked bar
    age_seg["Total"] = age_seg.groupby("CustomerAgeGroup")[
        "PurchaseAmount"
    ].transform("sum")
    age_seg["Percentage"] = (
        age_seg["PurchaseAmount"] / age_seg["Total"] * 100
    )

    fig_100 = px.bar(
        age_seg,
        x="CustomerAgeGroup",
        y="Percentage",
        color="label",
        barmode="stack",
        color_discrete_map=SEGMENT_COLORS,
        title="Segment Proportion Across Age Groups",
        labels={
            "CustomerAgeGroup": "Age Group",
            "Percentage": "Share (%)",
            "label": "Segment",
        },
    )
    fig_100.update_layout(margin=dict(t=40, b=20, l=20, r=20))
    st.plotly_chart(fig_100, use_container_width=True)

  st.caption("Small sample ($n=100$): treat findings as directional.")


# ==========================================
# TAB 5: RECOMMENDATIONS
# ==========================================
with tab5:
  st.markdown("### Strategic Commercial Recommendations (Goal C)")
  st.caption(
      "Actionable commercial and marketing priorities backed by live"
      " filtered metrics for Sophia Martinez."
  )

  # Compute live supporting metrics for recommendations
  growth_promising_rev = df_filtered[
      df_filtered["label"].isin(["Growth", "Promising"])
  ]["PurchaseAmount"].sum()
  decline_share = (
      (decline_rev / total_revenue * 100) if total_revenue > 0 else 0
  )

  recs = [
      {
          "title": (
              "1. Launch Proactive Retention Campaigns for At-Risk Customers"
          ),
          "priority": "High",
          "target": "Decline Segment",
          "metric": (
              f"Decline Segment Revenue at Risk: ${decline_rev:,.2f}"
              f" ({decline_share:.1f}% of total)"
          ),
          "action": (
              "Deploy targeted win-back incentives, VIP discount codes, and"
              " personalized customer success surveys for accounts flagged as"
              " Decline or showing satisfaction $\\le 2$."
          ),
          "outcome": "Retention & Churn Reduction",
      },
      {
          "title": "2. Double Down on High-Margin Growth & Promising Categories",
          "priority": "High",
          "target": "Growth & Promising Segments",
          "metric": (
              "Combined Growth/Promising Revenue:"
              f" ${growth_promising_rev:,.2f}"
          ),
          "action": (
              "Increase digital advertising and inventory allocation for top"
              " performing categories (e.g., Electronics, Clothing & Fashion)"
              " tailored to high-engagement customer cohorts."
          ),
          "outcome": "Revenue Maximization & Upsell",
      },
      {
          "title": "3. Optimize Omnichannel Channel Mix & Cross-Selling",
          "priority": "Medium",
          "target": "Online & Physical Store Customers",
          "metric": f"Online Revenue Share: {pct_online:.1f}%",
          "action": (
              "Implement seamless 'Buy Online, Pick Up In Store' (BOPIS) and"
              " personalized online product recommendations to bridge offline"
              " and online purchasing patterns."
          ),
          "outcome": "Channel Shift & Basket Size Expansion",
      },
      {
          "title": "4. Address Regional Whitespace Opportunities",
          "priority": "Medium",
          "target": "Underperforming Regions",
          "metric": (
              f"Total Filtered Revenue Across {unique_customers} Customers:"
              f" ${total_revenue:,.2f}"
          ),
          "action": (
              "Investigate regional conversion bottlenecks and deploy localized"
              " promotional campaigns in lagging geographic markets."
          ),
          "outcome": "Geographic Penetration",
      },
      {
          "title": "5. Standardize Customer Experience Feedback Loops",
          "priority": "Low",
          "target": "All Segments (Focus on Satisfaction $\\le 2$)",
          "metric": f"Current Average Satisfaction: {avg_satisfaction:.2f} / 5.0",
          "action": (
              "Establish automated post-purchase feedback loops within 48 hours"
              " of transaction to catch dissatisfaction early and resolve"
              " fulfillment issues."
          ),
          "outcome": "Customer Lifetime Value (LTV) Protection",
      },
  ]

  for rec in recs:
    badge_color = (
        "red"
        if rec["priority"] == "High"
        else ("orange" if rec["priority"] == "Medium" else "blue")
    )
    with st.container():
      col_r_a, col_r_b = st.columns([4, 1])
      with col_r_a:
        st.markdown(f"#### {rec['title']}")
        st.markdown(
            f"**Target Segment:** `{rec['target']}` | **Supporting Metric:**"
            f" *{rec['metric']}*"
        )
        st.markdown(f"**Recommended Action:** {rec['action']}")
        st.markdown(f"**Expected Outcome:** 🎯 `{rec['outcome']}`")
      with col_r_b:
        st.markdown(
            f"<br><span"
            f" style='background-color:#e2e8f0;padding:6px 12px;border-radius:4px;font-weight:bold;color:#1e3a8a;'>Priority:"
            f" {rec['priority']}</span>",
            unsafe_allow_html=True,
        )
      st.markdown("---")


# ==========================================
# TAB 6: ASK THE DATA (AI CHATBOT, GROQ)
# ==========================================
with tab6:
  st.markdown("### Ask the Data – AI Customer Intelligence Assistant")
  st.caption(
      "Powered by Groq SDK. Ask questions about the currently filtered dataset."
  )

  # Check API Key
  groq_api_key = None
  try:
    if "GROQ_API_KEY" in st.secrets:
      groq_api_key = st.secrets["GROQ_API_KEY"]
  except Exception:
    pass

  if not groq_api_key:
    st.warning(
        "⚠️ `GROQ_API_KEY` is missing from Streamlit Secrets. To enable the AI"
        " chatbot, add your Groq API key in your app's Streamlit Cloud Settings"
        " -> Secrets as:\n\n```toml\nGROQ_API_KEY = 'your_actual_groq_api_key'\n```"
    )

  # Suggested questions
  st.markdown("**💡 Suggested Questions (Click to copy & paste below):**")
  sug_cols = st.columns(2)
  with sug_cols[0]:
    st.markdown("• *Which segment is most at risk and why?*")
    st.markdown("• *Where should we invest to grow revenue?*")
  with sug_cols[1]:
    st.markdown(
        "• *What is the average satisfaction in the Growth segment?*"
    )
    st.markdown("• *Which product category drives the highest sales?*")

  st.markdown("---")

  # Initialize chat history
  if "messages" not in st.session_state:
    st.session_state.messages = []

  # Clear chat button
  if st.button("🗑️ Clear Chat History"):
    st.session_state.messages = []
    st.rerun()

  # Display chat messages
  for message in st.session_state.messages:
    with st.chat_message(message["role"]):
      st.markdown(message["content"])

  # Build grounding context from filtered data
  context_summary = f"""
  CURRENT DATASET FILTER SLICE (NovaRetail):
  - Filtered Rows: {len(df_filtered)} out of {len(df_raw)} total rows
  - Total Revenue: ${total_revenue:,.2f}
  - Unique Customers: {unique_customers}
  - Revenue per Customer: ${revenue_per_customer:,.2f}
  - Avg Line-Item Value: ${avg_line_item:,.2f} - Avg Satisfaction: {avg_satisfaction:.2f} / 5.0 - Online Revenue Share: {pct_online:.1f}\% - Decline Segment Revenue (Risk):${decline_rev:,.2f} ({pct_decline:.1f}%)

  REVENUE BY SEGMENT:
  {df_filtered.groupby('label')['PurchaseAmount'].sum().to_string()}

  REVENUE BY REGION:
  {df_filtered.groupby('CustomerRegion')['PurchaseAmount'].sum().to_string()}

  REVENUE BY CATEGORY:
  {df_filtered.groupby('ProductCategory')['PurchaseAmount'].sum().to_string()}

  REVENUE BY RETAIL CHANNEL:
  {df_filtered.groupby('RetailChannel')['PurchaseAmount'].sum().to_string()}
  """

  # Chat input
  if prompt := st.chat_input(
      "Ask a question about the filtered NovaRetail data..."
  ):
    st.session_state.messages.append({"role": "user", "content": prompt})
    with st.chat_message("user"):
      st.markdown(prompt)

    with st.chat_message("assistant"):
      message_placeholder = st.empty()
      message_placeholder.markdown("Analyzing filtered data...")

      if not groq_api_key or Groq is None:
        error_msg = (
            "Groq API key is missing or `groq` package is not installed."
            " Please configure `GROQ_API_KEY` in Streamlit Secrets."
        )
        message_placeholder.markdown(error_msg)
        st.session_state.messages.append(
            {"role": "assistant", "content": error_msg}
        )
      else:
        # Try models with automatic fallback
        response_text = None
        used_model = None

        system_prompt = (
            "You are an expert business analytics consultant for NovaRetail."
            " You are assisting Sophia Martinez, Director of Customer"
            " Intelligence. Answer the user's question STRICTLY based on the"
            " provided dataset context below. If the answer cannot be found in"
            " the context, state clearly that the data does not contain this"
            " information. Always quote exact numbers where applicable, mention"
            " the small sample size ($n=100$) caveat when giving strategic"
            " advice, and keep answers concise and professional.\n\nDATA CONTEXT:\n"
            + context_summary
        )

        for model_name in ST_MODELS:
          try:
            client = Groq(api_key=groq_api_key)
            completion = client.chat.completions.create(
                model=model_name,
                messages=[
                    {"role": "system", "content": system_prompt},
                    {
                        "role": "user",
                        "content": prompt,
                    },
                ],
                temperature=0.3,
                max_tokens=800,
            )
            response_text = completion.choices[0].message.content
            used_model = model_name
            break
          except Exception as ex:
            continue

        if response_text:
          full_response = (
              f"{response_text}\n\n*— Answered via Groq model: `{used_model}`*"
          )
          message_placeholder.markdown(full_response)
          st.session_state.messages.append(
              {"role": "assistant", "content": full_response}
          )
        else:
          err_fallback = (
              "⚠️ Failed to connect to Groq API across all fallback models."
              " Please check your API key and network connection."
          )
          message_placeholder.markdown(err_fallback)
          st.session_state.messages.append(
              {"role": "assistant", "content": err_fallback}
          )
