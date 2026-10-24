import pandas as pd
import plotly.express as px
import streamlit as st

st.set_page_config(page_title="Sales dashboard", layout="wide")
df = pd.read_parquet("sales_agg")                       # folder written by Spark (Example 4)

st.sidebar.header("Filters")
cities = st.sidebar.multiselect("City", sorted(df.city.unique()), default=sorted(df.city.unique()))
cats = st.sidebar.multiselect("Category", sorted(df.category.unique()), default=sorted(df.category.unique()))
months = st.sidebar.slider("Months", 1, 12, (1, 12))
f = df[df.city.isin(cities) & df.category.isin(cats) & df.month.between(*months)]

c1, c2, c3 = st.columns(3)
c1.metric("Revenue (Rs)", f"{f.revenue.sum():,.0f}")
c2.metric("Orders", f"{f.orders.sum():,}")
c3.metric("Average order (Rs)", f"{f.revenue.sum() / max(f.orders.sum(), 1):,.0f}")

left, right = st.columns(2)
left.plotly_chart(px.line(f.groupby("month", as_index=False).revenue.sum(), x="month", y="revenue",
                          title="Monthly revenue"), use_container_width=True)
right.plotly_chart(px.bar(f.groupby("city", as_index=False).revenue.sum(), x="city", y="revenue",
                          title="Revenue by city"), use_container_width=True)
st.subheader("Drill-down")
st.dataframe(f.groupby(["city", "category"], as_index=False)[["orders", "revenue"]].sum())
