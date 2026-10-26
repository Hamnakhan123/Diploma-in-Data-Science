"""Advanced pandas: method chaining, groupby-agg, merge, pivot_table and pd.cut."""
import numpy as np
import pandas as pd

rng = np.random.default_rng(7)
n = 1200
sales = pd.DataFrame({
    "date": pd.date_range("2026-01-01", periods=n, freq="7h"),
    "shop_id": rng.integers(1, 7, n),
    "product": rng.choice(["Tea", "Biscuits", "Rice", "Cooking oil"], n, p=[.35, .3, .2, .15]),
    "qty": rng.integers(1, 6, n),
})
shops = pd.DataFrame({"shop_id": range(1, 7),
                      "city": ["Karachi", "Karachi", "Lahore", "Lahore", "Islamabad", "Karachi"]})
price = {"Tea": 450, "Biscuits": 120, "Rice": 380, "Cooking oil": 650}

report = (sales
          .merge(shops, on="shop_id", how="left")                 # add the city column
          .assign(revenue=lambda d: d["qty"] * d["product"].map(price),
                  month=lambda d: d["date"].dt.month_name().str[:3])
          .groupby(["city", "product"], as_index=False)
          .agg(orders=("qty", "size"), units=("qty", "sum"), revenue=("revenue", "sum"))
          .sort_values(["city", "revenue"], ascending=[True, False]))
print(report.to_string(index=False))

# Pivot: cities as rows, products as columns
pivot = (sales.merge(shops, on="shop_id")
         .assign(revenue=lambda d: d["qty"] * d["product"].map(price))
         .pivot_table(index="city", columns="product", values="revenue", aggfunc="sum", margins=True))
print("\nRevenue pivot (Rs):\n", pivot.round(0).astype(int))

# Binning a numeric column into categories
sales["basket"] = pd.cut(sales["qty"], bins=[0, 2, 4, 5], labels=["small", "medium", "large"])
print("\nBasket sizes:\n", sales["basket"].value_counts().sort_index())
print("\nMemory before/after category dtype:",
      sales["product"].memory_usage(deep=True), "->", sales["product"].astype("category").memory_usage(deep=True), "bytes")
