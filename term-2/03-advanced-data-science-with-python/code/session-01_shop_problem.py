"""Real-life problem: clean a shop's sales log and turn it into a weekly stock plan."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

rng = np.random.default_rng(2026)
products = {"Milk 1L": 280, "Bread": 180, "Eggs (dozen)": 420, "Tea 200g": 450, "Sugar 1kg": 160}
base = {"Milk 1L": 40, "Bread": 30, "Eggs (dozen)": 15, "Tea 200g": 6, "Sugar 1kg": 10}
weekday_boost = np.array([1.0, 0.9, 0.9, 1.0, 1.3, 1.5, 1.2])        # Mon..Sun, weekends busier
rows = []
for day in pd.date_range("2026-07-06", periods=84):
    for p, mu in base.items():
        rows.append((day, p, rng.poisson(mu * weekday_boost[day.dayofweek])))
log = pd.DataFrame(rows, columns=["date", "product", "qty"])
# inject real-world dirt: duplicated rows and a typo quantity
log = pd.concat([log, log.sample(15, random_state=1)]).reset_index(drop=True)
log.loc[10, "qty"] = 999

# 1. Clean
before = len(log)
log = log.drop_duplicates()
log = log[log["qty"].between(0, 200)]
print(f"rows: {before} -> {len(log)} after removing duplicates and impossible quantities")

# 2. Revenue by product
log["revenue"] = log["qty"] * log["product"].map(products)
by_product = log.groupby("product")["revenue"].sum().sort_values(ascending=False)
print("\nRevenue over 12 weeks (Rs):\n", by_product.to_string())

# 3. Busiest weekdays
log["weekday"] = log["date"].dt.day_name().str[:3]
order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
by_day = log.groupby("weekday")["revenue"].mean().reindex(order) * len(products)
print("\nAverage daily revenue by weekday (Rs):\n", by_day.round(0).to_string())

# 4. Weekly stock plan = mean + 1.65 * std of weekly units
weekly = log.groupby([pd.Grouper(key="date", freq="W-SUN"), "product"])["qty"].sum().unstack()
plan = pd.DataFrame({"avg_week": weekly.mean().round(0), "std": weekly.std().round(1)})
plan["stock_to_keep"] = np.ceil(plan["avg_week"] + 1.65 * plan["std"]).astype(int)
print("\nWeekly stock plan:\n", plan.to_string())

fig, ax = plt.subplots(1, 2, figsize=(11, 4))
by_product.plot.barh(ax=ax[0], color="#FF6B1A"); ax[0].set_title("Revenue by product (12 weeks)"); ax[0].set_xlabel("Rs")
by_day.plot.bar(ax=ax[1], color="#A78BFA"); ax[1].set_title("Average daily revenue by weekday"); ax[1].set_ylabel("Rs")
fig.tight_layout(); fig.savefig("session-01_shop_insights.png", dpi=110); plt.show()
