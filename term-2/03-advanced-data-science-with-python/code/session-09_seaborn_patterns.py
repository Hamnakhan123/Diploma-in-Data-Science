"""Seaborn patterns: hourly x weekday heatmap, violin plots by group, and a pairplot."""
import numpy as np
import pandas as pd
import seaborn as sns
import matplotlib.pyplot as plt

rng = np.random.default_rng(4)
n = 4000
df = pd.DataFrame({"hour": rng.choice(range(7, 23), n, p=np.r_[[2, 4, 6, 5, 4, 6, 8, 6, 5, 6, 8, 9, 10, 9, 7, 5]] / 100),
                   "weekday": rng.choice(["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"], n, p=[.12, .12, .12, .13, .15, .19, .17])})
df["basket_rs"] = (rng.gamma(2.5, 220, n) * np.where(df.weekday.isin(["Sat", "Sun"]), 1.3, 1)).round()
df["items"] = (df.basket_rs / 150 + rng.poisson(1, n)).round().clip(1)
df["customer"] = np.where(rng.random(n) < .4, "member", "walk-in")
df.loc[df.customer == "member", "basket_rs"] *= 1.2

order = ["Mon", "Tue", "Wed", "Thu", "Fri", "Sat", "Sun"]
grid = df.pivot_table(index="weekday", columns="hour", values="basket_rs", aggfunc="size").reindex(order)
day, hour = grid.stack().idxmax()
print(f"busiest weekday-hour: {day} {int(hour)}:00 with {int(grid.values.max())} transactions")
print(df.groupby("customer")["basket_rs"].describe()[["mean", "50%", "max"]].round(0))

sns.set_theme(style="whitegrid")
fig = plt.figure(figsize=(15, 4.5))
ax1 = fig.add_subplot(1, 2, 1)
sns.heatmap(grid, cmap="YlOrRd", ax=ax1, cbar_kws={"label": "transactions"}); ax1.set_title("When do customers shop? (weekday x hour)")
ax2 = fig.add_subplot(1, 2, 2)
sns.violinplot(data=df, x="weekday", y="basket_rs", hue="customer", split=True, order=order, inner="quart", ax=ax2,
               palette=["#FF6B1A", "#A78BFA"])
ax2.set_ylim(0, 2500); ax2.set_title("Basket value by weekday: members vs walk-ins")
fig.tight_layout(); fig.savefig("session-09_seaborn_patterns.png", dpi=100); plt.show()

g = sns.pairplot(df.sample(600, random_state=0)[["basket_rs", "items", "hour", "customer"]], hue="customer",
                 palette=["#FF6B1A", "#A78BFA"], plot_kws={"s": 12, "alpha": .6}, height=2.2)
g.figure.suptitle("Pairplot: relationships between basket value, items and hour", y=1.02)
g.savefig("session-09_pairplot.png", dpi=90); plt.show()
