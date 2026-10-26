"""Build a 2x2 exploratory figure with Matplotlib and Seaborn and save it as a PNG."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

rng = np.random.default_rng(1)
n = 400
df = pd.DataFrame({
    "area_sqft": rng.normal(1400, 400, n).clip(500, 3000).round(),
    "city": rng.choice(["Karachi", "Lahore", "Islamabad"], n),
})
city_premium = df["city"].map({"Karachi": 1.0, "Lahore": 0.9, "Islamabad": 1.2})
df["price_lakh"] = (0.11 * df["area_sqft"] * city_premium + rng.normal(0, 20, n)).round(1)

sns.set_theme(style="whitegrid")
fig, axes = plt.subplots(2, 2, figsize=(11, 8))
sns.histplot(df, x="price_lakh", bins=30, kde=True, ax=axes[0, 0]).set_title("Distribution of price")
sns.scatterplot(df, x="area_sqft", y="price_lakh", hue="city", alpha=.7, ax=axes[0, 1]).set_title("Area vs price")
sns.boxplot(df, x="city", y="price_lakh", ax=axes[1, 0]).set_title("Price by city")
sns.heatmap(df[["area_sqft", "price_lakh"]].corr(), annot=True, cmap="rocket_r", ax=axes[1, 1]).set_title("Correlation")
fig.suptitle("Exploratory data analysis of 400 house listings", fontsize=14)
fig.tight_layout()
fig.savefig("session-01_eda_dashboard.png", dpi=110)
plt.show()

print(df.groupby("city")["price_lakh"].describe()[["count", "mean", "50%", "max"]].round(1))
