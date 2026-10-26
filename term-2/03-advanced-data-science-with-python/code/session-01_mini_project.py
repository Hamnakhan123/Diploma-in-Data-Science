"""Mini project: automatic data profile and ML-type recommendation for any DataFrame."""
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.datasets import load_diabetes

def profile(df: pd.DataFrame, target: str | None = None) -> None:
    print(f"Rows: {len(df)}  Columns: {df.shape[1]}")
    info = pd.DataFrame({"dtype": df.dtypes.astype(str), "missing": df.isna().sum(),
                         "unique": df.nunique()})
    print(info.to_string())
    print("\nNumeric summary:\n", df.describe().T[["mean", "std", "min", "max"]].round(3).to_string())
    if target is None:
        print("\nRecommendation: no target column -> unsupervised learning (clustering, PCA).")
    elif df[target].nunique() <= 10:
        print(f"\nRecommendation: target '{target}' has {df[target].nunique()} classes -> classification.")
    else:
        print(f"\nRecommendation: target '{target}' is numeric with many values -> regression.")

def histograms(df: pd.DataFrame, path: str) -> None:
    num = df.select_dtypes("number").columns
    cols = 4; rows = int(np.ceil(len(num) / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(12, 2.6 * rows))
    for ax, c in zip(axes.ravel(), num):
        ax.hist(df[c], bins=25, color="#FF6B1A", edgecolor="white"); ax.set_title(c)
    for ax in axes.ravel()[len(num):]:
        ax.axis("off")
    fig.tight_layout(); fig.savefig(path, dpi=100); plt.show()

data = load_diabetes(as_frame=True).frame        # 442 patients, 10 features, disease progression target
profile(data, target="target")
histograms(data, "session-01_mini_project_profile.png")
