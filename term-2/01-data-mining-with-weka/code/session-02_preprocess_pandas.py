import pandas as pd
import numpy as np

df = pd.DataFrame({
    "age": [25, 32, 41, 29, np.nan, 36, 23, 45, 27, 33],
    "salary": [45000, np.nan, 120000, 52000, 61000, 88000, 38000, 135000, 47000, 70000],
    "city": ["Karachi", "Lahore", "Karachi", None, "Islamabad", "Lahore", "Karachi", "Islamabad", "Lahore", "Karachi"],
    "experience": [1, 6, 15, 4, 7, 10, 0, 20, 3, np.nan],
    "promoted": ["no", "yes", "yes", "no", "yes", "yes", "no", "yes", "no", "no"],
})
print("Missing before:", df.isna().sum().to_dict())

num = ["age", "salary", "experience"]
df[num] = df[num].fillna(df[num].mean())            # like ReplaceMissingValues
df["city"] = df["city"].fillna(df["city"].mode()[0])
print("Missing after: ", df.isna().sum().to_dict())

norm = (df[num] - df[num].min()) / (df[num].max() - df[num].min())   # like Normalize
print("\nNormalised (first 4 rows):")
print(norm.head(4).round(3))

df["age_band"] = pd.cut(df["age"], bins=3, labels=["young", "middle", "senior"])  # like Discretize
print("\nAge bands:")
print(df[["age", "age_band"]].head(6))

encoded = pd.get_dummies(df[["city"]], prefix="city").astype(int)   # like NominalToBinary
print("\nOne-hot encoded city:")
print(encoded.head(4))
