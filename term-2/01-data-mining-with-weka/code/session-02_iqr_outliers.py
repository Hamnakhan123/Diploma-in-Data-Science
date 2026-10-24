import pandas as pd

marks = pd.Series([62, 70, 68, 75, 71, 66, 69, 300, 73, 5], name="marks")
q1, q3 = marks.quantile([0.25, 0.75])
iqr = q3 - q1
low, high = q1 - 1.5 * iqr, q3 + 1.5 * iqr
print(f"Q1={q1}, Q3={q3}, IQR={iqr}, fences=({low}, {high})")
print("Outliers:", marks[(marks < low) | (marks > high)].tolist())
print("Capped:  ", marks.clip(low, high).tolist())
