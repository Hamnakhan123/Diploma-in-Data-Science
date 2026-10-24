import io
import pandas as pd

csv = """attendance,assignments,shift,passed
92,8,morning,yes
85,7,evening,yes
60,3,evening,no
78,6,morning,yes
55,2,morning,no
70,5,evening,yes
48,4,evening,no
88,9,morning,yes
65,,morning,no
95,10,evening,yes
72,6,evening,yes
58,3,morning,no"""
df = pd.read_csv(io.StringIO(csv))

print("Shape:", df.shape)
print("\nMissing values per column:")
print(df.isna().sum())
print("\nNumeric summary:")
print(df.describe().round(2))
print("\nClass balance:")
print(df["passed"].value_counts())
print("\nMean attendance by outcome:")
print(df.groupby("passed")["attendance"].mean().round(1))
