import pandas as pd

log = pd.DataFrame([
    ("u1", "09:00", "/home"), ("u1", "09:01", "/programs"), ("u1", "09:03", "/programs/data-science"), ("u1", "09:06", "/admissions"),
    ("u2", "09:02", "/home"), ("u2", "09:02", "/fees"),
    ("u3", "09:05", "/news"), ("u3", "09:06", "/home"), ("u3", "09:07", "/programs"), ("u3", "09:09", "/programs/data-science"), ("u3", "09:12", "/admissions"), ("u3", "09:15", "/admissions/apply"),
    ("u4", "09:10", "/home"), ("u4", "09:11", "/programs"),
    ("u5", "09:20", "/programs/data-science"), ("u5", "09:24", "/admissions"), ("u5", "09:27", "/admissions/apply"),
], columns=["user", "time", "page"])

print("Page views:")
print(log["page"].value_counts().to_string())

entry = log.groupby("user")["page"].first().value_counts()
exitp = log.groupby("user")["page"].last().value_counts()
print("\nEntry pages:", entry.to_dict())
print("Exit pages: ", exitp.to_dict())

log["next"] = log.groupby("user")["page"].shift(-1)
paths = log.dropna().groupby(["page", "next"]).size().sort_values(ascending=False)
print("\nMost common page-to-page moves:")
print(paths.head(4).to_string())

applied = log.groupby("user")["page"].apply(lambda s: "/admissions/apply" in set(s))
print(f"\nConversion to application: {applied.mean():.0%} of {len(applied)} visitors")
