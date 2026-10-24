import pandas as pd

df = pd.DataFrame({
    "temperature": [30.5, 28.0, 35.2, 22.1],
    "humidity": [70, 65, 80, 50],
    "outlook": ["sunny", "rainy", "sunny", "overcast"],
    "play": ["no", "yes", "no", "yes"],
})

def to_arff(frame, relation):
    lines = [f"@relation {relation}", ""]
    for col in frame.columns:
        if pd.api.types.is_numeric_dtype(frame[col]):
            kind = "numeric"
        else:
            kind = "{" + ",".join(sorted(frame[col].dropna().unique())) + "}"
        lines.append(f"@attribute {col} {kind}")
    lines += ["", "@data"]
    for row in frame.itertuples(index=False):
        lines.append(",".join("?" if pd.isna(v) else str(v) for v in row))
    return "\n".join(lines)

print(to_arff(df, "weather"))
