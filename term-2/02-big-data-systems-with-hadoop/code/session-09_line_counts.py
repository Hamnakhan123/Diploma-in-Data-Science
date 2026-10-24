versions = {"MapReduce (Java)": 60, "Pig Latin": 4, "HiveQL / SQL": 1, "PySpark DataFrame": 1}
for lang, lines in versions.items():
    print(f"{lang:<18} {'#' * lines if lines < 40 else '#' * 40 + ' ...'} {lines}")
