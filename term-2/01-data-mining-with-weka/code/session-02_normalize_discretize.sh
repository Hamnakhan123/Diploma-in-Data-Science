#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.filters.unsupervised.attribute.ReplaceMissingValues -i employees.arff -o step1.arff
weka weka.filters.unsupervised.attribute.Normalize -i step1.arff -o step2.arff
weka weka.filters.unsupervised.attribute.Discretize -B 3 -R 1 -i step1.arff -o step3.arff
echo "--- normalised (first 4 rows) ---"
grep -A 5 "@data" step2.arff | tail -4
echo "--- age discretised into 3 bins ---"
grep "@attribute age" step3.arff
