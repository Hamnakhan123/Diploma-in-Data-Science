#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.filters.unsupervised.attribute.ReplaceMissingValues -i employees.arff -o filled.arff
grep -A 12 "@data" filled.arff
