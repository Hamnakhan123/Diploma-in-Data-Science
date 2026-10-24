#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
for c in functions.LinearRegression trees.M5P trees.REPTree lazy.IBk functions.SMOreg; do
  printf "%-28s" "$c"
  weka weka.classifiers.$c -t houses.arff -x 4 -o -v | grep -E "Correlation coefficient|Mean absolute error" | awk '{printf "%s=%s  ", $1, $NF}'
  echo
done
