#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
printf "%-22s %9s %11s %9s %8s\n" model accuracy recall_yes prec_yes roc
for c in rules.ZeroR trees.J48 bayes.NaiveBayes functions.Logistic trees.RandomForest; do
  weka weka.classifiers.$c -t churn.arff -x 10 -s 1 -o -v > out.txt
  acc=$(grep -m1 "Correctly Classified" out.txt | awk '{print $(NF-1) "%"}')
  grep -m1 -E " yes$" out.txt | awk -v m="$c" -v a="$acc" '{printf "%-22s %9s %11s %9s %8s\n", m, a, $1, $3, $7}'
done
