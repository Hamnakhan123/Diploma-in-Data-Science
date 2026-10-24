#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.classifiers.functions.Logistic -t churn.arff -no-cv -d churn_logistic.model | sed -n '/Odds Ratios/,/^$/p'
echo "--- predictions for new customers ---"
weka weka.classifiers.functions.Logistic -l churn_logistic.model -T new_customers.arff -classifications weka.classifiers.evaluation.output.prediction.PlainText | sed -n '/inst#/,$p'
