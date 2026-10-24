#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.core.Instances churn.arff | sed -n '/^Relation/,$p'
echo
echo "Churned (yes): $(grep -c ',yes$' churn.arff)   Stayed (no): $(grep -c ',no$' churn.arff)"
