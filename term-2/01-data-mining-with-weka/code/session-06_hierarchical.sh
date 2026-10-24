#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.clusterers.HierarchicalClusterer -t customers.arff -N 4 -L WARD | grep -A6 "^Clustered Instances"
