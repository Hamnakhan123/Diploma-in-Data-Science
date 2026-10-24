#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.clusterers.EM -t customers.arff -N -1 -S 100 | grep -E "Number of clusters selected|^Clustered Instances|^[0-9]+ +[0-9]+ \(|Log likelihood"
