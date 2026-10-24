#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.clusterers.SimpleKMeans -t customers.arff -N 4 -S 10 -init 1
