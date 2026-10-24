#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.clusterers.SimpleKMeans -t weather.arff -N 2 -S 10 -c last | sed -n '/Classes to Clusters/,$p'
