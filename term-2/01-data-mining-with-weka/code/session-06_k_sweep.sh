#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
for k in 2 3 4 5 6; do
  printf "k=%s  " "$k"
  weka weka.clusterers.SimpleKMeans -t customers.arff -N $k -S 10 -init 1 | grep "Within cluster sum" | awk '{print "SSE =", $NF}'
done
