#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.associations.Apriori -t shop.arff -N 5 -T 1 -C 1.2 -M 0.2 | sed -n '/Best rules/,$p'
