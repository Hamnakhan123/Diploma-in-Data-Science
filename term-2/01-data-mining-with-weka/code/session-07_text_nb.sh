#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.classifiers.meta.FilteredClassifier -t reviews.arff -x 5 -o \
  -F "weka.filters.unsupervised.attribute.StringToWordVector -C -L -W 1000" \
  -W weka.classifiers.bayes.NaiveBayesMultinomial | sed -n '/Stratified cross-validation/,$p'
