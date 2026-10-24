#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.filters.unsupervised.attribute.StringToWordVector -i reviews.arff -C -L -T -I \
  -stopwords-handler weka.core.stopwords.Rainbow -W 1000 > vec.arff
echo "Attributes created: $(grep -c '^@attribute' vec.arff)"
grep '^@attribute' vec.arff | head -12
