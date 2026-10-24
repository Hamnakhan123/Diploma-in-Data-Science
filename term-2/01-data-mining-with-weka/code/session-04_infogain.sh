#!/usr/bin/env bash
# Requires WEKA: weka() runs java -cp weka.jar
weka() { java -cp "${WEKA_HOME:-.}/weka.jar" "$@"; }
weka weka.attributeSelection.InfoGainAttributeEval -i weather.arff -s "weka.attributeSelection.Ranker" | sed -n '/Ranked attributes/,/Selected/p'
