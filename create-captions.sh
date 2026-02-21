#!/bin/bash
cd C:/projects/oykh-temp/kaggle-dataset-v2/images
for f in *.jpg; do
  echo "OYKHCHAR minimalist white stick figure character, 2.5D cell-shaded, simple black dot eyes with white reflections" > "${f%.jpg}.txt"
done
ls *.txt | wc -l
