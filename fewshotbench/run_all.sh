#!/bin/bash

# Define the methods and datasets as arrays
methods=("protonet" "maml" "matchingnet" "baseline")
datasets=("swissprot" "tabula_muris")

# Loop through each combination of method and dataset
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
        echo "Running with method: $method and dataset: $dataset"
        python run.py exp.name=exp1 method=$method dataset=$dataset
    done
done
