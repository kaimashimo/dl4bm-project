#!/bin/bash

# Define the methods and datasets as arrays
methods=("protonet" "maml" "matchingnet" "baseline" "baseline_pp")
datasets=("swissprot" "tabula_muris")
n_shots=("1" "5" "10")
n_ways=("5" "20")
# Loop through each combination of method and dataset
for method in "${methods[@]}"; do
    for dataset in "${datasets[@]}"; do
	for n_shot in "${n_shots[@]}"; do
	    for n_way in "${n_ways[@]}"; do
        	echo "Running with method: $method, dataset: $dataset, n_shot: $n_shot, n_way: $n_way"
        	exp_name=${method}_${dataset}_${n_shot}_${n_way}
		python run.py exp.name=$exp_name method=$method dataset=$dataset n_shot=$n_shot n_way=$n_way
	    done
        done	
    done
done
