for dataset in 'swissprot' 'tabula_muris'; do
for hidden_size in 64 128 256 512; do
  for dropout in 0 0.25 0.5 0.75; do
    for n_layers in 2 3 4; do
      output_file="${dataset}_${hidden_size}_${dropout}_${n_layers}.txt"
      exp_name=hyperparamrelationnet_${dataset}_hs${hidden_size}_d${dropout}_nl${n_layers}
      python run.py exp.name=$exp_name method=relationnet dataset=$dataset model_params.hidden_size=$hidden_size mmodel_params.dropout=$dropout model_params.n_layers=$n_layers > $output_file 2>&1 || echo "Run on $dataset with hidden size $hidden_size and dropout $dropout and n_layers $n_layers failed" >> $output_file
    done
  done
done
done
