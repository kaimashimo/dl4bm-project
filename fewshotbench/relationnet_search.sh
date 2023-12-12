for dataset in 'swissprot' 'tabula_muris'; do
for hidden_size in 256 512 1024; do
  for dropout in 0.25 0.5 0.75; do
    for n_layers in 2 3 4; do
      output_file="experiments/relationnet/${dataset}_${hidden_size}_${dropout}_${n_layers}.txt"
      python run.py exp.name=exp1 method=relationnet dataset=$dataset model.hidden_size=$hidden_size model.dropout=$dropout model.n_layers=$n_layers > $output_file 2>&1 || echo "Run on $dataset with hidden size $hidden_size and dropout $dropout and n_layers $n_layers failed" >> $output_file
    done
  done
done
done
