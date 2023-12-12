for hidden_size in 256 512 1024; do
  for dropout_rate in 0.25 0.5 0.75; do
    for n_layers in 2 3 4; do
      output_file="${hidden_size}_${dropout_rate}_${n_layers}.txt"
      echo "python run.py model.hidden_size=$hidden_size model.dropout_rate=$dropout_rate model.n_layers=$n_layers > $output_file"
    done
  done
done

