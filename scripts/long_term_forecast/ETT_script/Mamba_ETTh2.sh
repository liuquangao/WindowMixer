model_name=Mamba

for pred_len in 96 192 336 720
do

python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh2.csv \
  --model_id ETTh2_$pred_len'_'$pred_len \
  --model $model_name \
  --data ETTh2 \
  --features M \
  --seq_len $pred_len \
  --label_len 48 \
  --pred_len $pred_len \
  --e_layers 2 \
  --enc_in 7 \
  --expand 2 \
  --d_ff 16 \
  --d_conv 4 \
  --c_out 7 \
  --d_model 128 \
  --des 'Exp' \
  --itr 1 \

done