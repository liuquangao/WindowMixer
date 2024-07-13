export CUDA_VISIBLE_DEVICES=0
seq_len=336
model_name=WindowMixer
e_layers=3
individual=1
moving_avg=25

# Multivariate
for pred_len in 96 192 336 720
do
python -u run.py \
  --task_name long_term_forecast \
  --is_training 1 \
  --root_path ./dataset/ETT-small/ \
  --data_path ETTh1.csv \
  --model_id ETTh1_$seq_len'_'$pred_len \
  --model $model_name \
  --data ETTh1 \
  --features M \
  --seq_len $seq_len \
  --pred_len $pred_len \
  --enc_in 7 \
  --dec_in 7 \
  --c_out 7 \
  --des 'Exp' \
  --e_layers $e_layers\
  --moving_avg $moving_avg\
  --w_size 8\
  --d_model 128\
  --activation 'tanh'\
  --individual $individual\
  --itr 1 --train_epochs 30 --patience 5 --batch_size 128 --learning_rate 0.0001
done
