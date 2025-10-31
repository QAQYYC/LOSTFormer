export CUDA_VISIBLE_DEVICES=0

model_name=TLM
seq_len=96

for pred_len in 96 192 336 720
do
    python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --root_path ./dataset/weather/ \
    --data_path WTH.csv \
    --model_id LOSTFormer_WTH_96_$pred_len \
    --model $model_name \
    --data WTH \
    --features M \
    --train_epochs 50 \
    --patience 10 \
    --learning_rate 0.001 \
    --batch_size 16 \
    --seq_len 96 \
    --label_len 0 \
    --pred_len $pred_len \
    --d_layers 3 \
    --factor 3 \
    --num_nodes 12 \
    --channel 1 \
    --slice_size_per_day 24 \
    --enc_in 12 \
    --c_out 12 \
    --des 'Exp' \
    --n_heads 2 \
    --d_model 64 \
    --d_ff 152 \
    --lradj cosine \
    --itr 1
done
