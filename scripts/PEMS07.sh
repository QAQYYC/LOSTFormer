export CUDA_VISIBLE_DEVICES=0

model_name=LOSTFormer


python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --learning_rate 0.001 \
    --train_epochs 50 \
    --patience 10 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id LOSTFormer_PEMS07_48_48 \
    --model $model_name \
    --data PEMS07 \
    --features M \
    --seq_len 48 \
    --label_len 0 \
    --pred_len 48 \
    --batch_size 16 \
    --d_layers 3 \
    --factor 3 \
    --num_nodes 883 \
    --channel 1 \
    --slice_size_per_day 288 \
    --enc_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --n_heads 2 \
    --d_model 64 \
    --d_ff 152 \
    --lradj cosine \
    --itr 1 | tee logs/$model_name/PEMS07_48_48.log
#
python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --learning_rate 0.001 \
    --train_epochs 50 \
    --patience 10 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id LOSTFormer_PEMS07_60_60 \
    --model $model_name \
    --data PEMS07 \
    --features M \
    --seq_len 60 \
    --label_len 0 \
    --pred_len 60 \
    --batch_size 16 \
    --d_layers 3 \
    --factor 3 \
    --num_nodes 883 \
    --channel 1 \
    --slice_size_per_day 288 \
    --enc_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --n_heads 2 \
    --d_model 64 \
    --d_ff 152 \
    --lradj cosine \
    --itr 1 | tee logs/$model_name/PEMS07_60_60.log

python -u run.py \
    --task_name long_term_forecast \
    --is_training 0 \
    --learning_rate 0.001 \
    --train_epochs 50 \
    --patience 10 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id TLM_Linearattn_opt_PEMS07_96_96 \
    --model $model_name \
    --data PEMS07 \
    --features M \
    --seq_len 96 \
    --label_len 0 \
    --pred_len 96 \
    --batch_size 8 \
    --d_layers 3 \
    --factor 3 \
    --num_nodes 883 \
    --channel 1 \
    --slice_size_per_day 288 \
    --enc_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --n_heads 2 \
    --d_model 64 \
    --d_ff 152 \
    --lradj cosine \
    --itr 1 | tee logs/$model_name/PEMS07_96_96.log

python -u run.py \
    --task_name long_term_forecast \
    --is_training 1 \
    --learning_rate 0.001 \
    --train_epochs 50 \
    --patience 10 \
    --root_path ./dataset/PEMS/ \
    --data_path PEMS07.npz \
    --model_id TLM_Linearattn_opt_PEMS07_192_192 \
    --model $model_name \
    --data PEMS07 \
    --features M \
    --seq_len 192 \
    --label_len 0 \
    --pred_len 192 \
    --batch_size 4 \
    --d_layers 3 \
    --factor 3 \
    --num_nodes 883 \
    --channel 1 \
    --slice_size_per_day 288 \
    --enc_in 883 \
    --c_out 883 \
    --des 'Exp' \
    --n_heads 2 \
    --d_model 64 \
    --d_ff 152 \
    --lradj cosine \
    --itr 1 | tee logs/$model_name/PEMS07_192_192.log



