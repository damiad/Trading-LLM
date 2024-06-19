model_name=TradingLLM
train_epochs=40
learning_rate=0.01
llama_layers=32

master_port=1234
num_process=1
batch_size=6 #24
d_model=32
d_ff=128

comment='TradingLLM-GBPCAD'

# Update this path based on where your model is saved
model_path='./checkpoints/long_term_forecast_GBPCAD_TradingLLM_gbpcad_ftM_sl512_ll48_pl96_dm32_nh8_el2_dl1_df128_fc3_ebtimeF_Exp_0-TradingLLM-GBPCAD'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_saved_model.py \
	--root_path ./dataset/ETT-small/ \
	--data_path output.csv \
	--model_id GBPCAD \
	--model $model_name \
	--data gbpcad \
	--seq_len 512 \
	--label_len 48 \
	--pred_len 96 \
	--target 'close' \
	--itr 1 \
	--d_model $d_model \
	--d_ff $d_ff \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--llm_layers $llama_layers \
	--train_epochs $train_epochs \
	--model_checkpoint_path $model_path \
	--model_comment $comment \
	--lradj 'type3'
