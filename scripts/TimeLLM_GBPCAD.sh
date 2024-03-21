model_name=TradingLLM
train_epochs=40
learning_rate=0.01
llama_layers=32

master_port=1234
num_process=1
batch_size=6 #24
d_model=32
d_ff=128

comment='TimeLLM-GBPCAD'

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/ETT-small/ \
	--data_path output.csv \
	--model_id GBPCAD \
	--model $model_name \
	--data gbpcad \
	--features M \
	--seq_len 512 \
	--label_len 48 \
	--pred_len 96 \
	--factor 3 \
	--target 'close' \
	--enc_in 7 \
	--dec_in 7 \
	--c_out 7 \
	--itr 1 \
	--d_model $d_model \
	--d_ff $d_ff \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--llm_layers $llama_layers \
	--train_epochs $train_epochs \
	--model_comment $comment \
	--percent 100

