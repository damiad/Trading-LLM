model_name=TradingLLM
train_epochs=100
learning_rate=0.02
llama_layers=32

master_port=1234
num_process=1
batch_size=6 #24
d_model=32
d_ff=128
num_entries=40000

comment='${num_entries}-ending-5by2'

python3 dataset/ETT-small/cut.py $num_entries

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/ETT-small/ \
	--data_path output.csv \
	--model_id GBPCAD \
	--model $model_name \
	--data gbpcad \
	--seq_len 18 \
	--label_len 0 \
	--pred_len 5 \
	--seq_step 2 \
	--target 'close' \
	--itr 1 \
	--d_model $d_model \
	--d_ff $d_ff \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--llm_layers $llama_layers \
	--train_epochs $train_epochs \
	--model_comment $comment \
	--lradj 'type1' \
	--cg_value 5 \
	--patience 20

# patience add
