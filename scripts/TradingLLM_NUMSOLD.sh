model_name=TradingLLM
train_epochs=100
learning_rate=0.02
llama_layers=32

master_port=1234
num_process=1
batch_size=8 #24
d_model=32
d_ff=128
num_entries=2000

comment='testing'

python3 dataset/sells/cut.py $num_entries

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/sells/ \
	--data_path output.csv \
	--model_id NUMSOLD \
	--model $model_name \
	--data numsold \
	--seq_len 15 \
	--label_len 0 \
	--pred_len 5 \
	--seq_step  7 \
	--target 'number_sold' \
	--itr 1 \
	--d_model $d_model \
	--d_ff $d_ff \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--llm_layers $llama_layers \
	--train_epochs $train_epochs \
	--model_comment $comment \
	--lradj 'type3' \
	--cg_value 1 \
	--patience 10

# patience add
