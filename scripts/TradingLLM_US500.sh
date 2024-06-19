model_name=TradingLLM
train_epochs=100
learning_rate=0.002
llama_layers=32

master_port=1234
num_process=1
batch_size=4 #24
d_model=32
d_ff=128
num_entries=20000
seq_len=100
pred_len=15
seq_step=8
cg_value=15


comment="${num_entries}-ending-${pred_len}by${seq_step}-exp23-test"

python3 dataset/myexp/cut.py us500 head $num_entries

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/myexp/ \
	--data_path output.csv \
	--model_id US500 \
	--model $model_name \
	--data us500 \
	--seq_len $seq_len \
	--label_len 0 \
	--pred_len $pred_len \
	--seq_step $seq_step \
	--target 'close' \
	--itr 1 \
	--d_model $d_model \
	--d_ff $d_ff \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--llm_layers $llama_layers \
	--train_epochs $train_epochs \
	--model_comment $comment \
	--lradj 'type3' \
	--cg_value $cg_value \
	--patience 20

# patience add
