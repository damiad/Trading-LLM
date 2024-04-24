model_name=TradingLLM
train_epochs=100
learning_rate=0.02
llama_layers=32

master_port=1234
num_process=1
batch_size=4 #24
d_model=32
d_ff=128
num_entries=20000
seq_len=100
pred_len=6
seq_step=5
cg_value=6

comment="${num_entries}-ending-${pred_len}by${seq_step}"

python3 dataset/currencies/cut.py eurusd tail $num_entries

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/currencies/ \
	--data_path output.csv \
	--model_id EURUSD \
	--model $model_name \
	--data eurusd \
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
	--lradj 'type1' \
	--cg_value $cg_value \
	--patience 10

# patience add
