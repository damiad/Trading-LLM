model_name=TradingLLM
train_epochs=100
learning_rate=0.0001
llama_layers=32

master_port=1234
num_process=1
batch_size=5 #24
d_model=32
d_ff=128
num_entries=10000
seq_len=128
pred_len=5
seq_step=1
# cg_value=20

comment="${num_entries}-ending-${pred_len}by${seq_step}"

python3 dataset/currencies/cut.py gbpcad tail $num_entries

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/stocks/ \
	--data_path AAPL.csv \
	--model_id AAPL \
	--model $model_name \
	--data aapl \
	--seq_len $seq_len \
	--label_len 0 \
	--pred_len $pred_len \
	--seq_step $seq_step \
	--target 'Close' \
	--itr 1 \
	--d_model $d_model \
	--d_ff $d_ff \
	--batch_size $batch_size \
	--learning_rate $learning_rate \
	--llm_layers $llama_layers \
	--train_epochs $train_epochs \
	--model_comment $comment \
	--lradj 'type3' \
	--cg_value $pred_len \
	--patience 20

# patience add
