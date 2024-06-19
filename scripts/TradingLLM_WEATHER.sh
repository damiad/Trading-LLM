model_name=TradingLLM
train_epochs=100
learning_rate=0.02
llama_layers=32

master_port=1234
num_process=1
batch_size=4 #24
d_model=32
d_ff=128

seq_len=100
pred_len=5
seq_step=1
cg_value=5

comment="ending-${pred_len}by${seq_step}-exp16-train"

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/myexp/ \
	--data_path london_weather.csv \
	--model_id WEATHER \
	--model $model_name \
	--data weather \
	--seq_len $seq_len \
	--label_len 0 \
	--pred_len $pred_len \
	--seq_step $seq_step \
	--target 'mean_temp' \
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
