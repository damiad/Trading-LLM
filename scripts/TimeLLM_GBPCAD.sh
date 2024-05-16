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
seq_len=40
pred_len=6
seq_step=5
cg_value=6

<<<<<<< HEAD
comment="${num_entries}-ending-${pred_len}by${seq_step}"
=======
comment='${num_entries}-ending-5by2'
>>>>>>> 9e80016c18ec88e8f5c05aeb72331344499109d4

python3 dataset/currencies/cut.py gbpcad tail $num_entries

accelerate launch --mixed_precision bf16 --num_processes $num_process --main_process_port $master_port run_main.py \
	--root_path ./dataset/currencies/ \
	--data_path output.csv \
	--model_id GBPCAD \
	--model $model_name \
	--data gbpcad \
<<<<<<< HEAD
	--seq_len $seq_len \
	--label_len 0 \
	--pred_len $pred_len \
	--seq_step $seq_step \
=======
	--seq_len 18 \
	--label_len 0 \
	--pred_len 5 \
	--seq_step 2 \
>>>>>>> 9e80016c18ec88e8f5c05aeb72331344499109d4
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
<<<<<<< HEAD
	--cg_value $cg_value \
	--patience 10
=======
	--cg_value 5 \
	--patience 20
>>>>>>> 9e80016c18ec88e8f5c05aeb72331344499109d4

# patience add
