#! /bin/bash


python3 run_ee.py \
	--task_name="EE" \
	--do_lower_case=False \
	--do_train=True \
	--do_eval=True \
	--do_predict=True \
	--data_dir=data \
	--vocab_file=cased_L-12_H-768_A-12/vocab.txt \
	--bert_config_file=cased_L-12_H-768_A-12/bert_config.json \
	--init_checkpoint=cased_L-12_H-768_A-12/bert_model.ckpt \
	--max_seq_length=128 \
	--train_batch_size=32 \
	--learning_rate=2e-5 \
	--num_train_epochs=40.0 \
	--output_dir=./output/result_dir \
	--gpu_device=0


perl conlleval.pl -d '\t' < ./output/result_dir/label_test.txt
