#! /bin/bash

lambda_1_vars=(1.0 0.5 0.0)
lambda_2_vars=(1.0 0.5 0.0)

for lambda_1 in ${lambda_1_vars[@]}
do
	for lambda_2 in ${lambda_2_vars[@]}
	do
		echo $lambda_1
		echo $lambda_2
		python3 bert_ee_lm.py \
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
			--output_dir=./output/lm/result_dir/$lambda_1/$lambda_2 \
			--gpu_device=0 \
			--root_weight=$lambda_1 \
			--coarse_weight=$lambda_2

		python evaluation/evaluate_no_bio.py ./output/lm/result_dir/$lambda_1/$lambda_2/label_test.txt
	done
done


# perl ./evaluation/conlleval.pl -d '\t' < ./output/lm/result_dir/$lambda_1/$lambda_2/label_test.txt
