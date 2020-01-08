# BERT-Fine-Tune for Event Detection

## Folder Description:
```
BERT_Event_Detection
|____ bert                          # need git from [here](https://github.com/google-research/bert)
|____ cased_L-12_H-768_A-12	    # need download from [here](https://storage.googleapis.com/bert_models/2018_10_18/cased_L-12_H-768_A-12.zip)
|____ data		            # train data
|____ middle_data	            # middle data (label id map)
|____ output			    # output (final model, predict results)
|____ run_ee.py		    # mian code
|____ run_ee.sh		    # run model and eval result
|____ evaluation		    # evaluation code
	|____ conlleval.pl		    # eval code for BIO-stype data
	|____ evaluate_no_bio.py   	# run model and eval result

```

### Usage:
```
bash run_ee.sh
```

### What's in run_ner.sh:
```
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


perl ./evaluation/conlleval.pl -d '\t' < ./output/result_dir/label_test.txt

python evaluate_no_bio.py
```