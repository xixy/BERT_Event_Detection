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