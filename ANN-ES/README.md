## Use ANN-ES

### 1. Create conda environment and install requirements

(optional) It might be a good idea to use a separate conda environment. It can be created by running:
```
conda create -n annes37 -y python=3.7 && conda activate annes37
pip install -r requirements.txt
```

### 2. train ANN-ES
```console
python train_biencoder_sumy.py \
      --eval_interval=10000 \
	  --document_path=data/zeshel/documents \
	  --data_path /data/zeshel/blink_format_sumy \
	  --output_path models/zeshel/biencoder_base \
	  --learning_rate 1e-05 \
	  --num_train_epochs 1 \
	  --max_context_length 128 \
	  --max_cand_length 128 \
	  --train_batch_size 1 \
	  --eval_batch_size 64 \
	  --shuffle True \
	  --bert_model bert-base-uncased \
	  --type_optimization all_encoder_layers \
	  --k 63 \
	  --encoding_batch_size 256 \
	  --data_parallel
```

### 3. eval ANN-ES
```console
python eval_biencoder \
      --path_to_model model_path/pytorch_model.bin \
      --data_path data/zeshel/blink_format_sumy \
	  --cand_pool_path data/zeshel/blink_format_sumy/entity_dict_tensor.pkl \
	  --cand_encode_path model_path/cand_embeddings.t7 \
      --output_path output_path \
      --encode_batch_size 256 \
	  --eval_batch_size 1 \
	  --top_k 64 \
	  --save_topk_result \
      --bert_model bert-base-uncased \
	  --mode test,valid \
      --zeshel \
	  --data_parallel
```

## Troubleshooting

If the module cannot be found, preface the python command with `PYTHONPATH=.`

## License
ANN-ES is MIT licensed. See the [LICENSE](LICENSE) file for details.
