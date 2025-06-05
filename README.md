# Towards Unsupervised Deep Graph Structure Learning

This is the source code of WWW-2022 paper "[Towards Unsupervised Deep Graph Structure Learning](https://arxiv.org/pdf/2201.06367.pdf)" (SUBLIME). 

![The proposed framework](pipeline.png)


## INSTALL
```
chmod +x install.sh
./install.sh
```
This will create a conda env and install everything that you need. You can activate it with the following command:
```sh
conda activate OpenGSL
```

## TRAIN

Here is a sample training command:

```sh
python train_person_data.py -dataset neurolake_target_train_50k_v8_1k.csv -sparse 1 \
  -annotated_dataset neurolake_target_train_50k_v8.csv \
  -annotation_column target_value -annotation_loss_weight 10.0 \
  -relationship_dataset neurolake_relation_target_train_42k_v8.csv \
  --use_geo_graph 1 --latitude_col ifdmcpf004c00v02 --longitude_col ifdmcpf005c00v02 \
  --geo_k 30 \
  --geo_max_weight 10.0 \
  --geo_min_weight 0.1 \
  -drop_columns_file drop_columns.csv \
  -classification_head_layers 1 \
  -epochs 11500 -lr 0.001 -hidden_dim 256 -proj_dim 768 -rep_dim 48 \
  -dropout 0.15 -nlayers 1 -eval_freq 5 \
  -maskfeat_rate_learner 0.03 -maskfeat_rate_anchor 0 \
  --embedding_only_epochs 100000 \
  --graph_learner_only_epochs 2 \
  -contrast_batch_size 7000 \
  -dropedge_rate 0 -k 30 -gamma 0.65 \
  -use_layer_norm 1 -use_residual 0 \
  -w_decay 0 -type_learner mlp -use_one_cycle 1 -use_arcface 0 -grad_accumulation_steps 1 \
  -arcface_num_samples 4000 -use_sampled_arcface 1 \
  -checkpoint_dir "${BASE_OUTPUT_DIR}/config165geok30w10" \
  --wandb_experiment_name config165geok30w10 \
  -checkpoint_freq 60000 \
  --eval_xgb_freq 500 \
  -output_dir "saved_models/config165geok30w10" > training_logs/config165geok30w10.log 2>&1
```


## Cite
If you compare with, build on, or use aspects of SUBLIME framework, please cite the following:
```
@inproceedings{liu2022towards,
  title={Towards unsupervised deep graph structure learning},
  author={Liu, Yixin and Zheng, Yu and Zhang, Daokun and Chen, Hongxu and Peng, Hao and Pan, Shirui},
  booktitle={Proceedings of the ACM Web Conference 2022},
  pages={1392--1403},
  year={2022}
}
```
