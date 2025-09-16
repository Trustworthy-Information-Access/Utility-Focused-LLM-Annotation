
#!/bin/bash
model_name_or_path="/root/paddlejob/workspace/env_run/model/models/Qwen/Qwen3-32B/"
dataset_path="/root/paddlejob/workspace/env_run/output/dpr-nq/results/annotation_candidate.json"
output_dir="/root/paddlejob/workspace/env_run/output/dpr-nq/results/annotation_relevance.jsonl"
mkdir $output_dir
train_group_size=16
python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --train_group_size $train_group_size --batch_size 4096 --output_dir $output_dir
