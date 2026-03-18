#!/bin/bash
# sleep 6s
# export VLLM_WORKER_MULTIPROC_METHOD=spawn
model_name_or_path="model/llama31-8b-instruct"
dataset_path="output/dev_rank/nq_hotpot_qa/NQ_hotpotQA/models_Qwen_utility_ranking/results/hotpotqa_result_final.jsonl"
output_dir="output/dev_rank/nq_hotpot_qa/NQ_hotpotQA/models_Qwen_utility_ranking/results/hotpotqa_result_final_top5_answer.jsonl"
python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --batch_size 512 --topk 5 --output_dir $output_dir

