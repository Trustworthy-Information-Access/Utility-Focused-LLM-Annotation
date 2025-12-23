# Utility Annotation via LLM 
## Pipeline
1. Relevance-based selection
   
   ```
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.jsonl"
   output_dir="results/annotation_relevance.jsonl"
   mkdir $output_dir
   python relevance_selection.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path   --batch_size 4096 --output_dir $output_dir
   ``` 


3. pseudo-answer generation
   Process the ``annotation_relevance.jsonl'' get the annotation_candidate_relevance.jsonl
   ```
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate_relevance.jsonl"
   output_dir="results/annotation_answer.jsonl"
   mkdir $output_dir
   python pseudo_answer.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path  --batch_size 4096 --output_dir $output_dir
   ```
   
   
5. utility-based selection
   Process the ``annotation_relevance.jsonl'' using the process.ipynb
   ```
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.json"
   output_dir="results/annotation_utility.jsonl"
   mkdir $output_dir
   train_group_size=16
   python utility_selection.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --train_group_size $train_group_size --batch_size 4096 --output_dir $output_dir
   ```

