# Utility Annotation via LLM 
## Pipeline
1. Relevance-based selection
   
   ```
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.json"
   output_dir="results/annotation_relevance.jsonl"
   mkdir $output_dir
   train_group_size=16
   python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --train_group_size $train_group_size --batch_size 4096 --output_dir $output_dir
   ``` 


3. pseudo-answer generation
   
   ```
   Change the function "EncodeDataset(Dataset)" in the dataset.py  and "llm-label-utility.py".  Then, 
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.json"
   output_dir="results/annotation_answers.jsonl"
   mkdir $output_dir
   train_group_size=16
   python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --train_group_size $train_group_size --batch_size 4096 --output_dir $output_dir
   ```
   
   
5. utility-based selection

   ```
   Change the function "EncodeDataset(Dataset)", "get_answers()" in the dataset.py and "llm-label-utility.py". Then, 
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.json"
   output_dir="results/annotation_utility.jsonl"
   mkdir $output_dir
   train_group_size=16
   python llm-label-utility.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path --train_group_size $train_group_size --batch_size 4096 --output_dir $output_dir
   ```

