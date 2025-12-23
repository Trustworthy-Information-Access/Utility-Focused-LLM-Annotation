# Utility Annotation via LLM 
## Pipeline
1. Relevance-based selection
   
   ```
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.jsonl"
   output_dir="results/annotation_relevance.jsonl"
   mkdir $output_dir
   python relevance_selection.py --model_name_or_path $model_name_or_path --dataset_path $dataset_path  --train_group_size $train_group_size   --batch_size 4096 --output_dir $output_dir
   ``` 


3. pseudo-answer generation
   Process the ``annotation_relevance.jsonl'' get the annotation_candidate_final.jsonl using the precoee.ipynb
   ```
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.jsonl"
   passages_corpus="results/annotation_candidate.jsonl"
   relevence_file_path="results/annotation_candidate_final.jsonl"
   output_dir="results/annotation_answer.jsonl"
   mkdir $output_dir
   train_group_size=16
   python pseudo_answer.py --model_name_or_path $model_name_or_path --passages_corpus $passages_corpus --relevence_file_path $relevence_file_path  --dataset_path $dataset_path --train_group_size $train_group_size  --batch_size 4096 --output_dir $output_dir
   ```
   
   
5. utility-based selection

   ```
   model_name_or_path="models/Qwen/Qwen3-32B/"
   dataset_path="results/annotation_candidate.jsonl"
   passages_corpus="results/annotation_candidate.jsonl"
   relevence_file_path="results/annotation_candidate_final.jsonl"
   answer_file_path="results/annotation_answer.jsonl"
   output_dir="results/annotation_utility.jsonl"
   mkdir $output_dir
   train_group_size=16
   python utility_selection.py --model_name_or_path $model_name_or_path --passages_corpus $passages_corpus --relevence_file_path $relevence_file_path --answer_file_path $answer_file_path  --dataset_path $dataset_path --train_group_size $train_group_size --batch_size 4096 --output_dir $output_dir
   ```

