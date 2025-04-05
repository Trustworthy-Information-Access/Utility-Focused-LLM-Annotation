model_path="model/retromae"
echo "====== retromae_bn128epoch4"
corpus_file="examples/retriever/msmarco/data/BertTokenizer_data/corpus"
train_query_file="examples/retriever/msmarco/data/BertTokenizer_data/train_query"
train_qrels="examples/retriever/msmarco/data/BertTokenizer_data/postive_real.tsv"
neg_file="examples/retriever/msmarco/data/BertTokenizer_data/train_negs.tsv"
teacher_score="teacher_score/score_v2_binray.txt"
output_model="./models/retromae_bn128epoch4_l3"
mkdir $output_model # --stop_after_n_steps 1000 \
torchrun --nproc_per_node 8  --master_port 12349  \
    -m bi_encoder.run \
    --output_dir $output_model \
    --model_name_or_path $model_path \
    --do_train  \
    --corpus_file $corpus_file \
    --train_query_file $train_query_file \
    --train_qrels $train_qrels \
    --neg_file $neg_file \
    --query_max_len 32 \
    --passage_max_len 140 \
    --fp16  \
    --per_device_train_batch_size 16 \
    --train_group_size 16 \
    --teacher_score $teacher_score \
    --sample_neg_from_topk 200 \
    --learning_rate 2e-5 \
    --num_train_epochs 4 \
    --negatives_x_device  \
    --dataloader_num_workers 6 
output_path=$output_model'/encode/'
mkdir $output_path
torchrun --nproc_per_node 8  --master_port 12349  \
    -m bi_encoder.run \
    --output_dir retromae_msmarco_passage_fintune \
    --model_name_or_path $output_model \
    --corpus_file $corpus_file \
    --passage_max_len 140 \
    --fp16  \
    --do_predict \
    --prediction_save_path $output_path \
    --per_device_eval_batch_size 256 \
    --dataloader_num_workers 6 \
    --eval_accumulation_steps 100 

query_path="examples/retriever/msmarco/data/BertTokenizer_data/dev_query"

torchrun --nproc_per_node 8  --master_port 12349  \
-m bi_encoder.run \
--output_dir retromae_msmarco_passage_fintune \
--model_name_or_path $output_model \
--test_query_file $query_path \
--query_max_len 32 \
--fp16  \
--do_predict \
--prediction_save_path $output_path \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100 

qrels_file="examples/retriever/msmarco/data/BertTokenizer_data/train_qrels.txt"
python bi_encoder/test.py \
--query_reps_path $output_path'query_reps' \
--passage_reps_path $output_path'passage_reps' \
--qrels_file $qrels_file \
--ranking_file  $output_path'dev_ranking.txt' \
--use_gpu 


qrels="examples/retriever/msmarco/data/msmarco-pass/qrels.dev.tsv"

echo "====== Evaluation"
python utils/convert_result_to_trec.py \
--input $output_path/dev_ranking.txt \
--output $output_path/dev_rank.txt.trec
python bi_encoder/trec_eval.py --qrel_file $qrels --run_file $output_path/dev_rank.txt.trec


output_path=$output_model'/encode19/'
mkdir $output_path
query_path="examples/retriever/msmarco/data/BertTokenizer_data/dev_query_19"

torchrun --nproc_per_node 8  --master_port 12349  \
-m bi_encoder.run \
--output_dir retromae_msmarco_passage_fintune \
--model_name_or_path $output_model \
--test_query_file $query_path \
--query_max_len 32 \
--fp16  \
--do_predict \
--prediction_save_path $output_path \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100 

qrels_file="examples/retriever/msmarco/data/BertTokenizer_data/train_qrels_19.txt"
python bi_encoder/test.py \
--query_reps_path $output_path'query_reps' \
--passage_reps_path  $output_model'/encode/passage_reps' \
--qrels_file $qrels_file \
--ranking_file  $output_path'dev_ranking.txt' \
--use_gpu 
qrels="examples/retriever/msmarco/data/test_dataset/trec_dl/dl19/2019qrels-pass.txt"
echo "====== Evaluation"
python utils/convert_result_to_trec.py \
--input $output_path/dev_ranking.txt \
--output $output_path/dev_rank.txt.trec
python bi_encoder/trec_eval.py --qrel_file $qrels --run_file $output_path/dev_rank.txt.trec
output_path=$output_model'/encode20/'
mkdir $output_path
query_path="examples/retriever/msmarco/data/BertTokenizer_data/dev_query_20"

torchrun --nproc_per_node 8  --master_port 12349  \
-m bi_encoder.run \
--output_dir retromae_msmarco_passage_fintune \
--model_name_or_path $output_model \
--test_query_file $query_path \
--query_max_len 32 \
--fp16  \
--do_predict \
--prediction_save_path $output_path \
--per_device_eval_batch_size 256 \
--dataloader_num_workers 6 \
--eval_accumulation_steps 100 

qrels_file="examples/retriever/msmarco/data/BertTokenizer_data/train_qrels_20.txt"
python bi_encoder/test.py \
--query_reps_path $output_path'query_reps' \
--passage_reps_path  $output_model'/encode/passage_reps' \
--qrels_file $qrels_file \
--ranking_file  $output_path'dev_ranking.txt' \
--use_gpu 
qrels="/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl20/2020qrels-pass.txt"
echo "====== Evaluation"
python utils/convert_result_to_trec.py \
--input $output_path/dev_ranking.txt \
--output $output_path/dev_rank.txt.trec
python bi_encoder/trec_eval.py --qrel_file $qrels --run_file $output_path/dev_rank.txt.trec

