
for epoch in 40
do
    for lr in 1e-5
    do 
        for score_file in "qwen_utility_generationA_score" 
        do 
            model_path="/root/paddlejob/workspace/env_run/model/RetroMAE_wiki"
            echo "======./models_shuffle_pos_all/"$score_file"_bn128_l3_"$epoch"_lr_"$lr
            corpus_file="/root/paddlejob/workspace/env_run/output/dpr-nq/utility_in_retrieval/examples/retriever/msmarco/dataset_retroMAE_wiki/corpus"
            train_query_file="/root/paddlejob/workspace/env_run/output/dpr-nq/utility_in_retrieval/examples/retriever/msmarco/dataset_retroMAE_wiki/train_query"
            train_qrels="/root/paddlejob/workspace/env_run/output/dpr-nq/results/llm_annotation/qwen3-32b-nq-annotation_utility_final.tsv"
            neg_file="/root/paddlejob/workspace/env_run/output/dpr-nq/results/llm_annotation/nq_all_passage_ids.tsv"
            teacher_score="/root/paddlejob/workspace/qwen3-32b-nq-teacher_score_utility.tsv"
            output_model="./models_shuffle_pos_all/"$score_file"_bn128_l3_"$epoch"_lr_"$lr
            mkdir $output_model # --stop_after_n_steps 1000 \
            torchrun --nproc_per_node 8  --master_port 12345  \
                -m bi_encoder.run \
                --output_dir $output_model \
                --model_name_or_path $model_path \
                --do_train  \
                --corpus_file $corpus_file \
                --train_query_file $train_query_file \
                --hard_neg_file $hard_neg_file \
                --train_qrels $train_qrels \
                --neg_file $neg_file \
                --teacher_score_files $teacher_score \
                --query_max_len 32 \
                --passage_max_len 156 \
                --loss_type "myloss" \
                --fp16  \
                --per_device_train_batch_size 16 \
                --train_group_size 8 \
                --learning_rate $lr \
                --num_train_epochs $epoch \
                --negatives_x_device  \
                --dataloader_num_workers 6 
            output_path=$output_model'/encode/'
            mkdir $output_path
            torchrun --nproc_per_node 8  --master_port 12345  \
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

            query_path="/root/paddlejob/workspace/env_run/output/utility_in_retrieval/examples/retriever/msmarco/contriever_data/dev_query"

            torchrun --nproc_per_node 8  --master_port 12345  \
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

            qrels_file="/root/paddlejob/workspace/env_run/output/utility_in_retrieval/examples/retriever/msmarco/data/BertTokenizer_data/train_qrels.txt"
            python bi_encoder/test.py \
            --query_reps_path $output_path'query_reps' \
            --passage_reps_path $output_path'passage_reps' \
            --qrels_file $qrels_file \
            --ranking_file  $output_path'dev_ranking.txt' \
            --use_gpu 


            qrels="/root/paddlejob/workspace/env_run/data/msmarco-pass/qrels.dev.tsv"

            echo "====== Evaluation"
            python utils/convert_result_to_trec.py \
            --input $output_path/dev_ranking.txt \
            --output $output_path/dev_rank.txt.trec
 


        done
    done
done


CUDA_VISIBLE_DEVICES=0,1,2,3,4,5,6,7 python /root/paddlejob/workspace/env_run/run.py --size 40000 --gpus 8 --interval 0.0


