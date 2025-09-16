import argparse
import json
import os
from pathlib import Path
from typing import List, Tuple
# 完全禁用音频支持
os.environ["AUDIO_DEPS_IGNORED"] = "1"
import sys
sys.modules['soundfile'] = None  # 阻止加载
# 在 datasets 导入前添加
from unittest.mock import MagicMock
sys.modules['_soundfile_data'] = MagicMock()
from datasets import load_dataset
from transformers import AutoTokenizer


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--tokenizer_name", type=str)
    parser.add_argument("--output_dir", type=str)
    parser.add_argument("--max_seq_length", type=int, default=512)
    parser.add_argument("--use_title", action='store_true', default=False)
    return parser.parse_args()


def save_to_json(input_file, output_file, id_file):
    with open(output_file, 'w', encoding='utf-8') as f, open(id_file, 'w', encoding='utf-8') as fid:
        cnt = 0
        for line in open(input_file, encoding='utf-8'):
            line = line.strip('\n').split('\t')
            if len(line) == 2:
                data = {"id": line[0], 'text': line[1]}
            else:
                data = {"id": line[0], 'title': line[1], 'text': line[2]}
            f.write(json.dumps(data) + '\n')
            fid.write(line[0] + '\t' + str(cnt) + '\n')
            cnt += 1

def save_to_id(input_file, id_file):
    with open(id_file, 'w', encoding='utf-8') as fid:
        cnt = 0
        for line in open(input_file, encoding='utf-8'):
            js = json.loads(line)
            fid.write(js["docid"] + '\t' + str(cnt) + '\n')
            cnt += 1

# 19335 Q0 1017759 0
# def preprocess_qrels(train_qrels, output_file):
#     with open(output_file, 'w', encoding='utf-8') as f:
#         for line in open(train_qrels, encoding='utf-8'):
#             line = line.strip().split('\t')
#             f.write(line[0] + '\t' + line[2] + '\n')
def preprocess_qrels(train_qrels, output_file):
    with open(output_file, 'w', encoding='utf-8') as f:
        for line in open(train_qrels, encoding='utf-8'):
            line = line.strip().split(' ')
            f.write(line[0] + '\t' + line[2] + '\n')


def tokenize_function(examples):
    if 'title' in examples and args.use_title:
        content = []
        for title, text in zip(examples['title'], examples['text']):
            content.append(title + tokenizer.sep_token + text)
        return tokenizer(content, add_special_tokens=False, truncation=True, max_length=max_length,
                         return_attention_mask=False,
                         return_token_type_ids=False)
    else:
        return tokenizer(examples["text"], add_special_tokens=False, truncation=True, max_length=max_length,
                         return_attention_mask=False,
                         return_token_type_ids=False)


args = get_args()
tokenizer = AutoTokenizer.from_pretrained(args.tokenizer_name)
max_length = args.max_seq_length

if __name__ == '__main__':
    Path(args.output_dir).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'corpus')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'train_query')).mkdir(parents=True, exist_ok=True)
    Path(os.path.join(args.output_dir, 'dev_query')).mkdir(parents=True, exist_ok=True)

    # preprocess_qrels('/root/paddlejob/workspace/env_run/data/msmarco-pass/qrels.train.tsv', os.path.join(args.output_dir, 'train_qrels.txt'))

    save_to_id('/root/paddlejob/workspace/env_run/output/dpr-nq/wikipedia-nq-corpus/corpus.jsonl', os.path.join(args.output_dir, 'corpus/mapping_id.txt'))
    corpus = load_dataset('json', data_files='/root/paddlejob/workspace/env_run/output/dpr-nq/wikipedia-nq-corpus/corpus.jsonl', split='train')
    corpus = corpus.map(tokenize_function, num_proc=8, remove_columns=["title", "text"], batched=True)
    corpus.save_to_disk(os.path.join(args.output_dir, 'corpus'))
    print('corpus dataset:', corpus)

    save_to_json('/root/paddlejob/workspace/env_run/output/dpr-nq/wikipedia-nq/nq-train-query.tsv', '/root/paddlejob/workspace/env_run/output/dpr-nq/wikipedia-nq/train.query.json',
                 os.path.join(args.output_dir, 'train_query/mapping_id.txt'))
    # save_to_id('/root/paddlejob/workspace/env_run/output/dpr-nq/wikipedia-nq/nq-train.jsonl', os.path.join(args.output_dir, 'train_query/mapping_id.txt') )
    train_query = load_dataset('json', data_files='/root/paddlejob/workspace/env_run/output/dpr-nq/wikipedia-nq/train.query.json', split='train')
    train_query = train_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    train_query.save_to_disk(os.path.join(args.output_dir, 'train_query'))
    print('train query dataset:', train_query)



    # save_to_json('/root/paddlejob/workspace/env_run/data/msmarco-pass/queries.train.tsv', '/root/paddlejob/workspace/env_run/data/msmarco-pass/train.query.json',
    #              os.path.join(args.output_dir, 'train_query/mapping_id.txt'))
    # train_query = load_dataset('json', data_files='/root/paddlejob/workspace/env_run/data/msmarco-pass/train.query.json', split='train')
    # train_query = train_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    # train_query.save_to_disk(os.path.join(args.output_dir, 'train_query'))
    # print('train query dataset:', train_query)

    # save_to_json('/root/paddlejob/workspace/env_run/data/msmarco-pass/queries.dev.tsv', '/root/paddlejob/workspace/env_run/data/msmarco-pass/queries.dev.json',
    #              os.path.join(args.output_dir, 'dev_query/mapping_id.txt'))
    # dev_query = load_dataset('json', data_files='/root/paddlejob/workspace/env_run/data/msmarco-pass/queries.dev.json', split='train')
    # dev_query = dev_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    # dev_query.save_to_disk(os.path.join(args.output_dir, 'dev_query'))
    # print('dev query dataset:', dev_query)



    # save_to_json('/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl19/msmarco-test2019-queries.tsv', '/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl19/msmarco-test2019-queries.dev.json',
    #              os.path.join(args.output_dir, 'dev_query_19/mapping_id.txt'))
    # dev_query = load_dataset('json', data_files='/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl19/msmarco-test2019-queries.dev.json', split='train')
    # dev_query = dev_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    # dev_query.save_to_disk(os.path.join(args.output_dir, 'dev_query_19'))
    # print('dev query dataset:', dev_query)


    # save_to_json('/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl20/msmarco-test2020-queries.tsv', '/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl20/msmarco-test2020-queries.dev.json',
    #              os.path.join(args.output_dir, 'dev_query_20/mapping_id.txt'))
    # dev_query = load_dataset('json', data_files='/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl20/msmarco-test2020-queries.dev.json', split='train')
    # dev_query = dev_query.map(tokenize_function, num_proc=8, remove_columns=["text"], batched=True)
    # dev_query.save_to_disk(os.path.join(args.output_dir, 'dev_query_20'))
    # print('dev query dataset:', dev_query)
    # # preprocess_qrels('/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl19/2019qrels-pass.txt', os.path.join(args.output_dir, 'train_qrels_19.txt'))
    # preprocess_qrels('/root/paddlejob/workspace/env_run/data/test_dataset/trec_dl/dl20/2020qrels-pass.txt', os.path.join(args.output_dir, 'train_qrels_20.txt'))
