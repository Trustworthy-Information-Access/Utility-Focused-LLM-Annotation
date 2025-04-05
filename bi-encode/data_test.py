import collections
import logging
import os.path
import random
from dataclasses import dataclass
from typing import List, Tuple, Any, Optional

import datasets
import torch
from torch.utils.data import Dataset
from transformers import DataCollatorWithPadding
from transformers import PreTrainedTokenizer, BatchEncoding
from datasets import load_dataset

from .arguments import DataArguments


def read_mapping_id(id_file):  # 读取id文件，返回字典id->offset
    id_dict = {}
    for line in open(id_file, encoding='utf-8'):
        id, offset = line.strip().split('\t')
        id_dict[id] = int(offset)
    return id_dict


def read_train_file(train_file):  # 读取训练文件，每个query对应的正样本（一般就一个）
    train_data = []
    for line in open(train_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        pos = line[1].split(',')
        train_data.append((qid, pos))  # TODO：这有bug，和后面的random choice矛盾
    return train_data



def read_neg_file(neg_file):  # 读取负样本文件，每个query对应的负样本（多个）
    neg_data = collections.defaultdict(list)  # 并没有涵盖所有的query，只有出现在neg_file中的query才有负样本
    for line in open(neg_file, encoding='utf-8'):
        line = line.strip('\n').split('\t')
        qid = line[0]
        neg = line[1].split(',')
        neg_data[qid].extend(neg)
    return neg_data


def read_teacher_score(score_files):  # 读取teacher score文件，返回字典{qid: {did: score}}
    #teacher_score = collections.defaultdict(dict)  # TODO：感觉有问题，如果是不存在的did会报错？ collections.defaultdict(lambda: collections.defaultdict(float)) 会更好
    teacher_score = collections.defaultdict(lambda: collections.defaultdict(float))
    for file in score_files.split(','):  # 读取每一个文件
        if not os.path.exists(file):
            logging.info(f"There is no score file:{file}, skip reading the score")
            return None
        for line in open(file):  # 读取文件中的每一行
            qid, did, score = line.strip().split()  # 按空格分割，得到query_id, doc_id, score
            score = float(score.strip('[]'))
            teacher_score[qid][did] = score  # 将score存入teacher_score字典中
    return teacher_score


def generate_random_neg(qids, pids, k=30):  # 从pids中随机选取k个作为负样本
    qid_negatives = {}
    for q in qids:
        negs = random.sample(pids, k)
        qid_negatives[q] = negs
    return qid_negatives


class TrainDataset_test(Dataset):
    def __init__(
            self,
            args: DataArguments,
            tokenizer: PreTrainedTokenizer, trainer = None,
    ):
        print(args.train_query_file)
        self.corpus_dataset = datasets.Dataset.load_from_disk(args.corpus_file)  
        self.query_dataset = datasets.Dataset.load_from_disk(args.train_query_file)
        self.train_qrels = read_train_file(args.train_qrels)
        self.corpus_id = read_mapping_id(args.corpus_id_file)
        self.query_id = read_mapping_id(args.train_query_id_file)
        if args.hard_neg_file:
            self.hard_train_negative = read_neg_file(args.hard_neg_file)
        else:
            self.hard_train_negative = {}
        if args.neg_file:  # 如果存在负样本文件
            self.train_negative = read_neg_file(args.neg_file)
        else:  # 随机生成负样本
            self.train_negative = generate_random_neg(list(self.query_id.keys()), list(self.corpus_id.keys()))

        self.teacher_score = None
        if args.teacher_score_files is not None:  # 如果存在teacher score文件
            self.teacher_score = read_teacher_score(args.teacher_score_files)  # cross encoder对q-d的打分

        self.tokenizer = tokenizer
        self.args = args
        self.total_len = len(self.train_qrels)

    def __len__(self):
        return self.total_len

    def create_query_example(self, id: Any):
        item = self.tokenizer.encode_plus(
            self.query_dataset[self.query_id[id]]['input_ids'],  # 从query_dataset中取出query的input_ids
            truncation='only_first',
            max_length=self.args.query_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def create_passage_example(self, id: Any):
        item = self.tokenizer.encode_plus(
            self.corpus_dataset[self.corpus_id[id]]['input_ids'],
            truncation='only_first',
            max_length=self.args.passage_max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item

    def __getitem__(self, item) -> Tuple[BatchEncoding, List[BatchEncoding], Optional[List[int]]]:
        group = self.train_qrels[item]  # 取一组查询和正样本 (qid, [pos_id1, ...])
        qid = group[0]
        query = self.create_query_example(qid)

        teacher_scores = None
        passages = []
        pos_id = group[1][0]
        # pos_id = random.choice(group[1])  # 随机选一个正样本（一般也就一个）
        passages.append(self.create_passage_example(pos_id))
        if self.teacher_score:
            teacher_scores = []
            teacher_scores.append(self.teacher_score[qid][pos_id])  # 取出teacher的q-d打分
        if qid in self.hard_train_negative and self.hard_train_negative[qid]!= [''] and self.hard_train_negative != {}:
            # print(self.hard_train_negative[qid])
            hard_negative = self.hard_train_negative[qid]
        else:
            hard_negative = []
        if len(hard_negative) >= self.args.train_group_size - len(passages): 
            hard_negative = random.sample(hard_negative, k=self.args.train_group_size - len(passages))
        for id in hard_negative:
            passages.append(self.create_passage_example(id))
            if self.teacher_score:  # score都是数字
                teacher_scores.append(self.teacher_score[qid][id])
            
        # 选取一点量negs
        query_negs = self.train_negative[qid][:self.args.sample_neg_from_topk] 
        # print(query_negs)
        negative_nums = self.args.train_group_size - len(passages)
        if len(query_negs) < negative_nums:
            negs = random.sample(self.corpus_id.keys(), k=negative_nums - len(query_negs))
            negs.extend(query_negs)
        else:
            negs = random.sample(query_negs, k=negative_nums)
        for id in negs:
            passages.append(self.create_passage_example(id))
            if self.teacher_score:  # score都是数字
                teacher_scores.append(self.teacher_score[qid][id])
        # print("passages: ", len(passages))
        # assert 1 > 2
        return query, passages, teacher_scores


class PredictionDataset(Dataset):
    def __init__(self, data_path: str, tokenizer: PreTrainedTokenizer, max_len=128):
        self.encode_data = datasets.Dataset.load_from_disk(data_path)
        self.tokenizer = tokenizer
        self.max_len = max_len

    def __len__(self):
        return len(self.encode_data)

    def __getitem__(self, item) -> BatchEncoding:
        item = self.tokenizer.encode_plus(
            self.encode_data[item]['input_ids'],
            truncation='only_first',
            max_length=self.max_len,
            padding=False,
            return_attention_mask=False,
            return_token_type_ids=False,
        )
        return item


@dataclass
class BiCollator(DataCollatorWithPadding):
    """
    Wrapper that does conversion from List[Tuple[encode_qry, encode_psg]] to List[qry], List[psg]
    and pass batch separately to the actual collator.
    Abstract out data detail for the model.
    """
    query_max_len: int = 32
    passage_max_len: int = 128

    def __call__(self, features):
        # 分别取出query、passage、teacher_score
        query = [f[0] for f in features]
        # assert len(query_set) == len(query)

        passage = [f[1] for f in features]  
        teacher_score = [f[2] for f in features]
        if teacher_score[0] is None:
            teacher_score = None
        else:
            teacher_score = torch.FloatTensor(teacher_score)

        # 将query和passage展平，比如[[1,2],[3,4]] -> [1,2,3,4]？
        if isinstance(query[0], list):
            query = sum(query, [])
        if isinstance(passage[0], list):
            passage = sum(passage, [])

        q_collated = self.tokenizer.pad(
            query,
            padding=True,
            max_length=self.query_max_len,
            return_tensors="pt",
        )
        d_collated = self.tokenizer.pad(
            passage,
            padding=True,
            max_length=self.passage_max_len,
            return_tensors="pt",
        )

        return {"query": q_collated, "passage": d_collated, "teacher_score": teacher_score}


@dataclass
class PredictionCollator(DataCollatorWithPadding):
    is_query: bool = True

    def __call__(
            self, features
    ):
        if self.is_query:
            return {"query": super().__call__(features), "passage": None}
        else:
            return {"query": None, "passage": super().__call__(features)}
