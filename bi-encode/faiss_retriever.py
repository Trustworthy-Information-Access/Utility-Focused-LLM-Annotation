import logging
import os

import faiss
import numpy as np
from tqdm import tqdm

logger = logging.getLogger(__name__)
logging.basicConfig(
    format="%(asctime)s - %(levelname)s - %(name)s -   %(message)s",
    datefmt="%m/%d/%Y %H:%M:%S",
    level=logging.INFO,
)


class BaseFaissIPRetriever:
    def __init__(self, reps_dim: int):
        index = faiss.IndexFlatIP(reps_dim)
        self.index = index

    def add(self, p_reps: np.ndarray):
        self.index.add(p_reps)

    def search(self, q_reps: np.ndarray, k: int):
        return self.index.search(q_reps, k)

    def batch_search(self, q_reps: np.ndarray, k: int, batch_size: int):
        num_query = q_reps.shape[0]  # number of queries
        all_scores = []
        all_indices = []
        for start_idx in tqdm(range(0, num_query, batch_size), total=num_query // batch_size):
            nn_scores, nn_indices = self.search(q_reps[start_idx: start_idx + batch_size], k)  # 一次性搜索batch_size个query，取topk
            all_scores.append(nn_scores)
            all_indices.append(nn_indices)
        all_scores = np.concatenate(all_scores, axis=0)  # 将所有query的topk结果拼接
        all_indices = np.concatenate(all_indices, axis=0)  # 将所有query的topk结果拼接

        return all_scores, all_indices


def search_queries(retriever, q_reps, p_lookup, depth, batch_size):  # 调用retriever的search方法，搜索query
    if batch_size > 0:
        all_scores, all_indices = retriever.batch_search(q_reps, depth, batch_size)
    else:
        all_scores, all_indices = retriever.search(q_reps, depth)

    psg_indices = [[str(p_lookup[x]) for x in q_dd] for q_dd in all_indices]  # 将检索到的passage的index转换为id
    psg_indices = np.array(psg_indices)
    return all_scores, psg_indices


def write_ranking(corpus_indices, corpus_scores, q_lookup, ranking_save_file):
    with open(ranking_save_file, 'w') as f:
        for qid, q_doc_scores, q_doc_indices in zip(q_lookup, corpus_scores, corpus_indices):
            score_list = [(s, idx) for s, idx in zip(q_doc_scores, q_doc_indices)]  # 将检索到的passage的index和score对应起来
            score_list = sorted(score_list, key=lambda x: x[0], reverse=True)  # 按照score排序
            rank = 1
            for s, idx in score_list:
                f.write(f'{qid}\t{idx}\t{rank}\t{s}\n')  # 写入文件，格式为query_id \t passage_id \t rank \t score
                rank += 1


def read_id(id_file):
    ids = []
    for line in open(id_file):
        offset, id = line.strip().split('\t')
        ids.append(id)
    return ids


def search_by_faiss(query_reps_path, passage_reps_path, save_file, batch_size=512, depth=1000, use_gpu=False):
    logger.info('Loading Passage Reps')
    p_reps = np.load(os.path.join(passage_reps_path, 'passage.npy'))
    p_reps = np.array(p_reps).astype('float32')
    p_lookup = read_id(os.path.join(passage_reps_path, 'offset2passageid.txt'))
    print("shape of passage", np.shape(p_reps))

    logger.info('Building Faiss Index')
    retriever = BaseFaissIPRetriever(np.shape(p_reps)[-1])

    faiss.omp_set_num_threads(64)
    if use_gpu:
        print('use GPU for Faiss')
        co = faiss.GpuMultipleClonerOptions()
        co.shard = True
        co.useFloat16 = True
        retriever.index = faiss.index_cpu_to_all_gpus(
            retriever.index,
            co=co
        )
    retriever.add(p_reps)

    logger.info('Loading Query Reps')
    q_reps = np.load(os.path.join(query_reps_path, 'query.npy'))
    q_reps = np.array(q_reps).astype('float32')
    q_lookup = read_id(os.path.join(query_reps_path, 'offset2queryid.txt'))
    print("shape of query", np.shape(q_reps))

    logger.info('Index Search Start')
    all_scores, psg_indices = search_queries(retriever, q_reps, p_lookup, depth, batch_size)
    logger.info('Index Search Finished')

    write_ranking(psg_indices, all_scores, q_lookup, save_file)
