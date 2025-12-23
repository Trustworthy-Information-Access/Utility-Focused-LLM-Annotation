import random
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Tuple
from arguments import DataArguments
import json
#####################################################################
def get_direct_judge_list_utility_ranking(query, passages, answer, max_length=300):
    num = len(passages)
    messages = get_prefix_direct_judge_list_utility_ranking(query, num, answer)
    rank = 0
    for passage in passages:
        rank += 1
        if len(passage.split(" ")) > int(max_length):
            passage = " ".join(passage.split(" ")[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {passage}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_prompt_utility_ranking(query, num, answer)})
    return messages

def get_prefix_direct_judge_list_utility_ranking(query, num, answer):
    return [{'role': 'user',
            'content': "You are RankGPT, an intelligent assistant that can rank passages based on their utility in generating the given reference answer to the question."},
            {'role': 'assistant',
            'content': "Yes, i am RankGPT."},
            {'role': 'user',
            'content': f"I will provide you with {num} passages, each indicated by number identifier [].  I will also give you a reference answer to the question. \nRank the passages based on their utility in generating the reference answer to the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]

def get_post_prompt_utility_ranking(query, num, answer):
    return f"Question: {query}. \n\n Reference answer: {answer}\n\n Rank the {num} passages above based on their utility in generating the reference answer to the question. The passages should be listed in utility descending order using identifiers.  The passages that have utility generating the reference answer to the question should be listed first. The output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain."
#####################################################################


def format_query(query: str) -> str:
    return f'{query.strip()}'.strip()

def format_passage(text: str, title: str = '') -> str:
    return f'{title.strip()} {text.strip()}'.strip()
#####################################################################
def get_prefix_direct_judge_list_utility(query, num):
    return [{'role': 'user',
             'content': "You are the utility judger, an intelligent assistant that can select the passages that have utility in answering the question."},
            {'role': 'assistant', 'content': 'Yes, I am the utility judger.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \n I will also provide you with a reference answer to the question. \nSelect the passages that have utility in generating the reference answer to the following question from the {num} passages: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages and the reference answer.'}]
def get_post_direct_judge_list_utility(query, instruct, answer):
    return f"Question: {query}. \n Reference answer: {answer}. \n\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n"+instruct

def get_direct_judge_list_utility(question, instruct, passages, answer, max_length=300):
    messages = get_prefix_direct_judge_list_utility(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        if len(content.split(" ")) > int(max_length):
            content = " ".join(content.split(" ")[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_utility(question, instruct, answer)})
    return messages
#####################################################################
def get_prefix_direct_judge_list_relevance(query, num):
    return [{'role': 'user',
             'content': "You are the relevance judger, an intelligent assistant that can select the passages that are relevant to the question."},
            {'role': 'assistant', 'content': 'Yes, I am the relevance judger.'},
            {'role': 'user',
             'content': f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages that are relevant to the question: {query}."},
            {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]

# def get_prefix_direct_judge_list_relevance(query, num):
#     return [{'role': 'user',
#              'content': "You are the relevance judger, an intelligent assistant that can select the passages that relevant to the question."},
#             {'role': 'assistant', 'content': 'Yes, i am the utility judger.'},
#             {'role': 'user',
#              'content': f"I will provide you with {num} passages, each indicated by number identifier []. Rank them based on their relevance to query: {query}."},
#             {'role': 'assistant', 'content': 'Okay, please provide the passages.'}]
def get_post_direct_judge_list_relevance(query, instruct):
    return f"Search Query: {query}."+instruct
def get_direct_judge_list_relevance(question, instruct, passages, max_length=300):
    messages = get_prefix_direct_judge_list_relevance(question, len(passages))
    rank = 0
    for content in passages:
        rank += 1
        if len(content.split(" ")) > int(max_length):
            content = " ".join(content.split(" ")[:int(max_length)])
        messages.append({'role': 'user', 'content': f"[{rank}] {content}"})
        messages.append({'role': 'assistant', 'content': f'Received passage [{rank}].'})
    messages.append({'role': 'user', 'content': get_post_direct_judge_list_relevance(question, instruct)})
    return messages
def get_passage_ids(passages_corpus_file):
    id_passages = {}
    id_query = {}
    with open(passages_corpus_file, "r", encoding="utf-8") as file:
        for line in file:
            js = json.loads(line)
            id_query[js["query_id"]]  = js["query"]
            for passage in js["passages"]:
                id_passages[passage["docid"]] = format_passage(passage["text"], passage["title"])
    return id_passages, id_query

def get_relevance_ids(relevance_file):
    id_relevance = {}
    with open(relevance_file, "r", encoding="utf-8") as file:
        for line in file:
            js = json.loads(line)
            passages = []
            for id, passage_id in enumerate(js["passages_ids"]):
                if js["relevance_score"][id] == 1:
                    passages.append(passage_id)
            id_relevance[js["query_id"]] = passages
    return id_relevance

def get_answers(answer_file):
    id_answers = {}
    with open(answer_file, "r", encoding="utf-8") as file:
        for line in file:
            js = json.loads(line)
            if "</think>" in js["answer_output"]:
                id_answers[js["query_id"]] = js["answer_output"].split("</think>")[1]
            else:
                id_answers[js["query_id"]] = js["answer_output"]
    return id_answers


def generate_answer_prompt_passages(question, passages):
    pas = '\n'.join(passages)
    return [{'role': 'user', 'content': f"You are a faithful question and answer assistant. Answer the question based on the given information with one or a few sentences without the source."}, 
            {'role': 'assistant', 'content': 'Yes, i am the faithful question and answer assistant.'}, 
            {'role': 'user', 'content': f"Given the information: \n{pas}\n Answer the following question based on the given information with one or a few sentences without the source.\n Question: {question}\n\n Answer:"},]

class RelevanceEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.utility_instruct = """
        Directly output the passages you selected that have utility in generating the reference answer to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.relevance_instruct = """
        Directly output the passages you selected that are relevant to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )



    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        # query = group['query']
        query = self.id_querys[group['query_id']]
        labels = []
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"]
        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])
        messages = generate_answer_prompt_passages(query, formated_passages)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        

        messages = get_direct_judge_list_relevance(query, self.relevance_instruct, formated_passages)
        relevance_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return utility_prompt, relevance_prompt, labels, formated_passages, formated_passages_ids, query_id


class AnswerEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.utility_instruct = """
        Directly output the passages you selected that have utility in generating the reference answer to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.relevance_instruct = """
        Directly output the passages you selected that are relevant to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """

        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        self.id_passages, self.id_querys = get_passage_ids(self.data_args.passages_corpus)
        self.id_relevance = get_relevance_ids(self.data_args.relevence_file_path)
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )



    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        # query = group['query']
        query = self.id_querys[group['query_id']]
        labels = []
        formated_passages = []
        formated_passages_ids = []
        group_negatives = group["passages"]
        for passage in group_negatives:
            formated_passages.append(format_passage(passage["text"], passage["title"]))
            formated_passages_ids.append(passage["docid"])
        messages = generate_answer_prompt_passages(query, formated_passages)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        

        messages = get_direct_judge_list_relevance(query, self.relevance_instruct, formated_passages)
        relevance_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return utility_prompt, relevance_prompt, labels, formated_passages, formated_passages_ids, query_id


class UtilityEncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
        self.utility_instruct = """
        Directly output the passages you selected that have utility in generating the reference answer to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.relevance_instruct = """
        Directly output the passages you selected that are relevant to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
        """
        self.tokenizer = tokenizer
        self.train_data = load_dataset(
            self.data_args.dataset_name,
            self.data_args.dataset_config,
            data_files=self.data_args.dataset_path,
            split=self.data_args.dataset_split,
            cache_dir=self.data_args.dataset_cache_dir,
        )
        if self.data_args.dataset_number_of_shards > 1:
            self.encode_data = self.encode_data.shard(
                num_shards=self.data_args.dataset_number_of_shards,
                index=self.data_args.dataset_shard_index,
            )
        self.id_passages, self.id_querys = get_passage_ids(self.data_args.passages_corpus)
        self.id_relevance = get_relevance_ids(self.data_args.relevence_file_path)
        self.id_answers = get_answers(self.data_args.answer_file_path)


    def __len__(self):
        return len(self.train_data)

    def __getitem__(self, item) -> Tuple[str, List[int]]:
        _hashed_seed = hash(item)
        group = self.train_data[item]
        query_id = group['query_id']
        query = self.id_querys[group['query_id']]
        labels = []
        formated_passages = []
        formated_passages_ids = []
        ids = self.id_relevance[query_id]
        for id in ids:
            formated_passages.append(self.id_passages[id])
            formated_passages_ids.append(id)
        answer_generation = self.id_answers[query_id]
        
        messages = get_direct_judge_list_utility(query, self.utility_instruct, formated_passages, answer_generation)
        utility_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        

        messages = get_direct_judge_list_relevance(query, self.relevance_instruct, formated_passages)
        relevance_prompt = self.tokenizer.apply_chat_template(messages, tokenize=False, add_generation_prompt=True, enable_thinking=True)
        return utility_prompt, relevance_prompt, labels, formated_passages, formated_passages_ids, query_id
