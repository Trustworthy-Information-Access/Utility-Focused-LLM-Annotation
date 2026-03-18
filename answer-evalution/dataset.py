import random
from torch.utils.data import Dataset
from datasets import load_dataset
from typing import List, Tuple
from arguments import DataArguments
def format_query(query: str) -> str:
    return f'{query.strip()}'.strip()

def format_passage(text: str, title: str = '') -> str:
    return f'{title.strip()} {text.strip()}'.strip()

def generate_answer_prompt_passages(question, passages):
    # pas = '\n'.join(passages)
    return [{'role': 'user', 'content': f"You are a faithful question and answer assistant. Answer the question based on the given information with one or few sentences without the source."}, 
            {'role': 'assistant', 'content': 'Yes, i am the faithful question and answer assistant.'}, 
            {'role': 'user', 'content': f"Given the information: \n{pas}\n Answer the following question based on the given information with one or few sentences without the source.\n Question: {question}\n\n Answer:"},]
    # pas = '\n'.join(passages)
    # return [{'role': 'user', 'content': f"You are a faithful question and answer assistant. Answer the question based on the given information with one or few words without the source."}, 
    #         {'role': 'assistant', 'content': 'Yes, i am the faithful question and answer assistant.'}, 
    #         {'role': 'user', 'content': f"Given the information: \n{pas}\n Answer the following question based on the given information with one or few words without the source.\n Question: {question}\n\n Answer:"},]
    # pas = '\n'.join(passages)
    # return [
    #         {'role': 'user', 'content': f"Question: {question} \n Passages: {pas} \n  Answer the question based on the given passages with one or few words without the explanation. Answer: "},]

class EncodeDataset(Dataset):
    def __init__(self, data_args: DataArguments, tokenizer):
        print(data_args)
        self.data_args = data_args
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
        query = group['query']
        query_id = group['query_id']
        labels = []
        formated_passages = [format_passage(passage['text'], passage['title']) for passage in group['top_passages'][:self.data_args.topk]]
        labels = [0]*len(formated_passages)
        formated_passages_ids = [passage['docid'] for passage in group['top_passages'][:5]]
        messages = generate_answer_prompt_passages(query, formated_passages)
        utility_prompt = messages
        return utility_prompt, labels, formated_passages, formated_passages_ids, query_id