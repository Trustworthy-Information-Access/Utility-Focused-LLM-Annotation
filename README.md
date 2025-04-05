# utility-focused-annotation

# Overview
This repository contains the code, datasets models used in our paper: "Leveraging LLMs for Utility-Focused Annotation: Reducing Manual Effort for Retrieval and RAG". 

Retrieval models typically rely on costly human-labeled query-document relevance annotations for training and evaluation. 
To reduce this cost and leverage the potential of Large Language Models (LLMs) in relevance judgments, we aim to explore whether LLM-generated annotations can effectively replace human annotations in training retrieval models. 
Retrieval usually emphasizes relevance, which indicates ``topic-relatedness'' of a document to a query, while in RAG, the value of a document (or utility), depends on how it contributes to answer generation. 
Recognizing this mismatch, some researchers use LLM performance on downstream tasks with documents as labels, but this approach requires manual answers for specific tasks, leading to high costs and limited generalization. 
In another line of work, prompting LLMs to select useful documents as RAG references eliminates the need for human annotation and is not task-specific. If we leverage LLMs’ utility judgments to annotate retrieval data, we may retain cross-task generalization without human annotation in large-scale corpora. 
Therefore, we investigate utility-focused annotation via LLMs for large-scale retriever training data across both in-domain and out-of-domain settings on the retrieval and RAG tasks. 
To reduce the impact of low-quality positives labeled by LLMs, we design a novel loss function, i.e., \orfunction. 
Our experiments reveal that: (1) Retrievers trained on utility-focused annotations significantly outperform those trained on human annotations in the out-of-domain setting on both tasks, demonstrating superior generalization capabilities. 
(2) LLM annotation does not replace human annotation in the in-domain setting. 
However, incorporating just 20% human-annotated data enables retrievers trained with utility-focused annotations to match the performance of models trained entirely with human annotations, while adding 100% human annotations further significantly enhances performance on both tasks.
We hope our work inspires others to design automated annotation solutions using LLMs, especially when human annotations are unavailable. 

# LLMs Annotations
The prompts used in our paper are shown in prompts.md. All annotated labels can be downloaded in [Huggiface hub](https://huggingface.co/hengranZhang/Utility_focused_annotation).

# Download dataset 
We use in-domain settings ([MSMARCO v1 and TREC-DL](https://microsoft.github.io/msmarco/Datasets)) and out-of-domain settings ([BEIR](https://github.com/beir-cellar/beir)) on both the retrieval and RAG tasks. 

# Retrievers Training 
We use the RetroMAE as our retriever backbone, which can be downloaded on [RetroMAE Pre-training on MSMARCO Passage](https://github.com/staoxiao/RetroMAE/blob/master/examples/pretrain/README.md)

```
sh run.sh
```

# Checkpoint
Retriever trained on different annotations can be directly downloaded in [Huggiface hub](https://huggingface.co/hengranZhang/Utility_focused_annotation)



