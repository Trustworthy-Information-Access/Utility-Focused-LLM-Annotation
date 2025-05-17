# utility-focused-annotation

# Overview
This repository contains the code, datasets models used in our paper: "Utility-Focused LLM Annotation for Retrieval and Retrieval-Augmented Generation". 

This paper explores the use of large language models (LLMs) for annotating document utility in training retrieval and retrieval-augmented generation (RAG) systems, aiming to reduce dependence on costly human annotations. We address the gap between retrieval relevance and generative utility by employing LLMs to annotate document utility. To effectively utilize multiple positive samples per query, we introduce a novel loss that maximizes their summed marginal likelihood. Using the Qwen-2.5-32B model, we annotate utility on the MS MARCO dataset and conduct retrieval experiments on MS MARCO and BEIR, as well as RAG experiments on MS MARCO QA, NQ, and HotpotQA. Our results show that LLM-generated annotations enhance out-of-domain retrieval performance and improve RAG outcomes compared to models trained solely on human annotations or downstream QA metrics. Furthermore, combining LLM annotations with just 20\% of human labels achieves performance comparable to using full human annotations. Our study offers a comprehensive approach to utilizing LLM annotations for initializing QA systems on new corpora. 

# Download dataset 
We utilize in-domain settings ([MSMARCO v1 and TREC-DL](https://microsoft.github.io/msmarco/Datasets)) and out-of-domain settings ([BEIR](https://github.com/beir-cellar/beir)) on both the retrieval and RAG tasks. 

# LLMs Annotations
We use the hard negative samples provided by [Tevatron](https://www.dropbox.com/scl/fi/pkm1mtgfobae9kuesp7dr/train-tevatron.jsonl?rlkey=2thutc4zkozr9jp4zbbrz5rvi&dl=0). 
The prompts used in our paper are shown in prompts.md. 


# Retrievers Training 
We use the RetroMAE and Contriever as our retriever backbone, which can be downloaded on [RetroMAE Pre-training on MSMARCO Passage](https://github.com/staoxiao/RetroMAE/blob/master/examples/pretrain/README.md) and [Contriever](https://huggingface.co/facebook/contriever)

```
sh run.sh
```

# Checkpoint
Currently, all the LLM annotated positive labels and the models' checkpoints are in the author's huggingface account. After the anonymity period, we will add the corresponding link.




