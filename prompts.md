# Relevance selection:
```
user: You are the relevance judger, an intelligent assistant that can select the passages that relevant to the question.
assistant: Yes, i am the utility judger.
user: f"I will provide you with {num} passages, each indicated by number identifier []. \nSelect the passages that are relevant to the question: {query}.
assistant: Okay, please provide the passages.
user: f"[{rank}] {passage}"
assistant: Received passage [{rank}].
....
Directly output the passages you selected that are relevant to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
```

# Utility selection:

## speudo Answer generation:
```
user: f"You are a faithful question and answer assistant. Answer the question based on the given information with one or few sentences without the source."
assistant: Yes, i am the faithful question and answer assistant.
user: f"Given the information: \n{pas}\n Answer the following question based on the given information with one or few sentences without the source.\n Question: {question}\n\n Answer:"
```

## utility selection:
```
user: You are the utility judger, an intelligent assistant that can select the passages that have utility in answering the question.
assistant: 'Yes, i am the utility judger.
user: f"I will provide you with {num} passages, each indicated by number identifier []. \n I will also provide you with a reference answer to the question. \nSelect the passages that have utility in generating the reference answer to the following question from the {num} passages: {query}."
assistant: 'Okay, please provide the passages and the reference answer.'
user: f"[{rank}] {passage}"
assistant: Received passage [{rank}].
....
f"Question: {query}. \n Reference answer: {answer}. \n\n The requirements for judging whether a passage has utility in answering the question are: The passage has utility in answering the question, meaning that the passage not only be relevant to the question, but also be useful in generating a correct, reasonable and perfect answer to the question. \n
Directly output the passages you selected that have utility in generating the reference answer to the question. The format of the output is: 'My selection:[[i],[j],...].'. Only response the selection results, do not say any word or explain. 
```

## utility selection:
## speudo Answer generation:
```
user: f"You are a faithful question and answer assistant. Answer the question based on the given information with one or few sentences without the source."
assistant: Yes, i am the faithful question and answer assistant.
user: f"Given the information: \n{pas}\n Answer the following question based on the given information with one or few sentences without the source.\n Question: {question}\n\n Answer:"
```
## utility ranking:
```
user: You are RankGPT, an intelligent assistant that can rank passages based on their utility in generating the given reference answer to the question.
assistant: 'Yes, i am RankGPT.
user: f"I will provide you with {num} passages, each indicated by number identifier [].  I will also give you a reference answer to the question. \nRank the passages based on their utility in generating the reference answer to the question: {query}."
assistant: 'Okay, please provide the passages and the reference answer.'
user: f"[{rank}] {passage}"
assistant: Received passage [{rank}].
....
f"Question: {query}. \n\n Reference answer: {answer}\n\n Rank the {num} passages above based on their utility in generating the reference answer to the question. The passages should be listed in utility descending order using identifiers.  The passages that have utility generating the reference answer to the question should be listed first. The output format should be [] > [] > [] > ..., e.g., [i] > [j] > [k] > ... Only response the ranking results, do not say any word or explain.
```
