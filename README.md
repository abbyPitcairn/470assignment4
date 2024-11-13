## Assignment 4 for Intro to Information Retrieval

Version: November 12, 2024

Author: Abigail Pitcairn

This assignment finds the 100 top search results for each query in a set of queries from a collection of documents. 
First, a tf-idf search is conducted to retrieve 100 top results. 
Then, an LLM information retrieval method is used to rerank the results from tf-idf search.
NOTE: the Answers.json file used in this project is not hosted on github because the file size is too large.

### Terminal Command: 
python3 Main.py <Answers.json> <topics_1.json> <topics_2.json>

### Output before session times out:
Starting main...
Building inverted index...
Searching...
Building inverted index...
Searching...
Reranking with LLM...
The attention mask is not set and cannot be inferred from input because pad token is same as eos token. As a consequence, you may observe unexpected behavior. Please pass your input's `attention_mask` to obtain reliable results.
Starting from v4.46, the `logits` model output will have the same type as the model (except at train time, where it will always be FP32)


