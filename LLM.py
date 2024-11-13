import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import BaseSearch

# Rerank search results using LLM.
# Version: November 12, 2024
# Author: Abigail Pitcairn


# Automate rerank and save output.
def llm_search(model_name, prompt, base_results, topics, answers, output_file_path):
    # Rerank base results with LLM search.
    print("Reranking with LLM...")
    reranked_results = rerank_documents_with_llm(model_name, prompt, base_results, topics, answers)
    # Save the reranked results to an output file.
    BaseSearch.save_to_result_file(reranked_results, output_file_path)


# Rerank using LLM.
def rerank_documents_with_llm(model_name, prompt, base_results, queries, docs, batch_size=8):
    # Initialize model.
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name, padding_side='left') # Padding size = left for decoder-only

    # Initialize dictionary for results.
    reranked_results = {}

    # Rerank documents for query by query.
    for query in queries:
        # Prepare query text.
        query_id = query['Id']
        query_text = f"{query['Title']} {query['Body']}"

        # Prepare document texts.
        doc_ids = list(base_results[query_id].keys())
        # Create dictionary mapping doc IDs to doc data
        docs_dict = {doc['Id']: doc for doc in docs}
        # Use the above dictionary to access doc text by doc ID
        doc_texts = [docs_dict[doc_id]['Text'] for doc_id in doc_ids]

        # Create prompts for the LLM.
        prompts = create_rerank_prompts(query_text, doc_texts, prompt)

        # Initialize dictionary of scores by document.
        scores = {}
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tokenizer(
                batch_prompts,
                return_tensors="pt",
                padding=True,
                truncation=True,
                return_attention_mask=True
            )

            with torch.no_grad():
                outputs = model.generate(inputs["input_ids"], max_new_tokens=10)

            for j, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True).strip()
                # Set the score to an integer representation of the model's ranking response.
                try:
                    score = int(response)
                # If no score is given, default to 0.
                except ValueError:
                    score = 0
                scores[doc_ids[i + j]] = score

        # Sort and save re-ranked results
        reranked_results[query_id] = {doc_id: scores[doc_id] for doc_id in sorted(scores, key=scores.get, reverse=True)}

    return reranked_results


# Create a list of prompts for all documents related to one query. Specify which prompt to use.
def create_rerank_prompts(query_text, doc_texts, prompt):
    prompts = []
    # For each document, create a prompt with the query and that document's text.
    for doc_text in doc_texts:
        if prompt == '1':
            prompt = (f"Query: {query_text}\nDocument: {doc_text}\nRate the relevance of this document "
                        "on a scale from 1 (least relevant) to 5 (most relevant). Return only the number.")
        else:
            prompt = (f"For this query about travel, {query_text}, how would you rate this document, {doc_text}, in"
                        "terms of relevance on a scale from 1 to 5, with 5 being the most relevant? Return only"
                        "the number.")
        # Add the prompt to the list of prompts.
        prompts.append(prompt)
    # Return the list of prompts to be passed to the LLM.
    return prompts
