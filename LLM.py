import torch
from transformers import AutoModelForCausalLM, AutoTokenizer
import BaseSearch

# Rerank search results using LLM
# Version: November 12, 2024
# Author: Abigail Pitcairn


def rerank_documents_with_llm(queries, initial_results, documents, model_name, prompt, batch_size=8):
    model = AutoModelForCausalLM.from_pretrained(model_name)
    tokenizer = AutoTokenizer.from_pretrained(model_name)

    reranked_results = {}

    for query_id, doc_scores in initial_results.items():
        # Prepare query text
        query_text = f"{queries[query_id]['Title']} {queries[query_id]['Body']}"

        # Prepare document texts
        doc_ids = list(doc_scores.keys())
        doc_texts = [documents[doc_id]['Text'] for doc_id in doc_ids]

        # Create prompts for the LLM
        prompts = create_rerank_prompts(query_text, doc_texts, prompt)

        scores = {}
        for i in range(0, len(prompts), batch_size):
            batch_prompts = prompts[i:i + batch_size]
            inputs = tokenizer(batch_prompts, return_tensors="pt", padding=True, truncation=True)

            with torch.no_grad():
                outputs = model.generate(inputs["input_ids"], max_new_tokens=10)

            for j, output in enumerate(outputs):
                response = tokenizer.decode(output, skip_special_tokens=True).strip()
                try:
                    score = int(response)
                except ValueError:
                    score = 0
                scores[doc_ids[i + j]] = score

        # Sort and save re-ranked results
        reranked_results[query_id] = {doc_id: scores[doc_id] for doc_id in sorted(scores, key=scores.get, reverse=True)}

    return reranked_results


def create_rerank_prompts(query_text, doc_texts, prompt):
    prompts = []
    for doc_text in doc_texts:
        if prompt == '1':
            prompt = (f"Query: {query_text}\nDocument: {doc_text}\nRate the relevance of this document "
                        "on a scale from 1 (least relevant) to 5 (most relevant). Return only the number.")
        else:
            prompt = (f"For this query about travel, {query_text}, how would you rate this document, {doc_text}, in"
                        "terms of relevance on a scale from 1 to 5, with 5 being the most relevant? Return only"
                        "the number.")
        prompts.append(prompt)
    return prompts


# Automate search, rerank, and save results.
def llm_search(model, prompt, answers, topics, output_file_path):
    base_results = BaseSearch.tf_idf_search(topics, answers)
    print("Base results returned.")

    # Rerank base results with LLM search.
    print("Reranking with LLM...")
    reranked_results1 = rerank_documents_with_llm(topics, base_results, answers, model, prompt)

    # Save the reranked results to an output file.
    BaseSearch.save_to_result_file(reranked_results1, output_file_path)