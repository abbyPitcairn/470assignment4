import sys
import BaseSearch
import LLM
from Evaluation import evaluate_search_result
import os

# Main method to run COS470 Intro to IR Assignment 4
# Retrieve documents from collection for a set of queries
# Search with base model then rerank with LLM
# Version: November 12, 2024
# Author: Abigail Pitcairn

def main(answers_filepath, topics_1_filepath, topics_2_filepath):
    print("Starting main...")

    # Load files.
    answers = BaseSearch.load_json_file(answers_filepath)
    topics1 = BaseSearch.load_json_file(topics_1_filepath)
    topics2 = BaseSearch.load_json_file(topics_2_filepath)

    # Base search for each topic file.
    base_result1 = BaseSearch.tf_idf_search(topics1, answers)
    base_result2 = BaseSearch.tf_idf_search(topics2, answers)

    # Set OS environment to better handle workload.
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'

    # Initialize LLM model.
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    # Name output files.
    prompt1_1_output = 'prompt1_1.tsv'
    prompt2_1_output = 'prompt2_1.tsv'
    prompt1_2_output = 'prompt1_2.tsv'
    prompt2_2_output = 'prompt2_2.tsv'

    # Conduct reranking on base results and save to output files.
    LLM.llm_search(model_name, '1', base_result1, topics1, answers, prompt1_1_output)
    LLM.llm_search(model_name, '2', base_result1, topics1, answers, prompt2_1_output)
    print(f"Search results for topics 1 saved to {prompt1_1_output} and {prompt2_1_output}")
    LLM.llm_search(model_name, '1', base_result2, topics2, answers, prompt1_2_output)
    LLM.llm_search(model_name, '2', base_result2, topics2, answers, prompt2_2_output)
    print(f"Search results for topics 2 saved to {prompt1_2_output} and {prompt2_2_output}")

    # COMMENT OUT EVALUATION BEFORE SUBMISSION.
    # Print evaluation metrics for topics 1 with qrel 1 for prompts 1 and 2.
    evaluate_search_result("qrel_1.tsv", prompt1_1_output)
    evaluate_search_result("qrel_1.tsv", prompt2_1_output)


# Terminal Command: python3 Main.py Answers.json topics_1.json
# OR python3 Main.py Answers.json topics_2.json
if __name__ == "__main__":
    if len(sys.argv) != 4:
        print("Usage: python3 Main.py answers.json topics_1.json topics_2.json")
        sys.exit(1)

    answers_file = sys.argv[1]
    topics_1_file = sys.argv[2]
    topics_2_file = sys.argv[3]

    main(answers_file, topics_1_file, topics_2_file)