import sys
from transformers import AutoModelForCausalLM, AutoTokenizer
import LLM
from Evaluation import evaluate_search_result
import os

# Main method to run COS470 Intro to IR Assignment 4
# Retrieve documents from collection for a set of queries
# Search with base model then rerank with LLM
# Version: November 12, 2024
# Author: Abigail Pitcairn


def main(answers, topics_1, topics_2):
    print("Starting main...")

    # Set OS environment to better handle workload
    os.environ['TRANSFORMERS_CACHE'] = '/mnt/netstore1_home/'

    # Initialize LLM model
    model_name = "HuggingFaceTB/SmolLM2-1.7B-Instruct"

    # Name output files
    prompt1_1_output = 'prompt1_1.tsv'
    prompt2_1_output = 'prompt2_1.tsv'
    prompt1_2_output = 'prompt1_2.tsv'
    prompt2_2_output = 'prompt2_2.tsv'

    # Conduct searches with base search, rerank, and file saving.
    # Order of searches:
    # Topic 1 prompt 1, Topic 1 prompt 2, Topic 2 prompt 1, Topic 2 prompt 2.
    LLM.llm_search(model_name, '1', answers, topics_1, prompt1_1_output)
    LLM.llm_search(model_name, '2', answers, topics_1, prompt2_1_output)
    print(f"Search results for topics 1 saved to {prompt1_1_output} and {prompt2_1_output}")
    LLM.llm_search(model_name, '1', answers, topics_2, prompt1_2_output)
    LLM.llm_search(model_name, '2', answers, topics_2, prompt2_2_output)
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