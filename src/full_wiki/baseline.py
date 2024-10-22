import argparse
import json
import os
import re

from dotenv import load_dotenv
from tqdm import tqdm

from src.full_wiki.dpr_retriever import DPRRetriever
from src.utils.datasets import load_hotpotqa
from src.utils.hotpotqa_eval import eval
from src.utils.model_factory import Model

def run_baseline(
    dataset_name="full_wiki",
    model_name="gpt-4o-mini",
    num_training_samples=1000,
    batch_size=32
):
    if model_name.lower() in ["meta-llama-3-8b-instruct", "gpt-4o-mini"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    retriever = DPRRetriever()
    
    dataset = load_hotpotqa(dataset_name)
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    offset = 1
    dataset_slice = dataset[offset : num_training_samples + offset]
    
    # Prepare batches of questions
    questions = [qa_pair['question'] for qa_pair in dataset_slice]
    sub_questions = []
    for q in questions:
        prompt = f"""
Decompose the given question into two sub-questions.
Question: {q}

The questions should be answerable independently. You can guess what the answer to the first question is if the second question depends on it

Example:
Question: What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?
Sub-question 1: Which women portrayed Corliss Archer in the film Kiss and Tell?
Sub-question 2: What government position was held by Shirley Temple.
"""
    
    
    total_percentage = 0
    total_samples = 0
    
    # Process questions in batches
    for i in tqdm(range(0, len(questions), batch_size), desc="Processing batches"):
        batch_questions = questions[i:i+batch_size]
        batch_retrieved_documents = retriever.retrieve(batch_questions, top_k=100)
        
        for j, retrieved_documents in enumerate(batch_retrieved_documents):
            qa_pair = dataset_slice[i+j]
            
            correct_titles = set(entry[0] for entry in qa_pair['supporting_facts'])
            retrieved_titles = set(doc['title'] for doc in retrieved_documents)
            
            correct_titles_found = correct_titles.intersection(retrieved_titles)
            percentage_found = (len(correct_titles_found) / len(correct_titles)) * 100
            
            total_percentage += percentage_found
            total_samples += 1
            
            print(f"Sample {i+j+1}:")
            print(f"  Question: {qa_pair['question']}")
            print(f"  Correct titles: {', '.join(correct_titles)}")
            print(f"  Percentage of correct titles found: {percentage_found:.2f}%")
            print(f"  Correct titles found: {', '.join(correct_titles_found)}")
            print(f"  Correct titles not found: {', '.join(correct_titles - correct_titles_found)}")
            print("-" * 50)
    
    # Calculate and print the average percentage
    avg_percentage = total_percentage / total_samples if total_samples > 0 else 0
    print(f"\nAverage percentage of correct titles found: {avg_percentage:.2f}%")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Optimized Baseline on HotpotQA Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="full_wiki",
        help="Name of the dataset to use (default: full_wiki)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4o-mini",
        help="Model to use (e.g., gpt-4, llama3) (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--num_training_samples",
        type=int,
        default=7000,
        help="Number of training samples to process (default: 1000)",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=32,
        help="Batch size for processing questions (default: 32)",
    )

    args = parser.parse_args()

    run_baseline(
        dataset_name=args.dataset,
        model_name=args.model,
        num_training_samples=args.num_training_samples,
        batch_size=args.batch_size,
    )

    # Example Commands:
    # python3 /path/to/your/script.py --dataset full_wiki --model gpt-4o-mini --num_training_samples 1000 --batch_size 32