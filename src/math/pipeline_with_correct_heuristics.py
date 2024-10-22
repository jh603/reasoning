import os
import json
import glob
import re
import csv
from typing import Dict, List, Tuple, Set

import jsonlines
import pickle
import numpy as np
from sklearn.metrics.pairwise import cosine_similarity
from tqdm import tqdm

import torch
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

from src.utils.model_factory import Model

llm_model = Model('gpt-3.5-turbo')
# llm_model = Model('gpt-4o-mini')


def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its content as a dictionary."""
    try:
        with open(file_path, 'r', encoding='utf-8') as file:
            data = json.load(file)
        return data
    except json.JSONDecodeError as e:
        print(f"Error decoding JSON for file {file_path}: {e}")
        return {}
    except Exception as e:
        print(f"Unexpected error loading file {file_path}: {e}")
        return {}

def load_heuristics_from_jsonl(heuristics_file_path: str) -> Dict[str, List[str]]:
    """
    Load heuristics from a JSONL file mapping problems to their heuristics.

    Args:
        heuristics_file_path (str): Path to the heuristics JSONL file.

    Returns:
        Dict[str, List[str]]: A dictionary mapping each problem to a list of its heuristics.
    """
    heuristics_dict = {}

    try:
        with jsonlines.open(heuristics_file_path, mode='r') as reader:
            for idx, obj in enumerate(reader, 1):
                if not isinstance(obj, dict) or 'problem' not in obj or 'heuristic' not in obj:
                    print(f"Line {idx} in heuristics file is malformed: {obj}")
                    continue

                problem = obj['problem'].strip()
                heuristic = obj['heuristic'].strip()

                if not problem or not heuristic:
                    print(f"Line {idx} in heuristics file is malformed: {obj}")
                    continue

                if problem not in heuristics_dict:
                    heuristics_dict[problem] = []
                heuristics_dict[problem].append(heuristic)

        print(f"Loaded heuristics for {len(heuristics_dict)} problems from '{heuristics_file_path}'.")
        return heuristics_dict

    except FileNotFoundError:
        print(f"Heuristics file not found at '{heuristics_file_path}'. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Error loading heuristics from '{heuristics_file_path}': {e}")
        exit(1)


def generate_llm_solution(problem: str, heuristics: List[str] = None) -> str:
    """
    Generate a solution to the problem using the LLM, optionally with provided heuristics.

    Args:
        problem (str): The problem statement.
        heuristics (List[str], optional): A list of heuristics to include in the prompt. Defaults to None.

    Returns:
        str: The solution generated by the LLM.
    """
    if heuristics:
        # Prepare heuristics string
        heuristics_str = "\n".join([f"{idx + 1}. {heuristic}" for idx, heuristic in enumerate(heuristics)])

        # print(f'heuristics_str: {heuristics_str}')
        # Construct the prompt with heuristics included
        prompt = (
            "Solve the math problem step by step. First consider the merit of the advice at the bottom.:"
            "Make sure the final answer satisfies the problem's constraints and present it in LaTeX box formatting like \\boxed{...}.\n\n"
            f"Problem: {problem}\n\n"
            f"{heuristics_str}\n\n"
        )
    else:
        # Construct the prompt without heuristics
        prompt = (
            f"Solve the following problem and provide a detailed solution.\n"
            f"Ensure that the final answer is enclosed within LaTeX box formatting like \\boxed{{...}}.\n\n"
            f"Problem: {problem}"
            "\nThink step by step."
        )

    print(prompt)
    response = llm_model.get_response(prompt)
    print(response)
    return response.strip()

def extract_boxed_answer(solution: str) -> str:
    """
    Extracts the content within \boxed{...} from a solution.

    Args:
        solution (str): The full solution text.

    Returns:
        str: The content inside \boxed{...}, or an empty string if not found.
    """
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    return match.group(1).strip() if match else ""

def evaluate_correctness(provided_solution: str, llm_solution: str) -> str:
    """
    Use the LLM to evaluate whether the provided solution and the LLM-generated solution are correct and equivalent.
    Allows different formatting if the answers are conceptually the same.

    Args:
        provided_solution (str): The solution provided in the JSON file.
        llm_solution (str): The solution generated by the LLM.

    Returns:
        str: 'Equivalent', 'Incorrect', or 'Cannot Determine'.
    """
    # Extract boxed answers
    provided_boxed = extract_boxed_answer(provided_solution)
    llm_boxed = extract_boxed_answer(llm_solution)

    # Debugging: Print extracted boxed answers
    print(f"    Provided boxed answer: {provided_boxed}")
    print(f"    LLM boxed answer: {llm_boxed}")

    # Check if both boxed answers are present
    if not provided_boxed or not llm_boxed:
        print("    Missing boxed answer in provided or LLM solution.")
        return "Cannot Determine"

    # Prepare evaluation prompt for LLM
    evaluation_prompt = (
        "You are an AI assistant that evaluates the correctness of mathematical solutions based on the following heuristics:\n\n"
        "Here are two final answers enclosed within \\boxed{...}. Determine if they are mathematically equivalent.\n\n"
        f"Answer 1: \\boxed{{{provided_boxed}}}\n\n"
        f"Answer 2: \\boxed{{{llm_boxed}}}\n\n"
        "Consider different representations (e.g., fractions vs. decimals) as equivalent if they represent the same value.\n"
        "Respond with only one of the following options exactly as stated and without any additional text:\n"
        "'Equivalent' - if both answers represent the same value.\n"
        "'Incorrect' - if the answers represent different values.\n"
        "'Cannot Determine' - if you cannot determine the equivalence based on the provided information."
    )
    # print(f"    Evaluation prompt:\n{evaluation_prompt}\n")

    # Get evaluation from LLM
    evaluation_response = llm_model.get_response(evaluation_prompt).strip().lower()

    # Map LLM response to standardized categories
    if "equivalent" in evaluation_response:
        return "Equivalent"
    elif "incorrect" in evaluation_response:
        return "Incorrect"
    else:
        # If the response is unclear or 'Cannot Determine'
        return "Cannot Determine"

def save_results_to_jsonl(output_path: str, results: List[Dict[str, str]]) -> None:
    """
    Save the evaluation results to a JSON Lines (JSONL) file.

    Args:
        output_path (str): Path to the JSONL file.
        results (List[Dict[str, str]]): List of result dictionaries.
    """
    try:
        with jsonlines.open(output_path, mode='w') as writer:
            for result in results:
                writer.write(result)
        print(f"Successfully saved results to '{output_path}'.")
    except Exception as e:
        print(f"Error saving results to '{output_path}': {e}")

def evaluate_solutions(folder_path: str, heuristics_dict: Dict[str, List[str]], max_problems: int) -> Tuple[int, int, float, int, int, float, List[Dict[str, str]]]:
    """
    Evaluate JSON files in a specific folder up to a maximum number of problems.
    Evaluates both with and without heuristics.

    Args:
        folder_path (str): Path to the category folder containing JSON files.
        heuristics_dict (Dict[str, List[str]]): A dictionary mapping each problem to its heuristics.
        max_problems (int): Maximum number of problems to evaluate in this folder.

    Returns:
        Tuple[int, int, float, int, int, float, List[Dict[str, str]]]: 
            - Number of correct solutions without heuristics
            - Number of evaluated solutions without heuristics
            - Accuracy percentage without heuristics
            - Number of correct solutions with heuristics
            - Number of evaluated solutions with heuristics
            - Accuracy percentage with heuristics
            - List of result dictionaries
    """
    results = []
    correct_without = 0
    evaluated_without = 0
    correct_with = 0
    evaluated_with = 0

    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    # Limit to max_problems
    json_files = json_files[:max_problems]

    print(f"Found {len(json_files)} JSON files in '{folder_path}'.")

    for idx, file_path in enumerate(json_files, 1):
        relative_file_path = os.path.relpath(file_path, folder_path)
        print(f"  Processing file {idx}/{len(json_files)}: {relative_file_path}")

        try:
            data = load_json_file(file_path)
            problem = data.get("problem", "").strip()
            provided_solution = data.get("solution", "").strip()

            if not problem or not provided_solution:
                print(f"    Skipping due to missing problem or solution.\n")
                continue

            # Step 1: Generate solution without heuristics
            initial_solution = generate_llm_solution(problem)
            print(f"    Initial solution generated without heuristics.")

            # Step 2: Evaluate correctness of initial solution
            correctness_without = evaluate_correctness(provided_solution, initial_solution)
            print(f"    Correctness without heuristics: {correctness_without}")

            if correctness_without == "Equivalent":
                correct_without += 1
            if correctness_without in ["Equivalent", "Incorrect"]:
                evaluated_without += 1  # Only count if evaluation was possible

            # Step 3: Retrieve heuristics for the current problem
            top_k_heuristics = heuristics_dict.get(problem, [])
            if top_k_heuristics:
                print(f"    Retrieved {len(top_k_heuristics)} heuristics for the problem.")
            else:
                print(f"    No heuristics found for the problem.")

            # Step 4: Generate final solution with heuristics
            final_solution = generate_llm_solution(problem, heuristics=top_k_heuristics if top_k_heuristics else None)
            print(f"    Final solution generated with heuristics.")

            # Step 5: Evaluate correctness of final solution
            correctness_with = evaluate_correctness(provided_solution, final_solution)
            print(f"    Correctness with heuristics: {correctness_with}")

            if correctness_with == "Equivalent":
                correct_with += 1
            if correctness_with in ["Equivalent", "Incorrect"]:
                evaluated_with += 1  # Only count if evaluation was possible

            # Append the result
            results.append({
                'problem': problem,
                'provided_solution': provided_solution,
                'llm_initial_solution': initial_solution,
                'correctness_without_heuristics': correctness_without,
                'heuristics_used': top_k_heuristics,
                'llm_final_solution': final_solution,
                'correctness_with_heuristics': correctness_with
            })

            print(f"    Result without heuristics: {correctness_without}")
            print(f"    Result with heuristics: {correctness_with}\n")

        except Exception as e:
            print(f"    Error processing file {relative_file_path}: {e}\n")
            continue

    # Calculate accuracies
    accuracy_without = (correct_without / evaluated_without) * 100 if evaluated_without > 0 else 0.0
    accuracy_with = (correct_with / evaluated_with) * 100 if evaluated_with > 0 else 0.0

    return correct_without, evaluated_without, accuracy_without, correct_with, evaluated_with, accuracy_with, results

def main():
    # Configuration Parameters
    ROOT_TEST_FOLDER = "/home/cpp/jerryhuang/math/MATH/test/"  # Root folder containing subfolders
    CURRENT_WORKING_DIR = os.getcwd()  # Directory where the script is run from
    MAX_PROBLEMS_PER_FOLDER = 1  # Set to an integer to limit the number of problems per folder
    HEURISTICS_FILE_PATH = "/home/cpp/jerryhuang/reasoning/src/math/heuristics_with_problems.jsonl"

    # Step 1: Load heuristics from the CSV file
    heuristics_dict = load_heuristics_from_jsonl(HEURISTICS_FILE_PATH)
    # print(heuristics_dict)

    # Traverse all subdirectories in ROOT_TEST_FOLDER
    for root, dirs, files in os.walk(ROOT_TEST_FOLDER):
        # Extract category name relative to ROOT_TEST_FOLDER
        category = os.path.relpath(root, ROOT_TEST_FOLDER)
        if category == '.':
            category = 'root'  # Files directly under ROOT_TEST_FOLDER, if any

        # Define the output JSONL file path in the current working directory
        # Replace any OS-specific path separators with underscores to avoid issues
        sanitized_category = category.replace(os.sep, '_')
        
        if category == 'number_theory':
            output_jsonl_filename = f"{sanitized_category}_heuristics_results.jsonl"
            output_jsonl_path = os.path.join(CURRENT_WORKING_DIR, output_jsonl_filename)

            print(f"Evaluating category: '{category}'")

            # Evaluate solutions in the current category folder
            correct_without, evaluated_without, accuracy_without, correct_with, evaluated_with, accuracy_with, results = evaluate_solutions(
                root, heuristics_dict, MAX_PROBLEMS_PER_FOLDER
            )

            # Save results to JSONL
            save_results_to_jsonl(output_jsonl_path, results)

            # Print accuracies for the current category
            print(f"  Accuracy without heuristics for '{category}': {correct_without}/{evaluated_without} = {accuracy_without:.2f}%")
            print(f"  Accuracy with heuristics for '{category}': {correct_with}/{evaluated_with} = {accuracy_with:.2f}%")
            print(f"  Results saved to: {output_jsonl_path}\n")

        # print("Evaluation completed for all categories.")

if __name__ == "__main__":
    main()
