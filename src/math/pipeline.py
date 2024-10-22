"""
Base pipeline without any heuristics

"""


import csv
import os
import json
import glob
import difflib
import re
from typing import Dict, List, Tuple

import jsonlines

from src.utils.model_factory import Model


model = Model('gpt-3.5-turbo')
# model = Model('gpt-4o-mini')
# model = Model('gpt-4o')


def load_json_file(file_path: str) -> dict:
    """Load a JSON file and return its content as a dictionary."""
    with open(file_path, 'r', encoding='utf-8') as file:
        data = json.load(file)
    return data

def generate_llm_solution(problem: str) -> str:
    """
    Pass the problem to the LLM and get the solution.

    Args:
        problem (str): The problem statement.

    Returns:
        str: The solution generated by the LLM.
    """
    prompt = (
        f"Solve the following problem and provide a detailed solution.\n"
        f"Ensure that the final answer is enclosed within LaTeX box formatting like \\boxed{{...}}.\n\n"
        f"Problem: {problem}\n"
        f"Think step by step"
    )
    response = model.get_response(prompt)
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
        return "Cannot Determine"

    # Prepare evaluation prompt for LLM
    evaluation_prompt = (
        "You are an AI assistant that evaluates the correctness of mathematical solutions.\n\n"
        "Here are two final answers enclosed within \\boxed{...}. Determine if they are mathematically equivalent.\n\n"
        f"Answer 1: \\boxed{{{provided_boxed}}}\n\n"
        f"Answer 2: \\boxed{{{llm_boxed}}}\n\n"
        "Consider different representations (e.g., fractions vs. decimals) as equivalent if they represent the same value.\n"
        "Respond with one of the following options exactly as stated and without additional text:\n"
        "'Equivalent' - if both answers represent the same value.\n"
        "'Incorrect' - if the answers represent different values.\n"
        "'Cannot Determine' - if you cannot determine the equivalence based on the provided information."
    )
    print(evaluation_prompt)

    # Get evaluation from LLM
    evaluation_response = model.get_response(evaluation_prompt).strip().lower()

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
    with jsonlines.open(output_path, mode='w') as writer:
        for result in results:
            writer.write(result)

def evaluate_solutions(
    folder_path: str, 
    max_problems: int
) -> Tuple[int, int, float, List[Dict[str, str]]]:
    """
    Evaluate JSON files in a specific folder up to a maximum number of problems.

    Args:
        folder_path (str): Path to the category folder containing JSON files.
        max_problems (int): Maximum number of problems to evaluate in this folder.

    Returns:
        Tuple[int, int, float, List[Dict[str, str]]]: 
            - Number of correct solutions
            - Number of evaluated files
            - Accuracy percentage
            - List of result dictionaries
    """
    results = []
    correct = 0
    evaluated = 0

    # Find all JSON files in the folder
    json_files = glob.glob(os.path.join(folder_path, '*.json'))

    # Limit to max_problems
    json_files = json_files[:max_problems]

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

            # Generate solution using LLM
            llm_solution = generate_llm_solution(problem)

            # Evaluate correctness by comparing boxed answers via LLM
            correctness = evaluate_correctness(provided_solution, llm_solution)

            # Determine if the evaluation is considered correct
            # Only 'Equivalent' is treated as correct
            if correctness == "Equivalent":
                correct += 1

            evaluated += 1

            # Append the result
            results.append({
                'problem': problem,
                'provided_solution': provided_solution,
                'llm_generated_solution': llm_solution,
                'correctness': correctness
            })

            print(f"    Result: {correctness}\n")

        except Exception as e:
            print(f"    Error processing file {relative_file_path}: {e}\n")
            continue

    accuracy = (correct / evaluated) * 100 if evaluated > 0 else 0.0
    return correct, evaluated, accuracy, results

def main():
    # Configuration Parameters
    ROOT_TEST_FOLDER = "/home/cpp/jerryhuang/math/MATH/test/"  # Root folder containing subfolders
    CURRENT_WORKING_DIR = os.getcwd()  # Directory where the script is run from
    MAX_PROBLEMS_PER_FOLDER = 100  # Set the maximum number of problems to evaluate per folder
    
    for root, dirs, files in os.walk(ROOT_TEST_FOLDER):
        # Extract category name relative to ROOT_TEST_FOLDER
        category = os.path.relpath(root, ROOT_TEST_FOLDER)
        if category == '.':
            category = 'root'  # Files directly under ROOT_TEST_FOLDER, if any
        
        # Define the output JSONL file path in the current working directory
        # Replace any OS-specific path separators with underscores to avoid issues
        sanitized_category = category.replace(os.sep, '_')
        
        if category == 'number_theory':
            output_jsonl_filename = f"{sanitized_category}_results.jsonl"
            output_jsonl_path = os.path.join(CURRENT_WORKING_DIR, output_jsonl_filename)
            
            print(f"Evaluating category: '{category}'")
            
            # Evaluate solutions in the current category folder
            correct, evaluated, accuracy, results = evaluate_solutions(root, MAX_PROBLEMS_PER_FOLDER)
            
            # Save results to JSONL
            save_results_to_jsonl(output_jsonl_path, results)
            
            # Print accuracy for the current category
            print(f"  Accuracy for '{category}': {correct}/{evaluated} = {accuracy:.2f}%")
            print(f"  Results saved to: {output_jsonl_path}\n")
    
    print("Evaluation completed for all categories.")

if __name__ == "__main__":
    main()