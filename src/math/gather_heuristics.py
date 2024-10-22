import os
import json
import glob
import re
from typing import List, Tuple
import jsonlines

from src.utils.model_factory import Model

# Initialize the LLM model
# Replace 'gpt-4o-mini' with your actual model identifier if different
model = Model('gpt-4o-mini')


def load_json_file(file_path: str) -> dict:
    """
    Load a JSON file and return its content as a dictionary.

    Args:
        file_path (str): Path to the JSON file.

    Returns:
        dict: Parsed JSON content.
    """
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


def extract_boxed_answer(solution: str) -> str:
    """
    Extract the content within \boxed{...} from a solution.

    Args:
        solution (str): The full solution text.

    Returns:
        str: The content inside \boxed{...}, or an empty string if not found.
    """
    match = re.search(r'\\boxed\{([^}]+)\}', solution)
    return match.group(1).strip() if match else ""

from typing import List

def generate_heuristics(problem: str, solution: str, model) -> List[str]:
    # Construct the initial prompt for the model
    prompt = (
        f"Problem: {problem}\n"
        f"Solution: {solution}\n\n"
        "What might a professor say to help his students solve problems like these. The advise should be short and concise and generalizable. The advise should focus on the most difficult or crucial parts of the problem."
    )

    # Get the response from the model
    response = model.get_response(prompt)

    print(f'reponse: {response}')
    # Initialize variables for parsing
    return [response]



def collect_heuristics_from_jsons(folder_path: str, max_problems: int = None) -> List[Tuple[str, str]]:
    """
    Iterate through all JSON files in the specified folder and collect general heuristics from each problem.

    Args:
        folder_path (str): Path to the folder containing JSON files.
        max_problems (int, optional): Maximum number of problems to process. Processes all if None.

    Returns:
        List[Tuple[str, str]]: A list of tuples containing (problem, heuristic).
    """
    heuristics_list = []

    # Find all JSON files in the folder and subfolders
    json_files = glob.glob(os.path.join(folder_path, '**', '*.json'), recursive=True)

    # Limit to max_problems if specified
    if max_problems:
        json_files = json_files[:max_problems]

    for idx, file_path in enumerate(json_files, 1):
        relative_file_path = os.path.relpath(file_path, folder_path)
        print(f"Processing file {idx}/{len(json_files)}: {relative_file_path}")

        data = load_json_file(file_path)

        problem = data.get("problem", "").strip()
        provided_solution = data.get("solution", "").strip()

        if not problem or not provided_solution:
            print(f"    Skipping file {relative_file_path} due to missing 'problem' or 'solution' field.\n")
            continue

        # Generate heuristics using the LLM
        heuristics = generate_heuristics(problem, provided_solution, model)

        if heuristics:
            print(f"    Generated {len(heuristics)} heuristic(s).")
            for heuristic in heuristics:
                if heuristic:  # Ensure it's not empty
                    heuristics_list.append((problem, heuristic))
        else:
            print(f"    No generalizable heuristics generated.\n")

    return heuristics_list

def save_heuristics_to_jsonl(heuristics: List[Tuple[str, str]], output_file: str) -> None:
    """
    Save the collected heuristics to a JSONL file, one heuristic per line in JSON format.
    Each line will contain a JSON object with 'problem' and 'heuristic'.

    Args:
        heuristics (List[Tuple[str, str]]): A list of tuples containing (problem, heuristic).
        output_file (str): Path to the output JSONL file.
    """
    try:
        with jsonlines.open(output_file, mode='w') as writer:
            for problem, heuristic in heuristics:
                writer.write({"problem": problem, "heuristic": heuristic})
        print(f"Successfully saved {len(heuristics)} heuristics to '{output_file}'.")
    except Exception as e:
        print(f"Error saving heuristics to '{output_file}': {e}")



def main():
    # Configuration Parameters
    ROOT_TEST_FOLDER = "/home/cpp/jerryhuang/math/MATH/test/number_theory"  # Update this path to your JSON files directory
    OUTPUT_HEURISTICS_FILE = "heuristics_with_problems.jsonl"  # Output file in the current working directory
    MAX_PROBLEMS_PER_FOLDER = 25

    # Collect heuristics from all JSON files
    heuristics = collect_heuristics_from_jsons(ROOT_TEST_FOLDER, MAX_PROBLEMS_PER_FOLDER)

    # Save the heuristics to a text file
    save_heuristics_to_jsonl(heuristics, OUTPUT_HEURISTICS_FILE)


if __name__ == "__main__":
    main()
