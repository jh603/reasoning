import pandas as pd
import json
from tqdm import tqdm
from src.agent_prompting.data_loader import load_gsm_ic_data
from src.agent_prompting.prompts import cot_prompt, zero_shot_cot_prompt, ltm_prompt
from src.utils.model_factory import ModelFactory

def decompose_problem(problem, model):
    decomposition_prompt = f"""
Decompose the following problem into a series of clear and concise steps needed to solve it. Do not perform any calculations or include any answers. Each step should explicitly state the action to be performed, and when applicable, mention the values to be used.

Ensure that the last step directly answers the original problem.

**Example:**

Problem: "Sarah is 4'8\". She grows 4 inches. How tall is she now in inches?"

Steps:
1. Convert Sarah's initial height from feet and inches to inches.
2. Note the number of inches Sarah grows.
3. Add the number of inches Sarah grows to her initial height in inches to find her new height.

**Now decompose the following problem:**

Problem: {problem}
"""
    # Call the LLM to get the decomposition (steps without calculations or answers)
    decomposition_response = model.get_response(decomposition_prompt)
    decomposition = extract_steps_from_response(decomposition_response)
    return decomposition_prompt, decomposition_response, decomposition


# Function to extract steps from the LLM's decomposition response
def extract_steps_from_response(response):
    # Split the response into lines
    lines = response.strip().split('\n')
    steps = []
    for line in lines:
        # Remove any numbering and whitespace
        line = line.strip()
        if line and (line[0].isdigit() or line.startswith('-')):
            # Remove numbering and bullets
            step = line.lstrip('0123456789.- ').strip()
            steps.append(step)
    return steps

# Sequential execution of each subproblem in the decomposed strategy
def execute_decomposed_steps(problem, model):
    """
    Executes each step in the decomposed problem sequentially.
    Each step is handled by a separate LLM call, and the previous answers
    are used to modify the next step.
    """
    decomposition_prompt, decomposition_response, decomposition = decompose_problem(problem, model)
    answers = []
    experiment_data = []

    for i, step in enumerate(decomposition):
        if i > 0:
            # Prepare the rewrite prompt for the LLM
            previous_answers = '\n'.join([f"Step {j+1} Answer: {answers[j]}" for j in range(i)])
            
            rewrite_prompt = f"""
Using the answers from previous steps, rewrite the following step by including the necessary values from previous answers. Only replace placeholders with corresponding answers. Do not change the intended action of the step. The problem is provided for reference only.

Problem: {problem}

Previous Answers:
{previous_answers}

Step:
{step}

Remember, only perform the instruction specified in the step. Do not solve the problem in this rewrite.
"""

            # Use the LLM to rewrite the step
            rewritten_step = model.get_response(rewrite_prompt).strip()
        else:
            rewritten_step = step

        # Prepare the prompt for solving the current step
        solve_prompt = f"""
Problem: {problem}

Instructions:
{rewritten_step}

Please provide only the answer to the instruction above. Do not include additional information or perform extra steps.
"""
        # Execute the LLM call for the current step
        response = model.get_response(solve_prompt)
        
        # Store the answer
        answer = response.strip()
        answers.append(answer)

        # Collect experiment data
        experiment_data.append({
            'step_number': i+1,
            'original_step': step,
            'rewritten_step': rewritten_step,
            'rewrite_prompt': rewrite_prompt if i > 0 else None,
            'solve_prompt': solve_prompt,
            'response': response,
            'answer': answer
        })
    # Return answers and experiment data
    return answers, decomposition_prompt, decomposition_response, experiment_data

def calculate_micro_accuracy(problems, predictions):
    correct_answers = 0
    total_problems = len(problems)
    
    for i, problem in enumerate(problems):
        correct_answer = str(problem['answer']).strip()
        predicted_answer = str(predictions[i]).strip()
        
        # Check if correct answer is within the predicted response
        if correct_answer in predicted_answer:
            correct_answers += 1
    
    micro_accuracy = (correct_answers / total_problems) * 100 if total_problems > 0 else 0
    return micro_accuracy

def calculate_macro_accuracy(problems, predictions):
    two_step_problems = [p for p in problems if p['n_steps'] == 2]
    multi_step_problems = [p for p in problems if p['n_steps'] > 2]

    correct_two_step = 0
    correct_multi_step = 0

    # Accuracy for 2-step problems
    two_step_total = len(two_step_problems)
    for i, problem in enumerate(two_step_problems):
        correct_answer = str(problem['answer']).strip()
        predicted_answer = str(predictions[i]).strip()

        if correct_answer in predicted_answer:
            correct_two_step += 1

    # Accuracy for >2-step problems
    multi_step_total = len(multi_step_problems)
    for i, problem in enumerate(multi_step_problems):
        correct_answer = str(problem['answer']).strip()
        index = two_step_total + i  # Adjust index for predictions list
        predicted_answer = str(predictions[index]).strip()

        if correct_answer in predicted_answer:
            correct_multi_step += 1

    # Calculate step accuracies
    two_step_accuracy = (correct_two_step / two_step_total) * 100 if two_step_total > 0 else None
    multi_step_accuracy = (correct_multi_step / multi_step_total) * 100 if multi_step_total > 0 else None

    # Collect accuracies that are not None
    accuracies = []
    if two_step_accuracy is not None:
        accuracies.append(two_step_accuracy)
    if multi_step_accuracy is not None:
        accuracies.append(multi_step_accuracy)

    # Calculate macro accuracy as the average of available accuracies
    if accuracies:
        macro_accuracy = sum(accuracies) / len(accuracies)
    else:
        macro_accuracy = 0

    # Replace None with 'N/A' for display purposes
    two_step_accuracy = two_step_accuracy if two_step_accuracy is not None else 'N/A'
    multi_step_accuracy = multi_step_accuracy if multi_step_accuracy is not None else 'N/A'

    return macro_accuracy, two_step_accuracy, multi_step_accuracy

def evaluate_prompts(problems, predictions):
    micro_acc = calculate_micro_accuracy(problems, predictions)
    macro_acc, two_step_acc, multi_step_acc = calculate_macro_accuracy(problems, predictions)
    
    return {
        "micro_accuracy": micro_acc,
        "macro_accuracy": macro_acc,
        "two_step_accuracy": two_step_acc,
        "multi_step_accuracy": multi_step_acc
    }

def run_prompting_technique(problem_statements, prompt_func, technique_name, model):
    predictions = []
    experiment_data = []
    for problem in tqdm(problem_statements, desc=f"Processing problems with {technique_name}"):
        if prompt_func is not None:
            prompt = prompt_func(problem)
        else:
            prompt = problem  # For techniques without a specific prompt function
        response = model.get_response(prompt)
        predictions.append(response.strip())

        # Collect experiment data
        experiment_data.append({
            'problem': problem,
            'prompt': prompt,
            'response': response,
            'technique': technique_name
        })

    return predictions, experiment_data

def evaluate_techniques(problems, model, context_type='original', dataset_type='', experiment_log=[]):
    """
    Evaluate different techniques (CoT, 0-CoT, LTM, and Decomposed) and return results as a list of dictionaries.
    """
    techniques = [
        ("Chain-of-Thought", cot_prompt),
        ("Zero-Shot Chain-of-Thought", zero_shot_cot_prompt),
        ("Least-to-Most", ltm_prompt),
        ("Decomposed Strategy", None)  # Decomposed Strategy doesn't need a prompt function
    ]

    results = []
    for name, prompt_func in techniques:
        # Select the appropriate context
        if context_type == 'original':
            problem_statements = [p['original_question'] for p in problems]
        elif context_type == 'new':
            problem_statements = [p['new_question'] for p in problems]

        # If it's the Decomposed Strategy, handle differently
        if name == "Decomposed Strategy":
            predictions = []
            for idx, problem in enumerate(tqdm(problem_statements, desc=f"Processing problems with {name}")):
                decomposition_answers, decomposition_prompt, decomposition_response, step_data = execute_decomposed_steps(problem, model)
                predictions.append(decomposition_answers[-1])  # Use the final answer

                # Collect experiment data
                experiment_log.append({
                    'technique': name,
                    'dataset_type': dataset_type,
                    'context_type': context_type,
                    'problem': problem,
                    'decomposition_prompt': decomposition_prompt,
                    'decomposition_response': decomposition_response,
                    'decomposition_steps': step_data,
                    'final_answer': decomposition_answers[-1]
                })
        else:
            # Get model predictions for this prompting technique
            predictions, technique_experiment_data = run_prompting_technique(problem_statements, prompt_func, name, model)
            # Collect experiment data
            for idx, problem in enumerate(problem_statements):
                experiment_log.append({
                    'technique': name,
                    'dataset_type': dataset_type,
                    'context_type': context_type,
                    'problem': problem,
                    'prompt': technique_experiment_data[idx]['prompt'],
                    'response': technique_experiment_data[idx]['response'],
                    'final_answer': predictions[idx]
                })

        # Evaluate micro and macro accuracy for this technique
        accuracy_results = evaluate_prompts(problems, predictions)

        # Store results
        results.append({
            'Technique': name,
            'Dataset': dataset_type,
            'Context': context_type,
            'Micro Accuracy': accuracy_results['micro_accuracy'],
            'Macro Accuracy': accuracy_results['macro_accuracy'],
            '2-Step Accuracy': accuracy_results['two_step_accuracy'],
            '>2-Step Accuracy': accuracy_results['multi_step_accuracy']
        })

    return results

if __name__ == "__main__":
    model = ModelFactory.create_model('gpt-3.5-turbo')
    # model = ModelFactory.create_model('llama-3.1-8b-instruct')

    # Collect all results into a list
    all_results = []
    experiment_log = []

    # Evaluate for both 2step and mstep datasets
    for dataset_type in ['2step', 'mstep']:
        problems = load_gsm_ic_data(dataset_type)[0:100]
        # Add dataset_type to each problem
        for p in problems:
            p['dataset_type'] = dataset_type
        print(f'Loaded {len(problems)} problems from {dataset_type} dataset...\n')

        for context_type in ['original', 'new']:
            print(f"Evaluating {dataset_type} Problems - {context_type.capitalize()} Questions...")
            results = evaluate_techniques(problems, model, context_type=context_type, dataset_type=dataset_type, experiment_log=experiment_log)
            all_results.extend(results)

    # Convert results to DataFrame
    results_df = pd.DataFrame(all_results)

    # Replace 'N/A' with NaN for proper numerical calculations
    results_df.replace('N/A', pd.NA, inplace=True)
    results_df['2-Step Accuracy'] = pd.to_numeric(results_df['2-Step Accuracy'], errors='coerce')
    results_df['>2-Step Accuracy'] = pd.to_numeric(results_df['>2-Step Accuracy'], errors='coerce')

    # Separate results by context
    results_original = results_df[results_df['Context'] == 'original']
    results_new = results_df[results_df['Context'] == 'new']

    # Aggregate results for original context
    grouped_original = results_original.groupby('Technique').agg({
        '2-Step Accuracy': 'mean',
        '>2-Step Accuracy': 'mean',
        'Micro Accuracy': 'mean',
        'Macro Accuracy': 'mean'
    }).reset_index()

    # Aggregate results for new context
    grouped_new = results_new.groupby('Technique').agg({
        '2-Step Accuracy': 'mean',
        '>2-Step Accuracy': 'mean',
        'Micro Accuracy': 'mean',
        'Macro Accuracy': 'mean'
    }).reset_index()

    # Format the accuracies to one decimal place and reorder columns
    for i, df in enumerate([grouped_original, grouped_new]):
        df['2-Step Accuracy'] = df['2-Step Accuracy'].round(1)
        df['>2-Step Accuracy'] = df['>2-Step Accuracy'].round(1)
        df['Micro Accuracy'] = df['Micro Accuracy'].round(1)
        df['Macro Accuracy'] = df['Macro Accuracy'].round(1)

        # Fill NaN values with 'N/A' for display
        df.fillna('N/A', inplace=True)

        # Reorder columns to match the desired format
        df = df[['Technique', '2-Step Accuracy', '>2-Step Accuracy', 'Micro Accuracy', 'Macro Accuracy']]

        # Assign back to the appropriate variable
        if i == 0:
            grouped_original = df
        else:
            grouped_new = df

    # Display the final aggregated results
    print("Final Aggregated Results for Original Questions:")
    print(grouped_original.to_string(index=False))

    print("\nFinal Aggregated Results for New Questions:")
    print(grouped_new.to_string(index=False))

    # Save experiment log to jsonl file
    with open('experiment_log_temp.jsonl', 'w') as f:
        for entry in experiment_log:
            f.write(json.dumps(entry) + '\n')
