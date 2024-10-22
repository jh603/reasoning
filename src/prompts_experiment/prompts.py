from src.utils.model_factory import Model

# Chain-of-Thought Prompt
def cot_prompt(problem):
    exemplar_problem = "Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?"
    exemplar_solution = "Anna has 2 more apples than Elsa, so Anna has 2 + 5 = 7 apples. Elsa and Anna have 5 + 7 = 12 apples together. The answer is 12."
    
    return (
        f"Solve grade school math problems. Feel free to ignore irrelevant information.\n"
        f"Q: {exemplar_problem}\n"
        f"A: {exemplar_solution}\n"
        f"Q: {problem}\nA: Let's think step by step."
    )

# Zero-Shot Chain-of-Thought
def zero_shot_cot_prompt(question):
    return f"Q: {question}\nA: Let's think step by step:"

# Least-to-Most Prompt
def ltm_prompt(problem):
    exemplar_problem = "Elsa has 5 apples. Anna has 2 more apples than Elsa. How many apples do they have together?"
    exemplar_solution = ("Let’s break down this problem:\n"
                         "1. How many apples does Anna have? "
                         "Anna has 2 more apples than Elsa. So Anna has 2 + 5 = 7 apples.\n"
                         "2. How many apples do Elsa and Anna have together? "
                         "Elsa and Anna have 5 + 7 = 12 apples together.\n"
                         "The answer is 12.")
    
    return (
        f"Solve grade school math problems. Feel free to ignore irrelevant information.\n"
        f"A: {exemplar_problem}\n"
        f"A: {exemplar_solution}\n"
        f"Q: {problem}\nA: Let’s break down this problem:"
    )

# Function to decompose the problem into steps
# def decompose_problem(problem, model):
#     decomposition_prompt = f"""
# Decompose the following problem into a series of steps without solving the problem. Each step should clearly outline what inputs it needs, using placeholders like {{answer from step 1}} to indicate inputs from previous steps.

# **Example:**
# Problem: "If John has 10 dollars and spends 3 dollars, how much money does he have left?"

# Steps:
# 1. Determine how much money John initially has.
# 2. Subtract 3 dollars from {{answer from step 1}} to find out how much money John has left.

# **Now decompose the following problem:**

# Problem: {problem}
# """
#     # Call the LLM to get the decomposition (steps with placeholders)
#     decomposition_response = model.get_response(decomposition_prompt)
    
#     # Extract the steps from the LLM response
#     decomposition = extract_steps_from_response(decomposition_response)
    
#     return decomposition

# Function to extract steps from the LLM's decomposition response
# def extract_steps_from_response(response):
#     # Split the response into lines and extract steps
#     lines = response.strip().split('\n')
#     steps = []
#     for line in lines:
#         line = line.strip()
#         if line and (line[0].isdigit() or line.startswith('-')):
#             # Remove numbering and any leading symbols
#             step = line.lstrip('0123456789.- ').strip()
#             steps.append(step)
#     return steps

# Sequential execution of each subproblem in the decomposed strategy
# def execute_decomposed_steps(problem, model):
#     # Get the decomposition of the problem from the LLM
#     decomposition = decompose_problem(problem, model)
#     print(f'Decomposition: {decomposition}\n')
    
#     answers = []
    
#     for i, step in enumerate(decomposition):
#         if i > 0:
#             # Prepare the rewrite prompt for the LLM
#             previous_answers = '\n'.join([f"Step {j+1} Answer: {answers[j]}" for j in range(i)])
            
#             rewrite_prompt = f"""
# Using the answers from previous steps, rewrite the following step by replacing the placeholders with the corresponding answers. The problem is provided for reference only.

# Problem: {problem}

# Previous Answers:
# {previous_answers}

# Step:
# {step}

# Remember, only perform the instruction specified in the step. Do not solve the problem in this rewrite.
# """
#             # Use the LLM to rewrite the step
#             step = model.get_response(rewrite_prompt).strip()
    
#         # Prepare the prompt for solving the current step
#         solve_prompt = f"""
# Problem: {problem}

# Instructions:
# {step}

# Please provide only the answer to the instruction above. Do not include additional information or perform extra steps.
# """
#         # Execute the LLM call for the current step
#         print(f'Step {i+1} Instructions:\n{step}')
#         response = model.get_response(solve_prompt)
#         print(f'Step {i+1} Answer: {response}\n')
        
#         # Store the answer
#         answers.append(response.strip())
    
#     return answers


# Function to extract the answer from the LLM's response
def extract_answer_from_response(response):
    # Assuming the LLM provides the answer directly
    return response.strip()

# Execute a specific prompt using the provided model and prompt function
def execute_prompt(question, prompt_func, model):
    prompt = prompt_func(question)
    print(f'Prompt: {prompt}')
    resp = model.get_response(prompt)
    print(f'Response: {resp}\n\n')
    return resp

# Run a specific prompting technique on a dataset
def run_prompting_technique(problems, prompt_func, technique_name, model, strategy="regular"):
    """
    Run a specific prompting technique on a set of problems.
    problems: List of questions (either original or with irrelevant context)
    prompt_func: The specific prompting function (e.g., CoT, Zero-shot CoT, etc.)
    model: The model instance used for querying
    strategy: The strategy to use ("regular" or "decomposed")
    """
    predictions = []

    # if strategy == "decomposed":
    #     # If using the decomposed strategy, handle it via the dynamic decomposition method
    #     for problem in problems:
    #         decomposition_answers = execute_decomposed_steps(problem, model)
    #         # For evaluation, we assume the final answer (from the last step) is the one needed
    #         predictions.append(decomposition_answers[-1])
    # else:
    
    
    # Regular strategy: CoT, LTM, etc.
    for problem in problems:
        response = execute_prompt(problem, prompt_func, model)
        predictions.append(response.strip())  # Collect predictions for evaluation

    return predictions  # Return the list of predictions for evaluation

# Main function for running the experiments
if __name__ == "__main__":
    model = Model('code-davinci-002')  # Initialize the model

    # Example problems (replace with actual dataset)
    problems = [
        "Elsa has 1221 apples. Anna has 2 more apples than twice of Elsa's. How many apples do they have together?"
    ]

    # Run the decomposed strategy
    decomposed_predictions = run_prompting_technique(problems, None, "Decomposed Strategy", model, strategy="decomposed")
    print(f"Decomposed Strategy Predictions: {decomposed_predictions}")
