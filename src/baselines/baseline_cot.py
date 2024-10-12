import argparse
import json
import os
import re

from dotenv import load_dotenv

from src.utils.datasets import load_hotpotqa
from src.utils.hotpotqa_eval import eval
from src.utils.model_factory import Model

load_dotenv()


def parse_response(response):
    answer = ""
    references = []

    # Extract FINAL_ANSWER using regex
    answer_match = re.search(r"FINAL_ANSWER:\s*(.+)", response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Extract FINAL_REFERENCES using regex
    references_match = re.search(
        r"FINAL_REFERENCES:\s*(\[.+\])", response, re.IGNORECASE | re.DOTALL
    )
    if references_match:
        ref_string = references_match.group(1)
        # Use regex to find all [Title, Number] patterns
        ref_list = re.findall(r"\[([^,\]]+),\s*(\d+)\]", ref_string)
        for ref in ref_list:
            title, index = ref
            references.append([title.strip(), int(index)])

    return answer, references


def validate_response(
    question, answer, references, model_instance, conversation_history
):
    """
    Validates the primary LLM agent's response using the Validator Agent.

    Args:
        question (str): The original question.
        correct_answer (str): The correct concise answer.
        correct_sp (list of lists): The correct supporting facts as [Title, Sentence Number].
        answer (str): The primary LLM agent's final answer.
        references (list of lists): The primary LLM agent's references as [Title, Sentence Number].
        model_instance (Model): The LLM model instance for the Validator Agent.
        conversation_history (list): The conversation history between agents.

    Returns:
        str: "PASS" if the response is correct and complete, otherwise "FAIL: [Reason]"
    """

    # Prepare the Validator Agent's prompt
    validator_prompt = f"""
You are a Validator Agent tasked with reviewing responses from the Primary LLM Agent. Your role is to ensure the correctness and sufficiency of the responses based solely on the conversation history.

**Instructions:**

1. **Extract Referenced Sentences**:
    - Identify and list all sentences referenced in the **FINAL_REFERENCES** section.

2. **Reconstruct the Answer**:
    - Using the sub-questions and the extracted referenced sentences, attempt to answer the original **Question**.

3. **Compare Answers**:
    - Compare your reconstructed answer with the **FINAL_ANSWER** provided by the Primary LLM Agent.

4. **Evaluate References**:
    - Assess whether the **FINAL_REFERENCES** sufficiently support the answer.

5. **Assess Entity Names**:
    - Verify that the entity names in the **FINAL_ANSWER** exactly match those in the original documents.

6. **Provide Feedback**:
    - If the reconstructed answer matches the **FINAL_ANSWER** and the references are sufficient, output "VALIDATION_RESULT: PASS".
    - If there are discrepancies or insufficient references, output "VALIDATION_RESULT: FAIL: [Reason for failure]".

**Example Responses:**

- **VALIDATION_RESULT: PASS**
- **VALIDATION_RESULT: FAIL: The FINAL_ANSWER does not match the reconstructed answer based on the sub-questions.**
- **VALIDATION_RESULT: FAIL: Missing reference [Document Title, Sentence Number].**
- **VALIDATION_RESULT: FAIL: The FINAL_ANSWER uses a different entity name than in the original documents.**

**Conversation History:**

**Question:** {question}

**Primary LLM Agent's Response:**

**Answer:** {answer}

**References:** {references}

**Please evaluate the response and provide feedback using the specified delimiter.**
"""

    # Append the validator prompt to the conversation history
    conversation_history.append({"role": "user", "content": validator_prompt})

    # Get the validator's feedback
    validator_response = model_instance.get_response_with_history(conversation_history)

    # Clean the response
    validator_feedback = validator_response.strip()

    print(f"validator_feedback: {validator_feedback}")
    return validator_feedback


def run_baseline(
    dataset_name="distractor",
    model_name="gpt-4o-mini",
    output_path=None,
    num_training_samples=1000,
    max_iterations=3,  # Maximum number of validation attempts
):
    # Load the dataset
    dataset = load_hotpotqa(dataset_name)
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    # Initialize the Primary and Validator LLM Agents (using the same model instance)
    if model_name.lower() in ["meta-llama-3-8b-instruct", "gpt-4o-mini"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    i = 0
    offset = 1
    for qa_pair in dataset[offset : num_training_samples + offset]:
        i += 1
        final_output = ""
        for entry in qa_pair["context"]:
            title = entry[0]
            sentences = entry[1]

            # Prepare the content with sentence indices
            flattened_content = "\n".join(
                [f"({i}) {sentence}" for i, sentence in enumerate(sentences)]
            )
            final_output += f"Title: {title}\nContent: {flattened_content}\n\n"

        # Initialize iteration counter and success flag
        iteration = 0
        success = False

        # Initialize conversation history for the model
        conversation_history = []

        while iteration < max_iterations and not success:
            # Define the prompt with the refined instructions
            prompt = f"""
Please answer the following question using only the information provided in the documents below.

**Instructions:**

1. **Paraphrase Documents**:
    - Rewrite **each** provided document **sentence by sentence** in your own words. Keep the language as simple as possible. Do not simply copy the sentences.
    - **Paraphrase every sentence** in each document, **regardless of whether it seems directly useful** for answering the question.
    - **Preserve Proper Nouns and Entity Names**: When paraphrasing, keep all names of people, places, organizations, titles, and other entities exactly as they appear in the original documents.
    - Maintain the original **sentence indices** `(0), (1), ...` for each sentence.
    - Ensure the format remains consistent, clearly separating each document.
    - **Do not omit or skip any documents or sentences.**

2. **Generate Sub-Questions**:
    - Create a sequence of **two or more sub-questions** that can help in answering the main multi-hop question.

3. **Answer Sub-Questions**:
    - Provide answers to each sub-question.
    - For each answer, include **all relevant citations** to the specific parts of the documents that support your response.
    - **List every sentence number used** to answer each sub-question.

4. **Explain Reasoning**:
    - Write a paragraph explaining your overall reasoning for answering the main question.
    - Incorporate the answers to the sub-questions with appropriate citations.
    - **Use the exact entity names** as presented in the original documents when referring to them.
    - **List every sentence number used** in this explanation.

5. **Provide Final Answer**:
    - The answer should be **very concise**, such as a year, a simple yes/no, or a person's name.
    - Use wording **exactly as it appears** in the original documents.
    - **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
    - **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.

6. **List References**:
    - **Aggregate** all references cited in both the sub-question answers and the explanation.
    - **Ensure that every referenced sentence is included** in this list.
    - Format each reference as `[Document Title, Sentence Number]`.
    - **You must cite at least 2 different document titles.**

7. **Handle Pronouns**:
    - If a sentence contains pronouns (e.g., 'he', 'she', 'they', 'it'), include the preceding sentence(s) that clarify who or what the pronoun refers to.
    - Ensure that all necessary context is captured for the answer to be fully understood.

8. **Use Unique Delimiters**:
    - Put "FINAL_ANSWER:" before your final answer.
    - Put "FINAL_REFERENCES:" before your final list of references.

**Examples:**

- **Good Concise Answers**:
    - 'an actor'
    - 'Yes'
    - 'October 1, 1776'
    - 'Steve Jobs'
    - 'a scholar'

- **Bad Concise Answers**:
    - 'an actor based in the US'
    - 'Yes they are the same'
    - 'a scholar during the past'

**Example Response:**

*Explanation:*  
The second sentence in the document titled "American History" states, "The United States was founded in 1776," indicating the founding year. The fourth sentence in the "George Washington" document mentions, "Washington was elected the first president the year the US was founded," linking Washington's election to the founding year. Therefore, based on these documents, the answer to the question "What year was George Washington elected president?" is 1776.

*Answer:*  
1776

*References:*  
[American History, 2], [George Washington, 4]

### START ANSWER
1. **Paraphrased Documents:**  
   *(Paraphrased content of the documents, maintaining sentence indices)*

2. **Sub-Questions:**  
   1. *(First sub-question)*  
   2. *(Second sub-question)*  
   *(Add more sub-questions if necessary)*

3. **Sub-Question Answers:**  
   1. ***[First Sub-Question]***  
      *Answer:* *(Answer to the first sub-question)*  
      *References:* [Document Title, Sentence Number], ...  

   2. ***[Second Sub-Question]***  
      *Answer:* *(Answer to the second sub-question)*  
      *References:* [Document Title, Sentence Number], ...  

   *(Add more sub-question answers if necessary)*

4. **Explanation:**  
   *(A paragraph explaining your reasoning for the main question, incorporating the sub-question answers and citations)*  
   *Referenced Sentences:* *(List of all sentence numbers used in this explanation)*

5. **Answer:**  
   FINAL_ANSWER: *(Final concise answer to the main question, following the guidelines for wording and support)*

6. **References:**  
   FINAL_REFERENCES: [Document Title, Sentence Number], [Document Title, Sentence Number], ...

**Question:** {qa_pair["question"]}

**Documents:**  
{final_output}

### END ANSWER
"""

            # Append the prompt to the conversation history
            conversation_history.append({"role": "user", "content": prompt})

            # Get the primary LLM's response
            response = model_instance.get_response_with_history(conversation_history)
            print(f"\n\n{response}")
            answer, references = parse_response(response)

            # Append the primary response to the conversation history
            conversation_history.append({"role": "assistant", "content": response})

            # Print the current iteration's details
            print(f'Question ID: {qa_pair["_id"]}')
            print(f"Iteration: {iteration + 1}")
            print(f"Answer: {answer}")
            print(f"References: {references}")

            # Validate the response
            validation_feedback = validate_response(
                question=qa_pair["question"],
                answer=answer,
                references=references,
                model_instance=model_instance,
                conversation_history=conversation_history.copy(),  # Pass a copy to avoid mutation
            )

            print(f"Validation Feedback: {validation_feedback}")

            start = validation_feedback.find("VALIDATION_RESULT:")
            result = validation_feedback[start + len("VALIDATION_RESULT:") :].strip()
            if result.upper() == "PASS":
                success = True
                predictions["answer"][qa_pair["_id"]] = answer
                predictions["sp"][qa_pair["_id"]] = references
            else:
                iteration += 1
                if iteration >= max_iterations:
                    # If max iterations reached without success, log as failure
                    predictions["answer"][qa_pair["_id"]] = answer
                    predictions["sp"][qa_pair["_id"]] = references
                    print(f"Failed to validate after {max_iterations} iterations.")
                else:
                    print(f"Validation failed: {validation_feedback}. Retrying...")

        print("-" * 50)

    if output_path is None:
        output_path = f"CoT_{dataset_name}_{model_name}.json"

    predictions_path = f"{output_path}_predictions.json"

    try:
        with open(predictions_path, "w") as pred_file:
            json.dump(predictions, pred_file, indent=4)
        print(f"Predictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

    # Evaluate predictions
    if dataset_name == "distractor":
        eval_path = os.getenv("HOTPOTQA_DEV_DISTRACTOR")
    else:
        eval_path = os.getenv("HOTPOTQA_DEV_FULLWIKI")
    eval(predictions_path, eval_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Distractor Baseline on HotpotQA Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="distractor",
        help="Name of the dataset to use (default: distractor)",
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
        default=10,
        help="Number of training samples to process (default: 10)",
    )

    args = parser.parse_args()

    run_baseline(
        dataset_name=args.dataset,
        model_name=args.model,
        num_training_samples=args.num_training_samples,
    )

    # Example Commands:
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot.py --dataset distractor --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot.py --dataset distractor --model gpt-4o-mini --num_training_samples 1000

    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot.py --dataset fullwiki --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot.py --dataset fullwiki --model gpt-4o-mini --num_training_samples 1000
