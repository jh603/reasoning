import argparse
import concurrent.futures
import json
import os
import re

from dotenv import load_dotenv

from src.utils.datasets import load_hotpotqa
from src.utils.hotpotqa_eval import eval
from src.utils.model_factory import Model

load_dotenv()


def fix_json_string(input_string):
    # Remove 'response:' prefix and '```json' markers if present
    input_string = re.sub(r"^response:\s*", "", input_string.strip())
    input_string = re.sub(r"^```json\s*|\s*```$", "", input_string, flags=re.MULTILINE)

    # Replace unescaped single quotes inside double-quoted strings
    input_string = re.sub(
        r'(?<=["])([^"]*?)\\\'([^"]*?)(?=["])', r"\1'\2", input_string
    )

    # Remove trailing commas in objects and arrays
    input_string = re.sub(r",\s*([\]}])", r"\1", input_string)

    input_string = input_string.strip()

    try:
        # Attempt to load and format the JSON string
        json_obj = json.loads(input_string)
        return json.dumps(json_obj, indent=4)
    except json.JSONDecodeError as e:
        return f"Error: Unable to parse JSON string. {str(e)}"


def parse_response(response):
    """
    Parses the AI model's response to extract the answer and references.

    Expected response format:
        Explanation: <explanation>
        Answer: <answer>
        References: [Title1, index1], [Title2, index2], ...
    """
    answer_pattern = r"Answer:\s*(.+)"
    references_pattern = r"References:\s*\[(.*)\]"

    answer_match = re.search(answer_pattern, response, re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""

    references_match = re.search(references_pattern, response, re.IGNORECASE)
    references = []

    if references_match:
        references_str = references_match.group(1)
        # Split by '], [' to separate different references
        references_list = re.split(r"\],\s*\[", references_str)

        for ref in references_list:
            ref = ref.replace("[", "").replace("]", "")  # Clean up brackets
            title_idx = ref.split(",")  # Split by comma to get title and index
            if len(title_idx) >= 2:
                title = ",".join(
                    title_idx[:-1]
                ).strip()  # In case title contains commas
                try:
                    idx = int(title_idx[-1].strip())  # Convert last part to int
                    references.append([title, idx])
                except ValueError:
                    print(f"Invalid reference index in: {ref}")

    return answer, references


def parse_important_sentences(response):
    """
    Parses the AI model's JSON response to extract important sentences.

    Expected JSON format:
    {
        "title": "Document Title",
        "important_sentences": [
            {"index": 1, "sentence": "First important sentence."},
            {"index": 3, "sentence": "Third important sentence."},
        ]
    }
    """
    try:
        # print(f'response: {response}')
        response = fix_json_string(response)
        # print(f'response: {response}')
        data = json.loads(response)
        title = data.get("title", "")
        important_sentences = data.get("important_sentences", [])
        # Validate the structure
        if not isinstance(title, str):
            print("Invalid or missing 'title' in JSON response.")
            return None, []
        if not isinstance(important_sentences, list):
            print("Invalid or missing 'important_sentences' in JSON response.")
            return None, []
        # Extract sentences
        sentences = []
        for item in important_sentences:
            idx = item.get("index")
            sentence = item.get("sentence")
            if isinstance(idx, int) and isinstance(sentence, str):
                sentences.append((idx, sentence.strip()))
            else:
                print(f"Invalid sentence entry: {item}")
        return title, sentences
    except json.JSONDecodeError:
        print("Failed to decode JSON response.")
        return None, []


def run_distractor_baseline(
    dataset_name="distractor",
    model_name="gpt-4",
    output_path=None,
    num_training_samples=1000,
):
    dataset = load_hotpotqa(dataset_name)
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    if model_name.lower() in ["meta-llama-3-8b-instruct"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    i = 0
    for qa_pair in dataset[:num_training_samples]:
        i += 1
        # if i % 25 == 0:
        print(f"{i} questions processed")
        question = qa_pair["question"]
        context = qa_pair["context"]  # List of [title, sentences]
        important_sentences = []  # List of [title, index, sentence]

        # Step 1: Identify important sentences from each document
        # Prepare prompts for all documents in the context
        prompts = []
        titles = []
        for entry in context:
            title = entry[0]
            sentences = entry[1]

            # Flatten the content with sentence indices
            flattened_content = "\n".join(
                [f"({i}) {sentence}" for i, sentence in enumerate(sentences)]
            )
            document = f"Title: {title}\nContent:\n{flattened_content}\n\n"

            # Prompt to identify important sentences in JSON format
            prompt = f"""
Please identify any important sentences in the following document that help answer the question. 

Provide the output in JSON format with the document title and a list of important sentences along with their indices. If there are no important sentences, include an empty list [].

Question: {question}

Document:
{document}

Example of the expected JSON format:
{{
    "title": "Document Title",
    "important_sentences": [
        {{"index": 1, "sentence": "First important sentence."}},
        {{"index": 3, "sentence": "Third important sentence."}},
    ]
}}
    
Important sentences:
"""
            prompts.append(prompt)
            titles.append(title)

        # Function to process a single prompt
        def process_prompt(prompt, expected_title):
            response = model_instance.get_response(prompt)
            parsed_title, extracted_sentences = parse_important_sentences(response)

            if parsed_title and parsed_title == expected_title:
                return extracted_sentences
            else:
                # You can choose to handle mismatches or parsing failures differently
                print(
                    f"Title mismatch or parsing failed for document titled '{expected_title}'."
                )
                return []

        # Use ThreadPoolExecutor to send prompts in parallel
        with concurrent.futures.ThreadPoolExecutor(max_workers=10) as executor:
            # Submit all prompts
            future_to_title = {
                executor.submit(process_prompt, prompt, title): title
                for prompt, title in zip(prompts, titles)
            }
            # Collect the results as they complete
            for future in concurrent.futures.as_completed(future_to_title):
                title = future_to_title[future]
                try:
                    extracted_sentences = future.result()
                    if extracted_sentences:
                        for idx, sentence in extracted_sentences:
                            important_sentences.append([title, idx, sentence])
                        # print(f"Identified {len(extracted_sentences)} important sentence(s) in '{title}'.")
                    else:
                        pass
                        # print(f"No important sentences identified in '{title}'.")
                except Exception as exc:
                    print(f"Generated an exception for title '{title}': {exc}")

        # If no important sentences found across all documents, skip to next QA pair
        if not important_sentences:
            print("No important sentences identified in any document.")
            predictions["answer"][qa_pair["_id"]] = ""
            predictions["sp"][qa_pair["_id"]] = []
            continue

        # Step 2: Aggregate important sentences and form the final prompt
        combined_content = ""
        references_set = set()  # To avoid duplicate references

        for item in important_sentences:
            title, idx, sentence = item
            combined_content += f"Title: {title}\nContent: ({idx}) {sentence}\n\n"
            references_set.add((title, idx))

        # Format references for the final answer
        formatted_references = [
            f"[{title}, {idx}]" for title, idx in sorted(references_set)
        ]
        formatted_references_str = ", ".join(formatted_references)

        # Final prompt to generate the answer
        final_prompt = f"""
Please answer this question based only on the important sentences provided by the documents below. You must first include a paragraph explaining your reasoning with citations, followed by a concise final answer to the question, and finally include the references you used to answer the question.

Here is a sample response and the format you must follow in your response:
    Explanation: The second sentence "The United States was founded in 1776" in the document titled "American History" tells us the year the US was founded. The fourth sentence "Washington was elected the first president the year the US was founded" in the document titled "George Washington" tells us that Washington was elected president the same year the US was founded. Therefore, the answer to the question 'What year was George Washington elected president?' based on these documents is 1776.
    Answer: 1776
    References: [American History, 2], [George Washington, 4]

Your final answer should be very short. For example, it may be a year, a simple yes or no, a person's name, etc.

Question: {question}

Documents:
{combined_content}
"""
        # Get the final answer from the model
        # print(f'final_prompt: {final_prompt}')
        final_response = model_instance.get_response(final_prompt)
        answer, references = parse_response(final_response)

        # print(f"Model Response:\n{final_response}")
        # print(f"Extracted Answer: {answer}")
        # print(f"Extracted References: {references}")

        # Store the predictions
        predictions["answer"][qa_pair["_id"]] = answer
        predictions["sp"][qa_pair["_id"]] = references

    if output_path is None:
        output_path = f"CoT2_{dataset_name}_{model_name}"

    predictions_path = f"{output_path}_predictions.json"

    try:
        with open(predictions_path, "w") as metrics_file:
            json.dump(predictions, metrics_file, indent=4)
        print(f"\nPredictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

    # Evaluate predictions
    if dataset_name == "distractor":
        eval_path = os.getenv("HOTPOTQA_DEV_DISTRACTOR")
    else:
        eval_path = os.getenv("HOTPOTQA_DEV_FULLWIKI")

    metrics = eval(predictions_path, eval_path)
    metrics_path = f"{output_path}_metrics.json"
    try:
        with open(metrics_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        print(f"\nMetrics saved to {metrics_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return


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
        default="gpt-4",
        help="Model to use (e.g., gpt-4, llama3) (default: gpt-4)",
    )
    # parser.add_argument(
    #     '--output_path',
    #     type=str,
    #     default=None,
    #     help='Path to save the predictions (default: distractor_<model>.json)'
    # )
    parser.add_argument(
        "--num_training_samples",
        type=int,
        default=10,
        help="Number of training samples to process (default: 10)",
    )

    args = parser.parse_args()

    run_distractor_baseline(
        dataset_name=args.dataset,
        model_name=args.model,
        # output_path=args.output_path,
        num_training_samples=args.num_training_samples,
    )

    # Example usage:
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot2.py --dataset distractor --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot2.py --dataset distractor --model gpt-4o-mini --num_training_samples 250
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot2.py --dataset distractor --model gpt-3.5-turbo-instruct --num_training_samples 250

    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot2.py --dataset fullwiki --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot2.py --dataset fullwiki --model gpt-4o-mini --num_training_samples 1000
