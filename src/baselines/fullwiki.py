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
    """
    Parses the AI model's response to extract the answer and references.

    Expected response format:
        Answer: <answer>
        References: [Title1, 1], [Title3, 4]
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


def run_distractor_baseline(
    dataset_name="distractor",
    model_name="gpt-4",
    output_path=None,
    num_training_samples=1000,
    model_path=None,  # Required if using Llama3
):
    """
    Runs the distractor baseline evaluation on the HotpotQA dataset using the specified model.
    """
    dataset = load_hotpotqa(dataset_name)
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    if model_name.lower() in ["meta-llama-3-8b-instruct", "gpt-4o-mini"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    for qa_pair in dataset[:num_training_samples]:
        final_output = ""
        for entry in qa_pair["context"]:
            title = entry[0]
            sentences = entry[1]

            flattened_content = " ".join(
                [f"({i}) {sentence}" for i, sentence in enumerate(sentences)]
            )
            final_output += f"Title: {title}\nContent: {flattened_content}\n\n"

        prompt = f"""
Please answer this question concisely based on the provided documents below. You must also include the references you use to answer the question.

Here is a sample response and the format you must follow in your response:
    Answer: Yes
    References: [Title1, 1], [Title3, 4]

Question: {qa_pair["question"]}

Documents: 
{final_output}
"""
        print(f'Correct answer: {qa_pair["answer"]}')

        response = model_instance.get_response(prompt)
        answer, references = parse_response(response)
        print(f"Answer: {answer}")
        print(f"References: {references}")

        predictions["answer"][qa_pair["_id"]] = answer
        predictions["sp"][qa_pair["_id"]] = references

    if output_path is None:
        output_path = f"distractor_{model_name}.json"

    predictions_path = f"{output_path}_predictions.json"

    try:
        with open(predictions_path, "w") as pred_file:
            json.dump(predictions, pred_file, indent=4)
        print(f"Predictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

    # Evaluate predictions
    eval_path = os.getenv("HOTPOTQA_DEV_DISTRACTOR")
    if eval_path:
        try:
            eval(predictions_path, eval_path)
        except Exception as e:
            print(f"Error during evaluation: {e}")
    else:
        print("Evaluation path not set in environment variables.")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Distractor Baseline on HotpotQA Dataset"
    )
    # parser.add_argument(
    #     '--dataset',
    #     type=str,
    #     default='distractor',
    #     help='Name of the dataset to use (default: distractor)'
    # )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model to use (e.g., gpt-4, llama3) (default: gpt-4)",
    )
    # parser.add_argument(
    #     '--model_path',
    #     type=str,
    #     default=None,
    #     help='Path to the Llama3 model (required if model is Llama3)'
    # )
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
        # dataset_name=args.dataset,
        model_name=args.model,
        # output_path=args.output_path,
        num_training_samples=args.num_training_samples,
        # model_path=args.model_path
    )

    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/distractor.py --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/distractor.py --model gpt-4o-mini --num_training_samples 1000
