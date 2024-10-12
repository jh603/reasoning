import argparse
import json
import os
import re
from collections import defaultdict

from dotenv import load_dotenv

from src.utils.datasets import load_hotpotqa
from src.utils.hotpotqa_eval import eval
from src.utils.model_factory import Model
from src.utils.models import DebertaQAModel

load_dotenv()


def parse_response(response):
    answer = ""
    references = []

    answer_match = re.search(
        r"FINAL_ANSWER:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL
    )
    if answer_match:
        answer = answer_match.group(1).strip()

    references_match = re.search(
        r"FINAL_REFERENCES:\s*(.+?)(?:\n|$)", response, re.IGNORECASE | re.DOTALL
    )
    if references_match:
        ref_string = references_match.group(1)
        ref_list = re.findall(r"\[([^\]]+?),\s*(\d+)\]", ref_string)
        for ref in ref_list:
            title, index = ref
            references.append([title.strip(), int(index)])

    return answer, references


def run_baseline(
    dataset_name="distractor",
    model_name="microsoft/deberta-v3-large",
    checkpoint_path="/home/cpp/jerryhuang/beam_retriever/output/10-10-2024/hotpotqa_reader_deberta_large-seed42-bsz4-fp16True-lr1e-05-decay0.0-warm0.1-valbsz32/checkpoint_best.pt",
    output_path=None,
    num_training_samples=1000,
):
    # Load the dataset
    dataset = load_hotpotqa(dataset_name)

    # Initialize the DebertaQAModel
    model_instance = DebertaQAModel(
        model_name=model_name, checkpoint_path=checkpoint_path
    )

    # Initialize predictions dictionary
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    i = 0
    offset = 77
    print(f"Processing {len(dataset)} samples.")

    for qa_pair in dataset[offset : num_training_samples + offset]:
        print(f"{i}: Processing question")
        i += 1

        # Build a dictionary mapping titles to sets of supporting sentence indices
        sp_dict = defaultdict(set)
        supporting_facts = qa_pair["supporting_facts"]
        for sp in supporting_facts:
            title, sent_idx = sp[0], sp[1]
            sp_dict[title].add(sent_idx)

        documents = ""
        for entry in qa_pair["context"]:
            title = entry[0]
            if title in sp_dict:
                sentences = entry[1]
                # Get the specific sentences indicated by the supporting facts
                selected_sentences = [
                    sentences[idx] for idx in sp_dict[title] if idx < len(sentences)
                ]
                flattened_content = " ".join(
                    selected_sentences
                )  # Combine selected sentences into a single string
                documents += f"{title}: {flattened_content}\n\n"

        print(f"documents: {documents}")
        # Use the reader model to get the answer
        response = model_instance.get_response(
            qa_pair["question"],
            documents,
            answer_merge=True,  # Enable answer merging
            topk=10,  # Adjust topk if desired
            max_ans_len=30,  # Adjust max_ans_len if desired
        )

        print(f"Response: {response}")
        print(f'Correct answer: {qa_pair["answer"]}')

        # Store the predicted answer
        predictions["answer"][qa_pair["_id"]] = response
        predictions["sp"][qa_pair["_id"]] = []
        # (Optional) You can implement logic here to predict the supporting facts (SP), but the reader usually doesn't do this directly

    # Save the predictions to the output file
    if output_path is None:
        output_path = f"CoT_ideal_reader_deb_{dataset_name}.json"
    predictions_path = f"{output_path}_predictions.json"

    try:
        with open(predictions_path, "w") as pred_file:
            json.dump(predictions, pred_file, indent=4)
        print(f"Predictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

    # Evaluate the results
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
    # parser.add_argument(
    #     '--model',
    #     type=str,
    #     default='gpt-4o-mini',
    #     help='Model to use (e.g., gpt-4, llama3) (default: gpt-4o-mini)'
    # )
    parser.add_argument(
        "--num_training_samples",
        type=int,
        default=10,
        help="Number of training samples to process (default: 10)",
    )

    args = parser.parse_args()

    run_baseline(
        dataset_name=args.dataset,
        # model_name=args.model,
        num_training_samples=args.num_training_samples,
    )

    # Example Commands:
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader_finetuned.py --dataset distractor --model deberta-v3-large --num_training_samples 1000

    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset fullwiki --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset fullwiki --model gpt-4o-mini --num_training_samples 1000
