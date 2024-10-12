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
    model_name="gpt-4o-mini",
    output_path=None,
    num_training_samples=1000,
):
    dataset = load_hotpotqa(dataset_name)
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    if model_name.lower() in ["meta-llama-3-8b-instruct", "gpt-4o-mini"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    i = 0
    offset = 1000
    print(len(dataset))
    for qa_pair in dataset[offset : num_training_samples + offset]:
        print(f"{i}: questions processed")
        i += 1
        final_output = ""

        sp_set = set()
        supporting_facts = qa_pair["supporting_facts"]
        for sp in supporting_facts:
            sp_set.add(sp[0])

        for entry in qa_pair["context"]:
            title = entry[0]
            sentences = entry[1]

            if title in sp_set:
                flattened_content = "\n".join(
                    [f"({i}) {sentence}" for i, sentence in enumerate(sentences)]
                )
                final_output += f"Title: {title}\nContent: {flattened_content}\n\n"

        prompt = f"""
Please answer the following question using only the information provided in the documents below.

**Instructions:**

1. **Explain Reasoning**:
    - Write a paragraph explaining your overall reasoning for answering the main question.
    - Incorporate the answers to the sub-questions with appropriate citations.
    - **Use the exact entity names** as presented in the original documents when referring to them.
    - **List every sentence number used** in this explanation.

2. **List References**:
    - **Aggregate** all references cited in both the sub-question answers and the explanation.
    - **Ensure that every referenced sentence is included** in this list.
    - Format each reference as `[Document Title, Sentence Number]`.
    - **You must cite at least 2 different document titles.**
    - If a sentence contains pronouns (e.g., 'he', 'she', 'they', 'it'), include the preceding sentence(s) that clarify who or what the pronoun refers to.
        e.g. For the refererence: "He had over 100 medals." This is missing a reference to who "he" refers to. You must also cite the sentence before this one that specifies who "he" refers to.
    - Ensure that all necessary context is captured for the answer to be fully understood.

3. **Provide Final Answer**:
    - The answer should be **very concise**, such as a year, a simple yes/no, or a person's name.
    - Use wording **exactly as it appears** in the original documents.
    - **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
    - **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.
    - Example of Good Concise Answers: "birdstrike", "Luigi Pirandello", "no", "3", "Farewell", "Alexis Tsipras", "water", "Al-Karaji", "Ufa, Russia", "603 performances", "two", "video game", "Mike Oldfield"

4. **Use Unique Delimiters**:
    - Put "FINAL_ANSWER:" before your final answer.
    - Put "FINAL_REFERENCES:" before your final list of references.

**Example Response:**

*Explanation:*  
The second sentence in the document titled "American History" states, "The United States was founded in 1776," indicating the founding year. The fourth sentence in the "George Washington" document mentions, "Washington was elected the first president the year the US was founded," linking Washington's election to the founding year. Therefore, based on these documents, the answer to the question "What year was George Washington elected president?" is 1776.

FINAL_ANSWER: 1776

FINAL_REFERENCES: [American History, 2], [George Washington, 4]

### START ANSWER
1. **Explanation:**
    - A paragraph explaining your reasoning for the main question
    - You can break down the multi-hop question into sub-question and answer them sequentially. Provide citations for each sub-question so you can keep track of them.
    - List of all document titles and sentence numbers used in this explanation

2. **List References**:
    - Aggregate all references cited in the explanation.
    - Ensure that every referenced sentence is included in this list.
    - Format each reference as `[Document Title, Sentence Number]`.
    - If a sentence contains pronouns (e.g., 'he', 'she', 'they', 'it'), include the preceding sentence(s) that clarify who or what the pronoun refers to.
    - Ensure that all necessary context is captured for the answer to be fully understood.

3. **Answer:**  
   FINAL_ANSWER: *(Final concise answer to the main question, following the guidelines for wording and support)*

4. **References:**  
   FINAL_REFERENCES: [Document Title, Sentence Number], [Document Title, Sentence Number], ...

**Question:** {qa_pair["question"]}

**Documents:**  
{final_output}
### END ANSWER
"""
        response = model_instance.get_response(prompt)
        answer, references = parse_response(response)

        predictions["answer"][qa_pair["_id"]] = answer
        predictions["sp"][qa_pair["_id"]] = references

    if output_path is None:
        output_path = f"CoT_ideal_reader_{dataset_name}_{model_name}.json"

    predictions_path = f"{output_path}_predictions.json"

    try:
        with open(predictions_path, "w") as pred_file:
            json.dump(predictions, pred_file, indent=4)
        print(f"Predictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

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
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset distractor --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset distractor --model gpt-4o-mini --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset distractor --model gpt-4o --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset distractor --model deberta-v3-large --num_training_samples 1000

    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset fullwiki --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_ideal_reader.py --dataset fullwiki --model gpt-4o-mini --num_training_samples 1000
