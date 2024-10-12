import json
from multiprocessing import Pool, set_start_method

from datasets import load_dataset

from src.utils.llama import Llama3
from src.utils.openai import get_chatgpt_response
from src.utils.utils import em_precision_recall_f1


def process_qa_pair(qa_pair, llama_model=None):
    """
    if llama_model is None, use gpt
    """
    question = qa_pair["question"]
    correct_answer = qa_pair["answer"]

    prompt = f"""
    You are given a factual question below. Provide a concise and precise answer with no unnecessary details.

    Question: {question}
    Answer (in a few words):
    """
    if llama_model:
        generated_answer = llama_model.get_llama_response(prompt)
    else:
        generated_answer = get_chatgpt_response(prompt)

    return {
        "question": question,
        "correct_answer": correct_answer,
        "generated_answer": generated_answer,
    }


def calculate_metrics_for_qa_pair(qa_pair):
    """
    Calculate the EM, Precision, Recall, and F1 metrics for a single QA pair.
    """
    pred_answer = qa_pair["generated_answer"]
    gold_answer = qa_pair["correct_answer"]
    em, precision, recall, f1 = em_precision_recall_f1(pred_answer, gold_answer)
    return em, precision, recall, f1


if __name__ == "__main__":
    set_start_method("spawn", force=True)

    NUM_TRAINING_SAMPLES = 1000

    hotpot_data = load_dataset("hotpot_qa", "fullwiki", split="validation").select(
        range(NUM_TRAINING_SAMPLES)
    )
    # llama_model = Llama3(
    #     "/home/cpp/jerryhuang/search2024/meta-llama/Meta-Llama-3-8B-Instruct"
    # )
    llama_model = None

    qa_pairs = []
    num_processed = 0
    for qa_pair in hotpot_data:
        result = process_qa_pair(qa_pair, llama_model)
        qa_pairs.append(result)
        num_processed += 1
        if num_processed % 10 == 1:
            print(f"Processed: {num_processed}")

    # Save the Q&A pairs to a JSON file
    with open("closed_book_gpt_qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f, indent=4)

    # Calculate EM, Precision, Recall, and F1 using multiprocessing for speedup
    with Pool(processes=4) as pool:
        metrics_results = pool.map(calculate_metrics_for_qa_pair, qa_pairs)

    total_em, total_precision, total_recall, total_f1 = map(sum, zip(*metrics_results))

    avg_em = total_em / NUM_TRAINING_SAMPLES
    avg_precision = total_precision / NUM_TRAINING_SAMPLES
    avg_recall = total_recall / NUM_TRAINING_SAMPLES
    avg_f1 = total_f1 / NUM_TRAINING_SAMPLES

    metrics = {
        "average_exact_match": avg_em,
        "average_precision": avg_precision,
        "average_recall": avg_recall,
        "average_f1_score": avg_f1,
    }

    with open("closed_book_gpt_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Average Exact Match (EM): {avg_em:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1 Score: {avg_f1:.2f}")
