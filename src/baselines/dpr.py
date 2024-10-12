import json
from multiprocessing import Pool, set_start_method

from datasets import load_dataset

from src.retrievers.DPR import GTRWikiRetriever
from src.utils.llama import Llama3
from src.utils.openai import get_chatgpt_response
from src.utils.utils import em_precision_recall_f1

NUM_TRAINING_SAMPLES = 1


def process_qa_pair(qa_pair, retriever, llama_model=None, conversation_history=None):
    question = qa_pair["question"]
    correct_answer = qa_pair["answer"]

    # Initialize conversation history if it's not provided
    if conversation_history is None:
        conversation_history = []

    # Add current question to the conversation history
    conversation_history.append(f"Question: {question}")

    # Prepare prompt with history context
    history_context = "\n".join(conversation_history)

    # Define the initial prompt with history context
    prompt = f"""
    Given the following conversation history, decide whether to generate a search query to retrieve more information or to provide a final answer directly. If you need more information, respond with a search query. If you can answer directly, provide the final answer.

    Conversation History:
    {history_context}

    New Question: {question}

    Your response should be in one of these two formats:
    1. SEARCH: <your search query>
    2. ANSWER: <your final answer>

    Answers should be concise. An example of (question, answer) pair looks like (Which genus contains more species, Ortegocactus or Eschscholzia?, Eschscholzia)
    
    Response:
    """

    # Get the response from the model (Llama or ChatGPT)
    if llama_model:
        response = llama_model.get_llama_response(prompt)
    else:
        response = get_chatgpt_response(prompt)

    print(f"response: {response}")

    # If the model decides a search is needed, retrieve documents
    if response.startswith("SEARCH:"):
        query = response[7:].strip()

        if retriever:
            documents = retriever.gtr_wiki_retrieval(query)
        else:
            documents = ["test test test"]

        # Update conversation history with the search query and the retrieved info
        conversation_history.append(f"Search Query: {query}")
        conversation_history.append(f"Retrieved Information: {documents}")

        # Ask again for a final answer based on the retrieved info
        final_answer_prompt = f"""
        Based on the following conversation history and retrieved information, provide a concise and precise final answer:

        Conversation History:
        {history_context}

        Retrieved Information:
        {documents}

        Final Answer:
        """

        if llama_model:
            final_answer = llama_model.get_llama_response(final_answer_prompt)
        else:
            final_answer = get_chatgpt_response(final_answer_prompt)

        # Add the final answer to the conversation history
        conversation_history.append(f"Final Answer: {final_answer}")

    else:
        # If the response is a direct answer, strip the "ANSWER:" part
        final_answer = response[7:].strip()  # Remove "ANSWER: " prefix
        conversation_history.append(f"Final Answer: {final_answer}")

    # Return the final output with conversation history for possible future use
    return {
        "question": question,
        "correct_answer": correct_answer,
        "generated_answer": final_answer,
        "conversation_history": conversation_history,  # Return history for next iteration
    }


def calculate_metrics_for_qa_pair(qa_pair):
    pred_answer = qa_pair["generated_answer"]
    gold_answer = qa_pair["correct_answer"]
    em, precision, recall, f1 = em_precision_recall_f1(pred_answer, gold_answer)
    return em, precision, recall, f1


if __name__ == "__main__":
    # retriever = GTRWikiRetriever()
    retriever = None
    set_start_method("spawn", force=True)

    hotpot_data = load_dataset("hotpot_qa", "fullwiki", split="validation").select(
        range(NUM_TRAINING_SAMPLES)
    )
    # llama_model = None  # Or initialize with Llama3 if needed
    llama_model = Llama3(
        "/home/cpp/jerryhuang/search2024/meta-llama/Meta-Llama-3-8B-Instruct"
    )

    qa_pairs = []
    num_processed = 0
    for qa_pair in hotpot_data:
        result = process_qa_pair(qa_pair, retriever, llama_model)
        qa_pairs.append(result)
        num_processed += 1
        if num_processed % 10 == 1:
            print(f"Processed: {num_processed}")

    with open("dpr_qa_pairs.json", "w") as f:
        json.dump(qa_pairs, f, indent=4)

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

    with open("dpr_metrics.json", "w") as f:
        json.dump(metrics, f, indent=4)

    print(f"Average Exact Match (EM): {avg_em:.2f}")
    print(f"Average Precision: {avg_precision:.2f}")
    print(f"Average Recall: {avg_recall:.2f}")
    print(f"Average F1 Score: {avg_f1:.2f}")
