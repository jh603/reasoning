# pipeline.py

import os

from src.bright_dataset_experiments.retriever import BrightRetriever
os.environ["CUDA_VISIBLE_DEVICES"] = "4,5,6,7"

import argparse
import pickle
import numpy as np
from sklearn.metrics import ndcg_score, average_precision_score
from tqdm import tqdm
from datasets import load_dataset
import re
import torch
import gc

def sanitize_filename(s: str) -> str:
    """Sanitize the string to be used as a filename."""
    return re.sub(r'[^a-zA-Z0-9_-]', '_', s)

def parse_args():
    """Parse command-line arguments."""
    parser = argparse.ArgumentParser(description="Document Retrieval Pipeline")
    parser.add_argument(
        "--subset",
        type=str,
        default="example",
        help="Dataset subset to use (e.g., llama3-70b_reason)",
    )
    parser.add_argument(
        "--split",
        type=str,
        default="stackoverflow",
        help="Dataset split to use (e.g., stackoverflow)",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        # default="intfloat/e5-mistral-7b-instruct",
        default="BAAI/bge-large-en-v1.5",
        help="Embedding model to use (e.g., intfloat/e5-mistral-7b-instruct)",
    )
    parser.add_argument(
        "--top_k",
        type=int,
        default=20,
        help="Number of top documents to retrieve",
    )
    parser.add_argument(
        "--ndcg_cutoffs",
        type=int,
        nargs='+',
        default=[10, 50],
        help="Cutoff values for nDCG (e.g., 10 50)",
    )
    parser.add_argument(
        "--cohere_api_key",
        type=str,
        default=None,
        help="Cohere API key if using Cohere embedding model",
    )
    parser.add_argument(
        "--batch_size",
        type=int,
        default=64,  # Adjusted for memory constraints
        help="Batch size for embedding",
    )
    return parser.parse_args()

def calculate_retrieval_metrics(results: dict, qrels: dict, k_values=[1, 5, 10, 25, 50, 100]) -> dict:
    """
    Calculate retrieval metrics using pytrec_eval.

    Args:
        results (dict): Retrieved results in the format {query_id: {doc_id: score}}.
        qrels (dict): Ground truth in the format {query_id: {doc_id: relevance}}.
        k_values (list, optional): List of cutoff values for metrics. Defaults to [1, 5, 10, 25, 50, 100].

    Returns:
        dict: Calculated metrics.
    """
    import pytrec_eval

    ndcg = {}
    _map = {}
    recall = {}
    precision = {}
    mrr = {"MRR": 0}

    for k in k_values:
        ndcg[f"NDCG@{k}"] = 0.0
        _map[f"MAP@{k}"] = 0.0
        recall[f"Recall@{k}"] = 0.0
        precision[f"P@{k}"] = 0.0

    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    evaluator = pytrec_eval.RelevanceEvaluator(qrels, {map_string, ndcg_string, recall_string, precision_string, "recip_rank"})
    scores = evaluator.evaluate(results)

    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id].get(f"ndcg_cut_{k}", 0.0)
            _map[f"MAP@{k}"] += scores[query_id].get(f"map_cut_{k}", 0.0)
            recall[f"Recall@{k}"] += scores[query_id].get(f"recall_{k}", 0.0)
            precision[f"P@{k}"] += scores[query_id].get(f"P_{k}", 0.0)
        mrr["MRR"] += scores[query_id].get("recip_rank", 0.0)

    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / len(scores), 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / len(scores), 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / len(scores), 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / len(scores), 5)
    mrr["MRR"] = round(mrr["MRR"] / len(scores), 5)

    output = {**ndcg, **_map, **recall, **precision, **mrr}
    print(output)
    return output

def main():
    args = parse_args()

    subset = args.subset
    split = args.split
    embedding_model = args.embedding_model
    top_k = args.top_k
    ndcg_cutoffs = args.ndcg_cutoffs
    cohere_api_key = args.cohere_api_key
    batch_size = args.batch_size

    embeddings_dir = "bright_embeddings"
    os.makedirs(embeddings_dir, exist_ok=True)

    # Sanitize filenames
    subset_sanitized = sanitize_filename(subset)
    split_sanitized = sanitize_filename(split)
    embedding_model_sanitized = sanitize_filename(embedding_model)

    # Define single file path
    data_file = os.path.join(
        embeddings_dir,
        f"{subset_sanitized}_{split_sanitized}_{embedding_model_sanitized}_data.pkl",
    )

    # Initialize Retriever
    print(f'embedding_model: {embedding_model}')
    retriever = BrightRetriever(
        embedding_model=embedding_model,
        batch_size=batch_size,
        max_length=512,
    )

    # Verify GPU setup
    available_gpus = torch.cuda.device_count()
    print(f"Number of GPUs available: {available_gpus}")
    for i in range(available_gpus):
        print(f"GPU {i}: {torch.cuda.get_device_name(i)}")

    # Check if data_file exists
    if os.path.exists(data_file):
        print("Data file exists. Loading from disk...")
        retriever.load_state(save_file_path=data_file)
    else:
        print("Data file not found. Creating embeddings...")
        # Load documents
        print(f"Loading documents for subset '{subset}' and split '{split}'...")
        dataset_documents = load_dataset("xlangai/BRIGHT", "documents")[split]

        # Extract document_ids and contents directly from the dataset_documents dictionary
        document_ids = dataset_documents["id"]
        contents = dataset_documents["content"]

        # Embed and add documents, which saves to data_file
        print("Embedding and adding documents...")
        retriever.embed_documents(save_file_path=data_file, document_ids=document_ids, contents=contents)
        print(f"Documents and embeddings saved to '{data_file}'.")

    # Build FAISS index
    retriever.build_faiss_index()

    # Load queries
    print(f"Loading queries for subset '{subset}' and split '{split}'...")
    dataset_queries = load_dataset("xlangai/BRIGHT", "examples")[split]
    queries = dataset_queries["query"]
    gold_ids_list = dataset_queries["gold_ids"]

    # Prepare ground truth for evaluation
    ground_truth = {}
    for query_id, gold_ids in zip(dataset_queries["id"], gold_ids_list):
        ground_truth[str(query_id)] = {str(doc_id): 1 for doc_id in gold_ids}

    # Perform retrieval and collect metrics
    print("Running retrieval and evaluating...")
    ndcg_scores = {cutoff: [] for cutoff in ndcg_cutoffs}
    recalls = []
    p_at_1 = []
    maps = []

    for query, query_id, gold_ids in tqdm(zip(queries, dataset_queries["id"], gold_ids_list), total=len(queries), desc="Processing Queries"):
        retrieved_docs = retriever.retrieve_faiss(query, top_k=top_k)
        retrieved_ids = [doc_id for doc_id, _, _ in retrieved_docs]
        retrieved_scores = [score for _, _, score in retrieved_docs]

        # Create relevance scores
        relevance = [1 if doc_id in gold_ids else 0 for doc_id in retrieved_ids]

        # Calculate nDCG for each cutoff
        for cutoff in ndcg_cutoffs:
            current_relevance = relevance[:cutoff]
            ideal_relevance = sorted(current_relevance, reverse=True)
            ndcg = ndcg_score(
                [ideal_relevance], [current_relevance], k=cutoff
            )
            ndcg_scores[cutoff].append(ndcg)

        # Calculate Recall
        retrieved_set = set(retrieved_ids)
        gold_set = set(gold_ids)
        recall = len(retrieved_set & gold_set) / len(gold_set) if gold_set else 0
        recalls.append(recall)

        # Calculate Precision@1
        p1 = 1.0 if retrieved_docs and retrieved_docs[0][0] in gold_set else 0.0
        p_at_1.append(p1)

        # Calculate MAP
        if gold_ids:
            y_true = [1 if doc_id in gold_set else 0 for doc_id in retrieved_ids]
            y_scores = [score for _, _, score in retrieved_docs]
            map_score = average_precision_score(y_true, y_scores)
            maps.append(map_score)
        else:
            maps.append(0.0)

        # Clear cache periodically to free memory
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
            gc.collect()

    # Aggregate metrics
    print("\nEvaluation Metrics:")
    for cutoff in ndcg_cutoffs:
        avg_ndcg = np.mean(ndcg_scores[cutoff])
        print(f"Average nDCG@{cutoff}: {avg_ndcg:.4f}")
    avg_recall = np.mean(recalls)
    print(f"Average Recall: {avg_recall:.4f}")
    avg_p_at_1 = np.mean(p_at_1)
    print(f"Average Precision@1: {avg_p_at_1:.4f}")
    avg_map = np.mean(maps)
    print(f"Average MAP: {avg_map:.4f}")


if __name__ == "__main__":
    main()


# Examples
# python3 '/home/cpp/jerryhuang/reasoning/src/bright_dataset_experiments/pipeline.py'
