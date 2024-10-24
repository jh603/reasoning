import os

os.environ["CUDA_VISIBLE_DEVICES"] = "0,1,2,3,4,5,6,7"

import json
import argparse
from datasets import load_dataset
from sklearn.metrics.pairwise import cosine_similarity
import re

from src.bright_dataset_experiments.retriever import BrightRetriever
from src.bright_dataset_experiments.utils import *


def sanitize_filename(s: str) -> str:
    return re.sub(r"[^a-zA-Z0-9_-]", "_", s)


def parse_args():
    parser = argparse.ArgumentParser(description="Document Retrieval Pipeline")
    parser.add_argument(
        "--subset", type=str, default="examples", help="Dataset subset to use"
    )
    parser.add_argument(
        "--split",
        type=str,
        # default="stackoverflow",
        default="biology",
        help="Dataset split to use",
    )
    parser.add_argument(
        "--embedding_model",
        type=str,
        # default="BAAI/bge-large-en-v1.5",
        # default="hkunlp/instructor-xl",
        default="intfloat/e5-mistral-7b-instruct",
        # default="Alibaba-NLP/gte-Qwen2-7B-instruct",
        help="Embedding model to use",
    )
    parser.add_argument(
        "--ndcg_cutoffs",
        type=int,
        nargs="+",
        default=[10, 50],
        help="Cutoff values for nDCG",
    )
    parser.add_argument(
        "--batch_size", type=int, default=128, help="Batch size for embedding"
    )
    parser.add_argument(
        "--output_dir",
        type=str,
        default="bright_embeddings",
        help="Output directory for results",
    )
    parser.add_argument(
        "--long_context",
        action="store_true",
        default=False,
        help="Whether to use long context",
    )
    return parser.parse_args()


def main():
    args = parse_args()

    embeddings_dir = args.output_dir
    os.makedirs(embeddings_dir, exist_ok=True)

    split_sanitized = sanitize_filename(args.split)
    model_sanitized = sanitize_filename(args.embedding_model)
    data_file = os.path.join(
        embeddings_dir, f"{split_sanitized}_{model_sanitized}_data.pkl"
    )

    retriever = BrightRetriever(
        embedding_model=args.embedding_model, batch_size=args.batch_size
    )

    # Load or create document embeddings
    if os.path.exists(data_file):
        print("Loading cached document embeddings...")
        retriever.load_state(data_file)
    else:
        print("Creating document embeddings...")
        dataset_documents = load_dataset("xlangai/BRIGHT", "documents")[args.split]
        retriever.embed_documents(
            document_ids=dataset_documents["id"],
            contents=dataset_documents["content"],
            save_file_path=data_file,
        )

    # Load queries
    print(f"Loading queries for subset '{args.subset}' and split '{args.split}'...")
    dataset_queries = load_dataset("xlangai/BRIGHT", args.subset)[args.split]

    # Debug print dataset structure
    print(f"Available keys: {dataset_queries.features.keys()}")
    print(f"Number of queries: {len(dataset_queries)}")

    query_ids = dataset_queries["id"]
    queries = dataset_queries["query"]

    # TODO: add instruction before each query
    # queries = ["Represent this Biology post for searching relevant passages: " + q for q in queries]

    # Get query embeddings
    query_embeddings = retriever.encode(queries)

    # Get document embeddings
    doc_embeddings = retriever.get_document_embeddings()
    doc_ids = retriever.get_document_ids()

    # Calculate similarities
    all_scores = cosine_similarity(query_embeddings, doc_embeddings)

    # Format excluded IDs and check their presence
    excluded_ids = {}
    gold_key = "gold_ids_long" if args.long_context else "gold_ids"

    # Process scores
    scores = get_scores(
        query_ids=query_ids,
        doc_ids=doc_ids,
        scores=all_scores,
        excluded_ids=excluded_ids,
    )

    # Build ground truth dictionary
    ground_truth = {}
    for example in dataset_queries:
        qid = str(example["id"])
        ground_truth[qid] = {}

        if gold_key not in example:
            print(f"WARNING: Missing {gold_key} for query {qid}")
            continue

        gold_ids = example[gold_key]

        for gid in gold_ids:
            str_gid = str(gid)
            ground_truth[qid][str_gid] = 1

    print("\nDEBUG: Sample entries before evaluation")
    sample_qid = next(iter(scores))
    print(f"\nScores for query {sample_qid}:")
    print(f"Number of scored documents: {len(scores[sample_qid])}")
    print(f"Sample scores: {dict(list(scores[sample_qid].items())[:3])}")

    print(f"\nGround truth for query {sample_qid}:")
    print(f"Number of relevant documents: {len(ground_truth[sample_qid])}")
    print(f"Relevant documents: {list(ground_truth[sample_qid].keys())}")

    # Verify data types
    for qid, doc_scores in scores.items():
        if not isinstance(qid, str):
            print(f"Converting query ID {qid} to string")
            scores[str(qid)] = doc_scores
            del scores[qid]
        for doc_id, score in doc_scores.items():
            if not isinstance(doc_id, str) or not isinstance(score, float):
                print(
                    f"Invalid types - Query: {qid}, Doc: {doc_id} ({type(doc_id)}), Score: {score} ({type(score)})"
                )

    try:
        results = calculate_retrieval_metrics(results=scores, qrels=ground_truth)
    except Exception as e:
        print("\nERROR: Metric calculation failed")
        print(f"Error message: {str(e)}")
        raise

    os.makedirs(args.output_dir, exist_ok=True)
    with open(os.path.join(args.output_dir, "results.json"), "w") as f:
        json.dump(results, f, indent=2)

    print("\nEvaluation Metrics:")
    print(json.dumps(results, indent=2))


if __name__ == "__main__":
    main()
