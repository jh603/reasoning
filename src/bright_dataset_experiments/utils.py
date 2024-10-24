import os.path
import time
import torch
import json
import cohere
import numpy as np
import vertexai
import pytrec_eval
import tiktoken
import voyageai
from tqdm import tqdm, trange
import torch.nn.functional as F
from gritlm import GritLM
from openai import OpenAI
from transformers import AutoTokenizer, AutoModel
from InstructorEmbedding import INSTRUCTOR
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from vertexai.language_models import TextEmbeddingInput, TextEmbeddingModel
from torchmetrics.functional.pairwise import pairwise_cosine_similarity


def get_scores(query_ids, doc_ids, scores, excluded_ids):
    """Process similarity scores into the format expected by pytrec_eval."""
    assert len(scores) == len(query_ids), f"{len(scores)}, {len(query_ids)}"
    assert len(scores[0]) == len(doc_ids), f"{len(scores[0])}, {len(doc_ids)}"

    emb_scores = {}
    for query_id, doc_scores in zip(query_ids, scores):
        str_query_id = str(query_id)
        cur_scores = {}

        # Convert excluded_ids to set of strings
        excluded = {str(did) for did in excluded_ids.get(str_query_id, [])}

        # Process scores with explicit float conversion
        for doc_id, score in zip(doc_ids, doc_scores):
            str_doc_id = str(doc_id)
            if str_doc_id not in excluded:
                # Convert numpy.float32 to Python float
                cur_scores[str_doc_id] = float(score)

        # Sort scores and ensure float type
        sorted_items = sorted(cur_scores.items(), key=lambda x: x[1], reverse=True)[
            :1000
        ]
        emb_scores[str_query_id] = {
            k: float(v) for k, v in sorted_items
        }  # Explicit float conversion

    # Verify all values are proper Python floats
    for qid in emb_scores:
        for did in emb_scores[qid]:
            assert isinstance(
                emb_scores[qid][did], float
            ), f"Score for {qid}/{did} is {type(emb_scores[qid][did])}"

    return emb_scores


def calculate_retrieval_metrics(results, qrels, k_values=[1, 5, 10, 25, 50, 100]):
    # Initialize metric dictionaries
    ndcg = {f"NDCG@{k}": 0.0 for k in k_values}
    _map = {f"MAP@{k}": 0.0 for k in k_values}
    recall = {f"Recall@{k}": 0.0 for k in k_values}
    precision = {f"P@{k}": 0.0 for k in k_values}
    mrr = {"MRR": 0}

    # Create evaluation strings
    map_string = "map_cut." + ",".join([str(k) for k in k_values])
    ndcg_string = "ndcg_cut." + ",".join([str(k) for k in k_values])
    recall_string = "recall." + ",".join([str(k) for k in k_values])
    precision_string = "P." + ",".join([str(k) for k in k_values])

    # Verify data types before evaluation
    verified_results = {}
    for qid, doc_scores in results.items():
        str_qid = str(qid)
        verified_results[str_qid] = {
            str(did): float(score) for did, score in doc_scores.items()
        }

    verified_qrels = {}
    for qid, doc_rels in qrels.items():
        str_qid = str(qid)
        verified_qrels[str_qid] = {str(did): int(rel) for did, rel in doc_rels.items()}

    try:
        evaluator = pytrec_eval.RelevanceEvaluator(
            verified_qrels,
            {map_string, ndcg_string, recall_string, precision_string, "recip_rank"},
        )
        scores = evaluator.evaluate(verified_results)
    except Exception as e:
        print("\nERROR during evaluation:")
        sample_qid = next(iter(verified_results))
        print(f"Sample results entry types:")
        print(f"Query ID type: {type(sample_qid)}")
        sample_did = next(iter(verified_results[sample_qid]))
        print(f"Doc ID type: {type(sample_did)}")
        print(f"Score type: {type(verified_results[sample_qid][sample_did])}")
        raise e

    # Process evaluation scores
    for query_id in scores.keys():
        for k in k_values:
            ndcg[f"NDCG@{k}"] += scores[query_id]["ndcg_cut_" + str(k)]
            _map[f"MAP@{k}"] += scores[query_id]["map_cut_" + str(k)]
            recall[f"Recall@{k}"] += scores[query_id]["recall_" + str(k)]
            precision[f"P@{k}"] += scores[query_id]["P_" + str(k)]
        mrr["MRR"] += scores[query_id]["recip_rank"]

    num_queries = len(scores)
    for k in k_values:
        ndcg[f"NDCG@{k}"] = round(ndcg[f"NDCG@{k}"] / num_queries, 5)
        _map[f"MAP@{k}"] = round(_map[f"MAP@{k}"] / num_queries, 5)
        recall[f"Recall@{k}"] = round(recall[f"Recall@{k}"] / num_queries, 5)
        precision[f"P@{k}"] = round(precision[f"P@{k}"] / num_queries, 5)
    mrr["MRR"] = round(mrr["MRR"] / num_queries, 5)

    return {**ndcg, **_map, **recall, **precision, **mrr}
