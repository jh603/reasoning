
import os
import re
import pickle
from typing import List, Tuple

import torch
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer

from sklearn.metrics.pairwise import cosine_similarity
import numpy as np


def load_heuristics(heuristics_file_path: str) -> List[str]:
    """
    Load heuristics from a text file, ensuring each heuristic is clean and well-formatted.

    Args:
        heuristics_file_path (str): Path to the heuristics.txt file.

    Returns:
        List[str]: A list of heuristics.
    """
    heuristics = []
    try:
        with open(heuristics_file_path, 'r', encoding='utf-8') as file:
            for line in file:
                # Remove numbering and leading/trailing whitespace
                heuristic = re.sub(r'^\d+\.\s*', '', line).strip()
                # Validate heuristic length to avoid capturing unintended lines
                if heuristic and len(heuristic) > 10:
                    heuristics.append(heuristic)
        print(f"Loaded {len(heuristics)} heuristics from '{heuristics_file_path}'.")
    except FileNotFoundError:
        print(f"Heuristics file not found at '{heuristics_file_path}'. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Error loading heuristics from '{heuristics_file_path}': {e}")
        exit(1)
    return heuristics


def embed_heuristics(
    heuristics: List[str],
    batch_size: int = 32,
    max_length: int = 512,
    model_name: str = 'facebook/dpr-ctx_encoder-single-nq-base',
    embeddings_file: str = 'heuristics_embeddings.pkl'
) -> None:
    """
    Embed heuristics using DPR and save the embeddings along with metadata.

    Args:
        heuristics (List[str]): A list of heuristics to embed.
        batch_size (int, optional): Batch size for embedding. Defaults to 32.
        max_length (int, optional): Maximum token length for the tokenizer. Defaults to 512.
        model_name (str, optional): Pretrained DPR model name. Defaults to 'facebook/dpr-ctx_encoder-single-nq-base'.
        embeddings_file (str, optional): Path to save the embeddings pickle file. Defaults to 'heuristics_embeddings.pkl'.
    """
    # Initialize tokenizer and model
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # If multiple GPUs are available, use DataParallel
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)

    embeddings = []
    valid_heuristics = []

    # Embed in batches
    for i in tqdm(range(0, len(heuristics), batch_size), desc="Embedding heuristics"):
        batch = heuristics[i:i + batch_size]
        # Tokenize
        inputs = tokenizer(
            batch,
            padding=True,
            truncation=True,
            return_tensors='pt',
            max_length=max_length
        )
        inputs = {k: v.to(device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get embeddings
            model_output = model(**inputs)
            batch_embeddings = model_output.pooler_output  # Shape: (batch_size, hidden_size)
            embeddings.append(batch_embeddings.cpu())

        # Keep track of heuristics that were successfully embedded
        valid_heuristics.extend(batch)

    if not embeddings:
        print("No heuristics were embedded. Please check your input data or parameters.")
        return

    # Concatenate all embeddings
    embeddings = torch.cat(embeddings, dim=0).numpy().astype('float16')  # Shape: (num_heuristics, hidden_size)

    # Create a dictionary with embeddings and metadata
    data_to_save = {
        'heuristics': valid_heuristics,
        'embeddings': embeddings
    }

    # Save as a pickle file
    try:
        with open(embeddings_file, 'wb') as f_emb:
            pickle.dump(data_to_save, f_emb)
        print(f"Embeddings and heuristics saved to '{embeddings_file}'.")
    except Exception as e:
        print(f"Error saving embeddings to '{embeddings_file}': {e}")


def load_embeddings(embeddings_file: str = 'heuristics_embeddings.pkl') -> Tuple[List[str], np.ndarray]:
    """
    Load heuristics and their embeddings from a pickle file.

    Args:
        embeddings_file (str, optional): Path to the embeddings pickle file. Defaults to 'heuristics_embeddings.pkl'.

    Returns:
        Tuple[List[str], np.ndarray]: A tuple containing the list of heuristics and their corresponding embeddings.
    """
    try:
        with open(embeddings_file, 'rb') as f_emb:
            data = pickle.load(f_emb)
        heuristics = data['heuristics']
        embeddings = data['embeddings']
        print(f"Loaded {len(heuristics)} heuristics and their embeddings from '{embeddings_file}'.")
        return heuristics, embeddings
    except FileNotFoundError:
        print(f"Embeddings file not found at '{embeddings_file}'. Exiting.")
        exit(1)
    except Exception as e:
        print(f"Error loading embeddings from '{embeddings_file}': {e}")
        exit(1)


def get_top_k_heuristics(query: str, heuristics: List[str], embeddings: np.ndarray, k: int = 5) -> List[Tuple[str, float]]:
    """
    Retrieve the top K most relevant heuristics based on the query.

    Args:
        query (str): The input query string.
        heuristics (List[str]): List of heuristics.
        embeddings (np.ndarray): Numpy array of heuristic embeddings.
        k (int, optional): Number of top heuristics to retrieve. Defaults to 5.

    Returns:
        List[Tuple[str, float]]: A list of tuples containing the heuristic and its similarity score.
    """
    # Initialize tokenizer and model
    model_name = 'facebook/dpr-ctx_encoder-single-nq-base'
    tokenizer = DPRContextEncoderTokenizer.from_pretrained(model_name)
    model = DPRContextEncoder.from_pretrained(model_name)
    model.eval()

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Tokenize the query
    inputs = tokenizer(
        [query],
        padding=True,
        truncation=True,
        return_tensors='pt',
        max_length=512
    )
    inputs = {k: v.to(device) for k, v in inputs.items()}

    with torch.no_grad():
        query_embedding = model(**inputs).pooler_output.cpu().numpy().astype('float16')  # Shape: (1, hidden_size)

    # Compute cosine similarity
    similarities = cosine_similarity(query_embedding, embeddings)[0]  # Shape: (num_heuristics,)

    # Get top K indices
    top_k_indices = similarities.argsort()[-k:][::-1]

    top_k_heuristics = [(heuristics[idx], float(similarities[idx])) for idx in top_k_indices]

    return top_k_heuristics


def main():
    """
    Main function to embed heuristics and provide retrieval functionality.
    """
    # Configuration Parameters
    HEURISTICS_FILE_PATH = "/home/cpp/jerryhuang/reasoning/src/math/heuristics.txt"  # Path to heuristics.txt
    EMBEDDINGS_FILE = "heuristics_embeddings.pkl"  # Output embeddings file
    BATCH_SIZE = 32  # Adjust based on your GPU memory
    TOP_K = 5  # Number of top heuristics to retrieve

    # Step 1: Load Heuristics
    heuristics = load_heuristics(HEURISTICS_FILE_PATH)

    # Step 2: Embed Heuristics and Save Embeddings
    embed_heuristics(
        heuristics=heuristics,
        batch_size=BATCH_SIZE,
        embeddings_file=EMBEDDINGS_FILE
    )

    # Step 3: Load Embeddings
    loaded_heuristics, loaded_embeddings = load_embeddings(EMBEDDINGS_FILE)

    # Step 4: Example Query to Retrieve Top K Heuristics
    example_query = "How can I solve a system of equations?"
    top_k = get_top_k_heuristics(example_query, loaded_heuristics, loaded_embeddings, k=TOP_K)

    print(f"\nTop {TOP_K} heuristics for the query '{example_query}':")
    for idx, (heuristic, score) in enumerate(top_k, 1):
        print(f"{idx}. {heuristic} (Similarity: {score:.4f})")


if __name__ == "__main__":
    main()
