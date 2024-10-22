import argparse
import torch
import numpy as np
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pickle

def load_embeddings(embeddings_file):
    # Load embeddings and metadata
    with open(embeddings_file, 'rb') as f_emb:
        data = pickle.load(f_emb)
    embeddings = data['embeddings']
    metadata = data['metadata']
    return metadata, embeddings

def retrieve(query, embeddings, metadata, top_k=5):
    # Initialize the DPR question encoder and tokenizer
    tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
    model.eval()
    
    # Check if multiple GPUs are available
    if torch.cuda.device_count() > 1:
        print(f"Using {torch.cuda.device_count()} GPUs!")
        model = torch.nn.DataParallel(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Embed the query
    inputs = tokenizer(query, return_tensors='pt', truncation=True, max_length=512)
    inputs = {k: v.to(device) for k, v in inputs.items()}
    with torch.no_grad():
        query_embedding = model(**inputs).pooler_output.cpu().numpy()

    # Normalize embeddings
    embeddings_norm = embeddings / np.linalg.norm(embeddings, axis=1, keepdims=True)
    query_embedding_norm = query_embedding / np.linalg.norm(query_embedding)

    # Compute cosine similarity
    scores = np.dot(embeddings_norm, query_embedding_norm.T).squeeze()

    # Get top-k results
    top_k_indices = scores.argsort()[-top_k:][::-1]
    results = []
    for idx in top_k_indices:
        results.append({
            'id': metadata[idx]['id'],
            'title': metadata[idx]['title'],
            'text': metadata[idx]['text'],
            'score': float(scores[idx])
        })

    return results

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Retrieve documents using DPR")
    parser.add_argument('--embeddings_file', type=str, default='/home/cpp/jerryhuang/reasoning/data/embeddings.pkl', help='Path to the embeddings pickle file')
    parser.add_argument('--query', type=str, required=True, help='Query string')
    parser.add_argument('--top_k', type=int, default=5, help='Number of top results to retrieve')
    args = parser.parse_args()

    metadata, embeddings = load_embeddings(args.embeddings_file)
    results = retrieve(args.query, embeddings, metadata, top_k=args.top_k)

    for rank, result in enumerate(results, start=1):
        print(f"Rank {rank}:")
        print(f"ID: {result['id']}")
        print(f"Title: {result['title']}")
        print(f"Score: {result['score']:.4f}")
        print(f"Text: {result['text'][:500]}...")  # Print first 500 characters of text
        print("-" * 40)