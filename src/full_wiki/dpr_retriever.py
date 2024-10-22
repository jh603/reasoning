import torch
import numpy as np
from transformers import DPRQuestionEncoder, DPRQuestionEncoderTokenizer
import pickle

class DPRRetriever:
    def __init__(self, embeddings_file='/home/cpp/jerryhuang/reasoning/data/embeddings.pkl', device=None):
        self.device = device or torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        self.metadata, self.embeddings = self.load_embeddings(embeddings_file)
        self.tokenizer = DPRQuestionEncoderTokenizer.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.model = DPRQuestionEncoder.from_pretrained('facebook/dpr-question_encoder-single-nq-base')
        self.model.to(self.device)
        self.model.eval()

        # Convert embeddings to PyTorch tensor and move to GPU
        self.embeddings = torch.tensor(self.embeddings, dtype=torch.float32).to(self.device)
        # Normalize embeddings
        self.embeddings_norm = self.embeddings / torch.norm(self.embeddings, dim=1, keepdim=True)

    def load_embeddings(self, embeddings_file):
        with open(embeddings_file, 'rb') as f_emb:
            data = pickle.load(f_emb)
        return data['metadata'], data['embeddings']

    def retrieve(self, queries, top_k=5, batch_size=32):
        all_results = [None] * len(queries)  # Pre-allocate results list

        for i in range(0, len(queries), batch_size):
            batch_queries = queries[i:i+batch_size]
            batch_indices = list(range(i, min(i+batch_size, len(queries))))
            
            # Tokenize and encode queries
            inputs = self.tokenizer(batch_queries, return_tensors='pt', padding=True, truncation=True, max_length=512)
            inputs = {k: v.to(self.device) for k, v in inputs.items()}

            # Compute query embeddings
            with torch.no_grad():
                query_embeddings = self.model(**inputs).pooler_output

            # Normalize query embeddings
            query_embeddings_norm = query_embeddings / torch.norm(query_embeddings, dim=1, keepdim=True)

            # Compute cosine similarity
            scores = torch.matmul(self.embeddings_norm, query_embeddings_norm.t())

            # Get top-k results
            top_k_values, top_k_indices = torch.topk(scores, k=top_k, dim=0)

            for batch_idx, query_idx in enumerate(batch_indices):
                query_results = []
                for rank in range(top_k):
                    idx = top_k_indices[rank, batch_idx].item()
                    score = top_k_values[rank, batch_idx].item()
                    query_results.append({
                        'id': self.metadata[idx]['id'],
                        'title': self.metadata[idx]['title'],
                        'text': self.metadata[idx]['text'],
                        'score': score
                    })
                all_results[query_idx] = query_results

        return all_results

def main():
    # Example usage
    retriever = OptimizedDPRRetriever()
    
    queries = [
        "What is the capital of France?",
        "Who wrote the play Hamlet?",
        "What is the boiling point of water?",
        "When was the first moon landing?",
        "What is the largest planet in our solar system?"
    ]
    top_k = 5
    results = retriever.retrieve(queries, top_k=top_k)

    for query_idx, query_results in enumerate(results):
        print(f"Results for query: '{queries[query_idx]}'")
        for rank, result in enumerate(query_results, start=1):
            print(f"Rank {rank}:")
            print(f"ID: {result['id']}")
            print(f"Title: {result['title']}")
            print(f"Score: {result['score']:.4f}")
            print(f"Text: {result['text'][:100]}...")  # Print first 100 characters of text
            print("-" * 40)
        print("\n")

if __name__ == "__main__":
    main()