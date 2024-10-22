import os
import pickle
import numpy as np
import faiss
from sentence_transformers import SentenceTransformer
from tqdm import tqdm
import torch
from torch.utils.data import Dataset, DataLoader

class ArticleDataset(Dataset):
    def __init__(self, ids, contents):
        self.ids = ids
        self.contents = contents

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.contents[idx]

class BrightRetriever:
    def __init__(self, embedding_model: str, batch_size: int = 64, max_length: int = 512):
        """
        Initializes the BrightRetriever with the specified embedding model.

        Args:
            embedding_model (str): The name of the embedding model to use.
            batch_size (int, optional): Batch size for embedding. Defaults to 64.
            max_length (int, optional): Maximum sequence length for tokenization. Defaults to 512.
        """
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.max_length = max_length
        self.document_embeddings = {}  # {id, content, embedding}
        self.index = None  # FAISS index
        
        print(f"Loading embedding model '{self.embedding_model}'...")
        self.model = SentenceTransformer(self.embedding_model)
        self.model.eval()

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
        num_gpus = torch.cuda.device_count()
        
        # Store base model for tokenization
        self.base_model = self.model
        
        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for embedding!")
            self.model = torch.nn.DataParallel(self.model)
        
        self.model.to(self.device)

    def embed_documents(self, document_ids: list, contents: list, save_file_path: str):
        """
        Embeds and adds documents to the retriever, saving them incrementally.

        Args:
            document_ids (list): List of document IDs.
            contents (list): List of document contents.
            save_file_path (str): Path to the pickle file to save documents and embeddings.
        """
        assert len(document_ids) == len(contents), "Document ids and contents must have the same length."

        num_documents = len(document_ids)
        print(f"Embedding and adding {num_documents} documents in batches of {self.batch_size}...")

        # Create dataset and dataloader
        dataset = ArticleDataset(document_ids, contents)
        dataloader = DataLoader(dataset, batch_size=self.batch_size, shuffle=False, num_workers=4)

        # Prepare to collect embeddings
        total_batches = len(dataloader)

        # Ensure the save file is empty if it exists
        if os.path.exists(save_file_path):
            os.remove(save_file_path)

        with tqdm(total=total_batches, desc="Embedding articles") as pbar:
            for batch_ids, batch_texts in dataloader:
                # Use base_model for tokenization
                inputs = self.base_model.tokenize(
                    batch_texts, 
                    # max_length=self.max_length, 
                    # truncation=True, 
                    # padding='longest'
                )
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                with torch.no_grad():
                    # Get embeddings using the DataParallel model
                    outputs = self.model(inputs)
                    embeddings_batch = outputs['sentence_embedding']

                    # Move embeddings to CPU and convert to numpy
                    embeddings_batch = embeddings_batch.cpu().numpy()

                # Save embeddings and metadata incrementally
                for doc_id, content, embedding in zip(batch_ids, batch_texts, embeddings_batch):
                    doc_data = {
                        "id": doc_id,
                        "content": content,
                        "embedding": embedding
                    }
                    self.document_embeddings.append(doc_data)

                    # Save to pickle file
                    with open(save_file_path, "ab") as f:
                        pickle.dump(doc_data, f)

                pbar.update(1)

        print(f"All documents embedded and saved to '{save_file_path}'.")

    def retrieve_faiss(self, query: str, top_k: int = 5) -> list:
        """
        Retrieves the top-K most similar documents to the given query using FAISS.

        Args:
            query (str): The query string.
            top_k (int, optional): Number of top documents to retrieve. Defaults to 5.

        Returns:
            list: List of tuples containing (document_id, content, similarity_score).
        """
        if self.index is None:
            raise ValueError("FAISS index not built. Call build_faiss_index() first.")

        # Use base_model for tokenization
        inputs = self.base_model.tokenize(
            [query], 
            # max_length=self.max_length, 
            # truncation=True, 
            # padding='longest'
        )
        inputs = {k: v.to(self.device) for k, v in inputs.items()}

        with torch.no_grad():
            # Get query embedding using the DataParallel model
            outputs = self.model(inputs)
            query_embedding = outputs['sentence_embedding'].cpu().numpy()

        # Normalize the query embedding
        query_embedding = query_embedding / np.linalg.norm(query_embedding, axis=1, keepdims=True)
        query_embedding = query_embedding.astype('float32')

        # Perform similarity search
        D, I = self.index.search(query_embedding, top_k)
        top_documents = []
        for score, idx in zip(D[0], I[0]):
            if idx == -1:
                continue  # No more neighbors
            doc = self.document_embeddings[idx]
            top_documents.append((doc["id"], doc["content"], float(score)))

        return top_documents

    def build_faiss_index(self):
        """
        Builds a FAISS index from the document embeddings for efficient similarity search.
        """
        print("Building FAISS index...")
        if not self.document_embeddings:
            raise ValueError("No documents to index. Please add documents first.")

        # Extract embeddings
        embeddings = np.array([doc["embedding"] for doc in self.document_embeddings]).astype('float32')
        # Normalize embeddings for cosine similarity
        faiss.normalize_L2(embeddings)

        # Initialize FAISS index (Inner Product for cosine similarity)
        dimension = embeddings.shape[1]
        self.index = faiss.IndexFlatIP(dimension)
        self.index.add(embeddings)
        print(f"FAISS index built with {self.index.ntotal} documents.")

    def save_state(self, save_file_path: str):
        """
        Saves the current state (documents and embeddings) to a single pickle file.
        """
        print(f"Saving state to '{save_file_path}'...")
        data = {
            "documents": self.document_embeddings
        }
        with open(save_file_path, "wb") as f:
            pickle.dump(data, f)
        print("State saved successfully.")

    def load_state(self, save_file_path: str):
        """
        Loads the state (documents and embeddings) from a single pickle file.
        """
        print(f"Loading state from '{save_file_path}'...")
        with open(save_file_path, "rb") as f:
            data = pickle.load(f)
            
        print(data.keys())
        self.document_embeddings = data
        print(f"Loaded {len(self.document_embeddings['id'])} documents.")

    def get_all_embeddings(self) -> np.ndarray:
        """
        Retrieves all document embeddings as a NumPy array.
        """
        return np.array([doc["embedding"] for doc in self.document_embeddings]).astype('float32')