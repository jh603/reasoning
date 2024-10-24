import os
import pickle
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm


class ArticleDataset(Dataset):
    def __init__(self, ids, contents):
        self.ids = ids
        self.contents = contents

    def __len__(self):
        return len(self.ids)

    def __getitem__(self, idx):
        return self.ids[idx], self.contents[idx]


class BrightRetriever:
    def __init__(self, embedding_model: str, batch_size: int = 128):
        self.embedding_model = embedding_model
        self.batch_size = batch_size
        self.document_embeddings = []

        print(f"Loading embedding model '{embedding_model}'...")
        if embedding_model == "BAAI/bge-large-en-v1.5":
            self.model = SentenceTransformer("BAAI/bge-large-en-v1.5")
        else:
            self.model = SentenceTransformer(embedding_model)

        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        num_gpus = torch.cuda.device_count()

        # Store base model for tokenization
        self.base_model = self.model

        if num_gpus > 1:
            print(f"Using {num_gpus} GPUs for embedding!")
            self.model = torch.nn.DataParallel(self.model)

        self.model.to(self.device)

    def encode(
        self, texts, batch_size=None, show_progress_bar=True, normalize_embeddings=True
    ):
        dataset = ArticleDataset(range(len(texts)), texts)
        dataloader = DataLoader(
            dataset,
            batch_size=batch_size or self.batch_size,
            shuffle=False,
            num_workers=4,
        )

        all_embeddings = []
        with torch.no_grad():
            for batch_ids, batch_texts in tqdm(
                dataloader, disable=not show_progress_bar, desc="Encoding"
            ):
                inputs = self.base_model.tokenize(batch_texts)
                inputs = {k: v.to(self.device) for k, v in inputs.items()}

                outputs = self.model(inputs)
                embeddings = outputs["sentence_embedding"].cpu().numpy()
                if normalize_embeddings:
                    embeddings = embeddings / np.linalg.norm(
                        embeddings, axis=1, keepdims=True
                    )
                all_embeddings.append(embeddings)

        return np.vstack(all_embeddings)

    def embed_documents(self, document_ids, contents, save_file_path):
        """Embed documents using multi-GPU support"""
        assert len(document_ids) == len(contents)

        print(f"Embedding {len(contents)} documents...")
        embeddings = self.encode(contents)

        self.document_embeddings = [
            {"id": doc_id, "content": content, "embedding": embedding}
            for doc_id, content, embedding in zip(document_ids, contents, embeddings)
        ]

        self.save_state(save_file_path)
        print(f"Documents embedded and saved to '{save_file_path}'")

    def get_document_embeddings(self):
        return np.array([doc["embedding"] for doc in self.document_embeddings])

    def get_document_ids(self):
        return [doc["id"] for doc in self.document_embeddings]

    def save_state(self, save_file_path):
        with open(save_file_path, "wb") as f:
            pickle.dump(self.document_embeddings, f)

    def load_state(self, save_file_path):
        print(f"Loading state from '{save_file_path}'...")
        with open(save_file_path, "rb") as f:
            self.document_embeddings = pickle.load(f)
        print(f"Loaded {len(self.document_embeddings)} documents")
