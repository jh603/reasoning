import csv
import os
import pickle

import torch
from dotenv import load_dotenv
from sentence_transformers import SentenceTransformer
from tqdm import tqdm

TOPK = 10

load_dotenv()


class GTRWikiRetriever:
    def __init__(self):
        self.device = "cuda" if torch.cuda.is_available() else "cpu"

        print("loading GTR encoder...")
        self.encoder = SentenceTransformer(
            "sentence-transformers/gtr-t5-xxl", device=self.device
        )
        self.docs = self._load_wikipedia_docs()
        self.gtr_emb = self._load_or_build_embeddings()

    def _load_wikipedia_docs(self):
        DPR_WIKI_TSV = os.environ.get("DPR_WIKI_TSV")
        with open(DPR_WIKI_TSV) as f:
            docs = [
                row[2] + "\n" + row[1]
                for i, row in enumerate(csv.reader(f, delimiter="\t"))
                if i > 0
            ]
        return docs

    def _load_or_build_embeddings(self):
        GTR_EMB = os.environ.get("GTR_EMB")
        if os.path.exists(GTR_EMB):
            print("gtr embeddings found, loading...")
            with open(GTR_EMB, "rb") as f:
                embs = pickle.load(f)
        else:
            print("gtr embeddings not found, building...")
            embs = self._build_index(self.docs)

        return torch.tensor(embs, dtype=torch.float16, device=self.device)

    def _build_index(self, docs):
        with torch.inference_mode():
            embs = self.encoder.encode(
                docs, batch_size=4, show_progress_bar=True, normalize_embeddings=True
            )
            embs = embs.astype("float16")

        GTR_EMB = os.environ.get("GTR_EMB")
        with open(GTR_EMB, "wb") as f:
            pickle.dump(embs, f)
        return embs

    def gtr_wiki_retrieval(self, question):
        with torch.inference_mode():
            queries = torch.tensor(
                self.encoder.encode(
                    [question],
                    batch_size=4,
                    show_progress_bar=True,
                    normalize_embeddings=True,
                ),
                dtype=torch.float16,
                device="cpu",
            )

        print("running GTR retrieval...")
        ret = []
        for q in tqdm(queries.to(self.device)):
            scores, idx = torch.topk(torch.matmul(self.gtr_emb, q), TOPK)

            idx = idx.tolist()
            ret.extend(
                "Title: {}\nText: {}".format(doc.split("\n")[0], doc.split("\n")[1])
                for doc, score in zip([self.docs[i] for i in idx], scores)
            )

        return ret


if __name__ == "__main__":
    retriever = GTRWikiRetriever()
    results = retriever.gtr_wiki_retrieval("who was barack obama")
    for result in results:
        print(result)
