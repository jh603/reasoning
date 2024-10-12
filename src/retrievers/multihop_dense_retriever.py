import argparse  # Import argparse to use Namespace
import json
from functools import partial

import faiss
import numpy as np
import torch
from mdr.qa.qa_dataset import QAEvalDataset, qa_collate
from mdr.qa.qa_model import QAModel
# Assuming you have the following modules from the multihop_dense_retrieval repository
from mdr.retrieval.models.mhop_retriever import RobertaRetriever
from mdr.retrieval.utils.utils import load_saved, move_to_cuda
from scripts.train_qa import \
    eval_final  # Make sure this function is accessible
from torch.utils.data import DataLoader
from transformers import AutoConfig, AutoTokenizer


class MultiHopQA:
    def __init__(self, retriever_args, reader_args):
        """
        Initialize the MultiHopQA system with retriever and reader configurations.

        :param retriever_args: Arguments required for the retriever initialization.
        :param reader_args: Arguments required for the reader initialization.
        """
        # Convert argument dictionaries to Namespace objects
        retriever_args = argparse.Namespace(**retriever_args)
        reader_args = argparse.Namespace(**reader_args)

        # Initialize device before initializing models
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

        # Initialize retriever components
        print(retriever_args.model_name)  # Debug print
        self.retriever_tokenizer = AutoTokenizer.from_pretrained(
            retriever_args.model_name
        )
        self.retriever = self._init_retriever(retriever_args)
        self.index = self._init_index(
            retriever_args.index_path, retriever_args.index_gpu
        )
        self.id2doc = self._load_corpus(retriever_args.corpus_dict)

        # Initialize reader components
        self.reader_tokenizer = AutoTokenizer.from_pretrained(reader_args.model_name)
        self.reader = self._init_reader(reader_args)

        # Other configurations
        self.top_k = getattr(retriever_args, "top_k", 20)
        self.max_q_len = getattr(retriever_args, "max_q_len", 70)
        self.max_q_sp_len = getattr(retriever_args, "max_q_sp_len", 350)

    def _init_retriever(self, args):
        """Initialize the retriever model."""
        config = AutoConfig.from_pretrained(args.model_name)
        retriever = RobertaRetriever(config, args)
        retriever = load_saved(retriever, args.model_path, exact=False)
        retriever.to(self.device)
        # If not using Apex, comment out the next line
        # retriever = amp.initialize(retriever, opt_level='O1')
        retriever.eval()
        return retriever

    def _init_index(self, index_path, index_gpu):
        """Initialize the FAISS index."""
        xb = np.load(index_path).astype("float32")
        index = faiss.IndexFlatIP(xb.shape[1])
        index.add(xb)
        if index_gpu != -1:
            res = faiss.StandardGpuResources()
            index = faiss.index_cpu_to_gpu(res, index_gpu, index)
        return index

    def _load_corpus(self, corpus_dict_path):
        """Load the document corpus mapping."""
        with open(corpus_dict_path, "r") as f:
            id2doc = json.load(f)
        return id2doc

    def _init_reader(self, args):
        """Initialize the reader model."""
        config = AutoConfig.from_pretrained(args.model_name)
        reader = QAModel(config, args)
        reader = load_saved(reader, args.reader_path, exact=False)
        reader.to(self.device)
        # If not using Apex, comment out the next line
        # reader = amp.initialize(reader, opt_level='O1')
        reader.eval()
        return reader

    def answer_question(self, question):
        """
        Answer a question using the multi-hop dense retriever and reader.

        :param question: The input question string.
        :return: A tuple containing the answer and supporting documents.
        """
        with torch.no_grad():
            # First-hop retrieval
            q_encoded = self.retriever_tokenizer.encode_plus(
                question,
                max_length=self.max_q_len,
                truncation=True,
                return_tensors="pt",
            ).to(self.device)
            q_embed = (
                self.retriever.encode_q(
                    q_encoded["input_ids"],
                    q_encoded["attention_mask"],
                    q_encoded.get("token_type_ids", None),
                )
                .cpu()
                .numpy()
            )

            # Search the index for the first hop
            scores_1, doc_ids_1 = self.index.search(q_embed, self.top_k)

            # Prepare query-document pairs for the second hop
            query_pairs = []
            for doc_id in doc_ids_1[0]:
                doc = self.id2doc[str(doc_id)]["text"]
                if not doc.strip():
                    doc = self.id2doc[str(doc_id)]["title"]
                query_pairs.append((question, doc))

            # Second-hop retrieval
            q_sp_encoded = self.retriever_tokenizer.batch_encode_plus(
                query_pairs,
                max_length=self.max_q_sp_len,
                truncation=True,
                padding=True,
                return_tensors="pt",
            ).to(self.device)
            q_sp_embed = (
                self.retriever.encode_q(
                    q_sp_encoded["input_ids"],
                    q_sp_encoded["attention_mask"],
                    q_sp_encoded.get("token_type_ids", None),
                )
                .cpu()
                .numpy()
            )

            # Search the index for the second hop
            scores_2, doc_ids_2 = self.index.search(q_sp_embed, self.top_k)

            # Reshape and combine scores
            scores_2 = scores_2.reshape(1, self.top_k, self.top_k)
            doc_ids_2 = doc_ids_2.reshape(1, self.top_k, self.top_k)
            path_scores = np.expand_dims(scores_1, axis=2) + scores_2
            search_scores = path_scores[0]

            # Rank the document pairs
            ranked_indices = np.unravel_index(
                np.argsort(search_scores.ravel())[::-1], (self.top_k, self.top_k)
            )
            ranked_pairs = np.vstack(ranked_indices).transpose()

            # Collect top document chains
            chains = []
            top_docs = {}
            for idx in range(self.top_k):
                first_idx, second_idx = ranked_pairs[idx]
                doc1_id = str(doc_ids_1[0, first_idx])
                doc2_id = str(doc_ids_2[0, first_idx, second_idx])
                doc1 = self.id2doc[doc1_id]
                doc2 = self.id2doc[doc2_id]
                chains.append([doc1, doc2])
                top_docs[doc1["title"]] = doc1["text"]
                top_docs[doc2["title"]] = doc2["text"]

            # Prepare input for the reader
            reader_input = [
                {"_id": 0, "question": question, "candidate_chains": chains}
            ]

            # Evaluate using the reader
            collate_fn = partial(qa_collate, pad_id=self.reader_tokenizer.pad_token_id)
            eval_dataset = QAEvalDataset(
                self.reader_tokenizer, reader_input, max_seq_len=512, max_q_len=64
            )
            eval_dataloader = DataLoader(
                eval_dataset, batch_size=self.top_k, collate_fn=collate_fn
            )

            qa_results = eval_final(
                None, self.reader, eval_dataloader, gpu=torch.cuda.is_available()
            )

            # Extract the answer and supporting passages
            answer_pred = qa_results["answer"][0]
            titles_pred = qa_results["titles"][0]
            supporting_passages = [
                {"title": title, "text": top_docs[title]} for title in titles_pred
            ]

            return answer_pred, supporting_passages


# Usage Example
if __name__ == "__main__":
    retriever_args = {
        "model_name": "roberta-base",
        "model_path": "/home/cpp/jerryhuang/reasoning/multihop_dense_retrieval/models/q_encoder.pt",
        "index_path": "/home/cpp/jerryhuang/reasoning/multihop_dense_retrieval/data/hotpot_index/wiki_index.npy",
        "corpus_dict": "/home/cpp/jerryhuang/reasoning/multihop_dense_retrieval/data/hotpot_index/wiki_id2doc.json",
        "topk": 20,  # Adjusted from 2 to 20
        "num_workers": 10,
        "max_q_len": 70,
        "max_c_len": 300,
        "max_q_sp_len": 350,
        "batch_size": 100,
        "beam_size": 5,
        "gpu": torch.cuda.is_available(),
        "save_index": False,
        "only_eval_ans": False,
        "shared_encoder": True,
        "save_path": "",
        "stop_drop": 0.0,
        "hnsw": False,
        "index_gpu": -1,
    }

    reader_args = {
        "model_name": "google/electra-large-discriminator",
        "reader_path": "/home/cpp/jerryhuang/reasoning/multihop_dense_retrieval/models/qa_electra.pt",
        "sp_weight": 0,
        "sp_pred": False,
        "max_ans_len": 30,
        "save_prediction": "",
    }

    # Initialize the MultiHopQA system
    multi_hop_qa = MultiHopQA(retriever_args, reader_args)

    # Input question
    question = "Who is the author of the book that inspired the movie Blade Runner?"

    # Get the answer and supporting documents
    answer, supporting_docs = multi_hop_qa.answer_question(question)

    # Print the results
    print(f"Question: {question}")
    print(f"Answer: {answer}")
    print("Supporting Documents:")
    for doc in supporting_docs:
        print(f"Title: {doc['title']}")
        print(f"Text: {doc['text']}\n")
