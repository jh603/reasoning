import os
import csv
import argparse
import torch
from tqdm import tqdm
from transformers import DPRContextEncoder, DPRContextEncoderTokenizer
import pickle
from torch.utils.data import DataLoader, Dataset

class ArticleDataset(Dataset):
    def __init__(self, titles, texts):
        self.titles = titles
        self.texts = texts

    def __len__(self):
        return len(self.titles)

    def __getitem__(self, idx):
        return self.titles[idx], self.texts[idx]

def embed_articles(input_tsv, embeddings_file, batch_size=32, max_articles=None):
    tokenizer = DPRContextEncoderTokenizer.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    model = DPRContextEncoder.from_pretrained('facebook/dpr-ctx_encoder-single-nq-base')
    model.eval()
    
    num_gpus = torch.cuda.device_count()
    if num_gpus > 1:
        print(f"Using {num_gpus} GPUs!")
        model = torch.nn.DataParallel(model)
    
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    # Read the TSV file
    articles = []
    with open(input_tsv, 'r', encoding='utf-8') as f:
        reader = csv.DictReader(f, delimiter='\t')
        for idx, row in enumerate(reader):
            if max_articles is not None and idx >= max_articles:
                break
            articles.append({
                'title': row['Title'],
                'text': row['Text']
            })

    # Prepare for embedding
    metadata = []
    titles = []
    texts = []

    for idx, article in enumerate(articles):
        titles.append(article['title'])
        texts.append(article['text'])
        metadata.append({
            'id': idx,
            'title': article['title'],
            'text': article['text']
        })

    # Create dataset and dataloader
    dataset = ArticleDataset(titles, texts)
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=False, num_workers=4)

    # Embed in batches
    all_embeddings = []
    total_batches = len(dataloader)
    
    with tqdm(total=total_batches, desc="Embedding articles") as pbar:
        for batch_titles, batch_texts in dataloader:
            inputs = tokenizer(
                batch_texts,
                batch_titles,
                padding=True,
                truncation=True,
                return_tensors='pt',
                max_length=512
            )

            inputs = {k: v.to(device) for k, v in inputs.items()}

            with torch.no_grad():
                embeddings_batch = model(**inputs).pooler_output
                all_embeddings.append(embeddings_batch.cpu())

            pbar.update(1)

    if not all_embeddings:
        print("No articles were embedded. Please check your input data or parameters.")
        return

    # Concatenate all embeddings
    all_embeddings = torch.cat(all_embeddings, dim=0)

    # Convert embeddings to numpy array and save as pickle
    embeddings_array = all_embeddings.numpy().astype('float16')
    
    # Create a dictionary with embeddings and full metadata
    data_to_save = {
        'embeddings': embeddings_array,
        'metadata': metadata
    }
    
    with open(embeddings_file, 'wb') as f_emb:
        pickle.dump(data_to_save, f_emb)

    print(f"Embeddings and full metadata for {len(metadata)} articles saved to {embeddings_file}")

if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Embed Wikipedia articles using DPR")
    parser.add_argument('--input_tsv', type=str, default='/home/cpp/jerryhuang/reasoning/data/wikipedia_articles.tsv', help='Path to the input TSV file')
    parser.add_argument('--embeddings_file', type=str, default='/home/cpp/jerryhuang/reasoning/data/embeddings.pkl', help='Path to save the embeddings pickle file')
    parser.add_argument('--batch_size', type=int, default=128, help='Batch size for embedding')
    parser.add_argument('--max_articles', type=int, default=None, help='Maximum number of articles to embed')
    args = parser.parse_args()

    embed_articles(args.input_tsv, args.embeddings_file, args.batch_size, args.max_articles)