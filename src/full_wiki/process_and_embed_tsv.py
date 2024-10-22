import os
import json
import bz2
import glob
import argparse
import re
from tqdm import tqdm
import torch
from transformers import AutoTokenizer, AutoModel

def process_wikipedia_dump(input_dir, output_file):
    """
    Processes the HotpotQA introductory paragraphs dataset by removing hyperlinks
    and preparing the text for embedding.
    """
    passages = []
    input_files = glob.glob(os.path.join(input_dir, '**/*.bz2'), recursive=True)
    for filepath in tqdm(input_files, desc="Processing files"):
        with bz2.open(filepath, 'rt', encoding='utf-8') as file:
            for line in file:
                data = json.loads(line)
                title = data.get('title', '')
                text = data.get('text', [])
                if not text:
                    continue
                # Flatten the list of sentences into paragraphs
                paragraphs = [''.join(sentences) for sentences in text]
                # Join paragraphs into a single text
                full_text = '\n'.join(paragraphs)
                # Remove hyperlinks
                clean_text = remove_hyperlinks(full_text)
                # Prepare passage
                passage = {
                    'title': title,
                    'text': clean_text
                }
                passages.append(passage)

    with open(output_file, 'w', encoding='utf-8') as f_out:
        for passage in passages:
            f_out.write(json.dumps(passage) + '\n')
    print(f"Processed {len(passages)} passages.")

def remove_hyperlinks(text):
    """
    Removes hyperlinks from the text.
    """
    # Remove <a href="...">...</a> tags
    clean_text = re.sub(r'<a href="[^"]+">([^<]+)</a>', r'\1', text)
    # Remove any remaining HTML tags (if any)
    clean_text = re.sub(r'<[^>]+>', '', clean_text)
    return clean_text

def generate_embeddings(passages_file, output_file, model_name, batch_size=16):
    """
    Generates embeddings for the passages using Hugging Face's DPR models and saves them in TSV format.
    """
    tokenizer = AutoTokenizer.from_pretrained(model_name)
    model = AutoModel.from_pretrained(model_name)
    model.eval()
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    embeddings = []
    texts = []
    titles = []

    # Read passages
    with open(passages_file, 'r', encoding='utf-8') as f_in:
        for line in f_in:
            passage = json.loads(line)
            texts.append(passage['text'])
            titles.append(passage['title'])

    # Open the TSV file for writing
    with open(output_file, 'w', encoding='utf-8') as f_out:
        # Write header
        f_out.write('id\ttitle\ttext\tembedding\n')
        # Process in batches
        for i in tqdm(range(0, len(texts), batch_size), desc="Generating embeddings"):
            batch_texts = texts[i:i+batch_size]
            batch_titles = titles[i:i+batch_size]
            inputs = tokenizer(batch_texts, padding=True, truncation=True, return_tensors='pt', max_length=512)
            inputs = {k: v.to(device) for k, v in inputs.items()}
            with torch.no_grad():
                outputs = model(**inputs)
                # Get the embeddings from the pooler output or last hidden state
                if hasattr(outputs, 'pooler_output') and outputs.pooler_output is not None:
                    batch_embeddings = outputs.pooler_output
                else:
                    # If the model does not have pooler_output, use the mean of the last hidden state
                    batch_embeddings = outputs.last_hidden_state.mean(dim=1)
                batch_embeddings = batch_embeddings.cpu().numpy()

            # Write embeddings to TSV
            for idx in range(len(batch_texts)):
                embedding = batch_embeddings[idx]
                emb_str = ' '.join(map(str, embedding.tolist()))
                line = f"{i + idx}\t{batch_titles[idx]}\t{batch_texts[idx]}\t{emb_str}\n"
                f_out.write(line)
    print(f"Generated embeddings for {len(texts)} passages and saved to {output_file}.")

if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('--input_dir', type=str, required=True, help='Path to the directory containing .bz2 files.')
    parser.add_argument('--output_passages', type=str, required=True, help='Path to save the processed passages.')
    parser.add_argument('--output_embeddings', type=str, required=True, help='Path to save embeddings TSV file.')
    parser.add_argument('--model_name', type=str, default="facebook/dpr-ctx_encoder-single-nq-base", help='Hugging Face model name.')
    parser.add_argument('--batch_size', type=int, default=16, help='Batch size for embedding generation.')
    args = parser.parse_args()

    process_wikipedia_dump(args.input_dir, args.output_passages)
    # generate_embeddings(args.output_passages, args.output_embeddings, args.model_name, args.batch_size)
