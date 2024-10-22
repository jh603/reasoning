import json
import bz2
import tarfile
import csv
import re
from typing import Dict
from io import BytesIO

class WikipediaArticle:
    def __init__(self, data: Dict):
        self.id = data['id']
        self.url = data['url']
        self.title = data['title']
        self.text = data['text']

    def get_plaintext(self) -> str:
        """Return the full plaintext of the article without HTML tags."""
        return ' '.join([''.join(paragraph) for paragraph in self.text])

def process_wikipedia_dump(file_path: str):
    with tarfile.open(file_path, 'r:bz2') as tar:
        for member in tar:
            if member.isfile() and member.name.endswith('.bz2'):
                f = tar.extractfile(member)
                if f is not None:
                    bz2_content = BytesIO(f.read())
                    with bz2.open(bz2_content, 'rt', encoding='utf-8') as bz2_file:
                        for line in bz2_file:
                            article_data = json.loads(line)
                            article = WikipediaArticle(article_data)
                            yield article

def remove_html_tags(text: str) -> str:
    """Remove HTML tags from the given text."""
    clean = re.compile('<.*?>')
    return re.sub(clean, '', text)

def main():
    dump_file = '/home/cpp/jerryhuang/reasoning/data/enwiki-20171001-pages-meta-current-withlinks-abstracts.tar.bz2'
    output_file = '/home/cpp/jerryhuang/reasoning/data/wikipedia_articles.tsv'
    
    print(f"Processing Wikipedia dump and writing to {output_file}...")
    with open(output_file, 'w', newline='', encoding='utf-8') as tsvfile:
        writer = csv.writer(tsvfile, delimiter='\t')
        writer.writerow(['Title', 'Text'])  # Write header
        
        for i, article in enumerate(process_wikipedia_dump(dump_file)):
            title = article.title
            text = remove_html_tags(article.get_plaintext())
            
            # Write to TSV
            writer.writerow([title, text])
            
            if i % 1000 == 0:
                print(f"Processed {i+1} articles")

    print(f"Processing complete. Data saved to {output_file}")

if __name__ == "__main__":
    main()