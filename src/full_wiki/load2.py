import csv

def count_articles(file_path):
    count = 0
    with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip the header row
        for row in reader:
            if len(row) >= 2:
                count += 1
    return count

def search_article(file_path, target_title):
    with open(file_path, 'r', newline='', encoding='utf-8') as tsvfile:
        reader = csv.reader(tsvfile, delimiter='\t')
        next(reader)  # Skip the header row
        
        for row in reader:
            if len(row) < 2:
                continue  # Skip malformed rows
            
            title, text = row[0], row[1]
            if title.lower() == target_title.lower():
                return title, text
    
    return None, None

def main():
    file_path = '/home/cpp/jerryhuang/reasoning/data/wikipedia_articles.tsv'
    target_title = "Meet Corliss Archer"
    
    total_articles = count_articles(file_path)
    print(f"Total number of articles: {total_articles}")
    
    print(f"Searching for article: {target_title}")
    found_title, found_text = search_article(file_path, target_title)
    
    if found_title:
        print(f"Article found: {found_title}")
        print("Text preview (first 500 characters):")
        print(found_text[:500] + "...")
    else:
        print(f"No article found with the title: {target_title}")

if __name__ == "__main__":
    main()