from datasets import load_dataset


SPLITS = [
    "biology",
    "earth_science",
    "economics",
    "psychology",
    "robotics",
    "stackoverflow",
    "sustainable_living",
    "pony",
    "leetcode",
    "aops",
    "theoremqa_theorems",
    "theoremqa_questions",
]

SUBSETS = [  # Which model augmented the query
    "examples",
    "Gemini-1.0_reason",
    "claude-3-opus_reason",
    "gpt4_reason",
    "grit_reason",
    "llama3-70b_reason",
]

dataset_task = "stackoverflow"
# examples = load_dataset("xlangai/BRIGHT", "examples")[dataset_task]

long_context = False
if long_context:
    doc_pairs = load_dataset("xlangai/BRIGHT", "long_documents")[dataset_task]
else:
    doc_pairs = load_dataset("xlangai/BRIGHT", "documents")[dataset_task]

# Extract document IDs and contents
print(len(doc_pairs))
print(doc_pairs[0].keys())

doc_ids = []
documents = []
for dp in doc_pairs:
    doc_ids.append(dp["id"])
    documents.append(dp["content"])

# Print some sample data
print(f"Number of examples: {len(examples)}")
print(f"Number of documents: {len(documents)}")
# print(f"Sample example: {examples[0]}")
# print(f"Sample document: {documents[0]}")
