from datasets import load_dataset

from src.retriever import GTRWikiRetriever
from src.utils.llama import Llama3

if __name__ == "__main__":
    NUM_TRAINING_SAMPLES = 1

    # 7405 samples
    hotpot_data = load_dataset("hotpot_qa", "fullwiki", split="validation")

    retriever = GTRWikiRetriever()
    llama_model = Llama3(
        "/home/cpp/jerryhuang/search2024/meta-llama/Meta-Llama-3-8B-Instruct"
    )

    prompt = ""
    for idx, qa_pair in enumerate(hotpot_data.select(range(NUM_TRAINING_SAMPLES))):
        question = qa_pair.get("question")
        documents = retriever.gtr_wiki_retrieval(question)
        print(f"documents: {documents}")

        documents_flattened = "\n".join(documents)
        prompt = f"""
        Answer the given question using only the provided documents below:
        
        Question: {question}
        
        Documents: {documents_flattened}
        """

        resp = llama_model.get_llama_response(prompt)
        print(f"resp: {resp}")
