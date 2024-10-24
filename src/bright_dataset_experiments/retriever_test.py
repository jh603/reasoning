from sentence_transformers import util
import torch

from src.bright_dataset_experiments.retriever import BrightRetriever


def calculate_string_similarity(
    str1: str, str2: str, model_name: str = "all-MiniLM-L6-v2"
) -> float:
    """
    Calculate the cosine similarity between two strings using a sentence transformer model.

    Args:
        str1 (str): First string
        str2 (str): Second string
        model_name (str): Name of the sentence transformer model to use

    Returns:
        float: Cosine similarity score between 0 and 1
    """
    # Initialize the retriever with the specified model
    retriever = BrightRetriever(embedding_model=model_name)

    # Tokenize both strings
    inputs1 = retriever.base_model.tokenize([str1])
    inputs2 = retriever.base_model.tokenize([str2])

    # Move inputs to the appropriate device
    inputs1 = {k: v.to(retriever.device) for k, v in inputs1.items()}
    inputs2 = {k: v.to(retriever.device) for k, v in inputs2.items()}

    # Get embeddings
    with torch.no_grad():
        embedding1 = retriever.model(inputs1)["sentence_embedding"]
        embedding2 = retriever.model(inputs2)["sentence_embedding"]

    # Calculate cosine similarity
    similarity = util.pytorch_cos_sim(embedding1, embedding2)

    return float(similarity[0][0])


# Example usage
if __name__ == "__main__":
    # Example strings
    # 0.7712
    # 0.8213
    # 0.8227

    # string1 = "Python type hints: what should I use for a variable that can be any iterable sequence? ``` def mysum(x)->int: s = 0 for i in x: s += i return s ``` The argument x can be list[int] or set[int], it can also be d.keys() where d is a dict, it can be range(10), as well as possibly many other sequences, where the item is of type int. What is the correct type-hint for x?"
    # string1 = "The problem is determining the correct type hint for the parameter x in the function mysum, where x is any iterable of integers. To solve this, we need to specify that x is an iterable and its elements are integers. The type hint for this is Iterable[int], which means an iterable containing integers."
    string1 = "I have this table and need to transform it to ... I don’t like UNPIVOT. Is there a better function in snowflake for this?"
    string2 = "The function FLATTEN flattens (explodes) compound values into multiple rows ...  FLATTEN( INPUT => <expr> "
    #     string2 = """
    # The document provides information on deprecated features in Python's typing module. Specifically:

    # 1. The class `Reversible` from `collections.abc` has been deprecated since Python 3.9. It now supports subscripting (`[]`), as per **PEP 585**.
    # 2. The class `Sized` from `typing` is a deprecated alias for `collections.abc.Sized`. As of Python 3.12, it is recommended to use `collections.abc.Sized` directly.

    # References to the respective PEPs and official Python documentation are provided for further details on these deprecations.
    # """
    #     string2 = """
    # Loading dataset subset 'documents' and split 'stackoverflow'...
    # Searching for document ID: Python_development_tools/typing_12_1.txt...

    # Content of Document ID Python_development_tools/typing_12_1.txt:

    # rsible
    # "collections.abc.Reversible") .

    # Deprecated since version 3.9:  [ ` collections.abc.Reversible  `
    # ](collections.abc.html#collections.abc.Reversible
    # "collections.abc.Reversible") now supports subscripting ( ` []  ` ). See  [
    # **PEP 585** ](https://peps.python.org/pep-0585/) and [ Generic Alias Type
    # ](stdtypes.html#types-genericalias) .

    # _ class  _ typing.  Sized  Â¶

    # Deprecated alias to [ ` collections.abc.Sized  `
    # ](collections.abc.html#collections.abc.Sized "collections.abc.Sized") .

    # Deprecated since version 3.12:  Use [ ` collections.abc.Sized  `
    # ](collections.abc.html#collections.abc.Sized "collections.abc.Sized") directly
    # instead.
    #     """

    model_name = "hkunlp/instructor-xl"
    similarity_score = calculate_string_similarity(string1, string2, model_name)
    print(f"\nString 1: {string1}")
    print(f"String 2: {string2}")
    print(f"Similarity score: {similarity_score:.4f}")
