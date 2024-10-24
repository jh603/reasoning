from typing import List, Dict, Any, Optional

from src.utils.model_factory import Model


class BrightReranker:
    def __init__(
        self,
        model: str = "gpt-3.5-turbo",
    ):
        self.model = Model(model)

    def rerank(self, documents: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        """
        Reranks the provided list of documents using the LLM.

        Args:
            documents (List[Dict[str, Any]]): A list of documents to rerank. Each document must have 'id', 'content', and 'score' keys.

        Returns:
            List[Dict[str, Any]]: The reranked list of documents sorted in descending order of the new scores.
        """
        if not documents:
            return []

        # Construct the prompt for the LLM
        prompt = self._construct_prompt(documents)

        # Call the LLM to get the reranked results
        response = self._call_llm(prompt)

        # Parse the LLM response to extract the reranked documents
        reranked_documents = self._parse_response(response, documents)

        return reranked_documents
