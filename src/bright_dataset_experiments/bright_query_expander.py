from src.utils.model_factory import Model


class BrightQueryExpander:
    def __init__(
        self,
        model: str = "gpt-4o-mini",
    ):
        self.model = Model(model)

    def expand(self, original_query: str):
        prompt = f"""
Can you rewrite this question in clear English:

{original_query}
"""
        question_clarified = self.model.get_response(prompt)

        prompt2 = f"""
What type of resource or answer is this person looking for to answer this question:

{question_clarified}    
"""
        response2 = self.model.get_response(prompt2)
        output = f"{question_clarified}\n{response2}"
        return output
