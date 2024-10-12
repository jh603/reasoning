from src.utils.models import DebertaQAModel, Llama3Model, OpenAIModel


class ModelFactory:
    @staticmethod
    def create_model(model_name: str, **kwargs):
        model_name = model_name.lower()
        if model_name in ["meta-llama-3-8b-instruct"]:
            return Llama3Model(model_path="meta-llama/Meta-Llama-3.1-8B-Instruct")
        elif model_name in ["gpt-4o", "gpt-4o-mini", "gpt-3.5-turbo-instruct"]:
            return OpenAIModel()
        elif model_name in ["deberta-v3-large"]:
            return DebertaQAModel(
                model_name="microsoft/deberta-v3-large",
                checkpoint_path="/home/cpp/jerryhuang/beam_retriever/output/10-10-2024/hotpotqa_reader_deberta_large-seed42-bsz4-fp16True-lr1e-05-decay0.0-warm0.1-valbsz32/checkpoint_best.pt",
            )
        else:
            raise ValueError(f"Unsupported model name: {model_name}")


class Model:
    def __init__(self, model_name: str, **kwargs):
        self.model_instance = ModelFactory.create_model(model_name, **kwargs)

    def get_response(self, query: str, context=None) -> str:
        if context:
            return self.model_instance.get_response(query, context)
        return self.model_instance.get_response(query)

    def get_response_with_history(self, history) -> str:
        return self.model_instance.get_response_with_history(history)
