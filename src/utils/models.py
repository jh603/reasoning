import os
import time

import torch
import transformers
from dotenv import load_dotenv
from openai import OpenAI
from tenacity import retry, stop_after_attempt, wait_random_exponential
from transformers import AutoConfig, AutoTokenizer

load_dotenv()

transformers.logging.set_verbosity_error()

SYSTEM_PROMPT_LLAMA = "You are a helpful AI assistant."
SYSTEM_PROMPT_OPENAI = "You are a helpful assistant. Please give concise answers."


class Llama3Model:
    def __init__(self, model_path: str):
        self.pipeline = transformers.pipeline(
            "text-generation",
            model=model_path,
            model_kwargs={"torch_dtype": torch.float16},
            device=0,  # Adjust based on your hardware (e.g., -1 for CPU)
        )
        self.tokenizer = self.pipeline.tokenizer
        self.model = self.pipeline.model

    def get_response(
        self,
        query: str,
        system_prompt: str = SYSTEM_PROMPT_LLAMA,
        max_tokens: int = 700,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        messages = [
            {"role": "system", "content": system_prompt},
            {"role": "user", "content": query},
        ]
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        response = outputs[0]["generated_text"][len(prompt) :].strip()
        return response

    def get_response_with_history(
        self,
        conversation_history: list,
        system_prompt: str = SYSTEM_PROMPT_LLAMA,
        max_tokens: int = 700,
        temperature: float = 0.7,
        top_p: float = 0.9,
    ) -> str:
        """
        Generates a response based on the conversation history.

        Args:
            conversation_history (list): A list of message dictionaries representing the conversation history.
            system_prompt (str): The system prompt to initialize the conversation.
            max_tokens (int): Maximum number of tokens to generate.
            temperature (float): Sampling temperature.
            top_p (float): Nucleus sampling probability.

        Returns:
            str: The assistant's reply.
        """
        # Initialize the prompt with the system prompt
        messages = [{"role": "system", "content": system_prompt}]

        # Append the conversation history
        for message in conversation_history:
            if message["role"] not in {"system", "user", "assistant"}:
                raise ValueError(f"Unknown role: {message['role']}")
            messages.append({"role": message["role"], "content": message["content"]})

        # Add the latest user query
        # Assuming the last message is from the user
        prompt = self.tokenizer.apply_chat_template(
            messages, tokenize=False, add_generation_prompt=True
        )

        # Generate the response
        outputs = self.pipeline(
            prompt,
            max_new_tokens=max_tokens,
            do_sample=True,
            temperature=temperature,
            top_p=top_p,
        )

        # Extract the generated response
        generated_text = outputs[0]["generated_text"][len(prompt) :].strip()

        # Optionally, append the assistant's response to the history
        # conversation_history.append({"role": "assistant", "content": generated_text})

        return generated_text


class OpenAIModel:
    def __init__(self):
        self.openai_client = OpenAI(
            api_key=os.getenv("OPENAI_API_KEY"),
            base_url=os.getenv("OPENAI_BASE_URL"),
        )
        self.system_prompt = SYSTEM_PROMPT_OPENAI

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response(
        self,
        query: str,
        system_prompt: str = None,
        model: str = "gpt-4o-mini",
    ) -> str:
        if system_prompt is None:
            system_prompt = self.system_prompt

        response = self.openai_client.chat.completions.create(
            model=model,
            messages=[
                {"role": "system", "content": system_prompt},
                {"role": "user", "content": query},
            ],
            temperature=0.7,
        )

        reply = response.choices[0].message.content
        return reply

    @retry(wait=wait_random_exponential(min=1, max=60), stop=stop_after_attempt(6))
    def get_response_with_history(
        self,
        conversation_history: list,
        model: str = "gpt-4o-mini",
    ) -> str:
        """
        Sends the conversation history to the LLM model and retrieves the assistant's response.

        Args:
            conversation_history (list): A list of message dictionaries representing the conversation history.
            model (str): The model to use for generating responses.

        Returns:
            str: The assistant's reply.
        """
        try:
            response = self.openai_client.chat.completions.create(
                model=model,
                messages=conversation_history,
                temperature=0.7,
            )

            reply = response.choices[0].message.content.strip()
            return reply

        except Exception as e:
            print(f"An error occurred while getting response from the model: {e}")
            return ""

    def embed_text(
        self, text: str, model: str = "text-embedding-ada-002", max_retries: int = 5
    ):
        retry_count = 0
        while retry_count < max_retries:
            try:
                return (
                    self.openai_client.embeddings.create(input=text, model=model)
                    .data[0]
                    .embedding
                )
            except Exception as e:
                retry_count += 1
                if retry_count < max_retries:
                    wait_time = 2**retry_count
                    print(
                        f"Retry {retry_count}/{max_retries} after error: {e}. Waiting for {wait_time} seconds."
                    )
                    time.sleep(wait_time)
                else:
                    print(f"Failed after {max_retries} retries.")
                    raise


import sys

sys.path.append("/home/cpp/jerryhuang/beam_retriever")
from qa.reader_model import Reader  # Import your custom Reader class


def merge_find_ans(
    start_logits, end_logits, ids, punc_token_list, topk=5, max_ans_len=20
):
    def is_too_long(span_id, punc_token_list):
        for punc_token_id in punc_token_list:
            if punc_token_id in span_id:
                return True
        return False

    start_candidate_val, start_candidate_idx = start_logits.topk(topk, dim=-1)
    end_candidate_val, end_candidate_idx = end_logits.topk(topk, dim=-1)
    pointer_s, pointer_e = 0, 0
    start = start_candidate_idx[pointer_s].item()
    end = end_candidate_idx[pointer_e].item()
    span_id = ids[start : end + 1]
    while (
        start > end
        or (end - start) > max_ans_len
        or is_too_long(span_id, punc_token_list)
    ):
        if pointer_s + 1 < topk and (
            start_candidate_val[pointer_s + 1] > end_candidate_val[pointer_e]
        ):
            pointer_s += 1
        elif pointer_e + 1 < topk:
            pointer_e += 1
        else:
            break
        start = start_candidate_idx[pointer_s].item()
        end = end_candidate_idx[pointer_e].item()
        span_id = ids[start : end + 1]
    return span_id


class DebertaQAModel:
    def __init__(self, model_name, checkpoint_path):
        # Load the tokenizer
        tokenizer_path = "/home/cpp/jerryhuang/beam_retriever/"
        self.tokenizer = AutoTokenizer.from_pretrained(tokenizer_path)

        # Load the configuration
        config = AutoConfig.from_pretrained(model_name)
        config.max_position_embeddings = 1024  # Adjust to match training if necessary

        # Initialize the Reader model
        self.model = Reader(config, model_name, task_type="hotpot")

        # Resize embeddings to match the tokenizer
        self.model.encoder.resize_token_embeddings(len(self.tokenizer))

        # Load the checkpoint
        print(f"Loading checkpoint from {checkpoint_path}")
        checkpoint = torch.load(checkpoint_path, map_location=torch.device("cpu"))

        # Load the state dictionary
        self.model.load_state_dict(checkpoint, strict=False)

        # Move model to device
        self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
        self.model.to(self.device)
        self.model.eval()

    def get_response(
        self, question: str, context: str, answer_merge=True, topk=5, max_ans_len=20
    ) -> str:
        max_len = 1024  # Adjust as per your training setup

        # Tokenize the question
        q_codes = self.tokenizer.encode(
            question, add_special_tokens=False, truncation=True, max_length=max_len
        )

        # Tokenize the context
        c_codes = []
        # Assuming context is a string; you can split it into paragraphs if needed
        # For this example, we'll treat it as a single string
        context_tokens = self.tokenizer.encode(
            context,
            add_special_tokens=False,
            truncation=True,
            max_length=max_len
            - len(q_codes)
            - 2,  # Reserve space for question and special tokens
        )
        c_codes.append(context_tokens)

        # Calculate total length and average length per context chunk
        total_len = len(q_codes) + sum([len(item) for item in c_codes])
        context_ids = [self.tokenizer.cls_token_id] + q_codes
        avg_len = (max_len - 2 - len(q_codes)) // len(c_codes) if c_codes else 0

        # Concatenate context tokens, truncating if necessary
        for item in c_codes:
            if total_len > max_len - 2:
                item = item[:avg_len]
            context_ids.extend(item)

        # Add the [SEP] token at the end
        context_ids = context_ids[: max_len - 1] + [self.tokenizer.sep_token_id]

        # Convert to tensors
        input_ids = torch.tensor(
            context_ids, dtype=torch.long, device=self.device
        ).unsqueeze(0)
        attention_mask = torch.ones(
            [1, len(context_ids)], dtype=torch.long, device=self.device
        )

        # Perform inference
        with torch.no_grad():
            outputs = self.model(input_ids=input_ids, attention_mask=attention_mask)

        # Check if 'output_answer_type' is in outputs
        output_answer_type = outputs.get("output_answer_type")
        if output_answer_type is not None:
            ans_type_logits = output_answer_type[0]
            ans_type = torch.argmax(ans_type_logits).item()
            # Adjust mapping based on your model
            if ans_type == 0:
                return "no"
            elif ans_type == 1:
                return "yes"
            # If ans_type == 2, proceed to extract the answer span

        # Extract start and end logits
        start_logits = outputs["start_qa_logits"][0]
        end_logits = outputs["end_qa_logits"][0]

        input_ids = input_ids[0]

        if answer_merge:
            # Prepare the punctuation token IDs to avoid in the answer
            punc_tokens = ["[CLS]", "[SEP]", "[PAD]", ".", "?", "!", ","]
            punc_token_list = self.tokenizer.convert_tokens_to_ids(punc_tokens)

            # Call the merge_find_ans function
            span_id = merge_find_ans(
                start_logits,
                end_logits,
                input_ids.tolist(),
                punc_token_list,
                topk=topk,
                max_ans_len=max_ans_len,
            )

            # Decode the answer
            answer = self.tokenizer.decode(span_id, skip_special_tokens=True)
        else:
            # Find the start and end indices
            start_idx = torch.argmax(start_logits).item()
            end_idx = torch.argmax(end_logits).item()

            # Ensure start_idx <= end_idx
            if start_idx > end_idx:
                end_idx = start_idx

            # Decode the answer
            answer_ids = input_ids[start_idx : end_idx + 1]
            answer = self.tokenizer.decode(answer_ids, skip_special_tokens=True)

        return answer.strip()

    def get_response_with_history(self, history):
        pass
