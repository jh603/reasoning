import argparse
import html
import json
import os
import re

from dotenv import load_dotenv

from src.utils.datasets import load_hotpotqa
from src.utils.hotpotqa_eval import eval
from src.utils.model_factory import Model

load_dotenv()


def clean_json_quotes(json_list):
    """
    Cleans JSON strings by removing escaped and regular quotes.

    Args:
        json_list (list): List of JSON items.

    Returns:
        list: Cleaned list of JSON items.
    """

    def clean_value(value):
        if isinstance(value, str):
            # Remove escaped quotes first, then regular quotes
            return value.replace('\\"', "").replace('"', "").replace("'", "")
        elif isinstance(value, dict):
            return {k: clean_value(v) for k, v in value.items()}
        elif isinstance(value, list):
            return [clean_value(item) for item in value]
        else:
            return value

    cleaned_list = []
    for item in json_list:
        if isinstance(item, str):
            # If the item is a JSON string, parse it first
            item = json.loads(item)
        cleaned_item = clean_value(item)
        cleaned_list.append(cleaned_item)

    return cleaned_list


def fix_json_string(input_string):
    """
    Cleans and formats a string to ensure it's valid JSON.

    Args:
        input_string (str): The input string to fix.

    Returns:
        str or None: Fixed JSON string or None if parsing fails.
    """
    # Remove all text before the first '{' only if the string doesn't start with '{'
    if not input_string.strip().startswith("{"):
        input_string = re.sub(r"^.*?(?={)", "", input_string, flags=re.DOTALL)

    # Remove 'response:' prefix and '```json' markers if present
    input_string = re.sub(r"^response:\s*", "", input_string.strip())
    input_string = re.sub(r"^```json\s*|\s*```$", "", input_string, flags=re.MULTILINE)

    # Replace smart quotes with standard double quotes
    input_string = (
        input_string.replace('"', '"')
        .replace('"', '"')
        .replace("'", "'")
        .replace("'", "'")
    )

    # Replace single quotes with double quotes for property names
    input_string = re.sub(r"(?<=\{|\,)\s*'(\w+)'\s*:", r'"\1":', input_string)

    # Escape double quotes inside sentences
    input_string = re.sub(
        r'(:\s*")([^"]*?)(")',
        lambda m: m.group(1) + m.group(2).replace('"', '\\"') + m.group(3),
        input_string,
    )

    # Replace unescaped single quotes inside double-quoted strings
    input_string = re.sub(
        r'(?<=["])([^"]*?)\\\'([^"]*?)(?=["])', r"\1'\2", input_string
    )

    # Remove trailing commas in objects and arrays
    input_string = re.sub(r",\s*([\]}])", r"\1", input_string)

    # Add missing commas between key-value pairs
    input_string = re.sub(r'("\s*:\s*"[^"]*")\s*(")', r"\1,\2", input_string)

    input_string = input_string.strip()

    try:
        json_obj = json.loads(input_string)
        return json.dumps(json_obj, indent=4)
    except json.JSONDecodeError as e:
        print(f"fix_json_string Error: Unable to parse JSON string. {str(e)}")
        print(f"Problematic Input: {input_string}")
        return None


def parse_important_sentences(response):
    """
    Parses the AI model's JSON response to extract important sentences.

    Expected response format:
    {
        "title": "Document Title",
        "important_sentences": [
            {"index": 1, "sentence": "First important sentence."},
            {"index": 3, "sentence": "Third important sentence."},
        ]
    }

    Args:
        response (str): The response string from the model.

    Returns:
        tuple: A tuple containing the title and a list of important sentences.
    """
    try:
        # Step 1: Extract text after "json:"
        response_after = ""
        match = re.search(r"json:\s*(\{.*\})", response, re.DOTALL | re.IGNORECASE)
        if match:
            response_after = match.group(1)
        else:
            print(
                "parse_important_sentences Error: 'json:' prefix not found in the response."
            )
            return None, []

        # Step 2: Clean and fix the JSON string
        fixed_json = fix_json_string(response_after)
        if not fixed_json:
            return None, []

        # Step 3: Load the JSON data
        data = json.loads(fixed_json)
        title = data.get("title", "")
        important_sentences = data.get("important_sentences", [])

        # Step 4: Validate the structure
        if not isinstance(title, str):
            print(
                "parse_important_sentences Error: Invalid or missing 'title' in JSON response."
            )
            return None, []
        if not isinstance(important_sentences, list):
            print(
                "parse_important_sentences Error: Invalid or missing 'important_sentences' in JSON response."
            )
            return None, []

        # Step 5: Extract sentences
        sentences = []
        for item in important_sentences:
            idx = item.get("index")
            sentence = item.get("sentence")
            if isinstance(idx, int) and isinstance(sentence, str):
                sentences.append((idx, sentence.strip()))
            else:
                print(
                    f"parse_important_sentences Error: Invalid sentence entry: {item}"
                )

        return title, sentences

    except json.JSONDecodeError as e:
        print(
            f"parse_important_sentences Error: Failed to decode JSON response. {str(e)}"
        )
        print(f"Response Content: {response}")
        return None, []
    except Exception as e:
        print(
            f"parse_important_sentences Error: An unexpected error occurred. {str(e)}"
        )
        return None, []


# ----------------- Multi-Agent Framework Implementation -----------------


class CoordinatorAgent:
    """
    Orchestrates the workflow by coordinating between different agents.
    """

    def __init__(self, model_instance):
        self.question_decomposition_agent = QuestionDecompositionAgent(model_instance)
        self.info_extraction_agent = InformationExtractionAgent(model_instance)
        self.reasoning_agent = ReasoningAgent(model_instance)
        self.answer_synthesis_agent = AnswerSynthesisAgent(model_instance)

    def process_question(self, question, documents):
        """
        Processes a single question through decomposition, information extraction, reasoning, and answer synthesis.

        Args:
            question (str): The main question to answer.
            documents (list): List of documents, each as [title, sentences].

        Returns:
            tuple: Final answer and combined references.
        """
        num_hops = 2
        sub_questions = self.question_decomposition_agent.decompose(
            question, documents, num_hops
        )
        rephrased_subquestions, subquestion_answers, top_documents = (
            self.info_extraction_agent.answer_subquestions(sub_questions, documents)
        )

        reasoning_results = self.reasoning_agent.reason(
            question,
            rephrased_subquestions,
            subquestion_answers,
            top_documents,
            documents,
        )
        answer = reasoning_results.get("answer", None)
        reasoning_references = reasoning_results.get("references", [])

        return answer, reasoning_references

    def combine_and_validate_references(self, reasoning_refs, top_documents):
        """
        Combines reasoning references, ensuring uniqueness and validity.

        Args:
            reasoning_refs (list): List of references from reasoning phase.
            top_documents (list): List of top documents for validation.

        Returns:
            list: Combined and validated references.
        """
        # Use a set of tuples to avoid duplicates
        reference_set = set()

        # Create a mapping of document titles to their sentence counts for validation
        # Using lowercase for case-insensitive matching
        doc_title_to_sentence_count = {
            title.lower(): len(sentences) for title, sentences in top_documents
        }

        # Function to validate a single reference
        def is_valid_reference(ref):
            title, idx = ref
            if title.lower() not in doc_title_to_sentence_count:
                print(
                    f"Validation Error: Document title '{title}' not found in the provided top documents."
                )
                return False
            if not (0 <= idx < doc_title_to_sentence_count[title.lower()]):
                print(
                    f"Validation Error: Sentence index '{idx}' out of range for document '{title}'."
                )
                return False
            return True

        # Combine references from reasoning
        all_refs = reasoning_refs

        for ref in all_refs:
            if is_valid_reference(ref):
                reference_set.add(tuple(ref))
            else:
                print(f"Invalid reference excluded: {ref}")

        # Convert back to list of lists
        combined_references = [list(ref) for ref in reference_set]

        # Sort references for consistency
        combined_references.sort(
            key=lambda x: (x[0].lower(), x[1])
        )  # Sort by title then index

        return combined_references


class QuestionDecompositionAgent:
    """
    Decomposes a complex question into simpler sub-questions with placeholders.
    """

    def __init__(self, model_instance):
        self.model = model_instance

    def decompose(self, question, documents, num_hops):
        prompt = f"""
Decompose the following multi-hop question into {num_hops} clear and specific sub-questions that can be either independently or sequentially using the documents below to help arrive at the final answer.

Each sub-question should:
- Be directly relevant to answering the original question.
- **Focus on a single aspect or piece of information necessary for the final answer**
- If the second question depends on the answer of the first question, ise placeholders like {{{{answer1}}}}, {{{{answer2}}}}, etc., to reference answers from previous sub-questions.

**Examples**:

1. **Original Question:**
   - *What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?*

   **Explanation**:
   We must first find out who played Corliss Archer in the film Kiss and Tell. After knowing the answer to this sub-question, we can see what government position was held by this woman.
   **Sub-questions:**
   - Who portrayed the role of Corliss Archer in the film Kiss and Tell?
   - What government position did {{{{answer1}}}} hold?


2. **Original Question:**
   - *Which countries have signed the Paris Agreement and what are their respective emission targets?*

   **Explanation**:
   We must first find out which countries have signed the Paris Agreement. After knowing the answer to this sub-question, we can look for the emission targets of each identified countries.
   **Sub-questions:**
   - Which countries have signed the Paris Agreement?
   - What are the emission targets for {{{{answer1}}}} of these countries under the Paris Agreement?

Provide your response in JSON format as follows:
Explanation: <explanation of how these sub-questions help us answer the original question>
{{
    "sub_questions": [
        "Sub-question 1",
        "Sub-question 2",
    ]
}}

**Original Question:**
{question}

**Documents:**
{documents}
"""
        response = self.model.get_response(prompt)
        response_json = fix_json_string(response)
        retry = 0
        while not response_json and retry < 3:
            retry += 1
            print(
                f"QuestionDecompositionAgent Retry {retry}: Attempting to get valid JSON response."
            )
            response = self.model.get_response(prompt)
            response_json = fix_json_string(response)
        try:
            if not response_json:
                print(
                    "QuestionDecompositionAgent Error: Received invalid JSON response."
                )
                return [question]
            data = json.loads(response_json)
            sub_questions = data.get("sub_questions", [])
            print(f"sub_questions: {sub_questions}")
            return sub_questions
        except json.JSONDecodeError:
            print("QuestionDecompositionAgent Error: Failed to parse JSON response.")
            return [question]


class InformationExtractionAgent:
    def __init__(self, model_instance):
        self.model = model_instance

    def answer_subquestions(self, sub_questions, documents):
        subquestion_answers = []
        rephrased_subquestions = []
        top_documents_set = set()

        for idx, sub_q in enumerate(sub_questions):
            if idx != 0:
                sub_q_filled = self.replace_placeholders(
                    sub_questions, subquestion_answers, idx
                )
            else:
                sub_q_filled = sub_q

            rephrased_subquestions.append(sub_q_filled)
            sub_q_answer, citations = self.generate_subquestion_answer(
                sub_q_filled, documents
            )
            for cit in citations:
                top_documents_set.add(cit)
            subquestion_answers.append(sub_q_answer)

        return rephrased_subquestions, subquestion_answers, list(top_documents_set)

    def replace_placeholders(
        self, sub_questions, subquestion_answers, current_sub_question_index
    ):
        if subquestion_answers:
            context_text = "\n".join(
                [
                    f"Sub-question {i+1}: {sub_questions[i]}\nAnswer {i+1}: {subquestion_answers[i]}"
                    for i in range(current_sub_question_index)
                ]
            )
        else:
            context_text = "No prior sub-questions or answers available."

        # The current sub-question to be rephrased
        current_sub_question = sub_questions[current_sub_question_index]

        # Prompt for the LLM to rephrase the sub-question concisely, using previous sub-questions and answers as context
        prompt = f"""
You are provided with a list of sub-questions that collectively aim to answer a complex question. Your task is to rephrase the current sub-question by incorporating answers from previous sub-questions **only** if there is a placeholder in the form of {{answer1}}, {{answer2}}, etc.

**Guidelines:**
1. **Placeholder Substitution:** If the current sub-question contains a placeholder (e.g., {{answer1}}), replace it with the corresponding answer from the previous sub-questions.
2. **No Placeholder:** If there is no placeholder in the current sub-question, leave it unchanged.
3. **Explanation:** Provide a brief explanation of whether a substitution was made and why.

**Example 1:**
Inputs:
- Sub-questions:
  1. "Who played the role Harry Potter in the films?"
  2. "How old is the actor in {{answer1}}?"
- Answers:
  - "Who played the role Harry Potter in the films?": Daniel Radcliffe

Output:
{{
    "explanation": "Replaced {{answer1}} with 'Daniel Radcliffe' from the previous sub-question.",
    "sub_question": "How old is the actor Daniel Radcliffe?"
}}

**Example 2:**
Inputs:
- Sub-questions:
  1. "Who played the role Harry Potter in the films?"
  2. "Who played Ron in the Harry Potter films?"
- Answers:
  - "Who played the role Harry Potter in the films?": Daniel Radcliffe

Output:
{{
    "explanation": "No placeholder found. The sub-question remains unchanged.",
    "sub_question": "Who played Ron in the Harry Potter films?"
}}

**Example 3:**
Inputs:
- Sub-questions:
  1. "What is the capital of France?"
  2. "What is the population of {{answer1}}?"
- Answers:
  - "What is the capital of France?": Paris

Output:
{{
    "explanation": "Replaced {{answer1}} with 'Paris' from the previous sub-question.",
    "sub_question": "What is the population of Paris?"
}}

**Instructions:**
Here is the original list of sub-questions: {sub_questions}

Here are the original sub-questions and their answers:
{context_text}

Now, rephrase the following sub-question concisely by incorporating any relevant information from the previous questions **only** if there is a placeholder present:

Sub-question to rephrase:
{current_sub_question}

**Important:** 
- **Do not** rephrase the sub-question if there is no placeholder.
- **Only** substitute placeholders with corresponding answers.
- **Do not** add any additional information beyond the substitution.

**Return the output strictly in JSON format as shown in the examples:**

{{
    "explanation": "Explanation of how you rephrased the question",
    "sub_question": "Rephrased sub-question",
}}
"""
        response = self.model.get_response(prompt)
        response_json = fix_json_string(response)
        retry = 0
        while not response_json and retry < 3:
            retry += 1
            print(
                f"InformationExtractionAgent Retry {retry}: Attempting to get valid JSON response for rephrased sub-question."
            )
            response = self.model.get_response(prompt)
            response_json = fix_json_string(response)

        if not response_json:
            print(
                "InformationExtractionAgent Warning: Fallback to original sub-question due to invalid JSON response."
            )
            return current_sub_question  # Fallback to original sub-question
        try:
            data = json.loads(response_json)
            rephrased_sub_question = data.get("sub_question", current_sub_question)
            print(f"Original Sub-question: {current_sub_question}")
            print(f"Rephrased Sub-question: {rephrased_sub_question}")
            return rephrased_sub_question
        except json.JSONDecodeError:
            print(
                "InformationExtractionAgent Error: Failed to parse JSON response for rephrased sub-question."
            )
            return current_sub_question  # Fallback to original sub-question

    def generate_subquestion_answer(self, sub_question, documents):
        """
        Generates an answer for the sub-question using the extracted important sentences.
        Ensures the answer is specific and aligns with expected precision.

        Args:
            sub_question (str): The current sub-question.
            extracted_sentences (list): List of important sentences.

        Returns:
            tuple: The generated answer and list of citations.
        """
        flattened_documents = "\n\n".join(
            [
                f"Title: {doc[0]}\n"
                + "\n".join([f"({i}) {sentence}" for i, sentence in enumerate(doc[1])])
                for doc in documents
            ]
        )

        prompt = f"""
Provide a concise and accurate answer to the given question. You must cite exactly one document and multiple sentences from within the chosen document.

If a sentence contains pronouns (such as 'he', 'she', 'they', 'it', etc.), include the preceding sentence(s) that clarify who or what the pronoun refers to. Ensure that all necessary context is captured for the answer to be fully understood.

Read carefully. There will be answers that are almost correct -- DON'T CHOOSE THESE DISTRACTOR

**Example**:
Question: When was the United States founded.
{{
    "explanation": [
        "American History (0): The United States is located in North America. This sentence tells us that 'the country' referred to in the next sentence is the United States.",
        "American History (3): The country was founded in 1776. This sentence tells us the year the US was founded.",
    ],
    "answer": "1776",
    "citations": ["American History 0", "American History 3"],
}}

**Important Sentences:**
{flattened_documents}

**Question:**
{sub_question}
"""
        response = self.model.get_response(prompt)
        response = fix_json_string(response)
        if not response:
            print("parse_response Error: Invalid JSON format.")
            return "", []
        response = json.loads(response)

        print(f"response: {response}")
        answer, citations = response["answer"], response["citations"]
        print(f"subquestion answer: {answer}")
        print(f"citations: {citations}")
        return answer, citations


# ----------------- End of Multi-Agent Framework Implementation -----------------


class ReasoningAgent:
    """
    Performs reasoning and inference based on extracted information and subquestion answers.
    """

    def __init__(self, model_instance):
        self.model = model_instance

    def reason(
        self, question, sub_questions, subquestion_answers, top_documents, documents
    ):
        """
        top_documents: set ['title idx']
        Returns:
            dict: Reasoning results containing explanation, answer, and references.
        """
        documents_dict = {}
        for doc in documents:
            title = doc[0]
            for idx, sentence in enumerate(doc[1]):
                documents_dict[f"{title} {idx}"] = sentence

        combined_content = ""
        combined_dict = {}

        result = []
        print(f"top_documents: {top_documents}")
        for doc in top_documents:
            title, idx = doc.rsplit(maxsplit=1)
            text = documents_dict.get(f"{title} {idx}", "Missing")
            if title in combined_dict:
                combined_dict[title].append(f"({idx}) {text}")
            else:
                combined_dict[title] = [f"({idx}) {text}"]

        for title, value in combined_dict.items():
            sorted_contents = sorted(value, key=lambda x: x.lower())
            combined_content = "\n".join(sorted_contents)
            result.append(f"{title}\n{combined_content}\n")
        combined_content = "\n".join(result)

        # Aggregate all citations from subquestion_answers
        all_citations = []
        for ans in subquestion_answers:
            if isinstance(ans, list):
                all_citations.extend(ans)

        # Prepare the reasoning prompt
        prompt = f"""
Based on the following important sentences extracted from documents, perform the necessary reasoning to answer the main question.

If a sentence contains pronouns (such as 'he', 'she', 'they', 'it', etc.), include the preceding sentence(s) that clarify who or what the pronoun refers to. Ensure that all necessary context is captured for the answer to be fully understood.

**Use the provided sub-questions and their answers to guide your reasoning. However, you must cite ALL necessary sources from the extracted information. You must cite 2 titles.**

Here is an example reasoning and answer format you must follow:

{{
    "explanation": [
        "American History (0): The United States is located in North America. This sentence tells us that 'the country' referred to in the next sentence is the United States.",
        "American History (1): The country was founded in 1776. This sentence tells us the year the US was founded.",
        "George Washington (3): Washington was elected the first president the year the US was founded. Washington was elected the first president the year the US was founded" tells us that Washington was elected president the same year the US was founded. Therefore, the answer to the question 'What year was George Washington elected president?' based on these documents is 1776. There is only one answer based on the current documents, so the answer is 1776. Ther questions asks for the year so 1776 is the most concise answer.",
    ],
    "answer": "1776",
    "references": [[American History, 0], [American History, 1], [George Washington, 3]],
}}

Your reasoning should connect the extracted sentences logically to derive the final answer. Important: Select only the most relevant documents that contribute directly to answering the given question.

Main Question: {question}

Sub-questions and Answers:
{chr(10).join([f"- {sq} - Answer: {ans}" for sq, ans in zip(sub_questions, subquestion_answers)])}

Extracted Information:
{combined_content}

Provide your response in the same format. The answer at the end refers to an entity, it should use wording that is the similar as to where the answer is found in the context. 

**If the final answer refers to an entity, it must match the wording used in the context as closely as possible. Only the name of the entity should be included, without additional details or modifiers.**

Examples of good concise answers are 'an actor', 'Yes', 'October 1, 1776', 'Steve Jobs', 'a scholar'
Examples of bad answer are 'an actor based in the US', 'Yes they are the same', 'a scholar during the past'

**If the answer is a list of potential correct answers, output the answer that is *most directly supported* by the given documents.**

**You must cite 2 titles**

**Provide your response in JSON format as follows:**
{{
    "explanation": [
        "title1 (index): sentence": "Reasoning.",
        "title2 (index2): sentence": "Reasoning.",
    ],
    "answer": "Your final answer.",
    "references": [["Title1", "index1"], ["Title2, index2"]],
}}
"""
        print(f"\n\nReasoning prompt:\n{prompt}\n\n")

        response = self.model.get_response(prompt)
        response_json = fix_json_string(response)
        retry = 0
        while not response_json and retry < 3:
            retry += 1
            print(
                f"ReasoningAgent Retry {retry}: Attempting to get valid JSON response."
            )
            response = self.model.get_response(prompt)
            response_json = fix_json_string(response)

        if not response_json:
            return {"explanation": None, "answer": None, "references": None}

        response_json = json.loads(response_json)
        print(f"response_json: {response_json}")
        explanation, answer, references = (
            response_json["explanation"],
            response_json["answer"],
            response_json["references"],
        )
        references = [[item[0], int(item[1])] for item in references]

        return {"explanation": explanation, "answer": answer, "references": references}


class AnswerSynthesisAgent:
    """
    Synthesizes the final answer based on reasoning results.
    """

    def __init__(self, model_instance):
        self.model = model_instance

    def synthesize(self, reasoning_results):
        """
        Synthesizes the final answer from reasoning results.

        Args:
            reasoning_results (dict): Dictionary containing explanation, answer, and references.

        Returns:
            str: The final synthesized answer.
        """
        explanation = reasoning_results.get("explanation", "")
        answer = reasoning_results.get("answer", "")
        references = reasoning_results.get("references", [])

        return answer


# ----------------- End of Multi-Agent Framework Implementation -----------------


def run_distractor_baseline(
    dataset_name="distractor",
    model_name="gpt-4",
    output_path=None,
    num_training_samples=1000,
):
    dataset = load_hotpotqa(dataset_name)
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    if model_name.lower() in ["meta-llama-3-8b-instruct"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    coordinator = CoordinatorAgent(model_instance)

    i = 0
    offset = 1000
    for qa_pair in dataset[offset : num_training_samples + offset]:
        i += 1
        print(f"\n\nProcessing Question {i}")
        question = qa_pair["question"]
        context = qa_pair["context"]  # List of [title, sentences]
        context = clean_json_quotes(context)

        final_answer, references = coordinator.process_question(question, context)

        print(f"\n\nQuestion: {question}")
        print(f'Correct Answer: {qa_pair["answer"]}')
        print(f"Final Answer: {final_answer}")
        print(f"References: {references}")
        print(f'Correct References: {qa_pair["supporting_facts"]}')
        # Store the predictions
        predictions["answer"][qa_pair["_id"]] = final_answer
        predictions["sp"][qa_pair["_id"]] = references

    if output_path is None:
        output_path = f"MultiAgent2_{dataset_name}_{model_name}"

    predictions_path = f"{output_path}_predictions.json"

    try:
        with open(predictions_path, "w") as metrics_file:
            json.dump(predictions, metrics_file, indent=4)
        print(f"\nPredictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

    # Evaluate predictions
    if dataset_name == "distractor":
        eval_path = os.getenv("HOTPOTQA_DEV_DISTRACTOR")
    else:
        eval_path = os.getenv("HOTPOTQA_DEV_FULLWIKI")

    metrics = eval(predictions_path, eval_path)
    metrics_path = f"{output_path}_metrics.json"
    try:
        with open(metrics_path, "w") as metrics_file:
            json.dump(metrics, metrics_file, indent=4)
        print(f"\nMetrics saved to {metrics_path}")
    except Exception as e:
        print(f"Error saving metrics: {e}")
        return


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Multi-Agent Baseline on HotpotQA Dataset"
    )
    parser.add_argument(
        "--dataset",
        type=str,
        default="distractor",
        help="Name of the dataset to use (default: distractor)",
    )
    parser.add_argument(
        "--model",
        type=str,
        default="gpt-4",
        help="Model to use (e.g., gpt-4, llama3) (default: gpt-4)",
    )
    parser.add_argument(
        "--num_training_samples",
        type=int,
        default=10,
        help="Number of training samples to process (default: 10)",
    )

    args = parser.parse_args()

    run_distractor_baseline(
        dataset_name=args.dataset,
        model_name=args.model,
        num_training_samples=args.num_training_samples,
    )

    # Example usage:
    # python3 baseline_multi_agents2.py --dataset distractor --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 baseline_multi_agents2.py --dataset distractor --model gpt-4o-mini --num_training_samples 250
    # python3 baseline_multi_agents2.py --dataset distractor --model gpt-3.5-turbo-instruct --num_training_samples 250

    # python3 baseline_multi_agents2.py --dataset fullwiki --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 baseline_multi_agents2.py --dataset fullwiki --model gpt-4o-mini --num_training_samples 1000
