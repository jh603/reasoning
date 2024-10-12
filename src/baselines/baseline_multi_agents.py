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

    input_string = input_string.strip()

    try:
        json_obj = json.loads(input_string)
        return json.dumps(json_obj, indent=4)
    except json.JSONDecodeError as e:
        print(f"fix_json_string Error: Unable to parse JSON string. {str(e)}")
        print(f"Problematic Input: {input_string}")
        return None


def parse_response(response):
    """
    Parses the AI model's response to extract the answer and references.

    Expected response format:
        Explanation: <explanation>
        Answer: <answer>
        References: [Title1, index1], [Title2, index2], ...

    Args:
        response (str): The response string from the model.

    Returns:
        tuple: A tuple containing the answer and a list of references.
    """
    answer_pattern = r"Answer:\s*(.+)"
    references_pattern = r"References:\s*\[(.*)\]"

    answer_match = re.search(answer_pattern, response, re.IGNORECASE)
    answer = answer_match.group(1).strip() if answer_match else ""

    references_match = re.search(references_pattern, response, re.IGNORECASE)
    references = []

    if references_match:
        references_str = references_match.group(1)
        # Split by '], [' to separate different references
        references_list = re.split(r"\],\s*\[", references_str)

        for ref in references_list:
            ref = ref.replace("[", "").replace("]", "")  # Clean up brackets
            title_idx = ref.split(",")  # Split by comma to get title and index
            if len(title_idx) >= 2:
                title = ",".join(
                    title_idx[:-1]
                ).strip()  # In case title contains commas
                try:
                    idx = int(title_idx[-1].strip())  # Convert last part to int
                    references.append([title, idx])
                except ValueError:
                    print(f"parse_response Error: Invalid reference index in: {ref}")

    return answer, references


def parse_important_sentences(response):
    """
    Parses the AI model's JSON response to extract important sentences.

    Expected response format:
    Response: {
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
        # Step 1: Set the number of hops to 2
        num_hops = 2

        # Step 2: Decompose the question into sub-questions with placeholders
        sub_questions = self.question_decomposition_agent.decompose(
            question, documents, num_hops
        )

        # Step 3: Extract information for each sub-question and get their answers
        extracted_info, rephrased_subquestions, subquestion_answers = (
            self.info_extraction_agent.extract(sub_questions, documents)
        )

        # Step 4: Perform reasoning and inference
        reasoning_results = self.reasoning_agent.reason(
            question,
            extracted_info,
            rephrased_subquestions,
            subquestion_answers,
            documents,
        )
        print(f"reasoning_results: {reasoning_results}")
        reasoning_references = reasoning_results.get("references", [])

        combined_references = self.combine_and_validate_references(
            reasoning_references, documents
        )

        # Step 5: Synthesize the final answer
        final_answer = self.answer_synthesis_agent.synthesize(reasoning_results)

        return final_answer, combined_references

    def combine_and_validate_references(self, reasoning_refs, documents):
        """
        Combines reasoning references, ensuring uniqueness and validity.

        Args:
            reasoning_refs (list): List of references from reasoning phase.
            documents (list): List of documents for validation.

        Returns:
            list: Combined and validated references.
        """
        # Use a set of tuples to avoid duplicates
        reference_set = set()

        # Create a mapping of document titles to their sentence counts for validation
        # Using lowercase for case-insensitive matching
        doc_title_to_sentence_count = {
            title.lower(): len(sentences) for title, sentences in documents
        }

        # Function to validate a single reference
        def is_valid_reference(ref):
            title, idx = ref
            if title.lower() not in doc_title_to_sentence_count:
                print(
                    f"Validation Error: Document title '{title}' not found in the provided documents."
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
        """
        Decomposes the main question into sub-questions.

        Args:
            question (str): The main question.
            documents (list): List of documents.
            num_hops (int): Number of sub-questions to generate.

        Returns:
            list: List of sub-questions.
        """
        # Remove duplicate documents based on title within this question pair
        unique_documents = {}
        for doc in documents:
            title = doc[0]
            if title not in unique_documents:
                unique_documents[title] = doc[1]
        documents = [
            [title, sentences] for title, sentences in unique_documents.items()
        ]

        # Introduce placeholders like {{answer1}}, {{answer2}}, etc.
        # Updated to use double curly braces by escaping them in the f-string
        prompt = f"""
Decompose the following multi-hop question into 2 clear and specific sub-questions that can be either independently or sequentially using the documents below to help arrive at the final answer.

Each sub-question should:
- Be directly relevant to answering the original question.
- **Focus on a single aspect or piece of information necessary for the final answer**
- Use placeholders like {{{{answer1}}}}, {{{{answer2}}}}, etc., to reference answers from previous sub-questions where necessary.

**Examples**:

1. **Original Question:**
   - *What government position was held by the woman who portrayed Corliss Archer in the film Kiss and Tell?*

   **Explanation**:
   We must first find out who played Corliss Archer in the film Kiss and Tell. After knowing the answer to this sub-question, we can see what government position was held by this woman.
   **Sub-questions:**
   - Who portrayed the role of Corliss Archer in the film Kiss and Tell?*
   - What government position did {{{{answer1}}}} hold?*


2. **Original Question:**
   - *Which countries have signed the Paris Agreement and what are their respective emission targets?*

   **Explanation**:
   We must first find out which countries have signed the Paris Agreement. After knowing the answer to this sub-question, we can look for the emission targets of each identified countries.
   **Sub-questions:**
   - Which countries have signed the Paris Agreement?*
   - What are the emission targets for {{{{answer1}}}} of these countries under the Paris Agreement?*

Provide your response in JSON format as follows:
{{
    "explanation": "explanation of how these sub-questions help us answer the original question",
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
        while not response and retry < 3:
            retry += 1
            response = self.model.get_response(prompt)
            response_json = fix_json_string(response)
        try:
            if not response_json:
                print(
                    "QuestionDecompositionAgent Error: Received invalid JSON response."
                )
                return [question]  # Fallback to original question
            data = json.loads(response_json)
            sub_questions = data.get("sub_questions", [])
            print(f"sub_questions: {sub_questions}")
            return sub_questions
        except json.JSONDecodeError:
            print("QuestionDecompositionAgent Error: Failed to parse JSON response.")
            return [question]  # Fallback to original question


class InformationExtractionAgent:
    """
    Extracts relevant information from the provided documents based on sub-questions.
    Processes sub-questions sequentially by replacing placeholders with prior answers.
    Generates answers for sub-questions using the extracted important sentences.
    """

    def __init__(self, model_instance):
        self.model = model_instance

    def extract(self, sub_questions, documents):
        """
        Extracts information and answers for each sub-question.

        Args:
            sub_questions (list): List of sub-questions.
            documents (list): List of documents.

        Returns:
            tuple: Extracted information and sub-question answers.
        """
        extracted_info = {}
        subquestion_answers = []  # To store answers to prior sub-questions
        rephrased_subquestions = []

        for idx, sub_q in enumerate(sub_questions):
            if idx != 0:
                sub_q_filled = self.replace_placeholders(
                    sub_questions, subquestion_answers, idx
                )
                print(f"Filled Sub-question {idx+1}: {sub_q_filled}")
            else:
                sub_q_filled = sub_q

            rephrased_subquestions.append(sub_q_filled)

            # Extract important sentences for the current sub-question
            extracted_sentences = self.extract_for_subquestion(sub_q_filled, documents)

            # Generate an answer for the current sub-question using the extracted sentences
            sub_q_answer = self.generate_subquestion_answer(
                sub_q_filled, extracted_sentences
            )

            # Store the extracted sentences and the subquestion answer
            extracted_info[f"sub_q_{idx+1}"] = extracted_sentences
            subquestion_answers.append(sub_q_answer)

        return extracted_info, rephrased_subquestions, subquestion_answers

    def replace_placeholders(
        self, sub_questions, subquestion_answers, current_sub_question_index
    ):
        """
        Uses an LLM to intelligently rephrase the current sub-question based on the original list of sub-questions
        and the answers to previous sub-questions.

        Args:
            sub_questions (list): The list of all sub-questions.
            subquestion_answers (list): The list of answers to previous sub-questions.
            current_sub_question_index (int): The index of the current sub-question to be rephrased.

        Returns:
            str: The rephrased sub-question with proper context from previous answers.
        """
        # Gather the previous sub-questions and their answers for context
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
You are given a list of sub-questions that may build towards answering a complex question. Your goal is to rephrase the current sub-question bsaed on the previous sub-questions and their answers.

**If the previous sub-questions' answers have no relation to the sub-question to rephrase, return the same sub-question**

*Example:* 
Given sub-questions, 'Who played the role Harry Potter in the films', 'How old is the actor in Answer1'
'Who played the role Harry Potter in the films': Daniel Radcliffe
The rephrased second question should be 'How old is Daniel Radcliffe'

Here is the original list of sub-questions: {sub_questions}

Here are the original sub-questions and their answers:
{context_text}

Now, rephrase the following sub-question concisely incorporating any relevant information from the previous question:

Sub-question to rephrase:
{current_sub_question}

Provide your response in JSON format as follows:
{{
    "explanation": "explanation of how you rephrased the question",
    "sub_question": "rephrased sub-question"
}}
"""

        response = self.model.get_response(prompt)
        response_json = fix_json_string(response)
        retry = 0
        while not response_json and retry < 3:
            retry += 1
            response = self.model.get_response(prompt)
            response_json = fix_json_string(response)

        if not response_json:
            return None, []
        response_json = json.loads(response_json)
        rephrased_sub_question = response_json["sub_question"]
        print(f"Original Sub-question: {current_sub_question}")

        return rephrased_sub_question

    def extract_for_subquestion(self, sub_question, documents):
        """
        Extracts important sentences for a given sub-question from the documents.

        Args:
            sub_question (str): The current sub-question.
            documents (list): List of documents.

        Returns:
            list: List of important sentences as [title, index, sentence].
        """
        important_sentences = []

        for entry in documents:
            title = entry[0]
            sentences = entry[1]

            # Flatten the content with sentence indices
            flattened_content = "\n".join(
                [f"({i}) {sentence}" for i, sentence in enumerate(sentences)]
            )
            document = f"Title: {title}\nContent:\n{flattened_content}\n\n"

            # Prompt to identify important sentences in JSON format
            prompt = f"""
Identify any important sentences in the following document that help to directly answer this question. If a sentence contains pronouns (such as 'he', 'she', 'they', etc.), include the preceding sentence(s) that clarify who or what the pronoun refers to. Ensure that all necessary context is captured for the answer to be fully understood.

Maintain a high standard and only include sentences that provide a high degree of information towrads answering the given question. Do not include sentences that provide background information that is not directly relevant to the given question.

**Output Format:**
Explanation: <Explanation>
json: {{
    "title": "Document Title",
    "important_sentences": [
        {{"index": <sentence_index>, "sentence": "<important_sentence>"}},
        ...
    ]
}}

**Full example 1:**
Input:
Question: When was George Washington born?
Document:
    Title: George Washington
    Content: (0) George Washington was the first president of the United States. (1) He led the revolutionary army.
    
Response:
Explanation: Sentences (0) tells use the George Washington was the first president of the United States. Sentence (1) and (2) tell us George Washington led the revolutionary army as (1) clarifies 'he' refers to George Washington. No sentence provides information relevant to the question.
json: {{
    "title": "George Washington",
    "important_sentences": []
}}

**Full example 2:**
User Input:

Question: Where did Michael Jackson perform?
Document:
    Title: Michael Jackson
    Content: (0) Michael Jackson was an American singer. (1) As an adult, he performed in the United States.
    
Response:
Explanation: Sentence (1) tells us Michael Jackson performed in the United States which answers the question directly, Sentence (1) contains pronouns ('he') that refer to Michael Jackson, which is clarified in Sentence 0. Both sentences are needed to fully understand who 'he' is.
json: {{
    "title": "Michael Jackson",
    "important_sentences": [
        {{"index": 0, "sentence": "Michael Jackson was an American singer."}},
        {{"index": 1, "sentence": "As an adult, he performed in the United States."}}
    ]
}}

**Inputs:**

Question: {sub_question}

Document:
{document}
"""

            response = self.model.get_response(prompt)
            parsed_title, extracted_sentences = parse_important_sentences(response)

            retry = 0
            while not parsed_title and retry < 5:
                print(f"Retrying extract_for_subquestion...")
                response = self.model.get_response(prompt)
                parsed_title, extracted_sentences = parse_important_sentences(response)

            if (
                parsed_title
                and html.unescape(title).strip().lower() == parsed_title.strip().lower()
            ):
                for idx, sentence in extracted_sentences:
                    important_sentences.append([title, idx, sentence])
            else:
                # Handle mismatches or parsing failures
                print(
                    f"extract_for_subquestion Warning: Title mismatch or parsing failed for document titled '{title}'."
                )
                continue

        return important_sentences

    def generate_subquestion_answer(self, sub_question, extracted_sentences):
        """
        Generates an answer for the sub-question using the extracted important sentences.
        Ensures the answer is specific and aligns with expected precision.

        Args:
            sub_question (str): The current sub-question.
            extracted_sentences (list): List of important sentences.

        Returns:
            str: The generated answer.
        """
        if not extracted_sentences:
            return "No relevant information found."

        # Aggregate the extracted sentences
        aggregated_sentences = "\n".join(
            [f"({idx}) {sentence}" for title, idx, sentence in extracted_sentences]
        )

        # Prepare the prompt for answer generation with specificity instructions
        prompt = f"""
Based on the following important sentences, provide a concise and accurate answer to the question. Only use the provided documents to answer the question.

**Important Sentences:**
{aggregated_sentences}

**Question:**
{sub_question}

**Answer:**
"""
        response = self.model.get_response(prompt)
        answer = response.strip()
        print(f"subquestion answer: {answer}")
        return answer


class ReasoningAgent:
    """
    Performs reasoning and inference based on extracted information and subquestion answers.
    """

    def __init__(self, model_instance):
        self.model = model_instance

    def reason(
        self, question, extracted_info, sub_questions, subquestion_answers, documents
    ):
        """
        Performs reasoning to derive the final answer based on extracted information and sub-question answers.

        Args:
            question (str): The main question.
            extracted_info (dict): Extracted important sentences for each sub-question.
            sub_questions (list): List of sub-questions.
            subquestion_answers (list): List of answers to sub-questions.

        Returns:
            dict: Reasoning results containing explanation, answer, and references.
        """
        print(f"extracted_info: {extracted_info}")

        documents_dict = {}
        for doc in documents:
            title = doc[0]
            sentences = doc[1]
            for idx, sentence in enumerate(sentences):
                documents_dict[f"{title}_{idx}"] = sentence

        combined_content = ""
        combined_docs = {}

        for _, items in extracted_info.items():
            for title, idx, content in items:
                if title not in combined_docs:
                    combined_docs[title] = set()
                temp = documents_dict.get(f"{title}_{idx}", "Missing")
                combined_docs[title].add(f"({idx}) {temp}")

        result = []
        for title, contents in combined_docs.items():
            sorted_contents = sorted(contents, key=lambda x: x.lower())
            combined_content = "\n".join(sorted_contents)
            result.append(f"Title: {title}\n{combined_content}\n")
        combined_content = "\n".join(result)

        prompt = f"""
Based on the following important sentences extracted from documents, perform the necessary reasoning to answer the main question.

**Use the provided sub-questions and their answers guide your reasoning. However, you must cite ALL necessary sources from the extracted information**

Here is an example reasoning and answer format you must follow:

Explanation: 
    (0) "American History": "The United States is located in North America." This sentence tell us the 'the country' referred to in the next sentence refers to the 'United States'
    (1) "American History": "The country was founded in 1776" tells us the year the US was founded.
    (3) "George Washington": "Washington was elected the first president the year the US was founded" tells us that Washington was elected president the same year the US was founded.
    Therefore, the answer to the question 'What year was George Washington elected president?' based on these documents is 1776. There is only answer based on the current documents so the answer is 1776.
Answer: 1776
References: [American History, 0], [American History, 1], [George Washington, 3]

Your reasoning should connect the extracted sentences logically to derive the final answer. Important: Select only the most relevant documents that contribute directly to answering the given question. 

Main Question: {question}

Sub-questions and Answers:
{chr(10).join([f"-{sq} - Answer: {ans}" for sq, ans in zip(sub_questions, subquestion_answers)])}

Extracted Information:
{combined_content}

Provide your response in the same format. The Answer at the end should be very concise. 

**If the answer is a list of potential correct answers, output the answer that is *most directly supported* by the given documents.**
"""
        print(f"\n\nReasoning prompt:\n{prompt}\n\n")

        response = self.model.get_response(prompt)
        explanation, answer, references = self.parse_reasoning_response(response)
        retry = 0
        while retry < 3 and answer == "" or len(references) == 0:
            print(f"Answer regenerated {retry}")
            retry += 1
            response = self.model.get_response(prompt)
            explanation, answer, references = self.parse_reasoning_response(response)

        return {"explanation": explanation, "answer": answer, "references": references}

    def parse_reasoning_response(self, response):
        """
        Parses the reasoning agent's response to extract explanation, answer, and references.

        Args:
            response (str): The response string from the model.

        Returns:
            tuple: Explanation, answer, and list of references.
        """
        explanation_pattern = r"Explanation:\s*([\s\S]+?)\nAnswer:"
        answer_pattern = r"Answer:\s*(.+)"
        references_pattern = r"References:\s*\[(.*)\]"

        explanation_match = re.search(explanation_pattern, response, re.IGNORECASE)
        explanation = explanation_match.group(1).strip() if explanation_match else ""

        answer_match = re.search(answer_pattern, response, re.IGNORECASE)
        answer = answer_match.group(1).strip() if answer_match else ""

        references_match = re.search(references_pattern, response, re.IGNORECASE)
        references = []

        if references_match:
            references_str = references_match.group(1)
            # Split by '], [' to separate different references
            references_list = re.split(r"\],\s*\[", references_str)

            for ref in references_list:
                ref = ref.replace("[", "").replace("]", "")  # Clean up brackets
                title_idx = ref.split(",")  # Split by comma to get title and index
                if len(title_idx) >= 2:
                    title = ",".join(
                        title_idx[:-1]
                    ).strip()  # In case title contains commas
                    try:
                        idx = int(title_idx[-1].strip())  # Convert last part to int
                        references.append([title, idx])
                    except ValueError:
                        print(
                            f"ReasoningAgent Error: Invalid reference index in: {ref}"
                        )

        return explanation, answer, references


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
    """
    Runs the multi-agent baseline on the specified HotpotQA dataset.

    Args:
        dataset_name (str): Name of the dataset to use (default: 'distractor').
        model_name (str): Model to use (e.g., 'gpt-4', 'llama3') (default: 'gpt-4').
        output_path (str, optional): Path to save the output files (default: None).
        num_training_samples (int): Number of training samples to process (default: 1000).
    """
    dataset = load_hotpotqa(dataset_name)
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    if model_name.lower() in ["meta-llama-3-8b-instruct"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    # Initialize Coordinator Agent
    coordinator = CoordinatorAgent(model_instance)

    i = 0
    offset = 3
    for qa_pair in dataset[offset : num_training_samples + offset]:
        i += 1
        print(f"\n\n{i} questions processed")
        question = qa_pair["question"]
        context = qa_pair["context"]  # List of [title, sentences]
        context = clean_json_quotes(context)

        final_answer, references = coordinator.process_question(question, context)

        print(f"\n\nQuestion: {question}")
        print(f'correct_answer: {qa_pair["answer"]}')
        print(f"final_answer: {final_answer}")
        print(f"references: {references}")
        print(f'correct_references: {qa_pair["supporting_facts"]}')
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
    # python3 baseline_multi_agents.py --dataset distractor --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 baseline_multi_agents.py --dataset distractor --model gpt-4o-mini --num_training_samples 250
    # python3 baseline_multi_agents.py --dataset distractor --model gpt-3.5-turbo-instruct --num_training_samples 250

    # python3 baseline_multi_agents.py --dataset fullwiki --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 baseline_multi_agents.py --dataset fullwiki --model gpt-4o-mini --num_training_samples 1000
