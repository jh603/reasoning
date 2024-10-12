import argparse
import json
import os
import re
from collections import defaultdict

from dotenv import load_dotenv

from src.utils.datasets import load_hotpotqa
from src.utils.hotpotqa_eval import eval
from src.utils.model_factory import Model

load_dotenv()


def parse_response(response, subquestions=None):
    answer = ""
    references = []

    # Extract FINAL_ANSWER using regex
    answer_match = re.search(r"FINAL_ANSWER:\s*(.+)", response, re.IGNORECASE)
    if answer_match:
        answer = answer_match.group(1).strip()

    # Extract FINAL_REFERENCES using regex
    references_match = re.search(
        r"FINAL_REFERENCES:\s*(\[.+\])", response, re.IGNORECASE | re.DOTALL
    )
    if references_match:
        ref_string = references_match.group(1)
        # Use regex to find all [Title, Number] patterns
        ref_list = re.findall(r"\[([^,\]]+),\s*(\d+)\]", ref_string)
        for ref in ref_list:
            title, index = ref
            references.append([title.strip(), int(index)])

    if not subquestions:
        subquestions = []
        section_match = re.search(
            r"Sub-Questions:(.*?)Sub-Question Answers:", response, re.DOTALL
        )
        if section_match:
            section_text = section_match.group(1).strip()

            # Split the text into lines
            lines = section_text.split("\n")

            # Filter out empty lines and lines containing "Sub-Questions:" or "Sub-Question Answers:"
            filtered_lines = [
                line.strip()
                for line in lines
                if line.strip()
                and "Sub-Questions:" not in line
                and "Sub-Question Answers:" not in line
            ]

            # Extract individual subquestions
            for line in filtered_lines:
                # Remove the number and leading space if present
                subq = re.sub(r"^\d+\.\s*", "", line)
                subquestions.append(subq)

        subquestions = subquestions[1:-1]

    print(f"subquestions: {subquestions}")
    return answer, references, subquestions


def validate_response(
    question, documents_dict, answer, references, subquestions, model_instance
):

    title_sentences = defaultdict(list)
    for title, idx in sorted(references, key=lambda x: f"{x[0]} {x[1]}"):
        key = f"{title} {idx}"
        if key in documents_dict:
            title_sentences[title].append(documents_dict[key])

    extracted_references_response = "\n\n".join(
        f"Reference: {' '.join(sentences)}" for _, sentences in title_sentences.items()
    )

    print(f"extracted_references_response: {extracted_references_response}")
    validator_prompt = f"""
    You are a Validator Agent tasked with reviewing responses from the Primary LLM Agent. Your role is to ensure the correctness, completeness, and precision of the responses based solely on the conversation history.
    
    **Instructions:**
    ***If there is a failure at an early step simply skip to Step 8 and provide feedback***
    
    0: **Check the completeness of each reference:**
        - Check for any ambiguous pronouns or references due to missing antecedents within each reference independently**. If a pronoun (e.g., "he", "she", "it") or reference is unclear and lacks an antecedent in the same document, it is a FAILURE skip to Step 8
        - **DO NOT** use any reasoning to infer how the pronouns refer to unless is it stated explicitly in a reference.
        - Sentences from separate references must be evaluated independently.
        - e.g. For the refererence: "He had over 100 medals." This is missing a reference to who "he" refers to. This is a FAILURE.

    1.1. **Answer Sub-Questions**:
        - Provide answers to each sub-question.
        - Choose the sentences in the documents that most closely align in wording to the sub-question
        Sub-questions: {subquestions}

    1.2. **Reconstruct the Answer**:
        - Using the original question and the list of references, attempt to answer the original **Question** without considering the provided answer. Ensure that your answer directly answers the given question.
        - If there are multiple constraints in th main question, ensure that each one is satsified by a reference. If any of them are not addressed, this is a FAILURE. Suggest a single constraint that is not met in your feedback.
        - Ensure that the reasoning is clear and explicit. No assumptions should be made beyond what is explicitly stated in the references. **Highlight any unresolved ambiguities.**
        - If any infomation is missing or is unclear this is a FAILURE.
    
    2. **Compare Answers**:
        - If the **FINAL_ANSWER** is not specified or missing is it a FAILURE. Ask the LLM to continue looking for the correct answer.
        - Compare your reconstructed answer with the **FINAL_ANSWER** provided by the Primary LLM Agent.
        - Check for completeness and precision. The **FINAL_ANSWER** should be the most precise and use the same language as the references.
    
    3. **Evaluate References**:
        - Assess whether the **FINAL_REFERENCES** sufficiently support the answer.
        - Ensure that all relevant references are included and that no critical references are missing.
    
    4. **Assess Entity Names**:
        - Verify that the entity names in the **FINAL_ANSWER** closely match those in the original documents.
        - Ensure that the terminology used is consistent with the source material.
        - The final answer should always be the name of an entity or a short answer like yes or no. It should never be a full sentence or longer.

    5. **Provide Feedback**:
        - If the reconstructed answer matches the **FINAL_ANSWER** and the references are sufficient, output "VALIDATION_RESULT: PASS".
        - **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
        - **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.
        - If there are discrepancies, incomplete answers, or insufficient references, output "VALIDATION_RESULT: FAIL: [Reason for failure]".
    
    6. **Provide a final ruling in the following form. Present an approach to fix this issue. Always identiy the sub-question that is not fully answered if there is a reasoning issue**: (This list is not exhaustive)
       - **VALIDATION_RESULT: PASS**
        - **VALIDATION_RESULT: FAIL: Missing reference [Document Title, Sentence Number]. Maybe look for a sentence that clarifies the sentence "<sentence>"**
        - **VALIDATION_RESULT: FAIL: The FINAL_ANSWER uses a different entity name than in the original documents. Consider the answer "<answer>"**
        - **VALIDATION_RESULT: FAIL: The subquestion question <sub-question> is answered incorrectly because <reason>**
    
    Show your thinking step by step
    
    **Conversation History:**
    
    **Question:** {question}
    
    **Primary LLM Agent's Response:**
    
    **Answer:** {answer}
    
    **References:** 
    {extracted_references_response}
    
    **Please evaluate the response and provide feedback using the specified delimiter.**
    """
    validator_response = model_instance.get_response(validator_prompt)

    print(f"validator_response.strip(): {validator_response.strip()}")
    return validator_response.strip()


def run_baseline(
    dataset_name="distractor",
    model_name="gpt-4o-mini",
    output_path=None,
    num_training_samples=1000,
    max_iterations=2,  # Maximum number of validation attempts
):
    # Load the dataset
    dataset = load_hotpotqa(dataset_name)
    initial_predictions = {
        "answer": {},  # _id: answer
        "sp": {},  # _id: [[title, idx],...]
    }
    predictions = {"answer": {}, "sp": {}}  # _id: answer  # _id: [[title, idx],...]

    if model_name.lower() in ["meta-llama-3-8b-instruct", "gpt-4o-mini"]:
        model_path = os.getenv(model_name.lower())
        model_instance = Model(model_name=model_name, model_path=model_path)
    else:
        model_instance = Model(model_name=model_name)

    i = 0
    offset = 68
    for qa_pair in dataset[offset : num_training_samples + offset]:
        print(f"{i} Questions processed")
        i += 1
        final_output = ""
        documents_dict = {}  # title idx: sentence
        for entry in qa_pair["context"]:
            title = entry[0]
            sentences = entry[1]
            flattened_content = "\n".join(
                [f"({i}) {sentence}" for i, sentence in enumerate(sentences)]
            )
            final_output += f"Title: {title}\nContent: {flattened_content}\n\n"

            for idx, sentence in enumerate(sentences):
                documents_dict[f"{title} {idx}"] = sentence

        print(documents_dict)
        print(f'Correct answer: {qa_pair["answer"]}')
        print(f'Correct references: {qa_pair["supporting_facts"]}')
        iteration = 0
        success = False

        conversation_history = []
        subquestions = None
        while iteration < max_iterations and not success:
            if iteration == 0:
                prompt_gen_subquestions = f"""
**Generate Sub-Questions**:
   **Assume you are a fifth grader trying to figure out how to answer a complex question**
   - Identify 2 facts you need to answer the main question. 
   - Use similar language to the original question.
   - When generating sub-questions, consider the context of key terms such as whether they refer to people, episodes, shows, films, or other entities. Avoid misinterpreting important context or generalizing key terms.
   - Create a sequence of **2 to 3 sub-questions** that can help in answering the main multi-hop question. The 3rd question should directly answer the main question.
   - Ensure that the last question can be **directly** used to answer the main question.
   - Use placeholders {{answer1}} where necessary.
   
**Question:** {qa_pair["question"]}

Output Format:
Reasoning: I need to first identify X. Then, given the answer to X, we must figure out who meets the constraints of Y, etc.

Questions:
Question 1.
Question 2.
Question 3.
"""
                subquestions = model_instance.get_response(prompt_gen_subquestions)
                print(f"subquestions: {subquestions}")

                start_index = subquestions.find("Questions:")

                if start_index == -1:
                    return "No 'Questions:' found in the text."

                # Extract the substring starting from "Questions:"
                questions_text = subquestions[start_index + len("Questions:") :]

                # Remove leading whitespace, including newlines
                questions_text = questions_text.lstrip()

                prompt = f"""
Please answer the following question using only the information provided in the documents below.

**Instructions:**
1. **Restate the Sub-Questions**:
    - Include the specific keywords the sub-questions are looking for
    
    e.g. Question 1. 
         Question 2. 
         Question 3. 

2. **Answer Sub-Questions**:
    - Provide answers to each sub-question. As you answer each sub-question. Redraft it based on the answer to previous sub-questions if necessary.
    - For each sub-question, provide a list of **ALL** documents and explain why each document is relevant or not
    - For each answer, include **all relevant citations** to the specific parts of the documents that support your response.
    - **List every sentence number used** to answer each sub-question.
    - You can only use two documents in total for all sub-questions, so choose wisely.

3. **Explain Reasoning**:
    - Assume you have the reasoning capabilities of a fifth grader. Only base your answers off what is directly stated in the references. Your answer should be directly a subset of the text referenced
    - Write a paragraph explaining your overall reasoning for answering the main question.
    - Incorporate the answers to the sub-questions with appropriate citations.
    - **Use the exact entity names** as presented in the original documents when referring to them.
    - **List every sentence number used** in this explanation.
    - The answer is never unspecified. If you cannot determine the answer, go back to Step 2 and construct new sub-questions

4. **Provide Final Answer**:
    - The answer should be **very concise**, such as a year, a simple yes/no, or a person's name.
    - Use wording **exactly as it appears** in the original documents.
    - **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
    - **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.

5. **List References**:
    - **Aggregate** all references cited in both the sub-question answers and the explanation. ONLY KEEP CITATIONS THAT DIRECTLY SUPPORT THE MAIN QUESTION.
    - **You must cite at exactly 2 different document titles.**
    - Ensure there are no duplicates in the list of references
    - **Ensure that every referenced sentence is included** in this list.
    - Format each reference as `[Document Title, Sentence Number]`.
    

6. **Handle Pronouns**:
    - If a sentence contains pronouns (e.g., 'he', 'she', 'they', 'it'), include the preceding sentence(s) that clarify who or what the pronoun refers to.
    - Ensure that all necessary context is captured for the answer to be fully understood.

7. **Use Unique Delimiters**:
    - Put "FINAL_ANSWER:" before your final answer.
    - Put "FINAL_REFERENCES:" before your final list of references.

**Examples:**

- **Good Concise Answers**:
    - 'an actor'
    - 'Yes'
    - 'October 1, 1776'
    - 'Steve Jobs'
    - 'a scholar'

- **Bad Concise Answers**:
    - 'an actor based in the US'
    - 'Yes they are the same'
    - 'a scholar during the past'

**Example Response:**

*Explanation:*  
The second sentence in the document titled "American History" states, "The United States was founded in 1776," indicating the founding year. The fourth sentence in the "George Washington" document mentions, "Washington was elected the first president the year the US was founded," linking Washington's election to the founding year. Therefore, based on these documents, the answer to the question "What year was George Washington elected president?" is 1776.

*Answer:*  
1776

*References:*  
[American History, 2], [George Washington, 4]

### START ANSWER

1. **Sub-Questions:**  
   1. *(First sub-question)*  
   2. *(Second sub-question)*  
   *(Add more sub-questions if necessary)*

2. **Sub-Question Answers:**
   **If there are multiple answers to a sub-question, track a list of all possible answers, and narrow down the list as you move from sub-question to sub-question**
   If there is not answer to a sub-question do not provide any references.
   1. ***[First Sub-Question]***  
      *Reasoning* 
        - Document 1 has the keyword "keyword" but it is not correct because... 
      *Answer:* *(Answer to the first sub-question)*  
      *References:* [Document Title, Sentence Number], ...  

   2. ***[Second Sub-Question]***  
      *Redrafted sub-question* *(sub-question 2 incorporating sub-question 1's answer if relevant)
      *Reasoning*
        - Document 1 has the keyword "keyword" but it is not correct because... 
      *Answer:* *(Answer to the second sub-question)*  
      *References:* [Document Title, Sentence Number], ...  

   *(Add more sub-question answers if necessary)*
   
   Example:
   1. ***Who played in the nba***  
      *Reasoning* 
        - Document 1 has the keyword "keyword" but it is not correct because... 
      *Answer:* According to the documents, Michael Jordan played in the NBA, Lebron James played in the NBA, and etc.
      *References:* [NBA, 0], [Lebron, 1], [Jordan, 0]
      
   2. ***Which NBA player scored the most baskets***  
      *Reasoning* 
        - Document 1:...
      *Answer:* Based on the answer of sub-question 1, Michael Jordan scored X points and Lebron James score Y points. Therefore, since X>Y we can eliminate Lebron James from consideration for the final answer.
      *References:* [NBA, 0], [Lebron, 1], [Jordan, 0]

3. **Explanation:**  
   *(A paragraph explaining your reasoning for the main question, incorporating the sub-question answers and citations)*  

4. **Answer:**  
   FINAL_ANSWER: *(Final concise answer to the main question, following the guidelines for wording and support)*

5. **References:**  
   FINAL_REFERENCES: [Document Title, Sentence Number], [Document Title, Sentence Number], ...

**Question:** {qa_pair["question"]}

**Subquestions** {subquestions}

**Documents:**  
{final_output}

### END ANSWER
"""
            else:
                prompt = f"""
Please address the feedback and use this output format:
1. **Explanation:**
    - First restate the document you wish to use by quoting it verbatim including the title and indices.
    - You must quote the new statements you wish to cite and then include the title and sentence index as well.
    - To clarify antecedents, you must include a sentence *prior* to the one that is unclear.
    
    - e.g. Michael Jordan: (0) Michael Jordan played for the Chicago, Bulls. (0) He had 6 championships.
    To support the statement that Michael Jordan had 6 championships, I will also cite the sentence (1) from the article "Michael Jordan," "He had 6 championships" ["Michael Jordan", 1]
    - e.g. Michael Jordan: (0) Michael Jordan played for the Chicago, Bulls. (0) He had 6 championships.
    To clarify who "he" refers to in the statement "he had 6 championships," I will also cite the sentence (0) from the article "Michael Jordan," "Michael Jordan played for the Chicago, Bulls" ["Michael Jordan", 0] as this sentence clarifies the pronoun in the subsequent sentence.

1.1 **Actions:**
    - State your intended action to fix the problem.
    - e.g. I will add the reference ["Michael Jordan", 1]
    - e.g. I will add the reference ["Michael Jordan", 0]

2. **Implement Improvements:**
    - Ensure there are no duplicates in the list of references
    - e.g. The previous references were ["Michael Jordan", 0], ["NBA", 0]. The proposed change in 1.1 is to add the reference ["Michael Jordan", 1]. Therefore, the improved list of references is ["Michael Jordan", 0], ["Michael Jordan", 1], ["NBA", 0]

3. **Answer:**  
   FINAL_ANSWER: *(An updated concise answer to the main question)*

4. **Updated list of References:**  
   FINAL_REFERENCES: [Document Title, Sentence Number], [Document Title, Sentence Number], ...

Keep in mind:
- The answer should be **very concise**, such as a year, a simple yes/no, or a person's name.
- Use wording **exactly as it appears** in the original documents.
- **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
- **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.

For reference errors:

**Feedback:**
{validation_feedback}
"""

            # print(f'conversation_history: {conversation_history}')
            print(f"prompt: {prompt}")
            print(f'question: {qa_pair["question"]}')
            conversation_history.append({"role": "user", "content": prompt})

            response = model_instance.get_response_with_history(conversation_history)
            print(f"response: {response}")
            answer, references, subquestions = parse_response(response, subquestions)

            # Append the primary response to the conversation history
            conversation_history.append({"role": "assistant", "content": response})

            # Print the current iteration's details
            print(f'Question ID: {qa_pair["_id"]}')
            print(f"Iteration: {iteration + 1}")
            print(f"Answer: {answer}")
            print(f"References: {references}")

            # Validate the response
            validation_feedback = validate_response(
                question=qa_pair["question"],
                documents_dict=documents_dict,
                answer=answer,
                references=references,
                subquestions=subquestions,
                model_instance=model_instance,
            )

            print(f"Validation Feedback: {validation_feedback}")

            start = validation_feedback.find("VALIDATION_RESULT:")
            if start == -1:
                start = validation_feedback.find("VALIDATION RESULT:")
            if start == -1:
                print("Validator did not return a proper validation result.")
                iteration += 1
                continue

            validation_feedback = validation_feedback[
                start + len("VALIDATION_RESULT:") :
            ].strip()

            print(f"\n\n\nadsdas{validation_feedback.upper()}")
            if iteration == 0:
                initial_predictions["answer"][qa_pair["_id"]] = answer
                initial_predictions["sp"][qa_pair["_id"]] = references

            if "FAIL" not in validation_feedback.upper():  # success
                success = True
                predictions["answer"][qa_pair["_id"]] = answer
                predictions["sp"][qa_pair["_id"]] = references
            else:
                iteration += 1
                if iteration >= max_iterations:
                    # If max iterations reached without success, log as failure
                    predictions["answer"][qa_pair["_id"]] = answer
                    predictions["sp"][qa_pair["_id"]] = references
                    print(f"Failed to validate after {max_iterations} iterations.")
                else:
                    print(f"Validation failed: {validation_feedback}. Retrying...")

        print("-" * 50)
        print(f'Correct answer: {qa_pair["answer"]}')
        print(f'Correct references: {qa_pair["supporting_facts"]}')

    if output_path is None:
        output_path = f"CoT_{dataset_name}_{model_name}.json"

    predictions_path = f"{output_path}_predictions.json"
    try:
        with open(predictions_path, "w") as pred_file:
            json.dump(predictions, pred_file, indent=4)
        print(f"Predictions saved to {predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

    if output_path is None:
        output_path = f"CoT_{dataset_name}_{model_name}.json"

    initial_predictions_path = f"{output_path}_initial_predictions.json"
    try:
        with open(initial_predictions_path, "w") as pred_file:
            json.dump(initial_predictions, pred_file, indent=4)
        print(f"Initial predictions saved to {initial_predictions_path}")
    except Exception as e:
        print(f"Error saving predictions: {e}")
        return

    # Evaluate predictions
    if dataset_name == "distractor":
        eval_path = os.getenv("HOTPOTQA_DEV_DISTRACTOR")
    else:
        eval_path = os.getenv("HOTPOTQA_DEV_FULLWIKI")
    print("initial")
    eval(initial_predictions_path, eval_path)
    print("after feedback")
    eval(predictions_path, eval_path)


if __name__ == "__main__":
    parser = argparse.ArgumentParser(
        description="Run Distractor Baseline on HotpotQA Dataset"
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
        default="gpt-4o-mini",
        help="Model to use (e.g., gpt-4, llama3) (default: gpt-4o-mini)",
    )
    parser.add_argument(
        "--num_training_samples",
        type=int,
        default=10,
        help="Number of training samples to process (default: 10)",
    )

    args = parser.parse_args()

    run_baseline(
        dataset_name=args.dataset,
        model_name=args.model,
        num_training_samples=args.num_training_samples,
    )

    # Example Commands:
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot_without_rewrite.py --dataset distractor --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot_without_rewrite.py --dataset distractor --model gpt-4o-mini --num_training_samples 1000

    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot.py --dataset fullwiki --model meta-llama-3-8b-instruct --num_training_samples 1000
    # python3 /home/cpp/jerryhuang/reasoning/src/baselines/baseline_cot.py --dataset fullwiki --model gpt-4o-mini --num_training_samples 1000
