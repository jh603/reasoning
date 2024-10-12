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
    references = []
    references_match = re.search(
        r"FINAL_REFERENCES:\s*(\[.+\])", response, re.IGNORECASE | re.DOTALL
    )
    if references_match:
        ref_string = references_match.group(1)
        # Use regex to find all [Title, Number] patterns, allowing commas in the title
        ref_list = re.findall(r"\[([^,\]]+(?:,[^,\]]+)*),\s*(\d+)\]", ref_string)
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
    You are a highly skeptical teacher.
    
    **Instructions:**
    ***If there is a failure at an early step simply skip to Step 6 and provide feedback***
    
    0: **Check the completeness of each reference:**
        - Check for any ambiguous pronouns or references due to missing antecedents within each reference**. If a pronoun (e.g., "he", "she", "it") or reference is unclear and lacks an antecedent in the same document, it is a FAILURE skip to Step 6
        - e.g. Reference: "He had over 100 bananas in the ocean." This is missing a reference to who "he" refers to. This is a FAILURE.
        - e.g. Reference: "Prior to the flying to Mars, it trained in China to prepare."  This is missing a reference to who "it" refers to. This is a FAILURE.

    1.1. **Answer Sub-Questions**:
        - Provide answers to each sub-question.
        - If an sub-question can have multiple possible answers, it is sufficient to identify one good answer as long as it is supported by the references.
        - E.g. The sub-question is asking for all the NBA players, but since the documents only support the fact that Michael Jackson was an NBA player. It is sufficient since this fact is supported.
        - Choose the sentences in the documents that most closely aligns with wording to the sub-question
        - E.g. The sub-question asks for a list of "NBA players." The documents mention only Lebron James as a basketball player. This is insufficient as basketball player is not the same as an NBA player. This is a FAILURE. Please look for a DIFFERENT document that clearly states "NBA player"
        - E.g. The sub-question asks for a list of "guest stars." If don't explicitly use the term "guest." This is not up to standard and is FAILURE. Please look for a DIFFERENT document that clearly uses the term "guest"
        
        Sub-questions: {subquestions}

    1.2. **Reconstruct the Answer**:
        - Using the original question and the list of references, attempt to answer the original **Question** without considering the provided answer. Ensure that your answer directly answers the given question.
        - If there are multiple constraints in th main question, ensure that each one is satsified by a reference. If any of them are not addressed, this is a FAILURE. Suggest a single constraint that is not met in your feedback.
        - E.G. The main question imposes constraints: constraint (1), constraint (2), constraint (3) and constraint (4). Constraint (1) is satsified by the sentence...
        - Ensure that the reasoning is clear and explicit. No assumptions should be made beyond what is explicitly stated in the references. **Highlight any unresolved ambiguities.**
        - If any infomation is missing or is unclear this is a FAILURE and skip to Step 6.
    
    2. **Compare Answers**:
        - If the **FINAL_ANSWER** is not specified or missing is it a FAILURE. Ask the LLM to continue looking for the correct answer.
        - Compare your reconstructed answer with the **FINAL_ANSWER** provided by the Primary LLM Agent.
        - Check for completeness and precision. The **FINAL_ANSWER** should be the most precise and use the same language as the references.
    
    3. **Evaluate References**:
        - Assess whether the **FINAL_REFERENCES** sufficiently support the answer.
        - Ensure that all relevant references are included and that no critical references are missing.
    
    4. **Assess Entity Names**:
        - Verify that the entity names in the **FINAL_ANSWER** exactly matches the best keyword match in a reference.
        - Ensure that the terminology used is consistent with the source material.
        - The final answer should always be the name of an entity or a short answer like yes or no. It should never be a full sentence or longer.

    5. **Provide Feedback**:
        - If the reconstructed answer matches the **FINAL_ANSWER** and the references are sufficient, output "VALIDATION_RESULT: PASS".
        - **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
        - **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.
        - If there are discrepancies, incomplete answers, or insufficient references, output "VALIDATION_RESULT: FAIL: [Reason for failure]".
    
    6. **Provide a final ruling in the following form. Present an approach to fix this issue. Always include the exact SUB_QUESTION or REFERENCE that has an issue in the feedback**: (This list is not exhaustive)
       - **VALIDATION_RESULT: PASS**
       - **VALIDATION_RESULT: FAIL: The pronoun <pronoun> in article titled <title> is unclear, consider citing a previous sentence in the reference as well.
       - **VALIDATION_RESULT: FAIL: The subquestion question <sub-question> is answered incorrectly because <reason>. The reference <reference> is incorrect because <reason>**
    
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
    max_iterations=3,
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
    offset = 100
    for qa_pair in dataset[offset : num_training_samples + offset]:
        print(f"{i} Questions processed")
        i += 1
        final_output = ""
        documents_dict = {}  # title idx: sentence
        for entry in qa_pair["context"]:
            title = entry[0]
            sentences = entry[1]
            flattened_content = " ".join(
                [f"({i}) {sentence}" for i, sentence in enumerate(sentences)]
            )
            final_output += f"Title: {title}: {flattened_content}\n\n"

            for idx, sentence in enumerate(sentences):
                documents_dict[f"{title} {idx}"] = sentence

        print(f"documents_dict: {documents_dict}")
        print(f'Correct answer: {qa_pair["answer"]}')
        print(f'Correct references: {qa_pair["supporting_facts"]}')
        iteration = 0
        success = False

        conversation_history = []
        subquestions = None
        while iteration < max_iterations and not success:
            print(f"iteration: {iteration}")
            if iteration == 0:
                prompt_gen_subquestions = f"""
**Generate Sub-Questions**:
   - First, rewrite the multi-hop question into a series of a few short, easy to understand English sentences. Explicitly infer what each part of the question is referring to. Consider the context of key terms such as whether they refer to people, episodes, shows, films, or other entities. Clarify what type of entity the question is asking for.
        - E.g. Which of Tara Strong major voice role in animated series is an American animated television series based on the DC Comics fictional superhero team, the "Teen Titans"?
            - The question can be clarified to be "Which of Tara Strong's major voice roles is in X" and "X is an American animated television series based on Y" and "Y is the DC Comics fictional superhero team, the Teen Titans."
        - E.g. Who is the younger brother of The episode guest stars of The Hard Easy?
            - This question can be clarified to be "Which guest stars were in the episode named "The Hard Easy?" and "Out of these guest stars, which guess star has a younger brother" and "Who is the younger brother called."
   - Use language similar to the original question in each of the sub-questions.
   - Make sure you include every constraint introduced in the question. 
   - Create a sequence of sub-questions that can help in answering the main multi-hop question. The last question should directly answer the main question. Each sub-question can impose a single constraint or look up a single fact.
   - Use placeholders {{answer1}}, {{answer2}}, etc., where necessary.
        
**Example**:
    **Question**: Which of Tara Strong major voice role in animated series is an American animated television series based on the DC Comics fictional superhero team, the "Teen Titans"?
    Reasoning: The question can be clarified to be "Which of Tara Strong's major voice roles is in X" and "X is an American animated television series based on Y" and "Y is the DC Comics fictional superhero team, the Teen Titans."

    Questions:
        1. What are Tara Strong's major roles are in animated series?
        2. Which of the actors roles in {{answer1}} are based on the DC Comics fictional superhero team, the "Teen Titans"?

**New Question**: {qa_pair["question"]}

**Output Format**:
Reasoning: The multi-hop question is asking "<rephrased question>". It seems that X in the question refers to a TV SHOW. We need to first identify X in the TV show. Once we have the answer to X, we can move on to Y, which appears to refer to a person or entity.

**Questions:**
1. X refers to a TV show. Question 1.
2. Question 2.
"""

                subquestions = model_instance.get_response(prompt_gen_subquestions)
                print(f"subquestions: {subquestions}")

                start_index = subquestions.find("Questions:")

                if start_index == -1:
                    return "No 'Questions:' found in the text."

                questions_text = subquestions[start_index + len("Questions:") :]
                subquestions = questions_text.lstrip()

                prompt = f"""
Please answer the following question using only the information provided in the documents below.

**Instructions:**

1. **Restate the Sub-Questions including the specific facts in front of the sub-questions when applicable**:
    e.g. Question 1. 
         Question 2. 
         Question 3. 

2. **Answer Sub-Questions**:
    - Provide answers to each sub-question. As you answer each sub-question, redraft it based on the answer to previous sub-questions if necessary.
    - **Be highly skeptical. If there is a document that does not exactly answer you subquestion, do not use it at all. There will be attempts to trick you such as mixing up "basketball player" vs "NBA player" or "painter" and "artist" or "guest star" and "main cast star". These entities should not be considered the same.
        - E.g. The sub-question asks for a list of "guest stars." If don't explicitly use the term "guest." This is not up to standard and is FAILURE. Please look for a DIFFERENT document that clearly uses the term "guest"
    - For each answer, quote the full document you wish to reference. You must do this.
    - If an sub-question can have multiple possible answers, it is sufficient to identify one good answer as long as it is supported by the references.
    - **List every sentence number used** to answer each sub-question.
    - You can only cite one reference per sub-question.

3. **Explain Reasoning**:
    - Assume you have the reasoning capabilities of a fifth grader. Only base your answers off what is directly stated in the references. Your answer should be directly a subset of the text referenced
    - Write a paragraph explaining your overall reasoning for answering the main question.
    - **List every sentence number used** in this explanation.

4. **Provide Final Answer**:
    - The answer should be **very concise**, such as a year, a simple yes/no, or a person's name.
    - Use wording **exactly as it appears** in the original documents.
    - **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
    - **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.
    - The final answer should always be the name of an entity or a short answer like yes or no. It should never be a full sentence or longer.


5. **List References**:
    - **Aggregate** all references cited in both the sub-question answers and the explanation. ONLY KEEP CITATIONS THAT DIRECTLY SUPPORT THE MAIN QUESTION.
    - Ensure there are no duplicates in the list of references
    - **Ensure that every referenced sentence is included** in this list.
    - Format each reference as `[Document Title, Sentence Number]`.

6. **Use Unique Delimiters**:
    - Put "FINAL_ANSWER:" before your final answer.
    - Put "FINAL_REFERENCES:" before your final list of references.

Answer format:

1. **Sub-Questions:**  
   1. *(First sub-question)* 
   2. *(Second sub-question)*  
   *(Add more sub-questions if necessary)*

2. **Sub-Question Answers:**
   **If there are multiple answers to a sub-question, track a list of all possible answers, and narrow down the list as you move from sub-question to sub-question**
   If there is not answer to a sub-question do not provide any references.
   1. ***First Sub-Question***
      *Reasoning* 
        - Why did I choose this document? Does this document use the fact provided by the oracle?
        - State the indices available for the document: (0), (1), ...
        - Entire Document: Quote the ENTIRE document verbatim
        - Document Title: (index) Quote the important sentences. Explain why this answers the question
      *Answer:* *(Answer to the first sub-question)*  
      *References:* [Document Title, Sentence Number], ...  
      
    2. ***Second Sub-Question***
      *Reasoning* 
        - Redraft second sub-question based on the first sub-question's answer if helpful
        - Why did I choose this document? Does this document use the fact provided by the oracle?
        - State the indices available for the document: (0), (1), ...
        - Entire Document: Quote the ENTIRE document verbatim
        - Document Title: (index) Quote the important sentences. Explain why this answers the question
      *Answer:* *(Answer to the first sub-question)*  
      *References:* [Document Title, Sentence Number], ...  

   *(Add more sub-question answers if necessary)*
   
   Example:
   1. ***What leagues did Michael Jordan play in?***  
      *Reasoning* 
        - The document "Lebron James" DOES NOT mention that Jordan is an "athlete." Therefore, I CANNOT reference it. 
        - Since I cannot use the document "Lebron James," I will thus, reference the article "NBA" as it refers to Michael Jordan as an "athlete."
        - Indices available for "NBA": (0), (1), (2), (3)
        - Entire Document: NBA: (0) Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ, is an American businessman and former professional basketball player. (1) He played 15 seasons in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA championships with the Chicago Bulls. (3) He was integral in popularizing basketball and the NBA around the world in the 1980s and 1990s, becoming a global cultural icon. (4) His profile on the NBA website states, "By acclamation, Michael Jordan is the greatest basketball player of all time."
        - NBA: (0) Michael Jeffrey Jordan (born February 17, 1963), also known by his initials MJ, is an American businessman and former professional basketball player. This tell us who "he" refers to in sentence (1)
        - NBA: (1) He played 15 seasons in the National Basketball Association (NBA) between 1984 and 2003, winning six NBA championships with the Chicago Bulls. This tells us Michael Jordan played in the NBA.
      *Answer:* According to the documents, Michael Jordan played in the NBA. Sentence (0) clarifies who "he" refers to in sentence (1). We do not know what other leagues Michael Jordan may have played by based on the given document, but the documents clearly state that he was a part of the NBA.
      *References:* [NBA, 0], [NBA, 1]

3. **Explanation:**  
   *(A paragraph explaining your reasoning for the main question, incorporating the sub-question answers and citations)*  
   You can only choose 2 documents to cite at the end.

4. **Answer:**  
   FINAL_ANSWER: *(Final concise answer to the main question, following the guidelines for wording and support)*

5. **References:**  
   FINAL_REFERENCES: [Document Title, Sentence Number], [Document Title, Sentence Number], ...

**Question:** {qa_pair["question"]}

**Subquestions** 
{subquestions}

**Documents:**  
{final_output}
"""
            else:
                prompt = f"""
Please address the feedback and use this output format. The feedback is from a the most accurate scholar in the world. Please be flexible and reconsider all the documents if there is feedback.
If the feedback is that a reference is unclear not because of an ambiguous pronoun but because the terminology does not exactly match the question, immediately remove that reference and find a better on.
- E.g. The sub-question asks for a list of "NBA players." The documents mention only Lebron James as a basketball player. This is insufficient as basketball player is not the same as an NBA player. This is a FAILURE. Please look for a DIFFERENT document that clearly states "NBA player"
- E.g. The sub-question asks for a list of "guest stars." THe documents only mentions normal stars. This is not up to standard and is FAILURE. Please look for a DIFFERENT document that clearly states "guest star"
        
**Do not include more background information in your final answer. Keep it concise. Do not redraft it unless it is wrong."

1. **Explanation:**
    - First restate the ENTIRE document you wish to use by quoting it verbatim including the title and indices.
    - You must quote the new statements you wish to cite and then include the title and sentence index as well.
    - To clarify antecedents, you must include a sentence *prior* to the one that is unclear.
    - e.g. Full articles that I will consider: 
        Michael Jordan: (0) Michael Jordan played for the Chicago, Bulls. (1) He had 6 championships. (3) He was the logo of the NBA.
        Lebron James: (0) Lebron is a team player. (1) He plays on the LA Lakers. (2) He has 3 championships.
        
        Because the feedback states that the article in "Lebron James" states that he is a "team player" and not explicitly that he is an NBA player. I will remove all references to this article.
        To support the statement that Michael Jordan had 6 championships, I will also cite the sentence (1) from the article "Michael Jordan," "He had 6 championships" ["Michael Jordan", 1]
    - e.g. Full articles that I will consider:
        Michael Jordan: (0) Michael Jordan played for the Chicago, Bulls. (1) He had 6 championships. (3) He was the logo of the NBA.
        
        To clarify who "he" refers to in the statement "he had 6 championships," I will also cite the sentence (0) from the article "Michael Jordan," "Michael Jordan played for the Chicago, Bulls" ["Michael Jordan", 0] as this sentence clarifies the pronoun in the subsequent sentence.

1.1 **Actions:**
    - State your intended action to fix the problem. Do not say you are going to add a reference if it was already in your list of references previously.
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
- **If the answer refers to an entity**, ensure the wording **matches exactly** how it is presented in the context.
- **If the answer is a list of potential correct answers**, select the one that is *most directly supported* by the given documents and use its exact name.

For reference errors:

**Feedback:**
{validation_feedback}
"""

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
                    pass
                    # print(f'Validation failed: {validation_feedback}. Retrying...')

        print("-" * 50)
        print(f"qa_pair: {qa_pair}")
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
