import copy
import logging
import re
from datetime import datetime

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
from datasets import load_dataset
from nltk.translate.bleu_score import sentence_bleu
from rouge_score import rouge_scorer
from sentence_transformers import SentenceTransformer
from sklearn.metrics.pairwise import cosine_similarity
from torch.utils.tensorboard import SummaryWriter

from networks import PolicyNetwork, ValueNetwork
from retrievers.DPR import GTRWikiRetriever
from utils.llama import Llama3

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize logging
logging.basicConfig(
    level=logging.DEBUG, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Models and Tokenizer
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
retriever = GTRWikiRetriever()
llama_model = Llama3(
    "/home/cpp/jerryhuang/search2024/meta-llama/Meta-Llama-3-8B-Instruct"
)
tokenizer = llama_model.get_tokenizer()

MAX_DEPTH = 250
TOP_K_ACTIONS = 3  # Initial TOP_K_ACTIONS value
MIN_TOP_K_ACTIONS = 1
STEP_TOKENS = 1  # num of tokens after which TOP_K_ACTIONS decreases
NUM_TRAINING_SAMPLES = 1
NUM_VALIDATION_SAMPLES = 1
NUM_TEST_SAMPLES = 1
MAX_QUERY_TOKENS = 259
DEPTH_PENALTY_WEIGHT = 0
MAX_ITERATIONS_TRAIN = 500
MAX_ITERATIONS_EVAL = 500
C_PARAM = 1.2  # Exploration parameter
SIMILARITY_THRESHOLD = 0.05

TEMPERATURE = 0.8


def get_dynamic_top_k(
    query_length,
    initial_top_k=TOP_K_ACTIONS,
    min_top_k=MIN_TOP_K_ACTIONS,
    step=STEP_TOKENS,
):
    """
    Dynamically adjusts TOP_K_ACTIONS based on the current query length.

    Parameters:
    - query_length (int): Number of tokens generated so far.
    - initial_top_k (int): Initial value of TOP_K_ACTIONS.
    - min_top_k (int): Minimum value of TOP_K_ACTIONS.
    - step (int): Number of tokens after which TOP_K_ACTIONS decreases.

    Returns:
    - int: Current TOP_K_ACTIONS value.
    """
    # Calculate how many times the step has been exceeded
    decrement_steps = query_length // step
    # Decrease TOP_K_ACTIONS by 1 for each step, not going below min_top_k
    dynamic_k = max(initial_top_k - decrement_steps, min_top_k)
    return dynamic_k


# Define the State class
class State:
    def __init__(
        self,
        question,
        draft_answer="None",
        previous_queries=None,
        retrieved_documents=None,
        depth=0,
        last_action=None,
    ):
        self.question = question
        self.draft_answer = draft_answer
        self.previous_queries = previous_queries if previous_queries else []
        self.retrieved_documents = retrieved_documents if retrieved_documents else []
        self.depth = depth
        self.reward = 0
        self.last_action = last_action
        self.partial_query_tokens = []  # For token-level query generation
        self.generating_query = (
            False  # Flag to indicate if currently generating a query
        )

    def to_text(self):
        # Text representation for answer generation
        context = "\n".join(self.previous_queries)
        documents = "\n".join(self.retrieved_documents)
        partial_answer = self.draft_answer
        return f"Question: {self.question}\nPrevious Queries: {context}\nRetrieved Documents: {documents}\n\nDraft Answer: {partial_answer}"

    def to_query_text(self):
        # Text representation for query generation
        partial_query = tokenizer.decode(self.partial_query_tokens)
        return f"Question: {self.question}\nDraft Query: {partial_query}"

    def is_terminal(self):
        # Terminal if depth limit reached or if final answer is submitted
        terminal = (
            self.depth >= MAX_DEPTH
            or self.last_action == "Submit Draft Answer as Final Answer"
        )
        if terminal:
            logging.info(
                f"Reached terminal state at depth {self.depth} with action '{self.last_action}'"
            )
        return terminal

    def get_possible_actions(self):
        actions = []

        if self.generating_query:
            # Token-level actions with dynamic TOP_K_ACTIONS
            input_text = self.to_query_text()
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(
                llama_model.model.device
            )
            logits = llama_model.get_next_token_logits(input_ids)
            probabilities = torch.softmax(logits, dim=-1).squeeze(
                0
            )  # Shape: [vocab_size]

            # Calculate current_top_k based on the number of tokens generated
            current_top_k = max(
                TOP_K_ACTIONS - len(self.partial_query_tokens), MIN_TOP_K_ACTIONS
            )

            # Get top K token indices
            top_probs, top_indices = torch.topk(probabilities, current_top_k * 2)
            top_indices = top_indices.tolist()
            top_probs = top_probs.tolist()

            # Exclude special tokens
            special_token_ids = set(tokenizer.all_special_ids)
            filtered_actions = [
                idx for idx in top_indices if idx not in special_token_ids
            ][:current_top_k]

            # Generate token actions
            token_actions = [
                f"Generate Query Token: {token_id}" for token_id in filtered_actions
            ]
            actions.extend(token_actions)
        else:
            # High-level actions
            actions.extend(
                [
                    "Generate Query and Revise Answer",
                    "Submit Draft Answer as Final Answer",
                ]
            )
        return actions

    def take_action(self, action):
        new_state = State(
            question=self.question,
            draft_answer=self.draft_answer,
            previous_queries=copy.deepcopy(self.previous_queries),
            retrieved_documents=copy.deepcopy(self.retrieved_documents),
            depth=self.depth + 1,
            last_action=action,
        )
        new_state.partial_query_tokens = self.partial_query_tokens.copy()
        new_state.generating_query = self.generating_query

        logging.debug(
            f"Creating new state from depth {self.depth} to {new_state.depth}"
        )

        if action == "Generate Query and Revise Answer":
            logging.info(f"Action: {action} - Starting query generation.")
            new_state.generating_query = True
            # Initialize partial_query_tokens with the refined prompt
            prompt = (
                "Please write a simple query that can be answered from a single page of an encyclopedia.\n"
                "Use the format:\n"
                "Query: [Your query here]?\n"
                "Ensure the query ends with a '?'\n"
                "Do not provide any answers or additional text."
            )
            new_state.partial_query_tokens = tokenizer.encode(
                prompt, add_special_tokens=False
            )
            logging.debug(
                "Generating Query Flag Set to True. Partial Query Tokens Initialized with Prompt."
            )
        elif action.startswith("Generate Query Token:"):
            token_id = int(action.split(":")[1].strip())
            token_text = tokenizer.decode([token_id])
            logging.debug(f"Action: {action} ({token_text})")

            # Append the token
            new_state.partial_query_tokens.append(token_id)

            # Decode the partial query
            partial_query = tokenizer.decode(new_state.partial_query_tokens)
            logging.debug(f"Updated Partial Query: '{partial_query}'")

            # Check for termination condition
            if len(
                new_state.partial_query_tokens
            ) >= MAX_QUERY_TOKENS or partial_query.strip().endswith("?"):
                logging.info("Termination condition met.")
                new_state.generating_query = False
                print(f"\n\nFormulating query and retrieving documents")

                # Formulate the complete query
                decoded_query = tokenizer.decode(new_state.partial_query_tokens)

                # Extract the **last** occurrence of 'Query:'
                matches = list(
                    re.finditer(
                        r"Query:\s*(.*?)\s*(?:\n|<END>|$)", decoded_query, re.DOTALL
                    )
                )
                if matches:
                    # Take the last match to get the actual generated query
                    complete_query = matches[-1].group(1).strip()
                    logging.debug(f"Extracted Complete Query: '{complete_query}'")
                else:
                    # Fallback extraction if regex fails
                    logging.warning(
                        "Regex failed to match. Using fallback extraction method."
                    )
                    query_start = decoded_query.rfind("Query:")
                    if query_start != -1:
                        complete_query = (
                            decoded_query[query_start + len("Query:") :]
                            .split("<END>")[0]
                            .strip()
                        )
                    else:
                        complete_query = decoded_query.replace("<END>", "").strip()
                    logging.debug(f"Fallback Complete Query: '{complete_query}'")

                print(f"Complete Query: {complete_query}\n")

                # Append to previous queries
                new_state.previous_queries.append(complete_query)

                # Retrieve documents
                retrieved = retriever.gtr_wiki_retrieval(complete_query)
                # retrieved = ['test test test']
                new_state.retrieved_documents.extend(retrieved)

                # Revise the draft answer based on retrieved information
                new_state.draft_answer = new_state.revise_answer(retrieved)
        elif action == "Submit Draft Answer as Final Answer":
            logging.info(f"Action: {action} - Submitting final answer.")
            # Terminal action; no further processing needed
        else:
            logging.warning(f"Unknown action: {action}")

        return new_state

    def revise_answer(self, retrieved_documents):
        logging.info("Revising the draft answer using LLaMA...")
        context = "\n".join(retrieved_documents)
        prompt = (
            "Revise the draft answer based on the new information if one is provided. "
            "Otherwise, write a draft answer to the question below that includes any new important information from the documents provided."
            "\nOnly include the draft answer in the output"
            f"\n\n{self.to_text()}\nRetrieved Documents: {context}"
        )
        revised_answer = llama_model.get_llama_response(prompt, temperature=TEMPERATURE)

        if revised_answer is None or revised_answer.strip() == "":
            logging.warning("LLaMA returned an empty answer. Assigning default answer.")
            revised_answer = "None"

        print(f"PROMPT: {prompt[:40]}")
        print(f"REVISED_ANSWER: {revised_answer}")
        return revised_answer

    def compute_reward(self, ground_truth_answer):
        if self.draft_answer:
            draft_embedding = embedding_model.encode(self.draft_answer)
            ground_truth_embedding = embedding_model.encode(ground_truth_answer)
            similarity = cosine_similarity([draft_embedding], [ground_truth_embedding])[
                0
            ][0]
        else:
            similarity = 0

        # Intermediate reward for each query token generated
        query_length = len(self.partial_query_tokens)
        query_length_reward = query_length * 0.01  # Adjust weight as necessary

        # Depth penalty
        depth_penalty = DEPTH_PENALTY_WEIGHT * max(self.depth, 1)
        self.reward = similarity + query_length_reward - depth_penalty
        logging.info(
            f"Similarity: {similarity}, Query Length Reward: {query_length_reward}, "
            f"Depth Penalty: {depth_penalty}, Reward: {self.reward}"
        )
        return self.reward

    def to_tensor(self):
        # Convert state to tensor representation for neural network inputs
        if self.generating_query:
            text_representation = self.to_query_text()
        else:
            text_representation = self.to_text()
        embedding = embedding_model.encode(text_representation)
        tensor = torch.tensor(embedding, dtype=torch.float32)
        return tensor  # .unsqueeze(0)  # Add batch dimension


# Define the Node class for MCTS
class Node:
    def __init__(self, state, parent=None, action=None, prior_probability=0.0):
        self.state = state  # The current state
        self.parent = parent  # Parent node
        self.children = []  # List of child nodes
        self.visit_count = 0
        self.total_value = 0.0
        self.untried_actions = state.get_possible_actions()
        self.action = action  # Action taken to reach this node
        self.prior_probability = prior_probability  # From the Policy Network

    def get_possible_actions(self):
        actions = []

        if self.state.generating_query:  # Correct reference
            # Token-level actions with dynamic TOP_K_ACTIONS
            input_text = self.state.to_query_text()
            input_ids = tokenizer.encode(input_text, return_tensors="pt").to(
                llama_model.model.device
            )
            logits = llama_model.get_next_token_logits(input_ids)
            probabilities = torch.softmax(logits, dim=-1).squeeze(
                0
            )  # Shape: [vocab_size]

            # Calculate current_top_k based on the number of tokens generated
            current_top_k = max(
                TOP_K_ACTIONS - len(self.state.partial_query_tokens), MIN_TOP_K_ACTIONS
            )

            # Get top K token indices
            top_probs, top_indices = torch.topk(probabilities, current_top_k * 2)
            top_indices = top_indices.tolist()
            top_probs = top_probs.tolist()

            # Exclude special tokens
            special_token_ids = set(tokenizer.all_special_ids)
            filtered_actions = [
                idx for idx in top_indices if idx not in special_token_ids
            ][:current_top_k]

            # Generate token actions
            token_actions = [
                f"Generate Query Token: {token_id}" for token_id in filtered_actions
            ]
            actions.extend(token_actions)
        else:
            # High-level actions
            actions.extend(
                [
                    "Generate Query and Revise Answer",
                    "Submit Draft Answer as Final Answer",
                ]
            )
        return actions

    def is_fully_expanded(self):
        return len(self.untried_actions) == 0

    def is_terminal_node(self):
        return self.state.is_terminal()

    def best_child(self, c_param=C_PARAM):
        choices_weights = []
        for child in self.children:
            if child.visit_count == 0:
                exploitation = 0  # No exploitation info yet
                exploration = (
                    c_param
                    * child.prior_probability
                    * np.sqrt(self.visit_count + 1)
                    / (1 + child.visit_count)
                )
            else:
                exploitation = child.total_value / child.visit_count
                exploration = (
                    c_param
                    * child.prior_probability
                    * np.sqrt(self.visit_count)
                    / (1 + child.visit_count)
                )
            total_weight = exploitation + exploration
            choices_weights.append(total_weight)
            logging.debug(
                f"Child Action='{child.action}', Exploitation={exploitation:.4f}, Exploration={exploration:.4f}, Total Weight={total_weight:.4f}"
            )

        if not choices_weights:
            logging.warning("No children to select as best child.")
            return self  # Return self if no children are available

        best_index = np.argmax(choices_weights)
        best = self.children[best_index]
        logging.debug(
            f"Selected best child: Action='{best.action}', Total Weight={choices_weights[best_index]:.4f}"
        )
        return best

    def expand(self, policy_network, state_tensor):
        action_texts = (
            self.untried_actions.copy()
        )  # Copy to avoid modifying during iteration
        if not action_texts:
            logging.debug("No actions to expand.")
            return

        action_embeddings = embedding_model.encode(action_texts)
        action_embeddings = torch.tensor(action_embeddings, dtype=torch.float32).to(
            device
        )

        num_actions = action_embeddings.size(0)

        # Ensure state_tensor is 2D
        if state_tensor.dim() == 1:
            state_tensor = state_tensor.unsqueeze(0)

        # Repeat state_tensor to match the number of actions
        state_embeddings = state_tensor.repeat(
            num_actions, 1
        )  # (num_actions, input_size)

        with torch.no_grad():
            log_probs = policy_network(
                state_embeddings, action_embeddings
            )  # (num_actions,)
            action_probs = (
                torch.exp(log_probs).cpu().numpy()
            )  # Convert to probabilities

        # Normalize the probabilities
        sum_probs = np.sum(action_probs)
        if sum_probs == 0:
            logging.warning(
                "Sum of action probabilities is zero. Assigning uniform probabilities."
            )
            action_probs = np.ones_like(action_probs) / len(action_probs)
        else:
            action_probs /= sum_probs

        # Select top K actions based on policy probabilities
        # Calculate current_top_k based on the number of tokens generated
        current_top_k = get_dynamic_top_k(len(self.state.partial_query_tokens))

        top_k_indices = np.argsort(action_probs)[-current_top_k:]
        top_actions = [action_texts[i] for i in top_k_indices]
        top_probs = action_probs[top_k_indices]

        logging.debug(f"Expanding node with actions: {top_actions}")

        for action, prob in zip(top_actions, top_probs):
            child_state = self.state.take_action(action)
            child_node = Node(
                state=child_state, parent=self, action=action, prior_probability=prob
            )
            self.children.append(child_node)
            self.untried_actions.remove(action)
            logging.debug(
                f"Added child node: Action='{action}', Prior Probability={prob:.4f}"
            )


# MCTS functions
def mcts_search(
    root,
    policy_network,
    value_network,
    max_iterations,
    ground_truth_answer,
    training_data,
    writer,
):
    logging.info("Starting MCTS search...")
    for iteration in range(max_iterations):
        logging.debug(f"MCTS iteration: {iteration+1}/{max_iterations}")
        node = tree_policy(root, policy_network, value_network)
        reward = default_policy(node.state, value_network, ground_truth_answer)
        backup(node, reward, writer, iteration)

        logging.debug(f"Iteration {iteration+1}: Reward = {reward}")

        # Log the reward for this iteration
        writer.add_scalar("MCTS/Iteration_Reward", reward, iteration)

        # Early stopping if high reward achieved
        if reward > 0.9:  # Threshold can be adjusted
            logging.info("Early stopping as high reward achieved.")
            writer.add_scalar("MCTS/Early_Stopping_Reward", reward, iteration)
            break

    # After MCTS iterations, traverse from root to terminal node, selecting best child at each step
    path_nodes = []
    current_node = root
    while not current_node.is_terminal_node() and current_node.children:
        best_child = current_node.best_child(c_param=0)
        if best_child == current_node:
            # Prevent infinite loop if best_child returns the same node
            logging.warning("Best child is the current node. Stopping traversal.")
            break
        path_nodes.append(best_child)
        current_node = best_child

    # Collect training data from the path_nodes
    for node in path_nodes:
        if node.action is not None:
            training_data.append(
                {
                    "state": node.state,
                    "action": node.action,
                    "prior_probability": node.prior_probability,
                    "visit_count": node.visit_count,
                    "total_value": node.total_value,
                }
            )
            # Log the action and its prior probability
            writer.add_scalar(
                "MCTS/Action_Prior_Probability",
                node.prior_probability,
                node.visit_count,
            )
            writer.add_scalar(
                "MCTS/Action_Visit_Count", node.visit_count, node.visit_count
            )
            writer.add_scalar(
                "MCTS/Action_Total_Value", node.total_value, node.visit_count
            )
            logging.debug(
                f"Added training sample: Action='{node.action}', Reward={node.total_value}"
            )

    logging.info(f"Collected {len(path_nodes)} training samples from MCTS.")
    return current_node  # Return the terminal node


def tree_policy(node, policy_network, value_network):
    while not node.is_terminal_node():
        if not node.is_fully_expanded():
            state_tensor = node.state.to_tensor().to(device)
            node.expand(policy_network, state_tensor)
            return node.children[-1]
        else:
            node = node.best_child()
    return node


def default_policy(state, value_network, ground_truth_answer):
    if state.is_terminal():
        return state.compute_reward(ground_truth_answer)
    state_tensor = state.to_tensor().to(device)
    with torch.no_grad():
        value_estimate = value_network(state_tensor).item()
    return value_estimate


def backup(node, reward, writer, iteration):
    # logging.debug(f"Backing up reward: {reward}")
    while node is not None:
        node.visit_count += 1
        node.total_value += reward

        # Log cumulative value and visit count at each node
        writer.add_scalar("MCTS/Node_Total_Value", node.total_value, iteration)
        writer.add_scalar("MCTS/Node_Visit_Count", node.visit_count, iteration)

        node = node.parent


def get_tree_structure(node, depth=0):
    indent = "  " * depth
    lines = [f"{indent}{node}"]
    for child in node.children:
        lines.extend(get_tree_structure(child, depth + 1))
    return lines


def get_action_path(node):
    """
    Traverse the tree from the given terminal node up to the root node,
    collecting all actions taken along the path.

    Parameters:
    - node (Node): The terminal node.

    Returns:
    - List[str]: A list of actions from the root to the terminal node.
    """
    actions = []
    current_node = node
    while current_node.parent is not None:
        actions.append(current_node.action)
        current_node = current_node.parent
    actions.reverse()
    return actions


def collect_training_data(node, training_data, writer):
    for node in nodes:
        if node.action is not None:
            training_data.append(
                {
                    "state": node.state,
                    "action": node.action,
                    "prior_probability": node.prior_probability,
                    "visit_count": node.visit_count,
                    "total_value": node.total_value,
                }
            )
            # Log the action and its prior probability
            writer.add_scalar(
                "MCTS/Action_Prior_Probability",
                node.prior_probability,
                node.visit_count,
            )
            writer.add_scalar(
                "MCTS/Action_Visit_Count", node.visit_count, node.visit_count
            )
            writer.add_scalar(
                "MCTS/Action_Total_Value", node.total_value, node.visit_count
            )
            logging.debug(
                f"Added training sample: Action='{node.action}', Reward={node.total_value}"
            )


# Training the Policy and Value Networks
def train_networks(
    policy_network,
    value_network,
    training_data,
    validation_data,
    writer,
    num_epochs=5,
    batch_size=32,
):
    policy_optimizer = torch.optim.Adam(
        policy_network.parameters(), lr=1e-4, weight_decay=1e-5
    )
    value_optimizer = torch.optim.Adam(
        value_network.parameters(), lr=1e-4, weight_decay=1e-5
    )
    loss_fn = nn.MSELoss()
    best_accuracy = 0.0

    policy_scheduler = torch.optim.lr_scheduler.StepLR(
        policy_optimizer, step_size=10, gamma=0.1
    )
    value_scheduler = torch.optim.lr_scheduler.StepLR(
        value_optimizer, step_size=10, gamma=0.1
    )

    global_step = 0
    for epoch in range(num_epochs):
        logging.info(f"Training epoch {epoch+1}/{num_epochs}")
        np.random.shuffle(training_data)
        for i in range(0, len(training_data), batch_size):
            batch = training_data[i : i + batch_size]
            if not batch:
                continue

            # Filter out samples with action=None (redundant if collect_training_data already did)
            filtered_batch = [
                data_point for data_point in batch if data_point["action"] is not None
            ]
            if not filtered_batch:
                continue

            state_tensors = torch.stack(
                [data_point["state"].to_tensor() for data_point in filtered_batch]
            ).to(
                device
            )  # (batch_size, embedding_size)
            action_texts = [data_point["action"] for data_point in filtered_batch]
            action_embeddings = embedding_model.encode(action_texts)
            action_embeddings = torch.tensor(action_embeddings, dtype=torch.float32).to(
                device
            )  # (batch_size, embedding_size)
            action_probs = np.array(
                [data_point["prior_probability"] for data_point in filtered_batch]
            )
            action_probs = torch.tensor(action_probs, dtype=torch.float32).to(
                device
            )  # (batch_size,)
            rewards = torch.tensor(
                [
                    data_point["total_value"] / (data_point["visit_count"] + 1e-8)
                    for data_point in filtered_batch
                ],
                dtype=torch.float32,
            ).to(
                device
            )  # (batch_size,)

            # Policy Network Training
            policy_optimizer.zero_grad()
            log_probs = policy_network(
                state_tensors, action_embeddings
            )  # (batch_size,)
            policy_loss = F.mse_loss(torch.exp(log_probs), action_probs)
            policy_loss.backward()
            torch.nn.utils.clip_grad_norm_(policy_network.parameters(), max_norm=1.0)
            policy_optimizer.step()

            # Value Network Training
            value_optimizer.zero_grad()
            values = value_network(state_tensors).squeeze(1)  # (batch_size,)
            value_loss = loss_fn(values, rewards)
            value_loss.backward()
            torch.nn.utils.clip_grad_norm_(value_network.parameters(), max_norm=1.0)
            value_optimizer.step()

            # Logging to TensorBoard
            writer.add_scalar("Policy_Loss", policy_loss.item(), global_step)
            writer.add_scalar("Value_Loss", value_loss.item(), global_step)
            writer.add_scalar("Reward_Mean", rewards.mean().item(), global_step)
            logging.info(
                f"Epoch {epoch+1}, Batch {i//batch_size +1}: Policy Loss={policy_loss.item():.4f}, Value Loss={value_loss.item():.4f}, Reward={rewards.mean().item():.4f}"
            )
            global_step += 1

        # Step schedulers
        policy_scheduler.step()
        value_scheduler.step()

        # Evaluate after each epoch
        accuracy = evaluate_model(
            policy_network, value_network, validation_data, writer
        )
        logging.info(f"Epoch {epoch+1}, Validation Accuracy: {accuracy}")
        writer.add_scalar("Validation_Accuracy", accuracy, epoch)
        if accuracy > best_accuracy:
            best_accuracy = accuracy
            torch.save(policy_network.state_dict(), "policy_network_best.pth")
            torch.save(value_network.state_dict(), "value_network_best.pth")
            logging.info(f"New best model saved at epoch {epoch+1}")


# Evaluation Function
def evaluate_model(policy_network, value_network, validation_data, writer):
    correct = 0
    total = 0
    scorer = rouge_scorer.RougeScorer(["rouge1", "rouge2", "rougeL"], use_stemmer=True)

    for idx, qa_pair in enumerate(validation_data):
        question = qa_pair["question"]
        ground_truth_answer = qa_pair["answer"]
        initial_state = State(question=question)
        root_node = Node(state=initial_state)

        best_child = mcts_search(
            root_node,
            policy_network,
            value_network,
            max_iterations=MAX_ITERATIONS_EVAL,
            ground_truth_answer=ground_truth_answer,
            training_data=[],
            writer=writer,  # Pass the writer to MCTS search
        )
        final_answer = best_child.state.draft_answer

        # Compute similarity to ground truth
        if final_answer:
            draft_embedding = embedding_model.encode(final_answer)
            ground_truth_embedding = embedding_model.encode(ground_truth_answer)
            similarity = cosine_similarity([draft_embedding], [ground_truth_embedding])[
                0
            ][0]
        else:
            similarity = 0.0

        # Additional metrics
        bleu = compute_bleu(final_answer, ground_truth_answer)
        rouge = compute_rouge(final_answer, ground_truth_answer, scorer)

        logging.info(
            f"Sample {idx+1}: Similarity={similarity:.4f}, BLEU={bleu:.4f}, ROUGE={rouge:.4f}"
        )

        # Log individual sample metrics
        writer.add_scalar("Validation/Similarity", similarity, idx)
        writer.add_scalar("Validation/BLEU", bleu, idx)
        writer.add_scalar("Validation/ROUGE", rouge, idx)

        # Define thresholds based on multiple metrics
        if similarity > 0.8 and bleu > 0.7 and rouge > 0.7:
            correct += 1
        total += 1

    accuracy = correct / total if total > 0 else 0
    logging.info(f"Validation Accuracy: {correct}/{total} = {accuracy*100:.2f}%")
    writer.add_scalar("Validation/Accuracy", accuracy, 0)
    return accuracy


def compute_bleu(prediction, reference):
    reference = reference.split()
    prediction = prediction.split()
    return sentence_bleu([reference], prediction)


def compute_rouge(prediction, reference, scorer):
    scores = scorer.score(reference, prediction)
    return scores["rougeL"].fmeasure  # Example using ROUGE-L


def load_hotpotqa_dataset(dataset_name, split=None):
    if dataset_name.lower() == "hotpotqa":
        return load_dataset("hotpot_qa", "fullwiki", split=split)
    else:
        raise ValueError("Unsupported dataset")


def main():
    logging.info("Loading HotpotQA training dataset...")
    hotpot_data = load_hotpotqa_dataset("hotpotqa", split="train")

    # Initialize Policy and Value Networks
    embedding_size = 384  # based on all-MiniLM-L6-v2
    hidden_size = 128
    policy_network = PolicyNetwork(
        input_size=embedding_size, hidden_size=hidden_size
    ).to(device)
    value_network = ValueNetwork(input_size=embedding_size, hidden_size=hidden_size).to(
        device
    )

    # Initialize TensorBoard SummaryWriter with a unique log directory
    current_time = datetime.now().strftime("%Y%m%d-%H%M%S")
    writer = SummaryWriter(log_dir=f"logs/{current_time}")

    training_data = []

    with open("mcts_paths.txt", "w") as f:
        logging.info("Starting training data generation...")
        for idx, qa_pair in enumerate(hotpot_data.select(range(NUM_TRAINING_SAMPLES))):
            logging.info(f"Processing training sample {idx+1}/{NUM_TRAINING_SAMPLES}")
            question = qa_pair["question"]
            print(f"Question: {question}")
            ground_truth_answer = qa_pair["answer"]
            initial_state = State(question=question)
            root_node = Node(state=initial_state)

            best_child = mcts_search(
                root_node,
                policy_network,
                value_network,
                max_iterations=MAX_ITERATIONS_TRAIN,
                ground_truth_answer=ground_truth_answer,
                training_data=training_data,
                writer=writer,  # Pass the writer to MCTS search
            )
            final_state = best_child.state
            reward = final_state.compute_reward(ground_truth_answer)

            # Get action path
            action_path = get_action_path(best_child)

            # Save question and action path
            print(f"Writing question: {question}")
            f.write(f"Question: {question}\n")
            f.write(f"Action Path: {action_path}\n")
            f.write(f"Final Answer: {final_state.draft_answer}\n\n")

            # Optionally, log the final reward for each training sample
            writer.add_scalar("Training/Final_Reward", reward, idx)

        print(f"training_data: {training_data}")
        logging.info(f"Total training samples collected: {len(training_data)}")

    print(f"Training data collected: {len(training_data)}")

    # return
    # Prepare validation data
    logging.info("Loading HotpotQA validation dataset...")
    hotpot_validation_data = load_hotpotqa_dataset("hotpotqa", split="validation")
    validation_data = hotpot_validation_data.select(range(NUM_VALIDATION_SAMPLES))

    # Train the Policy and Value Networks
    logging.info("Training networks...")
    train_networks(
        policy_network,
        value_network,
        training_data,
        validation_data,
        writer,  # Pass the writer to the training function
        num_epochs=5,
        batch_size=32,
    )

    # Prepare test data
    logging.info("Loading HotpotQA test dataset...")
    test_data = hotpot_validation_data.select(
        range(NUM_TEST_SAMPLES)
    )  # Adjust if you have a separate test split

    # Evaluate on the test set
    logging.info("Starting evaluation on test set...")
    accuracy = evaluate_model(
        policy_network,
        value_network,
        test_data,
        writer,  # Pass the writer to the evaluation function
    )
    logging.info(f"Final Test Accuracy: {accuracy*100:.2f}%")

    # Optionally, log the final test accuracy
    writer.add_scalar("Test/Final_Accuracy", accuracy, 0)

    # Close the TensorBoard writer
    writer.close()


if __name__ == "__main__":
    main()
