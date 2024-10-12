import logging

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

from src.utils.llama import Llama3

# Constants
QUERIES_TO_EXPLORE = 3
MAX_DEPTH = 10
QUERY_GEN_PENALTY = 0.1
QUERY_RUN_PENALTY = 0.05
DUPLICATE_QUERY_PENALTY = 0.2
SIMILARITY_THRESHOLD = 0.8

# Device Configuration
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Initialize logging
logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)

# Initialize Models
embedding_model = SentenceTransformer("all-MiniLM-L6-v2")
llama_model = Llama3(
    "/home/cpp/jerryhuang/search2024/meta-llama/Meta-Llama-3-8B-Instruct"
)
tokenizer = llama_model.get_tokenizer()


class State:
    def __init__(self, question, draft_answer="", previous_queries=None, depth=0):
        self.question = question
        self.draft_answer = draft_answer
        self.previous_queries = previous_queries or []
        self.depth = depth
        self.available_queries = []

    def generate_queries(self):
        prompt = f"""
        Given the question: "{self.question}"
        And the current draft answer: "{self.draft_answer}"
        Generate {QUERIES_TO_EXPLORE} unique and relevant queries to gather more information.
        Format your response as a Python list of strings, like this:
        ["query1", "query2", "query3"]
        """
        response = llama_model.get_llama_response(prompt)
        try:
            new_queries = eval(response)
            if not isinstance(new_queries, list) or not all(
                isinstance(q, str) for q in new_queries
            ):
                raise ValueError("Invalid response format")
            self.available_queries = [
                q for q in new_queries if q not in self.previous_queries
            ]
        except:
            logging.error(f"Failed to parse LLM response: {response}")
            self.available_queries = []

    def run_query(self, query):
        # Simulated retrieval function
        retrieved_docs = ["Sample retrieved document for query: " + query]

        prompt = f"""
        Question: {self.question}
        Current draft answer: {self.draft_answer}
        New information from query "{query}": {retrieved_docs[0]}
        Please revise the draft answer based on this new information.
        """
        revised_answer = llama_model.get_llama_response(prompt)

        new_state = State(
            question=self.question,
            draft_answer=revised_answer,
            previous_queries=self.previous_queries + [query],
            depth=self.depth + 1,
        )
        return new_state

    def is_terminal(self):
        return self.depth >= MAX_DEPTH

    def get_possible_actions(self):
        actions = ["generate_queries", "submit_answer"]
        actions.extend([f"run_query: {q}" for q in self.available_queries])
        return actions

    def take_action(self, action):
        if action == "generate_queries":
            self.generate_queries()
            return self
        elif action == "submit_answer":
            return self  # Terminal state
        elif action.startswith("run_query:"):
            query = action.split(": ", 1)[1]
            return self.run_query(query)
        else:
            raise ValueError(f"Unknown action: {action}")

    def compute_reward(self, ground_truth_answer):
        similarity = cosine_similarity(
            [embedding_model.encode(self.draft_answer)],
            [embedding_model.encode(ground_truth_answer)],
        )[0][0]

        query_penalty = len(self.previous_queries) * QUERY_RUN_PENALTY
        duplicate_penalty = (
            sum(self.previous_queries.count(q) - 1 for q in set(self.previous_queries))
            * DUPLICATE_QUERY_PENALTY
        )

        return similarity - query_penalty - duplicate_penalty

    def to_tensor(self):
        state_repr = f"Question: {self.question}\nDraft Answer: {self.draft_answer}\nPrevious Queries: {', '.join(self.previous_queries)}"
        return torch.tensor(embedding_model.encode(state_repr), dtype=torch.float32)


class Node:
    def __init__(self, state, parent=None, action=None):
        self.state = state
        self.parent = parent
        self.children = []
        self.action = action
        self.visit_count = 0
        self.total_value = 0
        self.prior_probability = 0

    def is_fully_expanded(self):
        return len(self.children) == len(self.state.get_possible_actions())

    def best_child(self, c_param=1.4):
        choices_weights = [
            (c.total_value / c.visit_count)
            + c_param
            * c.prior_probability
            * (np.sqrt(self.visit_count) / (1 + c.visit_count))
            for c in self.children
        ]
        return self.children[np.argmax(choices_weights)]

    def expand(self, policy_network):
        action = self.state.get_possible_actions()[len(self.children)]
        child_state = self.state.take_action(action)
        child_node = Node(child_state, parent=self, action=action)

        state_tensor = self.state.to_tensor().unsqueeze(0).to(device)
        action_tensor = (
            torch.tensor(embedding_model.encode(action)).unsqueeze(0).to(device)
        )
        with torch.no_grad():
            child_node.prior_probability = torch.exp(
                policy_network(state_tensor, action_tensor)
            ).item()

        self.children.append(child_node)
        return child_node


class PolicyNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(PolicyNetwork, self).__init__()
        self.fc1 = nn.Linear(
            input_size * 2, hidden_size
        )  # Concatenate state and action embeddings
        self.fc2 = nn.Linear(hidden_size, 1)  # Output a single score for each action
        self.dropout = nn.Dropout(p=0.1)

    def forward(self, state_embeddings, action_embeddings):
        # state_embeddings: (batch_size, input_size)
        # action_embeddings: (batch_size, input_size)
        combined = torch.cat(
            [state_embeddings, action_embeddings], dim=1
        )  # (batch_size, 2 * input_size)
        x = F.relu(self.fc1(combined))
        x = self.dropout(x)
        x = self.fc2(x)  # (batch_size, 1)
        output = F.log_softmax(x, dim=1).squeeze(1)  # (batch_size,)
        return output


class ValueNetwork(nn.Module):
    def __init__(self, input_size, hidden_size):
        super(ValueNetwork, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.dropout = nn.Dropout(p=0.1)
        self.fc2 = nn.Linear(hidden_size, 1)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.dropout(x)
        return self.fc2(x)


def mcts_search(
    root, policy_network, value_network, num_simulations, ground_truth_answer
):
    for _ in range(num_simulations):
        node = root
        search_path = [node]

        # Selection
        while node.is_fully_expanded() and not node.state.is_terminal():
            node = node.best_child()
            search_path.append(node)

        # Expansion
        if not node.state.is_terminal():
            node = node.expand(policy_network)
            search_path.append(node)

        # Simulation
        state_tensor = node.state.to_tensor().unsqueeze(0).to(device)
        with torch.no_grad():
            value_estimate = value_network(state_tensor).item()

        # Backpropagation
        for node in reversed(search_path):
            node.visit_count += 1
            node.total_value += value_estimate

    return root.best_child(c_param=0)


def train_networks(
    policy_network,
    value_network,
    training_data,
    validation_data,
    writer,
    num_epochs=5,
    batch_size=32,
):
    policy_optimizer = torch.optim.Adam(policy_network.parameters(), lr=1e-4)
    value_optimizer = torch.optim.Adam(value_network.parameters(), lr=1e-4)
    policy_criterion = nn.MSELoss()
    value_criterion = nn.MSELoss()

    for epoch in range(num_epochs):
        policy_network.train()
        value_network.train()
        total_policy_loss = 0
        total_value_loss = 0

        for i in range(0, len(training_data), batch_size):
            batch = training_data[i : i + batch_size]
            state_tensors = torch.stack(
                [data["state"].to_tensor() for data in batch]
            ).to(device)
            action_tensors = torch.stack(
                [embedding_model.encode(data["action"]) for data in batch]
            ).to(device)
            rewards = torch.tensor(
                [data["reward"] for data in batch], dtype=torch.float32
            ).to(device)

            # Train Policy Network
            policy_optimizer.zero_grad()
            policy_output = policy_network(state_tensors, action_tensors)
            policy_loss = policy_criterion(policy_output, rewards)
            policy_loss.backward()
            policy_optimizer.step()

            # Train Value Network
            value_optimizer.zero_grad()
            value_output = value_network(state_tensors).squeeze(1)
            value_loss = value_criterion(value_output, rewards)
            value_loss.backward()
            value_optimizer.step()

            total_policy_loss += policy_loss.item()
            total_value_loss += value_loss.item()

        avg_policy_loss = total_policy_loss / (len(training_data) / batch_size)
        avg_value_loss = total_value_loss / (len(training_data) / batch_size)
        writer.add_scalar("Loss/Policy", avg_policy_loss, epoch)
        writer.add_scalar("Loss/Value", avg_value_loss, epoch)

        # Validation
        accuracy = evaluate_model(
            policy_network, value_network, validation_data, writer, epoch
        )
        writer.add_scalar("Accuracy/Validation", accuracy, epoch)

        logging.info(
            f"Epoch {epoch+1}/{num_epochs}, Policy Loss: {avg_policy_loss:.4f}, Value Loss: {avg_value_loss:.4f}, Validation Accuracy: {accuracy:.4f}"
        )


def evaluate_model(policy_network, value_network, eval_data, writer, epoch):
    policy_network.eval()
    value_network.eval()
    correct = 0
    total = 0
    rouge_scorer_obj = rouge_scorer.RougeScorer(
        ["rouge1", "rouge2", "rougeL"], use_stemmer=True
    )

    with torch.no_grad():
        for data in eval_data:
            question = data["question"]
            ground_truth = data["answer"]
            initial_state = State(question)
            root = Node(initial_state)

            final_node = mcts_search(
                root,
                policy_network,
                value_network,
                num_simulations=50,
                ground_truth_answer=ground_truth,
            )
            final_answer = final_node.state.draft_answer

            # Compute metrics
            bleu_score = sentence_bleu([ground_truth.split()], final_answer.split())
            rouge_scores = rouge_scorer_obj.score(ground_truth, final_answer)
            rouge_l = rouge_scores["rougeL"].fmeasure

            similarity = cosine_similarity(
                [embedding_model.encode(final_answer)],
                [embedding_model.encode(ground_truth)],
            )[0][0]

            writer.add_scalar("Metrics/BLEU", bleu_score, total)
            writer.add_scalar("Metrics/ROUGE-L", rouge_l, total)
            writer.add_scalar("Metrics/Similarity", similarity, total)

            if similarity > 0.8:  # You can adjust this threshold
                correct += 1
            total += 1

    accuracy = correct / total
    return accuracy


def collect_training_data(
    question, ground_truth, policy_network, value_network, num_simulations
):
    initial_state = State(question)
    root = Node(initial_state)
    training_data = []

    for _ in range(num_simulations):
        node = root
        while not node.state.is_terminal():
            if not node.is_fully_expanded():
                new_node = node.expand(policy_network)
                state_tensor = new_node.state.to_tensor().unsqueeze(0).to(device)
                with torch.no_grad():
                    value_estimate = value_network(state_tensor).item()
                reward = new_node.state.compute_reward(ground_truth)
                training_data.append(
                    {
                        "state": new_node.state,
                        "action": new_node.action,
                        "reward": reward,
                        "value_estimate": value_estimate,
                    }
                )
                break
            else:
                node = node.best_child()

        # Backpropagate
        while node is not None:
            node.visit_count += 1
            node.total_value += reward
            node = node.parent

    return training_data


def main():
    # Load dataset
    dataset = load_dataset(
        "hotpot_qa", "fullwiki", split="train[:1000]"
    )  # Adjust split as needed

    # Initialize networks
    input_size = 384  # This should match the size of your sentence embeddings
    hidden_size = 256
    policy_network = PolicyNetwork(input_size, hidden_size).to(device)
    value_network = ValueNetwork(input_size, hidden_size).to(device)

    # Initialize TensorBoard writer
    writer = SummaryWriter()

    # Collect training data
    all_training_data = []
    for example in dataset:
        question = example["question"]
        answer = example["answer"]
        training_data = collect_training_data(
            question, answer, policy_network, value_network, num_simulations=10
        )
        all_training_data.extend(training_data)

    # Split data into training and validation sets
    split = int(0.8 * len(all_training_data))
    train_data = all_training_data[:split]
    val_data = all_training_data[split:]

    # Train networks
    train_networks(
        policy_network,
        value_network,
        train_data,
        val_data,
        writer,
        num_epochs=10,
        batch_size=32,
    )

    # Final evaluation
    test_dataset = load_dataset("hotpot_qa", "fullwiki", split="validation[:100]")  #
