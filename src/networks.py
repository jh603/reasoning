import torch
import torch.nn as nn
import torch.nn.functional as F


# Define the Policy Network
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


# Define the Value Network
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
