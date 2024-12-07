
import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F
import os

class NeuralNet(nn.Module):
    def __init__(self, input_dim, hidden_dim, output_dim):
        super().__init__()
        self.fc1 = nn.Linear(input_dim, hidden_dim)
        self.fc2 = nn.Linear(hidden_dim, output_dim)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = self.fc2(x)
        return x

    def save_model(self, filename='model.pth'):
        model_dir = './model'
        if not os.path.exists(model_dir):
            os.makedirs(model_dir)
        filepath = os.path.join(model_dir, filename)
        torch.save(self.state_dict(), filepath)


class QLearningTrainer:
    def __init__(self, model, learning_rate, discount_factor):
        self.learning_rate = learning_rate
        self.discount_factor = discount_factor
        self.model = model
        self.optimizer = optim.Adam(model.parameters(), lr=self.learning_rate)
        self.loss_function = nn.MSELoss()

    def perform_training_step(self, state, action, reward, next_state, done):
        state = torch.tensor(state, dtype=torch.float)
        next_state = torch.tensor(next_state, dtype=torch.float)
        action = torch.tensor(action, dtype=torch.long)
        reward = torch.tensor(reward, dtype=torch.float)

        if len(state.shape) == 1:
            state = state.unsqueeze(0)
            next_state = next_state.unsqueeze(0)
            action = action.unsqueeze(0)
            reward = reward.unsqueeze(0)
            done = (done, )

        predicted_q_values = self.model(state)
        target_q_values = predicted_q_values.clone()
        for idx in range(len(done)):
            new_q_value = reward[idx]
            if not done[idx]:
                new_q_value = reward[idx] + self.discount_factor * torch.max(self.model(next_state[idx]))
            target_q_values[idx][torch.argmax(action[idx]).item()] = new_q_value

        self.optimizer.zero_grad()
        loss = self.loss_function(target_q_values, predicted_q_values)
        loss.backward()
        self.optimizer.step()
