from typing import Dict, Any, Tuple
from torchvision import transforms
from torch import nn
from torch import optim
from torch.nn import MSELoss
import torch.nn.functional as F
import os
import numpy as np
import numpy.typing as npt
import torch
import random
import pickle
from env import Connect4Game

def get_possible_locations(board):
    locations = []
    for col in range(7):
        for row in range(5, -1, -1):
            if board[row][col] == 0 or row == 0:
                locations.append((row, col))
                break
    return locations

def get_valid_locations(board):
    locations = []
    for col in range(7):
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                locations.append((row, col))
                break
    return locations

def get_random_move(board):
    locations = []
    for col in range(7):
        for row in range(5, -1, -1):
            if board[row][col] == 0:
                locations.append(col)
                break
    return random.choice(locations)

class DQNAgent():
    def __init__(
            self, 
            env, 
            params: Dict[str, Any],
    ):        
        self.N = params["N"]
        self.M = params["M"]
        self.gamma = params["gamma"]
        self.device = params["device"]
        self.epsilon = params["epsilon"]
        self.best_model = params["best_model"]
        self.learning_rate = params["learning_rate"]
        self.minibatch_size = params["minibatch_size"]
        self.player = params["player"]

        self.env = env
        self.dqn = DQN().to(self.device)
        self.replay_buffer = ReplayMemory(
            capacity=self.N,
            device=self.device,
        )
        self.optimizer = optim.AdamW(
            self.dqn.parameters(),
            lr=self.learning_rate,
            betas=(0.9, 0.95),
        )
        self.criterion = MSELoss().to(self.device)
        self.history = []

    def get_action(self, random_enabled) -> int:
        # TODO:
        if random_enabled and random.random() < self.epsilon:
            action = self.env.action_space_sample()
        else:
            possible_actions = [xy[1] for xy in get_valid_locations(self.env.state.board)]
            output = self.dqn(torch.tensor(self.env.state.board))
            # print(output.shape)
            # print(possible_actions)
            mask = torch.full(output.shape, float('-inf')).to('cuda')
            mask[0][possible_actions] = output[0][possible_actions]
            action = torch.argmax(mask).item()
            # print(action)
            # print(self.env.state.board)
        return action

    def update_values(self, curr_state, next_state, action, reward, done):
        self.replay_buffer.push(torch.tensor(curr_state), action, reward, torch.tensor(next_state), done)
        minibatch = self.replay_buffer.sample(self.minibatch_size)
        outputs, labels = [], []
        for sample in minibatch:
            target = sample["reward"]
            target += 0 if sample["done"] else self.gamma * torch.max(self.dqn(sample["next_state"])).item() 
            output = self.dqn(sample["current_state"])[0][sample["action"]]
            outputs.append(output)
            labels.append(target)
        outputs = torch.stack(outputs).to(self.device)
        labels = torch.tensor(labels).to(self.device)
        loss = self.criterion(outputs, labels)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        stored_q_vals = {}
        q_values = self.dqn(torch.tensor(curr_state))[0]
        corresponding_locations = get_possible_locations(curr_state)
        for i in range(7):
            stored_q_vals[corresponding_locations[i]] = q_values[i]
        self.history.append((get_valid_locations(curr_state), stored_q_vals, corresponding_locations[action]))

    def update_env(self, env):
        self.env = env
    
    def reset(self, env, i, to_log):
        self.env = env
        if to_log:
            file_path = os.path.join("connect_4_data", "data_agent_" + str(self.player) + "_game_" + str(i) + ".pkl")
            with open(file_path, 'wb') as file:
                pickle.dump(self.history, file)
        self.history = []

class DQN(nn.Module):
    
    def __init__(self):
        super(DQN, self).__init__()
        self.conv1 = nn.Conv2d(1, 128, kernel_size=(2, 2), stride=1, padding=0) 
        self.fc1 = nn.Linear(128 * 5 * 6, 64)  # Adjusted for the output size of the convolution
        self.fc2 = nn.Linear(64, 64)
        self.output = nn.Linear(64, 7)  # Assuming 7 possible actions (one for each column)

    def forward(self, x):
        x = x.type(torch.float32)
        x = x.unsqueeze(0)
        x = x.to('cuda')
        x = F.relu(self.conv1(x))
        x = x.view(-1, 128 * 5 * 6)  # Flatten the output from conv layer
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = self.output(x)
        return x
    

class ReplayMemory():

    def __init__(self, capacity: int, device: str = "cpu"):
        self.capacity = capacity
        self.memory = []
        self.position = 0
        self.device = device

    def push(
            self,
            state: torch.Tensor, 
            action: int, 
            reward: float, 
            next_state: torch.Tensor, 
            done: bool
    ):
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = {
            "current_state": state.to(self.device),
            "action": action,
            "reward": reward,
            "next_state": next_state.to(self.device),
            "done": done,
        }
        self.position = (self.position + 1) % self.capacity

    def sample(
            self, 
            batch_size: int,
    ):
        return random.sample(self.memory, min(batch_size, len(self.memory)))
    
    def __len__(
            self
    ):
        return len(self.memory)
    
    