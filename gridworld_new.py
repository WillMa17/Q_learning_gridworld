import numpy as np
import random
import pickle
import copy 

x_dim = 9
y_dim = 9
#gridworld = np.array((x_dim, y_dim))
goal = ((x_dim - 1) / 2, (y_dim - 1) / 2)
global_count_agent = 0

def is_legal(state):
    return (state[0] >= 0) and (state[1] >= 0) and (state[0] < x_dim) and (state[1] < y_dim)

def load_data(filename):
    try:
        with open(filename, 'rb') as file:
            return pickle.load(file)
    except FileNotFoundError:
        return []

def make_move(state, action):
    new_state = None
    if action == "up":
        new_state = (state[0], state[1] + 1)
    elif action == "down":
        new_state = (state[0], state[1] - 1)
    elif action == "left":
        new_state = (state[0] - 1, state[1])
    elif action == "right":
        new_state = (state[0] + 1, state[1])
    if is_legal(new_state):
        return new_state
    return state

class Agent:
    def __init__(self, start):
        self.states = []
        self.start = start
        self.actions = ["up", "down", "left", "right"]
        self.state = start
        self.lr = 0.2
        self.exp_rate = 0.1
        self.decay_gamma = 0.9

        self.Q_values = {}
        for i in range(x_dim):
            for j in range(y_dim):
                self.Q_values[(i, j)] = {}
                for a in self.actions:
                    self.Q_values[(i, j)][a] = 0
        for a in self.actions:
            self.Q_values[goal][a] = 1000
    
    def chooseAction(self):
        if np.random.uniform(0, 1) <= self.exp_rate:
            action = np.random.choice(self.actions)
            return action
        else:
            mx_nxt_reward = float('-inf')
            potential_actions = []
            for a in self.actions:
                current_position = self.state
                nxt_reward = self.Q_values[current_position][a]
                if nxt_reward > mx_nxt_reward:
                    action = a
                    mx_nxt_reward = nxt_reward
                    potential_actions = [a]
                elif nxt_reward == mx_nxt_reward:
                    potential_actions.append(a)
            if len(potential_actions) == 1:
                return potential_actions[0]
            elif len(potential_actions) > 1:
                return potential_actions[random.randrange(len(potential_actions))]

    def play(self, rounds):
        i = 0
        cumulative_data = []
        while i < rounds:
            action = self.chooseAction()
            next_state = make_move(self.state, action)
            next_action = self.actions[np.argmax([self.Q_values[next_state][a] for a in self.actions])]
            self.Q_values[self.state][action] = self.Q_values[self.state][action] + self.lr * (-1 + self.decay_gamma * self.Q_values[next_state][next_action] - self.Q_values[self.state][action])
            self.states.append([(self.state, {'up': self.Q_values[self.state]['up'], 'down': self.Q_values[self.state]['down'], 'left': self.Q_values[self.state]['left'], 'right': self.Q_values[self.state]['right']}), action])
            #print(self.states[0])
            #print("------------------------------")
            self.state = next_state
            if self.state == goal:
                cumulative_data.append(self.states)
                self.states = []
                self.state = self.start
                i += 1
        file_name = "gridworld_agent_data_temp/data_agent_" + str(global_count_agent) + ".pkl"
        with open(file_name, 'wb') as file:
            pickle.dump(cumulative_data, file)
"""
for i in range(0, 10000):
    start = (0, 0)
    start_decider = i // 2500
    if start_decider == 1:
        start = (8, 8)
    elif start_decider == 2:
        start = (8, 0)
    elif start_decider == 3: 
        start = (0, 8)
    new_agent = Agent(start)
    #print(f"Agent {str(global_count_agent)}'s turn")
    new_agent.play(200)
    global_count_agent += 1
"""

test = load_data("gridworld_agent_data_temp/data_agent_0.pkl")
print(len(test))
print(test[0][0])
print(test[0])

            
