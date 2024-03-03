import numpy as np
import random
import pickle

start_x = 0
start_y = 0
x_dim = 8
y_dim = 8
#gridworld = np.array((x_dim, y_dim))
start = (start_x, start_y)
goal = (x_dim - 1, y_dim - 1)
global_count = 0
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
        new_state = (state[0] + 1, state[1])
    elif action == "right":
        new_state = (state[0] - 1, state[1])
    if is_legal(new_state):
        return new_state
    return state

class Agent:
    def __init__(self):
        self.states = [] 
        self.actions = ["up", "down", "left", "right"]
        self.state = start
        self.lr = 0.2
        self.exp_rate = 0.3
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

    def play(self, rounds, global_count):
        i = 0
        global_count_temp = global_count
        while i < rounds:
            action = self.chooseAction()
            next_state = make_move(self.state, action)
            next_action = self.actions[np.argmax([self.Q_values[next_state][a] for a in self.actions])]
            self.Q_values[self.state][action] = self.Q_values[self.state][action] + self.lr * (self.decay_gamma * self.Q_values[next_state][next_action] - self.Q_values[self.state][action])
            self.states.append([self.state, action, self.Q_values])
            self.state = next_state
            if self.state == goal:
                file_name = "data_agent_" + str(global_count_agent) + ".pkl"
                cumulative_data = load_data(file_name)
                cumulative_data += self.states
                with open(file_name, 'wb') as file:
                    pickle.dump(cumulative_data, file)
                self.states = []
                self.state = start
                i += 1
                print(f"Round {str(i)} complete")
            global_count_temp += 1
        return global_count_temp

# while (global_count < 1000000):
#     new_agent = Agent()
#     print(f"Agent {str(global_count_agent)}'s turn")
#     print(f"Global count: {str(global_count)}")
#     global_count = new_agent.play(50, global_count)
#     global_count_agent += 1

test = load_data("data_agent_0.pkl")
print(test)

            
