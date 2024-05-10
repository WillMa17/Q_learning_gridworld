import os
import pickle
import numpy as np

# test = []
# test_2 = []
# file_path = os.path.join("connect_4_data", "data_agent_1_game_1.pkl")
# with open(file_path, 'rb') as file:
#     test = pickle.load(file)

# file_path_2 = os.path.join("connect_4_data", "data_agent_-1_game_1.pkl")
# with open(file_path_2, 'rb') as file_2:
#     test_2 = pickle.load(file_2)

# list = np.zeros((6, 7))
# for i in range(max(len(test), len(test_2))):
#     if i < len(test):
#         list[test[i][-1]] = 1
#     if i < len(test_2):
#         list[test_2[i][-1]] = 2

# print(list)

test = []
test_2 = []
file_path = os.path.join("connect_4_data", "data_agent_1_games_10.pkl")
with open(file_path, 'rb') as file:
    test = pickle.load(file)

file_path_2 = os.path.join("connect_4_data", "data_agent_-1_games_10.pkl")
with open(file_path_2, 'rb') as file_2:
    test_2 = pickle.load(file_2)

print(test)
print(test_2)