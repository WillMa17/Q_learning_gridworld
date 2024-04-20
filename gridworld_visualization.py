import matplotlib.pyplot as plt
import numpy as np

def islegal(state, nrows, ncols):
    return state[0] >= 0 and state[1] >= 0 and state[0] < nrows and state[1] < ncols

def visualize_gridworld(state, actions, title="Gridworld", goal = (4, 4), dim = 9):
    nrows = ncols = dim
    grid = np.zeros((nrows, ncols))
    #actions = up, down, left, right, but for np arrays it's actually right, left, up, down
    normalized_actions = [((action - min(actions)) / (max(actions) - min(actions))) + 0.35 for action in actions]
    up_down_left_right = ((state[0], state[1] + 1), (state[0], state[1] - 1), (state[0] - 1, state[1]), (state[0] + 1, state[1]))
    for i in range(4): 
        if (islegal(up_down_left_right[i], nrows, ncols)):
            grid[up_down_left_right[i]] = normalized_actions[i]
    fig, ax = plt.subplots()
    cmap = plt.cm.Reds
    cax = ax.matshow(grid, cmap=cmap)

    fig.colorbar(cax, location='left')
    ax.plot(state[0], state[1], 'ro', label = 'Position')  
    ax.plot(goal[0], goal[1], 'bo', label = 'Goal')
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_title(title)

    plt.show()

if __name__ == '__main__':
    state = (2, 2)
    actions = [3, 2, 1, -1]
    visualize_gridworld(state, actions)