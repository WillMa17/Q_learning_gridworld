import matplotlib.pyplot as plt
import numpy as np

def islegal(state, nrows, ncols):
    return state[0] >= 0 and state[1] >= 0 and state[0] < nrows and state[1] < ncols

def direction(path):
    if len(path) == 1:
        return path[-1]
    distance = ((path[-1][0] - path[-2][0]) * 0.5, (path[-1][1] - path[-2][1]) * 0.5)
    return (path[-1][0] + distance[0], path[-1][1] + distance[1])

def visualize_gridworld(state, actions, path = [], title="Gridworld", goal = (4, 4), dim = 9):
    if len(path) == 0:
        path = [state]
    nrows = ncols = dim
    grid = np.zeros((nrows, ncols))
    #actions = up, down, left, right, but for np arrays it's actually right, left, up, down
    state = (state[1], state[0])
    normalized_actions = [((action - min(actions)) / (max(actions) - min(actions))) + 0.35 for action in actions]
    up_down_left_right = ((state[0] + 1, state[1]), (state[0] - 1, state[1]), (state[0], state[1] - 1), (state[0], state[1] + 1))
    state = (state[1], state[0])
    for i in range(4): 
        if (islegal(up_down_left_right[i], nrows, ncols)):
            grid[up_down_left_right[i]] = normalized_actions[i]
    fig, ax = plt.subplots()
    cmap = plt.cm.Reds
    cax = ax.matshow(grid, cmap=cmap)
    ax.xaxis.tick_bottom()
    ax.invert_yaxis()
    fig.colorbar(cax, location='left')
    ax.plot(state[0], state[1], 'ro', label = 'Position')  
    ax.plot(goal[0], goal[1], 'bo', label = 'Goal')
    ax.plot(*zip(*[(p[0], p[1]) for p in path]), 'k-', linewidth=2)  # Path
    for (m, n), (m_next, n_next) in zip(path, path[1:]):
        ax.annotate("", xy=(m_next, n_next), xytext=(m, n),
                    arrowprops=dict(arrowstyle="->, head_length=0.6, head_width=0.3", lw=1.5, color='black'))
    ax.legend(loc='upper left', bbox_to_anchor=(1, 1))
    ax.set_xticks(np.arange(-0.5, ncols, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, nrows, 1), minor=True)
    ax.grid(which='minor', color='black', linestyle='-', linewidth=1)

    # ax.set_xticks([])
    # ax.set_yticks([])
    ax.set_title(title)

    plt.show()

if __name__ == '__main__':
    state = (3, 2)
    actions = [3, 2, 1, -1]
    path = [(0, 0), (0, 1), (0, 2), (1, 2), (2, 2), (3, 2)]
    visualize_gridworld(state, actions, path = path)