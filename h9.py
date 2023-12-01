import numpy as np
from termcolor import colored

# Environment size
rows = 7
cols = 10

# Wind strength
wind = [0, 0, 0, 1, 1, 1, 2, 2, 1, 0]

# Actions: up, down, left, right
actions = [(0, 1), (0, -1), (-1, 0), (1, 0)]
n_actions = len(actions)

# Initialize Q-table with zeros
Q = np.zeros((rows, cols, n_actions))

# Learning parameters
alpha = 0.5
gamma = 0.95
epsilon = 0.1
episodes = 5000

# Start and goal states
start = (3, 0)
goal = (3, 7)

def print_grid(state):
    grid = ""
    for i in range(rows):
        for j in range(cols):
            if (i, j) == state:
                grid += colored('1 ', 'red')
            else:
                grid += '0 '
        grid += '\n'
    print(grid)

for _ in range(episodes):
    # Initialize state
    state = start

    while state != goal:
        print_grid(state)

        # Choose action
        if np.random.uniform(0, 1) < epsilon:
            action = np.random.choice(n_actions)  # Explore
        else:
            action = np.argmax(Q[state])  # Exploit

        # Get next state
        next_state = (state[0] + actions[action][0], state[1] + actions[action][1])
        next_state = (max(min(next_state[0], rows - 1), 0), max(min(next_state[1], cols - 1), 0))
        next_state = (next_state[0] - wind[next_state[1]], next_state[1])
        next_state = (max(min(next_state[0], rows - 1), 0), next_state[1])

        # Update Q-value
        reward = -1
        Q[state][action] = Q[state][action] + alpha * (reward + gamma * np.max(Q[next_state]) - Q[state][action])

        # Update current state
        state = next_state

# Determine policy
policy = np.argmax(Q, axis=2)

print("Policy:")
print(policy)
