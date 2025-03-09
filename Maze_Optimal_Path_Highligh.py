import random
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as patches
from tqdm import tqdm
from matplotlib.animation import FuncAnimation

class Maze():
    def __init__(self):
        self.maze_size = 10
        self.start_space = (9, 0)
        self.terminal_space = (0, 9)
        self.state = self.start_space
        self.actions = [(-1, 0), (1, 0), (0, -1), (0, 1)]  # Down, Up, Left, Right
        self.obstacles = [(0, 3), (1, 1), (1, 5), (1, 6), (1, 7), (1, 8), (1, 9), (2, 1), (2, 2),
                          (2, 3), (2, 5), (3, 3), (3, 5), (3, 6), (3, 7), (3, 8), (4, 0), (4, 1),
                          (4, 3), (4, 4), (4, 5), (4, 8), (5, 1), (5, 5), (5, 6), (6, 1), (6, 2),
                          (6, 3), (6, 6), (6, 8), (7, 3), (7, 5), (7, 6), (7, 8), (8, 1), (8, 2),
                          (8, 3), (8, 6), (8, 7), (8, 8)]  # obstacles
    
    def reset(self):
        self.state = self.start_space
        return self.state

    def step(self, action):  # Action is (0 Down, 1 Up, 2 Left, 3 Right)
        next_state = (self.state[0] + self.actions[action][0], self.state[1] + self.actions[action][1])

        if next_state[0] < 0 or next_state[0] >= self.maze_size or next_state[1] < 0 or next_state[1] >= self.maze_size:
            next_state = self.state  # stay in the same state if it hits the game grid
        
        reward = -2  # Default reward for each move
        if next_state in self.obstacles:
            next_state = self.state  # Stay in the same state if next_state is an obstacle or out of bounds
            
        if next_state == self.terminal_space:
            reward = 100
            done = True
        else:
            done = False
        self.state = next_state
        return next_state, reward, done
    
def plot_maze(env):
    fig, ax = plt.subplots()

    # Plot the obstacles
    for obstacle in env.obstacles:
        rect = patches.Rectangle((obstacle[1], obstacle[0]), 1, 1, linewidth=1, edgecolor='black', facecolor='gray')
        ax.add_patch(rect)

    # Plot the start state
    start_rect = patches.Rectangle((env.start_space[1], env.start_space[0]), 1, 1, linewidth=1, edgecolor='black', facecolor='blue')
    ax.add_patch(start_rect)
    plt.text(env.start_space[1] + 0.5, env.start_space[0] + 0.5, 'S', horizontalalignment='center', verticalalignment='center', fontsize=20, color='white')
    
    # Plot the terminal state
    terminal_rect = patches.Rectangle((env.terminal_space[1], env.terminal_space[0]), 1, 1, linewidth=1, edgecolor='black', facecolor='green')
    ax.add_patch(terminal_rect)
    plt.text(env.terminal_space[1] + 0.5, env.terminal_space[0] + 0.5, 'T', horizontalalignment='center', verticalalignment='center', fontsize=20, color='white')

    # Set the limits and labels
    ax.set_xlim(0, env.maze_size)
    ax.set_ylim(0, env.maze_size)
    ax.set_xticks(range(env.maze_size + 1))
    ax.set_yticks(range(env.maze_size + 1))
    ax.set_xticklabels([])
    ax.set_yticklabels([])
    ax.grid(True)
    ax.set_aspect('equal')

    return fig, ax

# Initialize the maze
maze = Maze()
action_values = np.zeros(shape=(10, 10, 4))

# Epsilon-Greedy Policy
def policy(state, epsilon=0.1):
    if random.random() < epsilon:
        return np.random.randint(4)  # Random action
    else:
        av = action_values[state]
        return np.random.choice(np.flatnonzero(av == av.max()))

# N_steps_SARSA Algorithm
def n_step_sarsa(action_values, policy, episodes, alpha=0.1, gamma=0.99, epsilon=0.1, n=8):

    fig, ax = plot_maze(maze)
    agent_circle = patches.Circle((maze.start_space[1] + 0.5, maze.start_space[0] + 0.5), 0.3, color='red')
    ax.add_patch(agent_circle)
    plt.ion()
    plt.show()
    
    for episode in tqdm(range(1, episodes + 1)):
        state = maze.reset()
        action = policy(state, epsilon)
        transitions = []
        done = False
        t = 0 # Time step

        while t - n < len(transitions):
            if not done:
                next_state, reward, done = maze.step(action)
                next_action = policy(next_state, epsilon)
                transitions.append([state, action, reward])
            
            if t >= n:
                G = (1 - done) * action_values[next_state][next_action]
                for state_t, action_t, reward_t in reversed(transitions[t-n:]):
                    G = reward_t + gamma * G
                action_values[state_t][action_t] += alpha * (G- action_values[state_t][action_t])
            
            t += 1
            state = next_state
            action = next_action

            agent_circle.set_center((state[1] + 0.5, state[0] + 0.5))
            plt.pause(0.01)

    plt.ioff()
    
n_step_sarsa(action_values, policy, 30)

def is_terminal(state):
    return state == (0, 9)

# Extract optimal policy
def optimal_policy(action_values):
    path = []
    state = maze.start_space
    policy_symbols = ['↓', '↑', '←', '→']
    
    while state != maze.terminal_space:
        action = np.argmax(action_values[state])
        path.append(policy_symbols[action])
        next_state = (state[0] + maze.actions[action][0], state[1] + maze.actions[action][1])
        state = next_state
    
    return path

def plot_optimal_path(path, obstacles, start_state, terminal_state=(0, 9)):
    grid_size = maze.maze_size
    
    fig, ax = plt.subplots()
    
    # Adjust ticks to create grid
    ax.set_xticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.set_yticks(np.arange(-0.5, grid_size, 1), minor=True)
    ax.grid(which="minor", color="black", linestyle='-', linewidth=2)
    ax.tick_params(which="minor", size=0)
    
    # Set limits and aspect
    ax.set_xlim(-0.5, grid_size - 0.5)
    ax.set_ylim(-0.5, grid_size - 0.5)
    ax.set_xticks(np.arange(grid_size))
    ax.set_yticks(np.arange(grid_size))
    ax.set_xticklabels(np.arange(grid_size))
    ax.set_yticklabels(np.arange(grid_size))
    ax.set_aspect('equal')

    # Place policy symbols in the center of each cell and color obstacles, start, and terminal state
    for i in range(grid_size):
        for j in range(grid_size):
            if (i, j) in obstacles:
                rect = plt.Rectangle([j-0.5, i-0.5], 1, 1, fill=True, color='grey')
                ax.add_patch(rect)
            elif (i, j) == terminal_state:
                rect = plt.Rectangle([j-0.5, i-0.5], 1, 1, fill=True, color='green')
                ax.add_patch(rect)
                ax.text(j, i, 'T', ha='center', va='center', fontsize=20, color='white')
            elif (i, j) == start_state:
                rect = plt.Rectangle([j-0.5, i-0.5], 1, 1, fill=True, color='blue')
                ax.add_patch(rect)
                ax.text(j, i, 'S', ha='center', va='center', fontsize=20, color='white')

    # Draw the optimal path
    state = start_state
    for symbol in path:
        action = {'↓': 0, '↑': 1, '←': 2, '→': 3}[symbol]
        next_state = (state[0] + maze.actions[action][0], state[1] + maze.actions[action][1])
        if state == terminal_state:
            break
        if state != start_state and state != terminal_state:  # Do not plot symbol at the start or terminal state
            ax.text(state[1], state[0], symbol, ha='center', va='center', fontsize=20, color='red')
        state = next_state

    ax.set_title('Optimal Path')
    plt.show()

# Extract and plot the optimal policy
policy_optimal = optimal_policy(action_values)
plot_optimal_path(policy_optimal, maze.obstacles, maze.start_space)

def animate_policy(env, policy_optimal, interval=1000):
    step = 0
    fig, ax = plot_maze(env)
    state = env.start_space
    agent_circle = patches.Circle((state[1] + 0.5, state[0] + 0.5), 0.3, color='red') 
    ax.add_patch(agent_circle)

    def update(frame):
        nonlocal state, step
        
        if step >= len(policy_optimal): # Terminal state
            state = env.start_space  # Reset to start state
            agent_circle.set_center((state[1] + 0.5, state[0] + 0.5))
            step = 0
            return

        action_symbol = policy_optimal[step]

        action = {'↓': 0, '↑': 1, '←': 2, '→': 3}[action_symbol]
        next_state = (state[0] + env.actions[action][0], state[1] + env.actions[action][1])
        agent_circle.set_center((next_state[1] + 0.5, next_state[0] + 0.5))
        state = next_state
        step += 1

    anim = FuncAnimation(fig, update, frames=range(1000), interval=interval, repeat=True)
    plt.show()
    return anim  # Return the animation object to keep it alive

anim = animate_policy(maze, policy_optimal)