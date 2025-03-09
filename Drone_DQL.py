import copy
import torch
import random
import numpy as np
from tqdm import tqdm
from torch import nn
from torch.optim import AdamW
import torch.nn.functional as F
import matplotlib.pyplot as plt

# Define the Drone Environment using NumPy
class DroneEnvironment:
    def __init__(self, space_size=20, num_obstacles=100):
        self.space_size = space_size
        self.start_position = np.array([0, 0, 0], dtype=float)
        self.goal_position = np.array([space_size, space_size, space_size], dtype=float)
        self.state = self.start_position.copy()
        self.num_obstacles = num_obstacles
        self.obstacles = self._generate_obstacles()

    def _generate_obstacles(self):
        obstacles = []
        for _ in range(self.num_obstacles):
            obstacle = np.random.uniform(0, self.space_size, 3)
            obstacles.append(obstacle)
        return np.array(obstacles)

    def reset(self):
        self.state = self.start_position.copy()
        return self.state

    def step(self, action):
        next_state = self.state + action

        # Ensure the drone stays within bounds
        next_state = np.clip(next_state, 0, self.space_size)

        # Check for collision with obstacles
        if any(np.linalg.norm(next_state - obs) < 0.1 for obs in self.obstacles):
            reward = -10
            done = True
        elif np.linalg.norm(next_state - self.goal_position) < 0.1:
            reward = 10
            done = True
        else:
            reward = -0.01
            done = False

        self.state = next_state
        return next_state, reward, done

    def render(self, ax):
        ax.clear()
        ax.set_xlim([0, self.space_size])
        ax.set_ylim([0, self.space_size])
        ax.set_zlim([0, self.space_size])

        # Set axis ticks to display 0, 10, and 20
        ax.set_xticks([0, self.space_size // 2, self.space_size])
        ax.set_yticks([0, self.space_size // 2, self.space_size])
        ax.set_zticks([0, self.space_size // 2, self.space_size])
        
        # Set minor ticks to integers
        ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
        ax.yaxis.set_minor_locator(plt.MultipleLocator(2))
        ax.zaxis.set_minor_locator(plt.MultipleLocator(2))

        ax.scatter(self.start_position[0], self.start_position[1], self.start_position[2], c='green', marker='X', s=100, label='Start')
        ax.scatter(self.goal_position[0], self.goal_position[1], self.goal_position[2], c='red', marker='X', s=100, label='Goal')
        for obs in self.obstacles:
            ax.bar3d(obs[0], obs[1], obs[2], 0.5, 0.5, 0.5, color='grey', alpha=0.8)
        ax.scatter(self.state[0], self.state[1], self.state[2], c='black', marker='o', s=100)
        
        plt.draw()

# Define the Action Selection Function
def sample_action():
    # Generate a random action in the range of [-1, 1] for each dimension
    action = np.random.uniform(-1, 1, 3)
    return action

# Create the environment
env = DroneEnvironment()
env.reset()
state_dims = 3  # Dimension for the state (3D position)
num_actions = 3  # Dimension for the action space (3D action)

# Wrapper to preprocess the environment's observations
class PreprocessEnv():
    def __init__(self, env):
        self.env = env

    def reset(self):
        obs = self.env.reset()
        return torch.from_numpy(obs).unsqueeze(dim=0).float()

    def step(self, action):
        action = action.numpy().squeeze()  # Convert tensor to numpy array
        next_state, reward, done = self.env.step(action)
        next_state = torch.from_numpy(next_state).unsqueeze(dim=0).float()
        reward = torch.tensor([reward], dtype=torch.float32).unsqueeze(0)
        done = torch.tensor([done], dtype=torch.float32).unsqueeze(0)
        return next_state, reward, done

    def render(self, ax):
        self.env.render(ax)

# Wrap the environment to preprocess observations
env = PreprocessEnv(env)

# Define the Q-network architecture
q_network = nn.Sequential(
    nn.Linear(state_dims, 128),
    nn.ReLU(),
    nn.Linear(128, 64),
    nn.ReLU(),
    nn.Linear(64, num_actions)
)

# Create a copy of the Q-network to serve as the target network
target_q_network = copy.deepcopy(q_network).eval()

# Define the epsilon-greedy policy
def policy(state, epsilon):
    if torch.rand(1).item() < epsilon:
        # Sample random actions within the range [-1, 1] for each dimension
        return torch.tensor(np.random.uniform(-1, 1, num_actions), dtype=torch.float32).unsqueeze(0)
    else:
        av = q_network(state).detach()
        return av  # Directly use the Q-values as actions

# Define the replay memory for experience replay
class ReplayMemory:
    def __init__(self, capacity=100000):
        self.capacity = capacity
        self.memory = []
        self.position = 0
    
    def insert(self, transition):
        # Insert a new transition in the memory
        if len(self.memory) < self.capacity:
            self.memory.append(None)
        self.memory[self.position] = transition
        self.position = (self.position + 1) % self.capacity
    
    def sample(self, batch_size):
        # Sample a batch of transitions from memory
        assert self.can_sample(batch_size)
        batch = random.sample(self.memory, batch_size)
        batch = zip(*batch)
        return [torch.cat(items) for items in batch]
    
    def can_sample(self, batch_size):
        # Check if there are enough samples to sample a batch
        return len(self.memory) >= batch_size
    
    def __len__(self):
        return len(self.memory)

# Define the Deep Q-Learning algorithm
def deep_q_learning(env, q_network, target_q_network, policy, episodes, alpha=0.01, batch_size=32, gamma=0.99, epsilon=0.2):
    optim = AdamW(q_network.parameters(), lr=alpha)
    memory = ReplayMemory()
    stats = {'MSE Loss': [], 'Returns': []}
    best_return = float('-inf')
    best_path = []

    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    plt.ion()
    plt.show()

    for episode in tqdm(range(1, episodes + 1)):
        state = env.reset()
        done = False
        ep_return = 0
        path = [state.numpy().flatten()]

        while not done:
            action = policy(state, epsilon)
            next_state, reward, done = env.step(action)
            memory.insert((state, action, reward, done, next_state))
            path.append(next_state.numpy().flatten())

            if memory.can_sample(batch_size):
                # Sample a batch of transitions from memory
                state_b, action_b, reward_b, done_b, next_state_b = memory.sample(batch_size)

                # Compute the Q-values for the current states
                q_values = q_network(state_b)

                # Compute the Q-values for the selected actions
                qsa_b = q_values.gather(1, action_b.argmax(dim=1, keepdim=True))

                # Compute the Q-values for the next states
                next_q_values = target_q_network(next_state_b)
                next_qsa_b = torch.max(next_q_values, dim=-1, keepdim=True)[0]

                # Compute the target Q-values
                target_b = reward_b + (1 - done_b) * gamma * next_qsa_b

                # Ensure the sizes match
                target_b = target_b.view(-1, 1)
                qsa_b = qsa_b.view(-1, 1)

                # Compute the loss
                loss = F.mse_loss(qsa_b, target_b)
                optim.zero_grad()
                loss.backward()
                optim.step()

                stats['MSE Loss'].append(loss.item())
            
            state = next_state
            ep_return += reward.item()

            # Render the environment
            env.render(ax)
            plt.pause(0.01)
        
        stats['Returns'].append(ep_return)

        # Update the target network every 10 episodes
        if episode % 10 == 0:
            target_q_network.load_state_dict(q_network.state_dict())

        # Save the best path
        if ep_return > best_return:
            best_return = ep_return
            best_path = path
    
    plt.ioff()
    return stats, best_path

# Train the Q-network using Deep Q-Learning
stats, best_path = deep_q_learning(env, q_network, target_q_network, policy, 500)

def plot_best_path(env, best_path):
    fig = plt.figure()
    ax = fig.add_subplot(111, projection='3d')
    ax.set_xlim([0, env.env.space_size])
    ax.set_ylim([0, env.env.space_size])
    ax.set_zlim([0, env.env.space_size])

    # Set axis ticks to display 0, 10, and 20
    ax.set_xticks([0, env.env.space_size // 2, env.env.space_size])
    ax.set_yticks([0, env.env.space_size // 2, env.env.space_size])
    ax.set_zticks([0, env.env.space_size // 2, env.env.space_size])
    
    # Set minor ticks to integers
    ax.xaxis.set_minor_locator(plt.MultipleLocator(2))
    ax.yaxis.set_minor_locator(plt.MultipleLocator(2))
    ax.zaxis.set_minor_locator(plt.MultipleLocator(2))
    
    # Plot start and goal points with X
    ax.scatter(env.env.start_position[0], env.env.start_position[1], env.env.start_position[2], c='green', marker='X', s=100, label='Start')
    ax.scatter(env.env.goal_position[0], env.env.goal_position[1], env.env.goal_position[2], c='red', marker='X', s=100, label='Goal')

    # Plot obstacles as 3D rectangles
    for i, obs in enumerate(env.env.obstacles):
        ax.bar3d(obs[0], obs[1], obs[2], 0.5, 0.5, 0.5, color='grey', alpha=0.8)
    # Add a legend entry for obstacles
    ax.bar3d(env.env.obstacles[0][0], env.env.obstacles[0][1], env.env.obstacles[0][2], 0.5, 0.5, 0.5, color='grey', alpha=0.8, label='Obstacles')

    # Plot the best path
    best_path = np.array(best_path)
    ax.plot(best_path[:, 0], best_path[:, 1], best_path[:, 2], color='black')
    ax.quiver(best_path[:-1, 0], best_path[:-1, 1], best_path[:-1, 2],
              best_path[1:, 0] - best_path[:-1, 0],
              best_path[1:, 1] - best_path[:-1, 1],
              best_path[1:, 2] - best_path[:-1, 2],
              color='blue', arrow_length_ratio=0.1, label='Direction')
    
    plt.legend()
    plt.show()

# Plot the best path
plot_best_path(env, best_path)
