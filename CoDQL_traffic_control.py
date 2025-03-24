
import os
import numpy as np
import torch
import torch.nn as nn
import torch.optim as optim
from collections import deque
import random
import gym
from gym import spaces
import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

class QNetwork(nn.Module):
    def __init__(self, state_dim, action_dim, hidden_dim=128):
        super().__init__()
        self.network = nn.Sequential(
            nn.Linear(state_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Linear(hidden_dim, action_dim)
        )

    def forward(self, state):
        return self.network(state)

class CoDQLAgent:
    def __init__(self, state_dim, action_dim, neighbors, device='cpu', alpha=0.6, gamma=0.99, tau=0.01):
        self.state_dim = state_dim
        self.action_dim = action_dim
        self.device = device
        self.alpha = alpha  # Reward allocation factor
        self.gamma = gamma
        self.tau = tau
        self.neighbors = neighbors  # Neighboring agent indices

        # Q Networks
        self.q_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network = QNetwork(state_dim, action_dim).to(device)
        self.target_network.load_state_dict(self.q_network.state_dict())

        # Optimizer and Replay Buffer
        self.optimizer = optim.Adam(self.q_network.parameters(), lr=1e-3)
        self.replay_buffer = deque(maxlen=100000)
        self.batch_size = 64

    def select_action(self, state, epsilon):
        if random.random() < epsilon:
            return random.randint(0, self.action_dim - 1)
        state_tensor = torch.FloatTensor(state).unsqueeze(0).to(self.device)
        with torch.no_grad():
            q_values = self.q_network(state_tensor)
        return q_values.argmax().item()

    def store_transition(self, state, action, reward, next_state, done):
        self.replay_buffer.append((state, action, reward, next_state, done))

    def train(self):
        if len(self.replay_buffer) < self.batch_size:
            return None

        # Sample batch
        batch = random.sample(self.replay_buffer, self.batch_size)
        states, actions, rewards, next_states, dones = zip(*batch)
        states = torch.FloatTensor(states).to(self.device)
        actions = torch.LongTensor(actions).to(self.device)
        rewards = torch.FloatTensor(rewards).to(self.device)
        next_states = torch.FloatTensor(next_states).to(self.device)
        dones = torch.FloatTensor(dones).to(self.device)

        # Q-value updates
        current_q = self.q_network(states).gather(1, actions.unsqueeze(1)).squeeze()
        with torch.no_grad():
            next_q = self.target_network(next_states).max(1)[0]
        target_q = rewards + (1 - dones) * self.gamma * next_q

        # Update Q network
        loss = nn.MSELoss()(current_q, target_q)
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

        # Soft update target network
        for target_param, param in zip(self.target_network.parameters(), self.q_network.parameters()):
            target_param.data.copy_(self.tau * param.data + (1 - self.tau) * target_param.data)

        return loss.item()

class TrafficEnvironment(gym.Env):
    def __init__(self, num_agents=4, max_queue=10):
        super().__init__()
        self.num_agents = num_agents
        self.state_dim = 2  # [queue length, average wait time]
        self.action_dim = 2  # [switch phase, keep phase]
        self.max_queue = max_queue

        # Observation and Action Spaces
        self.observation_space = spaces.Box(low=0, high=max_queue, shape=(self.state_dim,), dtype=np.float32)
        self.action_space = spaces.Discrete(self.action_dim)

        self.state = np.zeros((num_agents, self.state_dim))
        self.timestep = 0

    def reset(self):
        self.state = np.zeros((self.num_agents, self.state_dim))
        self.timestep = 0
        return self.state

    def step(self, actions):
        rewards = []
        next_state = self.state.copy()

        for i, action in enumerate(actions):
            if action == 0:  # Switch phase
                next_state[i, 0] = max(0, self.state[i, 0] - 1)  # Reduce queue length
                next_state[i, 1] = max(0, self.state[i, 1] - 0.5)  # Reduce wait time
            else:  # Keep phase
                next_state[i, 0] = min(self.max_queue, self.state[i, 0] + 1)
                next_state[i, 1] = min(self.max_queue, self.state[i, 1] + 0.5)

            rewards.append(-next_state[i, 0] - 0.1 * next_state[i, 1])  # Reward based on congestion reduction

        self.state = next_state
        self.timestep += 1
        done = self.timestep >= 100  # Simulation ends after 100 timesteps
        return self.state, rewards, done, {}

def train_agents(num_episodes=500, num_agents=4):
    env = TrafficEnvironment(num_agents=num_agents)
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    agents = [CoDQLAgent(env.state_dim, env.action_dim, neighbors=list(range(num_agents)), device=device) for _ in range(num_agents)]

    epsilon = 1.0
    epsilon_decay = 0.995
    epsilon_min = 0.01

    for episode in range(num_episodes):
        states = env.reset()
        total_rewards = np.zeros(num_agents)
        done = False

        while not done:
            actions = [agent.select_action(state, epsilon) for agent, state in zip(agents, states)]
            next_states, rewards, done, _ = env.step(actions)

            for i, agent in enumerate(agents):
                agent.store_transition(states[i], actions[i], rewards[i], next_states[i], done)
                total_rewards[i] += rewards[i]

            for agent in agents:
                agent.train()

            states = next_states

        epsilon = max(epsilon_min, epsilon * epsilon_decay)
        logger.info(f"Episode {episode + 1}/{num_episodes}, Total Rewards: {total_rewards.sum()}")

    return agents

if __name__ == "__main__":
    trained_agents = train_agents()
    logger.info("Training complete.")
