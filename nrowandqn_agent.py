import gymnasium as gym
import numpy as np
import matplotlib
import matplotlib.pyplot as plt
import random
import torch
from torch import nn
import yaml
from experience_replay import ReplayMemory
from nrowandqn import NROWANDQN
from datetime import datetime, timedelta
import argparse
import itertools
import os

# Print date and time
DATE_FORMAT = "%m-%d %H:%M:%S"

# Directory for saving run info
RUNS_DIR = 'nrowandqn_runs'
os.makedirs(RUNS_DIR, exist_ok=True)

# Use "Agg" backend to generate plots as images
matplotlib.use('Agg')

device = 'cuda' if torch.cuda.is_available() else 'cpu'
device = 'cpu' # force cpu

class Agent:
    def __init__(self, hyperparameter_set):
        # Load hyperparameters
        with open('hyperparameters.yml', 'r') as file:
            all_hyperparameter_sets = yaml.safe_load(file)
            hyperparameters = all_hyperparameter_sets[hyperparameter_set]

        self.hyperparameter_set = hyperparameter_set

        self.env_id = hyperparameters['env_id']
        self.env_make_params = hyperparameters.get('env_make_params', {})  # Add this line
        self.env = gym.make(self.env_id, **self.env_make_params)

        # Define hyperparameters
        self.learning_rate_a = hyperparameters['learning_rate_a']
        self.replay_memory_size = hyperparameters['replay_memory_size']
        self.mini_batch_size = hyperparameters['mini_batch_size']
        self.discount_factor_g = hyperparameters['discount_factor_g']
        self.network_sync_rate = hyperparameters['network_sync_rate']
        self.initial_sigma = hyperparameters['initial_sigma']
        self.final_k_factor = hyperparameters['final_k_factor']
        self.adam_epsilon = hyperparameters['adam_epsilon']
        self.adam_beta1 = hyperparameters['adam_beta1']
        self.adam_beta2 = hyperparameters['adam_beta2']
        self.epsilon = hyperparameters['epsilon_init']
        self.epsilon_decay = hyperparameters['epsilon_decay']
        self.epsilon_min = hyperparameters['epsilon_min']


        # declare policy
        self.state_dim = self.env.observation_space.shape[0]
        self.action_dim = self.env.action_space.n

        # Logging and files
        self.LOG_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.log')
        self.MODEL_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.pt')
        self.GRAPH_FILE = os.path.join(RUNS_DIR, f'{hyperparameter_set}.png')
        self.k_factor = 1.0  # Online weight adjustment factor

        # Initialize networks and memory
        self.network = NROWANDQN(self.state_dim, self.action_dim,self.initial_sigma).to(device)
        self.target_network = NROWANDQN(self.state_dim, self.action_dim, self.initial_sigma).to(device)
        self.target_network.load_state_dict(self.network.state_dict())
        self.memory = ReplayMemory(self.replay_memory_size)

        # Optimizer
        self.optimizer = torch.optim.Adam(
            self.network.parameters(),
            lr=self.learning_rate_a,
            betas=(self.adam_beta1, self.adam_beta2),
            eps=self.adam_epsilon
        )


    def select_action(self, state):
        if random.random() < self.epsilon:
            return self.env.action_space.sample()
        else:
            state = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                return self.network(state).argmax().item()


    def apply_gaussian_noise(self):
        self.network.apply_gaussian_noise()

    def adjust_weights(self, total_reward, threshold=10):
        if total_reward > threshold:
            self.k_factor = max(self.k_factor * 0.99, 0.1)  # Reduce to a minimum of 0.1
            print(f"Updated k_factor: {self.k_factor:.4f}")

    def train(self, episodes):
        rewards_per_episode = []

        for episode in range(episodes):
            state = self.env.reset()[0]
            total_reward = 0
            done = False

            while not done:
                action = self.select_action(state)
                next_state, reward, terminated, truncated, _ = self.env.step(action)
                
                # Update `done` to account for both terminated and truncated conditions
                done = terminated or truncated

                self.memory.append((state, action, reward, next_state, done))
                state = next_state
                total_reward += reward

                if len(self.memory) >= self.mini_batch_size:
                    self.update_network()

            self.apply_gaussian_noise()
            self.adjust_weights(total_reward)

            self.epsilon = max(self.epsilon * self.epsilon_decay, self.epsilon_min)
            rewards_per_episode.append(total_reward)

            if (episode + 1) % self.network_sync_rate == 0:
                self.target_network.load_state_dict(self.network.state_dict())
                self.save_model()  # Save the model periodically

            # Logging progress
            print(f"Episode {episode + 1}, Total Reward: {total_reward}")


        # Save the final model after training completes
        print(f"Training completed. Saving final model to {self.MODEL_FILE}")
        self.save_model()


    def update_network(self):
        states, actions, rewards, next_states, dones = zip(*self.memory.sample(self.mini_batch_size))

        states = torch.FloatTensor(states).to(device)
        actions = torch.LongTensor(actions).unsqueeze(1).to(device)
        rewards = torch.FloatTensor(rewards).to(device)
        next_states = torch.FloatTensor(next_states).to(device)
        dones = torch.FloatTensor(dones).to(device)

        # Compute Q-targets
        with torch.no_grad():
            max_next_q_values = self.target_network(next_states).max(dim=1)[0]
            q_targets = rewards + (1 - dones) * self.discount_factor_g * max_next_q_values

        # Compute Q-values from the current policy
        q_values = self.network(states).gather(1, actions).squeeze()

        # Loss computation
        loss = nn.MSELoss()(q_values, q_targets)

        # Backpropagation
        self.optimizer.zero_grad()
        loss.backward()
        self.optimizer.step()

    def run(self, render):
        # Load the trained model
        self.network.load_state_dict(torch.load(self.MODEL_FILE, weights_only=True))
        self.network.eval()  # Set the network to evaluation mode

        # Initialize the environment with render_mode if rendering is enabled
        if render:
            self.env = gym.make(self.env_id, render_mode="human")

        state = self.env.reset()[0]  # Reset the environment
        done = False
        total_reward = 0

        while not done:
            # Render the environment if specified
            if render:
                self.env.render()

            # Select action based on the trained network
            state_tensor = torch.FloatTensor(state).unsqueeze(0).to(device)
            with torch.no_grad():
                action = self.network(state_tensor).argmax().item()

            # Take the action in the environment
            next_state, reward, terminated, truncated, _ = self.env.step(action)

            # Update the state and accumulate rewards
            state = next_state
            total_reward += reward
            done = terminated or truncated

        print(f"Test Episode Total Reward: {total_reward}")
        self.env.close()  # Close the environment

    def test(self, render=False):
        env = gym.make(self.env_id, render_mode="human" if render else None, **self.env_make_params)
        num_states = env.observation_space.shape[0]
        num_actions = env.action_space.n

        policy_dqn = NROWANDQN(num_states, num_actions, self.initial_sigma).to(device)
        policy_dqn.load_state_dict(torch.load(self.MODEL_FILE))
        policy_dqn.eval()

        state, _ = env.reset()
        state = torch.tensor(state, dtype=torch.float, device=device)
        terminated = False
        total_reward = 0

        while not terminated:
            with torch.no_grad():
                action = policy_dqn(state.unsqueeze(dim=0)).squeeze().argmax()
            new_state, reward, terminated, _, _ = env.step(action.item())
            total_reward += reward
            state = torch.tensor(new_state, dtype=torch.float, device=device)

        print(f"Test Episode Total Reward: {total_reward}")
        env.close()



    def save_model(self):
        if not os.path.exists(RUNS_DIR):
            os.makedirs(RUNS_DIR)
        torch.save(self.network.state_dict(), self.MODEL_FILE)
        print(f"Model saved to {self.MODEL_FILE}")

if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Train or test NROWAN-DQN model.')
    parser.add_argument('hyperparameters', help='Hyperparameter set to use.')
    parser.add_argument('--train', help='Train the agent', action='store_true')
    parser.add_argument('--test', help='Test the agent', action='store_true')
    args = parser.parse_args()

    agent = Agent(args.hyperparameters)
    if args.train:
        agent.train(episodes=500)
    elif args.test:
        agent.test(render=True)  # Assuming `run` is your test method
    else:
        print("Please specify either --train or --test.")